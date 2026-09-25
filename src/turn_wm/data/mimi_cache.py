from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MimiFeatureRecord:
    dataset: str
    recording_id: str
    path: str
    steps: int
    start_time_s: float
    start_index: int


def _recording_filename(
    dataset: str,
    recording_id: str,
) -> str:
    key = f"{dataset}\0{recording_id}".encode()

    digest = hashlib.sha256(key).hexdigest()[:24]

    return f"{digest}.safetensors"


def write_features(
    root: Path,
    *,
    dataset: str,
    recording_id: str,
    features: torch.Tensor,
) -> Path:
    """Write one recording's aligned Mimi features."""

    if features.ndim != 2:
        raise ValueError("features must have shape (time, feature_dim)")

    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")

    directory = root / dataset
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / _recording_filename(
        dataset,
        recording_id,
    )

    save_file(
        {
            "features": features.detach()
            .to(device="cpu", dtype=torch.float16)
            .contiguous()
        },
        str(path),
    )

    return path


class MimiFeatureStore:
    """Read precomputed Mimi features aligned to the action grid."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

        manifest_path = self.root / "manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Mimi feature manifest not found: {manifest_path}")

        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest["schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Mimi cache schema version: {manifest['schema_version']}"
            )

        self.metadata = manifest

        self._records = {
            (
                row["dataset"],
                row["recording_id"],
            ): MimiFeatureRecord(
                dataset=row["dataset"],
                recording_id=row["recording_id"],
                path=row["path"],
                steps=int(row["steps"]),
                start_time_s=float(row["start_time_s"]),
                start_index=int(row["start_index"]),
            )
            for row in manifest["recordings"]
        }

    def get(
        self,
        *,
        dataset: str,
        recording_id: str,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Read [start:end] from one recording without loading it all."""

        key = (dataset, recording_id)

        try:
            record = self._records[key]
        except KeyError as error:
            raise KeyError(f"No Mimi features for {key!r}") from error

        if not 0 <= start <= end <= record.steps:
            raise IndexError(
                f"Invalid feature slice [{start}:{end}] "
                f"for recording with {record.steps} steps"
            )

        path = self.root / record.path

        if not path.is_file():
            raise FileNotFoundError(f"Mimi feature file not found: {path}")

        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as file:
            features = file.get_slice("features")[start:end]

        return features


def write_manifest(
    root: Path,
    *,
    recordings: list[MimiFeatureRecord],
    model_name: str,
    model_revision: str | None,
    source_dataset_revision: str | None,
    model_resolved_revision: str | None = None,
    feature_rate_hz: float,
    feature_dim: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "resolved_revision": model_resolved_revision,
        },
        "source_dataset_revision": source_dataset_revision,
        "features": {
            "rate_hz": feature_rate_hz,
            "dim": feature_dim,
            "dtype": "float16",
            "alignment": "causal",
        },
        "recordings": [
            {
                "dataset": record.dataset,
                "recording_id": record.recording_id,
                "path": record.path,
                "steps": record.steps,
                "start_time_s": record.start_time_s,
                "start_index": record.start_index,
            }
            for record in sorted(
                recordings,
                key=lambda record: (
                    record.dataset,
                    record.recording_id,
                ),
            )
        ],
    }

    path = root / "manifest.json"

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return path
