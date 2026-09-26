from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE_SCHEMA_VERSION = 2
SUPPORTED_CACHE_SCHEMA_VERSIONS = frozenset({1, CACHE_SCHEMA_VERSION})


@dataclass(frozen=True)
class MimiAudioGap:
    """A true local media gap on the cache's canonical timeline."""

    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time_s) or not math.isfinite(self.end_time_s):
            raise ValueError("Audio-gap bounds must be finite")

        if self.end_time_s <= self.start_time_s:
            raise ValueError("Audio-gap end_time_s must be after start_time_s")

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass(frozen=True)
class MimiCacheExclusion:
    """A canonical recording intentionally omitted from a cache release."""

    dataset: str
    recording_id: str
    reason: str
    max_drift_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_drift_s) or self.max_drift_s < 0:
            raise ValueError("max_drift_s must be finite and non-negative")


@dataclass(frozen=True)
class MimiFeatureRecord:
    dataset: str
    recording_id: str
    path: str
    steps: int
    start_time_s: float
    start_index: int
    audio_gaps: tuple[MimiAudioGap, ...] = ()


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


def _audio_gap_from_manifest(row: dict[str, Any]) -> MimiAudioGap:
    start_time_s = float(row["start_time_s"])
    duration_s = float(row["duration_s"])
    end_time_s = float(row.get("end_time_s", start_time_s + duration_s))

    if not math.isclose(
        end_time_s - start_time_s,
        duration_s,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Audio-gap duration_s does not match end_time_s - start_time_s"
        )

    return MimiAudioGap(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    )


class MimiFeatureStore:
    """Read precomputed Mimi features aligned to the action grid."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

        manifest_path = self.root / "manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"Mimi feature manifest not found: {manifest_path}")

        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)

        schema_version = int(manifest["schema_version"])

        if schema_version not in SUPPORTED_CACHE_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported Mimi cache schema version: {schema_version}")

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
                audio_gaps=tuple(
                    _audio_gap_from_manifest(gap) for gap in row.get("audio_gaps", [])
                ),
            )
            for row in manifest["recordings"]
        }

        self._exclusions = tuple(
            MimiCacheExclusion(
                dataset=row["dataset"],
                recording_id=row["recording_id"],
                reason=row["reason"],
                max_drift_s=float(row["max_drift_s"]),
            )
            for row in manifest.get("excluded_recordings", [])
        )

    @property
    def feature_dim(self) -> int:
        return int(self.metadata["features"]["dim"])

    @property
    def feature_rate_hz(self) -> float:
        return float(self.metadata["features"]["rate_hz"])

    @property
    def dtype(self) -> str:
        return str(self.metadata["features"]["dtype"])

    @property
    def model_name(self) -> str:
        return str(self.metadata["model"]["name"])

    @property
    def model_revision(self) -> str | None:
        return self.metadata["model"].get("revision")

    @property
    def model_resolved_revision(self) -> str | None:
        return self.metadata["model"].get("resolved_revision")

    @property
    def source_dataset_revision(self) -> str | None:
        return self.metadata.get("source_dataset_revision")

    @property
    def schema_version(self) -> int:
        return int(self.metadata["schema_version"])

    @property
    def recording_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._records)

    @property
    def exclusions(self) -> tuple[MimiCacheExclusion, ...]:
        return self._exclusions

    @property
    def records(self) -> tuple[MimiFeatureRecord, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (record.dataset, record.recording_id),
            )
        )

    def record(self, *, dataset: str, recording_id: str) -> MimiFeatureRecord:
        key = (dataset, recording_id)

        try:
            return self._records[key]
        except KeyError as error:
            raise KeyError(f"No Mimi features for {key!r}") from error

    def get_by_index(
        self,
        *,
        dataset: str,
        recording_id: str,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        """Features of action-grid `decision_index` in [start_index, end_index).

        Row `k` of a recording's file is `decision_index = start_index + k`
        of its record; only the requested rows are read from disk.
        """

        record = self.record(dataset=dataset, recording_id=recording_id)
        first = record.start_index
        last = record.start_index + record.steps

        if not first <= start_index <= end_index <= last:
            raise IndexError(
                f"decision_index range [{start_index}, {end_index}) is outside "
                f"recording {recording_id!r} of {dataset!r}, which covers "
                f"[{first}, {last})"
            )

        return self.get(
            dataset=dataset,
            recording_id=recording_id,
            start=start_index - first,
            end=end_index - first,
        )

    def get(
        self,
        *,
        dataset: str,
        recording_id: str,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Read rows [start:end] of one recording's file without loading it all."""

        record = self.record(dataset=dataset, recording_id=recording_id)

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
    excluded_recordings: tuple[MimiCacheExclusion, ...] = (),
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
                "audio_gaps": [
                    {
                        "start_time_s": gap.start_time_s,
                        "end_time_s": gap.end_time_s,
                        "duration_s": gap.duration_s,
                    }
                    for gap in sorted(
                        record.audio_gaps,
                        key=lambda gap: (
                            gap.start_time_s,
                            gap.end_time_s,
                        ),
                    )
                ],
            }
            for record in sorted(
                recordings,
                key=lambda record: (
                    record.dataset,
                    record.recording_id,
                ),
            )
        ],
        "excluded_recordings": [
            {
                "dataset": exclusion.dataset,
                "recording_id": exclusion.recording_id,
                "reason": exclusion.reason,
                "max_drift_s": exclusion.max_drift_s,
            }
            for exclusion in sorted(
                excluded_recordings,
                key=lambda exclusion: (
                    exclusion.dataset,
                    exclusion.recording_id,
                    exclusion.reason,
                    exclusion.max_drift_s,
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


class MimiFeatureCaches:
    """Several Mimi caches read as one, e.g. the per-corpus caches of a release.

    Offers the reading interface of `MimiFeatureStore`; each recording is
    served by the cache that holds it.
    """

    def __init__(self, root: Path, stores: Sequence[MimiFeatureStore]) -> None:
        if not stores:
            raise ValueError(f"No Mimi cache under {root}")

        self.root = Path(root)
        self.stores = tuple(stores)
        self._by_key: dict[tuple[str, str], MimiFeatureStore] = {}

        for store in self.stores:
            for key in store.recording_keys:
                if key in self._by_key:
                    raise ValueError(f"Recording {key!r} is in several Mimi caches")

                self._by_key[key] = store

    @property
    def recording_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(self._by_key)

    @property
    def exclusions(self) -> tuple[MimiCacheExclusion, ...]:
        return tuple(e for store in self.stores for e in store.exclusions)

    def record(self, *, dataset: str, recording_id: str) -> MimiFeatureRecord:
        return self._store(dataset, recording_id).record(
            dataset=dataset, recording_id=recording_id
        )

    def get_by_index(
        self,
        *,
        dataset: str,
        recording_id: str,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        return self._store(dataset, recording_id).get_by_index(
            dataset=dataset,
            recording_id=recording_id,
            start_index=start_index,
            end_index=end_index,
        )

    def _store(self, dataset: str, recording_id: str) -> MimiFeatureStore:
        try:
            return self._by_key[(dataset, recording_id)]
        except KeyError as error:
            raise KeyError(
                f"No Mimi features for {(dataset, recording_id)!r}"
            ) from error


def store_datasets(store: MimiFeatureStore) -> frozenset[str]:
    """Corpora a cache covers: those of its recordings and exclusions."""

    return frozenset(
        {dataset for dataset, _ in store.recording_keys}
        | {exclusion.dataset for exclusion in store.exclusions}
    )


def open_mimi_cache(root: Path) -> MimiFeatureCaches:
    """Open one cache (`manifest.json`) or a release (`release_manifest.json`).

    A release root holds one cache per corpus, listed in its release manifest.
    """

    root = Path(root)

    if (root / "manifest.json").is_file():
        return MimiFeatureCaches(root, [MimiFeatureStore(root)])

    release = root / "release_manifest.json"

    if release.is_file():
        corpora = sorted(json.loads(release.read_text(encoding="utf-8"))["corpora"])

        return MimiFeatureCaches(
            root, [MimiFeatureStore(root / name) for name in corpora]
        )

    raise FileNotFoundError(
        f"No Mimi cache at {root}: expected manifest.json (one cache) or "
        "release_manifest.json (a release of per-corpus caches)"
    )


type MimiFeatures = MimiFeatureStore | MimiFeatureCaches
