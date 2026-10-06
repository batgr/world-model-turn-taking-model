"""
The directory of one training run: resolved config, its hash, metadata.

A run directory is `<output_root>/<model>/<timestamp>-<config hash>`; it
holds `config.yaml` (resolved) and `metadata.json` (git commit and status,
data and feature-cache identity), written before training starts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from turn_wm.data.feature_cache import (
    FeatureCaches,
)
from turn_wm.models.build import observation_source
from turn_wm.training.observations import (
    feature_cache_identity,
)


def resolved_config(cfg: DictConfig) -> dict:
    resolved = OmegaConf.to_container(
        cfg,
        resolve=True,
        enum_to_str=True,
    )

    if not isinstance(resolved, dict):
        raise TypeError("Resolved configuration must be a mapping")

    return resolved


def hash_config(cfg: DictConfig) -> str:
    payload = json.dumps(
        resolved_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def create_run_dir(
    cfg: DictConfig,
    config_hash: str,
) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config_hash[:8]}"

    run_dir = Path(cfg.experiment.output_root) / str(cfg.experiment.name) / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_id, run_dir


def git_metadata() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]

    try:
        commit = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
        }

    return {
        "commit": commit,
        "dirty": bool(status.strip()),
    }


def write_config(
    cfg: DictConfig,
    run_dir: Path,
) -> None:
    OmegaConf.save(
        config=cfg,
        f=run_dir / "config.yaml",
        resolve=True,
    )


def write_metadata(
    *,
    run_dir: Path,
    run_id: str,
    config_hash: str,
    cfg: DictConfig,
    git: dict[str, object],
    dataset_revision: str | None,
    feature_store: FeatureCaches | None = None,
) -> None:
    metadata: dict[str, object] = {
        "run_id": run_id,
        "seed": int(cfg.seed),
        "config_hash": config_hash,
        "git": git,
        "dataset": str(cfg.data.dataset),
        "dataset_revision": dataset_revision,
        "observation_source": observation_source(cfg),
    }

    if feature_store is not None:
        # What identifies the cache, not its whole manifest.
        metadata["feature_cache"] = {
            "root": str(feature_store.root),
            "caches": feature_cache_identity(feature_store),
        }

    path = run_dir / "metadata.json"

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
