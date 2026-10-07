"""
The directory of one training run: resolved config, its hash, metadata.

A run directory is `<output_root>/<experiment name>/<timestamp>-<config hash>`;
it holds `config.yaml` (resolved) and `metadata.json` (git commit and status,
host, data and feature-cache identity), written before training starts, then
`train.log`, `tensorboard/`, `fit-profile.txt` and `checkpoints/`. A run is resumed
from its directory (`turn-wm train --restore <run_dir>`).
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from turn_wm.config import upgrade_run_config
from turn_wm.data.feature_cache import FeatureCaches
from turn_wm.models.build import observation_source
from turn_wm.training.observations import feature_cache_identity


def resolved_config(cfg: DictConfig) -> dict:
    resolved = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    if not isinstance(resolved, dict):
        raise TypeError("Resolved configuration must be a mapping")

    return resolved


def hash_config(cfg: DictConfig) -> str:
    payload = json.dumps(
        resolved_config(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def create_run_dir(cfg: DictConfig, config_hash: str) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config_hash[:8]}"

    run_dir = Path(cfg.experiment.output_root) / str(cfg.experiment.name) / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_id, run_dir


def host() -> str:
    """The machine's name, recorded in the metadata (not in the run id)."""

    return socket.gethostname()


def git_metadata() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": bool(status.strip())}


def write_config(cfg: DictConfig, run_dir: Path) -> None:
    OmegaConf.save(config=cfg, f=run_dir / "config.yaml", resolve=True)


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
        "host": host(),
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
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True)
class RunRecord:
    """A training run as written to its run directory."""

    run_dir: Path
    cfg: DictConfig
    metadata: dict[str, Any]


def load_run(run_dir: Path) -> RunRecord:
    """Read a run's saved config and metadata; refuse an edited config."""

    run_dir = Path(run_dir).expanduser()
    config_path = run_dir / "config.yaml"
    metadata_path = run_dir / "metadata.json"

    for path in (config_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Not a training run directory, missing {path}")

    cfg = OmegaConf.load(config_path)

    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{config_path} must hold a mapping")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_hash = metadata.get("config_hash")

    if recorded_hash is not None and hash_config(cfg) != recorded_hash:
        raise ValueError(
            f"{config_path} does not match the config hash recorded in "
            f"{metadata_path}; it was edited after the run"
        )

    return RunRecord(run_dir=run_dir, cfg=upgrade_run_config(cfg), metadata=metadata)


def record_restore(run_dir: Path, *, checkpoint: Path) -> None:
    """Append one resumption (when, where, which code, from which checkpoint)."""

    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.setdefault("restores", []).append(
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "host": host(),
            "git": git_metadata(),
            "checkpoint": str(checkpoint),
        }
    )
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
