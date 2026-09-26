"""
Extract the representations of a finished training run into an artifact.

Everything is rebuilt from the run directory written by
`turn_wm.training.train.run`, never from the current defaults:

1. provenance: `config.yaml` (checked against the recorded config hash) and
   `metadata.json`, the dataset at the exact recorded revision, the
   checkpoint and its SHA-256;
2. data: the run's observations (`prepare_observations`) and the split's
   dataset (`build_run_dataset`), as training built them, with the cache,
   when there is one, checked against the one the run recorded;
3. extraction: `extract_snapshot` on a seeded fixed permutation of the
   split, so any `max_samples` prefix is a uniform sample of it.

The split is `validation` unless asked otherwise; `test` is never opened
implicitly. The model is whatever the run's config and checkpoint describe,
so later experiments (other encoders, normalization, context or horizons)
need no change here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.source import DATASETS, LoadedData, load_data
from turn_wm.evaluation.latent_analysis.extract import extract_snapshot, write_snapshot
from turn_wm.models.build import observation_source
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.training.lewm import LeWMModule
from turn_wm.training.train import (
    RunObservations,
    _config_hash,
    _git_metadata,
    build_run_dataset,
    mimi_cache_identity,
    prepare_observations,
)

DEFAULT_SPLIT = "validation"
DEFAULT_CHECKPOINT = "last.ckpt"


@dataclass(frozen=True)
class RunRecord:
    """A training run as written to its run directory."""

    run_dir: Path
    cfg: DictConfig
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: JEPA
    path: Path
    sha256: str
    epoch: int | None
    global_step: int | None


def extract_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    split: str = DEFAULT_SPLIT,
    max_samples: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str = "cpu",
    mimi_cache_root: Path | None = None,
    media_roots: dict[str, Path] | None = None,
) -> Path:
    """Write the anchor representations of `split` for one run; return the dir.

    `checkpoint` is a file name under `<run_dir>/checkpoints` or a path.
    `seed` (default: the run's) fixes the sample order; `batch_size` and
    `num_workers` do not change it. `mimi_cache_root` relocates the run's
    cache, which must still be the same cache; `media_roots` defaults to the
    `<DATASET>_MEDIA_ROOT` environment variables, as in training.
    """

    # 1. Provenance: the run as it was.
    record = load_run(run_dir)
    cfg = record.cfg

    if mimi_cache_root is not None:
        cfg = copy.deepcopy(cfg)
        OmegaConf.update(cfg, "data.mimi_cache.root", str(mimi_cache_root))

    seed = int(cfg.seed) if seed is None else seed
    L.seed_everything(seed, workers=True)

    # The checkpoint first: a wrong name fails before any download.
    loaded_checkpoint = load_checkpoint(record, checkpoint)
    loaded = load_run_data(record)

    # 2. Data: the split as the run saw it, in a fixed seeded order.
    observations = prepare_run_observations(
        record, cfg, loaded, media_roots=media_roots
    )
    dataset = build_run_dataset(cfg, loaded, observations, split=split, training=False)
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=batch_size or int(cfg.loader.batch_size),
            num_workers=num_workers,
            seed=seed,
            shuffle=True,
            drop_last=False,
        ),
    )

    # 3. Extraction.
    snapshot = extract_snapshot(
        loaded_checkpoint.model,
        loader,
        max_samples=max_samples,
        device=device,
    )

    if output_dir is None:
        output_dir = (
            record.run_dir
            / "latent_analysis"
            / f"{loaded_checkpoint.path.stem}-{split}"
        )

    return write_snapshot(
        snapshot,
        output_dir,
        provenance={
            "run": {
                "run_id": record.metadata["run_id"],
                "run_dir": str(record.run_dir),
                "config_hash": record.metadata.get("config_hash"),
                "git": record.metadata.get("git"),
                "seed": int(cfg.seed),
            },
            "data": {
                "dataset": record.metadata["dataset"],
                "dataset_revision": loaded.revision,
                "split": split,
                "observation_source": observation_source(cfg),
                "modalities": list(observations.modalities),
                "feature_caches": _feature_caches(observations),
                "context_steps": int(cfg.data.context_steps),
                "future_steps": int(cfg.data.future_steps),
                # Position of the anchor inside the context window.
                "anchor_step": int(cfg.data.context_steps) - 1,
                "split_samples": len(dataset),
                "split_samples_by_corpus": dataset.corpus_sizes(),
            },
            "sampling": {
                "order": "fixed_permutation",
                "seed": seed,
                "max_samples": max_samples,
                "samples": snapshot.samples,
            },
            "checkpoint": {
                "filename": loaded_checkpoint.path.name,
                "sha256": loaded_checkpoint.sha256,
                "epoch": loaded_checkpoint.epoch,
                "global_step": loaded_checkpoint.global_step,
            },
            "extraction": {
                "git": _git_metadata(),
                "device": device,
                "precision": "float32",
            },
        },
    )


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

    if recorded_hash is not None and _config_hash(cfg) != recorded_hash:
        raise ValueError(
            f"{config_path} does not match the config hash recorded in "
            f"{metadata_path}; it was edited after the run"
        )

    return RunRecord(run_dir=run_dir, cfg=cfg, metadata=metadata)


def load_run_data(record: RunRecord) -> LoadedData:
    """The run's dataset at exactly the revision it trained on."""

    name = record.metadata["dataset"]
    revision = record.metadata.get("dataset_revision")

    if name not in DATASETS:
        raise ValueError(f"Run dataset {name!r} is not one of {sorted(DATASETS)}")

    if revision is None:
        raise ValueError(
            f"Run {record.metadata['run_id']} recorded no dataset revision; "
            "its data cannot be rebuilt exactly"
        )

    loaded = load_data(replace(DATASETS[name], revision=revision))

    if loaded.revision != revision:
        raise ValueError(
            f"Loaded dataset revision {loaded.revision} instead of the run's {revision}"
        )

    return loaded


def prepare_run_observations(
    record: RunRecord,
    cfg: DictConfig,
    loaded: LoadedData,
    *,
    media_roots: dict[str, Path] | None = None,
) -> RunObservations:
    """The run's observations, with any feature cache being the run's own."""

    observations = prepare_observations(cfg, loaded, media_roots=media_roots)

    if observations.mimi_store is not None:
        recorded = (record.metadata.get("mimi_cache") or {}).get("caches")
        # Compared as JSON, the form the run recorded it in.
        found = json.loads(json.dumps(mimi_cache_identity(observations.mimi_store)))

        if recorded is None or found != recorded:
            raise ValueError(
                f"Mimi cache {observations.mimi_store.root} is not the cache the "
                f"run trained on: found {found}, recorded {recorded}"
            )

    return observations


def load_checkpoint(record: RunRecord, checkpoint: str | Path) -> LoadedCheckpoint:
    """The run's model with a checkpoint's weights, built from the run config."""

    path = Path(checkpoint).expanduser()

    if not path.is_file():
        path = record.run_dir / "checkpoints" / str(checkpoint)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(path, map_location="cpu", weights_only=True)

    # Built from the run's config, so its architecture (normalization, sizes,
    # context) is the checkpoint's; strict loading refuses any mismatch.
    module = LeWMModule(record.cfg)
    module.load_state_dict(state["state_dict"])

    return LoadedCheckpoint(
        model=module.model,
        path=path,
        sha256=_sha256(path),
        epoch=state.get("epoch"),
        global_step=state.get("global_step"),
    )


def _feature_caches(observations: RunObservations) -> dict[str, Any] | None:
    if observations.mimi_store is None:
        return None

    return {
        "mimi": {
            "root": str(observations.mimi_store.root),
            "caches": mimi_cache_identity(observations.mimi_store),
        }
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
