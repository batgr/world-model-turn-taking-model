"""
Training entry point for turn-taking world-model experiments.

This module wires together configuration, published dataset artifacts,
local raw media, PyTorch DataLoaders, the Lightning module and Trainer.
Model-specific losses remain in `turn_wm.training.lewm`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from turn_wm.data.build import build_dataset
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.source import DATASETS, LoadedData, load_data
from turn_wm.training.lewm import (
    LeWMModule,
    training_window,
    validate_config,
)


def run(
    cfg: DictConfig,
    *,
    media_roots: Mapping[str, Path] | None = None,
) -> None:
    """Run one training experiment."""

    validate_config(cfg)

    dataset_name = str(cfg.data.dataset)

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; expected one of {sorted(DATASETS)}"
        )

    # One experiment seed governs data order, workers and model initialization.
    L.seed_everything(cfg.seed, workers=True)

    loaded = load_data(DATASETS[dataset_name])

    roots = (
        _resolve_media_roots(loaded)
        if media_roots is None
        else _validate_media_roots(loaded, media_roots)
    )

    # The run directory is created only once the experiment can start, so a
    # rejected configuration or missing media leaves nothing behind.
    config_hash = _config_hash(cfg)
    run_id, run_dir = _create_run_dir(cfg, config_hash)

    _write_config(cfg, run_dir)
    _write_metadata(
        run_dir=run_dir,
        run_id=run_id,
        config_hash=config_hash,
        cfg=cfg,
        git=_git_metadata(),
        dataset_revision=loaded.revision,
    )

    window = training_window(cfg)
    modalities = tuple(cfg.data.modalities)

    train_dataset = build_dataset(
        loaded,
        split="train",
        window=window,
        training=True,
        media_roots=roots,
        modalities=modalities,
    )

    val_dataset = build_dataset(
        loaded,
        split="validation",
        window=window,
        training=False,
        media_roots=roots,
        modalities=modalities,
    )

    train_loader_config = DataLoaderConfig(
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=cfg.loader.persistent_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
        seed=cfg.seed,
    )

    val_loader_config = replace(
        train_loader_config,
        shuffle=False,
        drop_last=False,
    )

    train_loader = build_dataloader(
        train_dataset,
        loader=train_loader_config,
    )

    val_loader = build_dataloader(
        val_dataset,
        loader=val_loader_config,
    )

    module = LeWMModule(cfg)

    trainer_kwargs = OmegaConf.to_container(
        cfg.trainer,
        resolve=True,
    )

    if not isinstance(trainer_kwargs, dict):
        raise TypeError("cfg.trainer must resolve to a mapping")

    trainer_kwargs["default_root_dir"] = str(run_dir)

    callbacks = _build_callbacks(
        cfg,
        run_dir=run_dir,
    )

    logger = _build_logger(
        cfg,
        run_id=run_id,
        run_dir=run_dir,
    )

    trainer = L.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        logger=logger,
    )

    ckpt_path = cfg.checkpoint.resume_from

    if ckpt_path is not None:
        ckpt_path = str(Path(ckpt_path).expanduser())

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


def _resolve_media_roots(
    loaded: LoadedData,
) -> dict[str, Path]:
    """Resolve corpus media roots from environment variables."""

    roots = {}

    for name in loaded.names:
        variable = _media_root_variable(name)
        value = os.environ.get(variable)

        if value is None:
            raise ValueError(
                f"{variable} is not set; training requires local raw "
                f"media for corpus {name!r}"
            )

        path = Path(value).expanduser()

        if not path.is_dir():
            raise ValueError(f"{variable} does not point to a directory: {path}")

        roots[name] = path

    return roots


def _validate_media_roots(
    loaded: LoadedData,
    media_roots: Mapping[str, Path],
) -> dict[str, Path]:
    """Validate explicitly supplied media roots."""

    roots = {}

    for name in loaded.names:
        if name not in media_roots:
            raise ValueError(f"No media root supplied for corpus {name!r}")

        path = Path(media_roots[name]).expanduser()

        if not path.is_dir():
            raise ValueError(f"Media root for {name!r} is not a directory: {path}")

        roots[name] = path

    return roots


def _media_root_variable(dataset: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", dataset).upper()

    return f"{normalized}_MEDIA_ROOT"


def _build_callbacks(
    cfg: DictConfig,
    *,
    run_dir: Path,
) -> list[ModelCheckpoint]:
    if not cfg.checkpoint.enabled:
        return []

    checkpoint_dir = run_dir / "checkpoints"

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
    )

    return [checkpoint]


def _resolved_config(cfg: DictConfig) -> dict:
    resolved = OmegaConf.to_container(
        cfg,
        resolve=True,
        enum_to_str=True,
    )

    if not isinstance(resolved, dict):
        raise TypeError("Resolved configuration must be a mapping")

    return resolved


def _config_hash(cfg: DictConfig) -> str:
    payload = json.dumps(
        _resolved_config(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _create_run_dir(
    cfg: DictConfig,
    config_hash: str,
) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{config_hash[:8]}"

    run_dir = Path(cfg.experiment.output_root) / str(cfg.experiment.name) / run_id

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_id, run_dir


def _git_metadata() -> dict[str, object]:
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


def _write_config(
    cfg: DictConfig,
    run_dir: Path,
) -> None:
    OmegaConf.save(
        config=cfg,
        f=run_dir / "config.yaml",
        resolve=True,
    )


def _write_metadata(
    *,
    run_dir: Path,
    run_id: str,
    config_hash: str,
    cfg: DictConfig,
    git: dict[str, object],
    dataset_revision: str | None,
) -> None:
    metadata = {
        "run_id": run_id,
        "seed": int(cfg.seed),
        "config_hash": config_hash,
        "git": git,
        "dataset": str(cfg.data.dataset),
        "dataset_revision": dataset_revision,
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


def _build_logger(
    cfg: DictConfig,
    *,
    run_id: str,
    run_dir: Path,
):
    if not cfg.logging.wandb.enabled:
        return False

    # Lightning imports WandbLogger without wandb and only fails when it is
    # constructed, with a generic error; check the package itself.
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError(
            "WandB logging is enabled but the optional dependency "
            "is not installed. Run `uv sync --extra wandb`."
        )

    from lightning.pytorch.loggers import WandbLogger

    name = cfg.logging.wandb.name or run_id

    return WandbLogger(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        name=name,
        save_dir=str(run_dir),
        config=_resolved_config(cfg),
    )
