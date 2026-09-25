"""
Training entry point for turn-taking world-model experiments.

This module wires together configuration, published dataset artifacts,
local raw media, PyTorch DataLoaders, the Lightning module and Trainer.
Model-specific losses remain in `turn_wm.training.lewm`.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import replace
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

    # One experiment seed should govern model initialization and workers.
    L.seed_everything(cfg.seed, workers=True)

    loaded = load_data(DATASETS[dataset_name])

    roots = (
        _resolve_media_roots(loaded)
        if media_roots is None
        else _validate_media_roots(loaded, media_roots)
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

    callbacks = _build_callbacks(cfg)

    trainer = L.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
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


def _build_callbacks(cfg: DictConfig) -> list[ModelCheckpoint]:
    """Build training callbacks from the experiment configuration."""

    if not cfg.checkpoint.enabled:
        return []

    checkpoint = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
        filename="epoch={epoch:03d}-step={step}",
        auto_insert_metric_name=False,
    )

    return [checkpoint]
