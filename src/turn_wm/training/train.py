"""
Training entry point for turn-taking world-model experiments.

This module wires together configuration, published dataset artifacts,
local raw media, PyTorch DataLoaders, the Lightning module and Trainer.
Model-specific losses remain in `turn_wm.training.objective`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightning as L
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig, OmegaConf

from turn_wm.data.source import DATASETS, load_data
from turn_wm.training.config import validate_config
from turn_wm.training.datamodule import TurnTakingDataModule
from turn_wm.training.lewm import LeWMModule
from turn_wm.training.observations import (
    prepare_observations,
    require_mimi_cache_root,
)
from turn_wm.training.run_dir import (
    create_run_dir,
    git_metadata,
    hash_config,
    resolved_config,
    write_config,
    write_metadata,
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

    # Before any download: a cached run without a cache cannot start.
    require_mimi_cache_root(cfg)

    # One experiment seed governs data order, workers and model initialization.
    L.seed_everything(cfg.seed, workers=True)

    loaded = load_data(DATASETS[dataset_name])

    observations = prepare_observations(cfg, loaded, media_roots=media_roots)

    # The run directory is created only once the experiment can start, so a
    # rejected configuration or missing media leaves nothing behind.
    config_hash = hash_config(cfg)
    run_id, run_dir = create_run_dir(cfg, config_hash)

    write_config(cfg, run_dir)
    write_metadata(
        run_dir=run_dir,
        run_id=run_id,
        config_hash=config_hash,
        cfg=cfg,
        git=git_metadata(),
        dataset_revision=loaded.revision,
        mimi_store=observations.mimi_store,
    )

    module = LeWMModule(cfg)

    trainer_kwargs = OmegaConf.to_container(
        cfg.trainer,
        resolve=True,
    )

    if not isinstance(trainer_kwargs, dict):
        raise TypeError("cfg.trainer must resolve to a mapping")

    trainer_kwargs = cast(dict[str, Any], trainer_kwargs)
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

    if logger is not False:
        # The warmup/cosine LR per optimizer step, next to the losses.
        callbacks = [*callbacks, LearningRateMonitor(logging_interval="step")]

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
        datamodule=TurnTakingDataModule(cfg, loaded, observations),
        ckpt_path=ckpt_path,
    )


def _build_callbacks(
    cfg: DictConfig,
    *,
    run_dir: Path,
) -> list[Callback]:
    callbacks: list[Callback] = []

    if cfg.checkpoint.enabled:
        checkpoint_dir = run_dir / "checkpoints"

        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                save_top_k=cfg.checkpoint.save_top_k,
                save_last=cfg.checkpoint.save_last,
                every_n_epochs=cfg.checkpoint.every_n_epochs,
                filename="epoch={epoch:03d}-step={step}",
                auto_insert_metric_name=False,
            )
        )

    early = cfg.get("early_stopping")
    if early and early.get("enabled", False):
        callbacks.append(
            FullHorizonEarlyStopping(
                monitor=early.monitor,
                mode=early.mode,
                min_delta=early.min_delta,
                patience=early.patience,
                minimum_completed_epochs=early.minimum_completed_epochs,
            )
        )

    return callbacks


class FullHorizonEarlyStopping(EarlyStopping):
    """Start counting patience only after H=10 has had a full epoch."""

    def __init__(self, *, minimum_completed_epochs: int, **kwargs) -> None:
        super().__init__(check_on_train_epoch_end=False, **kwargs)
        if minimum_completed_epochs < 2:
            raise ValueError("minimum_completed_epochs must cover one full H=10 epoch")
        self.minimum_completed_epochs = minimum_completed_epochs

    def _run_early_stopping_check(self, trainer: L.Trainer) -> None:
        if trainer.current_epoch + 1 < self.minimum_completed_epochs:
            return
        super()._run_early_stopping_check(trainer)


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
        config=resolved_config(cfg),
    )
