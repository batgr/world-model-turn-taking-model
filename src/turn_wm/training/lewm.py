"""
Lightning module of the LeWM objective (`turn_wm.training.objective`).

Builds the model from the configuration, applies the rollout curriculum
during training, evaluates every horizon at validation, and logs the losses
and the validation metrics.
"""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig

from turn_wm.models.build import build_model
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.config import rollout_horizons_for_progress, validate_config
from turn_wm.training.metrics import ValidationMetrics
from turn_wm.training.objective import lejepa_forward
from turn_wm.training.scheduler import warmup_cosine_scheduler
from turn_wm.training.trajectories import trajectories


class LeWMModule(L.LightningModule):
    """Lightning wrapper: builds the model from `cfg` and logs every loss."""

    def __init__(self, cfg: DictConfig, model: JEPA | None = None) -> None:
        super().__init__()

        validate_config(cfg)

        # Seeding belongs to the experiment runner (training.train.run), which
        # seeds before building this module.
        self.cfg = cfg
        self.model = model if model is not None else build_model(cfg)
        self.sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        progress = self._training_progress()
        horizons = rollout_horizons_for_progress(self.cfg, progress)

        # Curriculum state, logged per step next to (not as) the losses.
        self.log("train/curriculum_progress", progress, on_step=True, on_epoch=False)
        self.log(
            "train/max_rollout_horizon",
            float(max(horizons)),
            on_step=True,
            on_epoch=False,
        )

        losses = lejepa_forward(
            self.model,
            self.sigreg,
            trajectories(batch),
            self.cfg,
            rollout_horizons=horizons,
        ).losses
        self._log_losses(losses, "train", batch)

        return losses["loss"]

    def on_validation_epoch_start(self) -> None:
        self._validation_metrics = self._new_validation_metrics()

    def _new_validation_metrics(self) -> ValidationMetrics:
        evaluation = self.cfg.get("evaluation") or {}

        return ValidationMetrics(
            self.cfg.prediction.rollout_horizons,
            persistence_baseline=evaluation.get("persistence_baseline", True),
            effective_rank_health=evaluation.get("effective_rank", True),
            latent_rank_samples=evaluation.get("latent_rank_samples", 8192),
            transition_metrics=evaluation.get("transition_metrics", False),
        )

    @property
    def validation_metrics(self) -> ValidationMetrics:
        """This validation epoch's accumulator (created on first use)."""

        if getattr(self, "_validation_metrics", None) is None:
            self._validation_metrics = self._new_validation_metrics()

        return self._validation_metrics

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # Every horizon, whatever the training curriculum: comparable metrics.
        output = lejepa_forward(self.model, self.sigreg, trajectories(batch), self.cfg)
        self._log_losses(output.losses, "val", batch)

        # Diagnostics from the tensors the losses used; no second forward.
        self.validation_metrics.update(
            latents=output.latents,
            tf_predictions=output.tf_predictions,
            rollout_predictions=output.rollout_predictions,
            context_steps=output.context_steps,
            datasets=batch["dataset"],
            context_action=batch.get("context_action"),
            context_valid=batch.get("context_valid"),
            future_action=batch.get("future_action"),
            future_valid=batch.get("future_valid"),
        )

        return output.losses["loss"]

    def on_validation_epoch_end(self) -> None:
        # Epoch-level ratios of aggregated errors, logged once per epoch.
        self.log_dict(
            {
                f"val/{name}": value
                for name, value in self.validation_metrics.compute().items()
            },
            on_step=False,
            on_epoch=True,
        )

    def _training_progress(self) -> float:
        """Progress over the first epoch (V2) or the whole run (V1), in [0, 1]."""

        if self.cfg.prediction.curriculum.get("progress_basis", "run") == "first_epoch":
            batches = self.trainer.num_training_batches
            if not math.isfinite(batches):
                raise ValueError(
                    "First-epoch curriculum requires a finite train loader"
                )
            total = math.ceil(batches / self.trainer.accumulate_grad_batches)
            # The first-epoch curriculum completes at the end of epoch one.
            done = total
        else:
            total = self._total_optimizer_steps()
            # The run curriculum completes at the last optimizer step.
            done = total - 1

        if total <= 1:
            return 1.0

        return min(1.0, max(0.0, self.global_step / done))

    def _total_optimizer_steps(self) -> int:
        # Optimizer steps, accounting for gradient accumulation, batch limits,
        # max_steps and devices, unlike epochs * len(dataloader).
        total = self.trainer.estimated_stepping_batches

        if not math.isfinite(total):
            raise ValueError(
                "The LR schedule and the curriculum need a finite number of "
                "optimizer steps; set trainer.max_epochs or trainer.max_steps"
            )

        return int(total)

    def transfer_batch_to_device(
        self,
        batch: dict[str, Any],
        device: torch.device,
        dataloader_idx: int,
    ) -> dict[str, Any]:
        # Media windows are frozen dataclasses, which Lightning's generic
        # transfer cannot rebuild; the encoder moves their waveforms itself.
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        config = self.cfg.optimizer
        optimizer_class = getattr(torch.optim, config.type)

        # The frozen encoder contributes no trainable parameters.
        parameters = [p for p in self.parameters() if p.requires_grad]

        optimizer = optimizer_class(
            parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scheduler = warmup_cosine_scheduler(
            optimizer,
            total_steps=self._total_optimizer_steps(),
            warmup_ratio=self.cfg.scheduler.warmup_ratio,
            min_lr=self.cfg.scheduler.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _log_losses(
        self, output: dict[str, torch.Tensor], stage: str, batch: dict[str, Any]
    ) -> None:
        self.log_dict(
            {f"{stage}/{name}": value.detach() for name, value in output.items()},
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["sample_id"]),
        )
