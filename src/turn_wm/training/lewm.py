"""
LeWM training objective on turn-taking trajectories.

A trajectory is the context window followed by the future window of one
sample: `T = context_steps + future_steps` grid steps of audio latents and
actions. Training uses a fixed context length (`data.context_steps`), so
every trajectory in a batch has the same length and no padding is needed.

Teacher forcing is dense over the ground-truth context: from z0..z(C-1) and
their actions the predictor predicts z1..zC, one step ahead at every
position (zC, the first future latent, is only a target). The rollout starts
at the context/future boundary from the ground-truth context and feeds its
own predictions back, never a ground-truth future latent; before each
prediction it keeps only the latest `prediction.rollout_context_size` states
and actions. The window is chosen by the inputs given to the predictor; its
attention stays plain causal attention.

Latent targets come from the audio of every step, including steps whose
turn-taking annotation is UNKNOWN or whose action is masked: the observation
is defined there, and masked actions have their own conditioning id.

The rollout loss is the weighted mean of the per-horizon losses of the
active horizons. During training a curriculum over optimizer steps activates
horizons progressively (`prediction.curriculum`); the mean keeps the rollout
term's magnitude independent of how many horizons are active, so the
curriculum changes the temporal difficulty only. Validation always evaluates
every `prediction.rollout_horizons`, so its metrics stay comparable across
the whole run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig
from torch import nn

from turn_wm.data.dataset import WindowConfig
from turn_wm.models.build import build_model, observation_source
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.metrics import ValidationMetrics
from turn_wm.training.scheduler import warmup_cosine_scheduler


@dataclass(frozen=True)
class Trajectories:
    """Model inputs for a batch of context + future trajectories.

    The observations are either precomputed encoder `features` (B, T, D) or
    raw `waveforms` with their `sample_rates`, never both.
    """

    actions: torch.Tensor  # (B, T)
    context_steps: int
    future_steps: int
    features: torch.Tensor | None = None  # (B, T, D), context then future
    waveforms: list[torch.Tensor] | None = None  # each (channels, samples)
    sample_rates: list[int] | None = None

    def __post_init__(self) -> None:
        has_features = self.features is not None
        has_audio = self.waveforms is not None and self.sample_rates is not None

        if has_features == has_audio or (
            not has_audio and (self.waveforms or self.sample_rates)
        ):
            raise ValueError(
                "Trajectories need exactly one observation source: features, or "
                "waveforms with sample_rates"
            )

    @property
    def total_steps(self) -> int:
        return self.context_steps + self.future_steps


def training_window(cfg: DictConfig) -> WindowConfig:
    """Fixed-context window the training data must be built with."""

    return WindowConfig(
        min_context_steps=cfg.data.context_steps,
        max_context_steps=cfg.data.context_steps,
        future_steps=cfg.data.future_steps,
    )


def validate_config(cfg: DictConfig) -> None:
    """Check the training recipe against the model before any data is read."""

    context_steps = cfg.data.context_steps
    future_steps = cfg.data.future_steps
    rollout_context_size = cfg.prediction.rollout_context_size
    horizons = list(cfg.prediction.rollout_horizons)
    weights = {str(key) for key in cfg.loss.rollout.horizon_weights}

    if context_steps < 1:
        raise ValueError(f"data.context_steps must be >= 1, got {context_steps}")

    if not 1 <= rollout_context_size <= context_steps:
        raise ValueError(
            "prediction.rollout_context_size must lie in "
            f"[1, data.context_steps={context_steps}], got {rollout_context_size}"
        )

    if cfg.model.predictor.num_frames < context_steps:
        raise ValueError(
            f"model.predictor.num_frames ({cfg.model.predictor.num_frames}) must "
            f"cover the teacher-forced context ({context_steps} steps)"
        )

    if not horizons or min(horizons) < 1:
        raise ValueError(
            f"prediction.rollout_horizons must be positive and non-empty: {horizons}"
        )

    if max(horizons) > future_steps:
        raise ValueError(
            f"data.future_steps ({future_steps}) must cover every rollout horizon: "
            f"{horizons}"
        )

    missing = [h for h in horizons if str(h) not in weights]

    if missing:
        raise ValueError(f"loss.rollout.horizon_weights has no weight for {missing}")

    non_positive = [
        h for h in horizons if not cfg.loss.rollout.horizon_weights[str(h)] > 0
    ]

    if non_positive:
        raise ValueError(
            f"loss.rollout.horizon_weights must be positive; got {non_positive}"
        )

    # Raises on an unknown source; the cache root is checked by the runner,
    # which opens the cache.
    observation_source(cfg)

    _validate_scheduler(cfg)

    if cfg.prediction.curriculum.enabled:
        _validate_curriculum(cfg.prediction.curriculum.stages, horizons=horizons)


def _validate_scheduler(cfg: DictConfig) -> None:
    scheduler = cfg.scheduler

    if scheduler.type != "warmup_cosine":
        raise ValueError(
            f"scheduler.type must be 'warmup_cosine', got {scheduler.type!r}"
        )

    if scheduler.interval != "step":
        raise ValueError(
            "scheduler.interval must be 'step' (the schedule and the curriculum "
            f"follow optimizer steps), got {scheduler.interval!r}"
        )

    if not 0 <= scheduler.warmup_ratio < 1:
        raise ValueError(
            f"scheduler.warmup_ratio must lie in [0, 1), got {scheduler.warmup_ratio}"
        )

    if not 0 < scheduler.min_lr <= cfg.optimizer.lr:
        raise ValueError(
            f"scheduler.min_lr must lie in (0, optimizer.lr={cfg.optimizer.lr}], "
            f"got {scheduler.min_lr}"
        )


def _validate_curriculum(stages: Sequence[Any], *, horizons: Sequence[int]) -> None:
    if not stages:
        raise ValueError("prediction.curriculum.stages needs at least one stage")

    previous_until = 0.0
    previous: set[int] = set()

    for index, stage in enumerate(stages):
        until = stage.until
        active = list(stage.horizons)
        name = f"prediction.curriculum.stages[{index}]"

        if not previous_until < until <= 1.0:
            raise ValueError(
                f"{name}.until must be increasing within (0, 1]; got {until} after "
                f"{previous_until}"
            )

        if not active:
            raise ValueError(f"{name} needs at least one horizon")

        unknown = [h for h in active if h not in horizons]

        if unknown:
            raise ValueError(
                f"{name} horizons {unknown} are not in prediction.rollout_horizons "
                f"{list(horizons)}"
            )

        if not previous <= set(active):
            raise ValueError(
                f"{name} drops horizons {sorted(previous - set(active))}; the "
                "curriculum is cumulative"
            )

        previous_until = until
        previous = set(active)

    if previous_until != 1.0:
        raise ValueError(
            f"the last prediction.curriculum stage must end at 1.0, got {previous_until}"
        )


def rollout_horizons_for_progress(cfg: DictConfig, progress: float) -> list[int]:
    """Rollout horizons active at `progress` in [0, 1] of the optimizer steps.

    A stage covers progress below its `until`; progress 1.0 is in the last
    stage. Without a curriculum every `prediction.rollout_horizons` is active.
    """

    curriculum = cfg.prediction.curriculum

    if not curriculum.enabled:
        return sorted(cfg.prediction.rollout_horizons)

    stages = list(curriculum.stages)

    for stage in stages:
        if progress < stage.until:
            return sorted(stage.horizons)

    return sorted(stages[-1].horizons)


def trajectories(batch: dict[str, Any]) -> Trajectories:
    """Assemble context + future trajectories from a collated data batch.

    Cached features (`context_features`/`future_features`) are used when the
    batch has them; otherwise the audio of the media windows.
    """

    has_features = "context_features" in batch

    if not has_features and "context_media" not in batch:
        raise ValueError(
            "Batch has no observations; build the dataset with a mimi_store or "
            "media_roots"
        )

    lengths = batch["context_lengths"]

    if not bool((lengths == lengths[0]).all()):
        raise ValueError(
            "Trajectories need one context length per batch; build the training "
            "dataset with training_window(cfg)"
        )

    context_steps = int(lengths[0])
    future_steps = int(batch["future_action"].shape[1])

    actions = torch.cat(
        [batch["context_action"][:, :context_steps], batch["future_action"]],
        dim=1,
    )

    if has_features:
        return Trajectories(
            actions=actions,
            context_steps=context_steps,
            future_steps=future_steps,
            # Cached features are float16; float32 before any op, as CPU
            # autocast (bf16-mixed) rejects float16 inputs.
            features=torch.cat(
                [
                    batch["context_features"][:, :context_steps].float(),
                    batch["future_features"].float(),
                ],
                dim=1,
            ),
        )

    waveforms = []
    sample_rates = []

    for index, (context, future) in enumerate(
        zip(batch["context_media"], batch["future_media"], strict=True)
    ):
        if context.audio is None or future.audio is None:
            raise ValueError(
                f"Sample {batch['sample_id'][index]} has no audio; "
                "LeWM trajectories need audio in every window"
            )

        if context.audio.sample_rate != future.audio.sample_rate:
            raise ValueError("Context and future audio must share a sample rate")

        # The windows are contiguous: context ends where the future starts.
        waveforms.append(
            torch.cat([context.audio.waveform, future.audio.waveform], dim=1)
        )
        sample_rates.append(context.audio.sample_rate)

    return Trajectories(
        actions=actions,
        context_steps=context_steps,
        future_steps=future_steps,
        waveforms=waveforms,
        sample_rates=sample_rates,
    )


def encode_trajectories(
    model: JEPA,
    batch: Trajectories,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encoder features and projected latents of every step, each (B, T, ·).

    The only place that knows where observations come from: cached encoder
    features are used as they are, raw audio goes through the encoder first.
    Everything downstream sees the same features and latents.
    """

    if batch.features is not None:
        features = batch.features
    else:
        features = model.encode_features(
            batch.waveforms,
            sample_rate=batch.sample_rates,
            target_length=batch.total_steps,
        )

    return features, model.project_features(features)


@dataclass(frozen=True)
class LeJEPAOutput:
    """Losses of one batch, with the tensors they were computed from.

    Validation metrics reuse these instead of running the predictor again.
    """

    losses: dict[str, torch.Tensor]
    latents: torch.Tensor  # (B, T, D) projected trajectory latents (targets)
    tf_predictions: torch.Tensor  # (B, C, D), predicting latents[:, 1 : C + 1]
    rollout_predictions: dict[int, torch.Tensor]  # h -> (B, D), latents[:, C + h - 1]
    context_steps: int


def lejepa_losses(
    model: JEPA,
    sigreg: nn.Module,
    batch: Trajectories,
    cfg: DictConfig,
    rollout_horizons: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Teacher-forcing, rollout and SIGReg losses for one batch.

    `rollout_horizons` are the active horizons (the training curriculum);
    by default every `prediction.rollout_horizons`. The rollout only runs up
    to the largest active horizon.
    """

    return lejepa_forward(
        model, sigreg, batch, cfg, rollout_horizons=rollout_horizons
    ).losses


def lejepa_forward(
    model: JEPA,
    sigreg: nn.Module,
    batch: Trajectories,
    cfg: DictConfig,
    rollout_horizons: Sequence[int] | None = None,
) -> LeJEPAOutput:
    """`lejepa_losses`, also returning its latents and predictions."""

    rollout_context_size = cfg.prediction.rollout_context_size
    rollout_horizons = sorted(
        cfg.prediction.rollout_horizons
        if rollout_horizons is None
        else rollout_horizons
    )
    rollout_stop_gradient = cfg.prediction.rollout_stop_gradient

    if not rollout_horizons:
        raise ValueError("At least one rollout horizon must be active")

    context_steps = batch.context_steps
    total_steps = batch.total_steps

    if context_steps < rollout_context_size:
        raise ValueError(
            f"context_steps ({context_steps}) is shorter than "
            f"rollout_context_size ({rollout_context_size})"
        )

    if max(rollout_horizons) > batch.future_steps:
        raise ValueError(
            f"rollout horizon {max(rollout_horizons)} exceeds "
            f"future_steps={batch.future_steps}"
        )

    # ---------------------------------------------------------
    # Encode the complete ground-truth trajectory
    # ---------------------------------------------------------

    _, emb = encode_trajectories(model, batch)
    # (B, T, D)

    act_emb = model.encode_actions(batch.actions.to(emb.device))
    # (B, T, D)

    if emb.size(1) != total_steps:
        raise ValueError(f"Expected {total_steps} latent steps, got {emb.size(1)}")

    # =========================================================
    # 1. DENSE TEACHER FORCING over the ground-truth context
    # =========================================================

    # z0..z(C-1) -> z1..zC: one-step supervision at every context position.
    tf_pred = model.predict(
        emb[:, :context_steps],
        act_emb[:, :context_steps],
    )

    tf_loss = F.mse_loss(tf_pred, emb[:, 1 : context_steps + 1])

    # =========================================================
    # 2. AUTOREGRESSIVE ROLLOUT from the context/future boundary
    # =========================================================

    max_horizon = max(rollout_horizons)

    rollout_emb = emb[:, :context_steps]
    rollout_act = act_emb[:, :context_steps]

    rollout_losses: dict[int, torch.Tensor] = {}
    rollout_predictions: dict[int, torch.Tensor] = {}

    for h in range(1, max_horizon + 1):
        # Only the latest `rollout_context_size` states and actions.
        pred = model.predict(
            rollout_emb[:, -rollout_context_size:],
            rollout_act[:, -rollout_context_size:],
        )[:, -1:]
        # prediction of z_(C + h - 1)

        if h in rollout_horizons:
            target_idx = context_steps + h - 1
            rollout_predictions[h] = pred[:, 0]

            rollout_losses[h] = F.mse_loss(
                pred,
                emb[:, target_idx : target_idx + 1],
            )

        next_emb = pred.detach() if rollout_stop_gradient else pred

        rollout_emb = torch.cat([rollout_emb, next_emb], dim=1)

        if h < max_horizon:
            action_idx = context_steps + h - 1

            rollout_act = torch.cat(
                [rollout_act, act_emb[:, action_idx : action_idx + 1]],
                dim=1,
            )

    # Weighted mean over the active horizons: activating more horizons makes
    # the task harder without scaling the rollout term up.
    weights = {
        h: float(cfg.loss.rollout.horizon_weights[str(h)]) for h in rollout_horizons
    }

    rollout_loss = sum(
        (weights[h] * rollout_losses[h] for h in rollout_horizons),
        emb.new_zeros(()),
    ) / sum(weights.values())

    # =========================================================
    # 3. SIGREG over the trajectory latents, (T, B, D)
    # =========================================================

    sigreg_loss = sigreg(emb.transpose(0, 1))

    # =========================================================
    # 4. TOTAL OBJECTIVE
    # =========================================================

    loss = (
        cfg.loss.teacher_forcing.weight * tf_loss
        + cfg.loss.rollout.weight * rollout_loss
        + cfg.loss.sigreg.weight * sigreg_loss
    )

    output = {
        "loss": loss,
        "tf_loss": tf_loss,
        "rollout_loss": rollout_loss,
        "sigreg_loss": sigreg_loss,
    }

    for h in rollout_horizons:
        output[f"rollout_{h}_loss"] = rollout_losses[h]

    return LeJEPAOutput(
        losses=output,
        latents=emb,
        tf_predictions=tf_pred,
        rollout_predictions=rollout_predictions,
        context_steps=context_steps,
    )


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

        return self._step(batch, "train", rollout_horizons=horizons)["loss"]

    def on_validation_epoch_start(self) -> None:
        self._validation_metrics = self._new_validation_metrics()

    def _new_validation_metrics(self) -> ValidationMetrics:
        evaluation = self.cfg.get("evaluation") or {}

        return ValidationMetrics(
            self.cfg.prediction.rollout_horizons,
            persistence_baseline=evaluation.get("persistence_baseline", True),
            cosine_similarity=evaluation.get("cosine_similarity", True),
            latent_health=evaluation.get("latent_health", True),
            latent_rank_samples=evaluation.get("latent_rank_samples", 8192),
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
        """Fraction of the run's optimizer steps done, in [0, 1]."""

        total_steps = self._total_optimizer_steps()

        if total_steps <= 1:
            return 1.0

        return min(1.0, max(0.0, self.global_step / (total_steps - 1)))

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

    def _step(
        self,
        batch: dict[str, Any],
        stage: str,
        rollout_horizons: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        output = lejepa_losses(
            self.model,
            self.sigreg,
            trajectories(batch),
            self.cfg,
            rollout_horizons=rollout_horizons,
        )
        self._log_losses(output, stage, batch)

        return output

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
