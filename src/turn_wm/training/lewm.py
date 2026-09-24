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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from turn_wm.data.dataset import WindowConfig
from turn_wm.models.build import build_model
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg


@dataclass(frozen=True)
class Trajectories:
    """Model inputs for a batch of context + future trajectories."""

    waveforms: list[torch.Tensor]  # each (channels, samples), context then future
    sample_rates: list[int]
    actions: torch.Tensor  # (B, T)
    context_steps: int
    future_steps: int

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


def trajectories(batch: dict[str, Any]) -> Trajectories:
    """Assemble context + future trajectories from a collated data batch."""

    if "context_media" not in batch:
        raise ValueError("Batch has no media; build the dataset with media_roots")

    lengths = batch["context_lengths"]

    if not bool((lengths == lengths[0]).all()):
        raise ValueError(
            "Trajectories need one context length per batch; build the training "
            "dataset with training_window(cfg)"
        )

    context_steps = int(lengths[0])
    future_steps = int(batch["future_action"].shape[1])

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

    actions = torch.cat(
        [batch["context_action"][:, :context_steps], batch["future_action"]],
        dim=1,
    )

    return Trajectories(
        waveforms=waveforms,
        sample_rates=sample_rates,
        actions=actions,
        context_steps=context_steps,
        future_steps=future_steps,
    )


def lejepa_losses(
    model: JEPA,
    sigreg: nn.Module,
    batch: Trajectories,
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    """Teacher-forcing, rollout and SIGReg losses for one batch."""

    rollout_context_size = cfg.prediction.rollout_context_size
    rollout_horizons = sorted(cfg.prediction.rollout_horizons)
    rollout_stop_gradient = cfg.prediction.rollout_stop_gradient

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

    emb = model.encode(
        batch.waveforms,
        sample_rate=batch.sample_rates,
        target_length=total_steps,
    )
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

    for h in range(1, max_horizon + 1):
        # Only the latest `rollout_context_size` states and actions.
        pred = model.predict(
            rollout_emb[:, -rollout_context_size:],
            rollout_act[:, -rollout_context_size:],
        )[:, -1:]
        # prediction of z_(C + h - 1)

        if h in rollout_horizons:
            target_idx = context_steps + h - 1

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

    rollout_loss = emb.new_zeros(())

    for h in rollout_horizons:
        weight = cfg.loss.rollout.horizon_weights[str(h)]
        rollout_loss = rollout_loss + weight * rollout_losses[h]

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

    return output


class LeWMModule(L.LightningModule):
    """Lightning wrapper: builds the model from `cfg` and logs every loss."""

    def __init__(self, cfg: DictConfig, model: JEPA | None = None) -> None:
        super().__init__()

        validate_config(cfg)

        # Before building the model, so its initialization is reproducible.
        L.seed_everything(cfg.seed, workers=True)

        self.cfg = cfg
        self.model = model if model is not None else build_model(cfg)
        self.sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")["loss"]

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.cfg.optimizer
        optimizer_class = getattr(torch.optim, optimizer.type)

        # The frozen encoder contributes no trainable parameters.
        parameters = [p for p in self.parameters() if p.requires_grad]

        return optimizer_class(
            parameters,
            lr=optimizer.lr,
            weight_decay=optimizer.weight_decay,
        )

    def _step(self, batch: dict[str, Any], stage: str) -> dict[str, torch.Tensor]:
        output = lejepa_losses(self.model, self.sigreg, trajectories(batch), self.cfg)

        self.log_dict(
            {f"{stage}/{name}": value.detach() for name, value in output.items()},
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["sample_id"]),
        )

        return output
