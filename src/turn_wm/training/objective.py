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

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from turn_wm.models.lewm.jepa import JEPA
from turn_wm.training.trajectories import Trajectories, encode_trajectories


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


def lejepa_forward(
    model: JEPA,
    sigreg: nn.Module,
    batch: Trajectories,
    cfg: DictConfig,
    rollout_horizons: Sequence[int] | None = None,
) -> LeJEPAOutput:
    """Teacher-forcing, rollout and SIGReg losses of one batch, with their tensors.

    `rollout_horizons` are the active horizons (the training curriculum);
    by default every `prediction.rollout_horizons`. The rollout only runs up
    to the largest active horizon.
    """

    rollout_context_size = cfg.prediction.rollout_context_size
    rollout_horizons = sorted(
        cfg.prediction.rollout_horizons
        if rollout_horizons is None
        else rollout_horizons
    )
    rollout_stop_gradient = cfg.prediction.rollout_stop_gradient

    context_steps = batch.context_steps

    # ---------------------------------------------------------
    # Encode the complete ground-truth trajectory
    # ---------------------------------------------------------

    _, emb = encode_trajectories(model, batch)
    # (B, T, D)

    act_emb = model.encode_actions(batch.actions.to(emb.device))
    # (B, T, D)

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

    if (cfg.get("evaluation") or {}).get("transition_metrics", False):
        output["weighted_sigreg_loss"] = cfg.loss.sigreg.weight * sigreg_loss

    for h in rollout_horizons:
        output[f"rollout_{h}_loss"] = rollout_losses[h]

    return LeJEPAOutput(
        losses=output,
        latents=emb,
        tf_predictions=tf_pred,
        rollout_predictions=rollout_predictions,
        context_steps=context_steps,
    )
