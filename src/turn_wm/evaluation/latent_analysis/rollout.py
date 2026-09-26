"""
Extract a trajectory snapshot of the validation rollout of a trained run.

The rollout is the validation rollout itself: every batch goes through
`lejepa_forward` (`turn_wm.training.lewm`), exactly as `validation_step` runs
it, and the snapshot keeps the latents and predictions that function
returns. There is no second predictor path here.

For each sample, with the anchor `t` the last context step (`C - 1`) and
`H` the run's `prediction.rollout_horizons` (grid steps):

- `anchor_latent` (N, D): z_t, the persistence baseline of every horizon;
- `true_future_latent` (N, |H|, D): z_(t + h), the rollout targets;
- `pred_future_latent` (N, |H|, D): the rollout prediction of z_(t + h);
- `rollout_action_ids` (N, C + max(H) - 1): the action ids the rollout
  reads, trajectory steps 0 .. C + max(H) - 2. Predicting z_(t + h) is
  conditioned on the latest `prediction.rollout_context_size` of them
  ending at step t + h - 1: for h > 1 these include the ground-truth
  actions of the future steps t + 1 .. t + h - 1.

Only the validation split is read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from turn_wm.data.mimi_precompute import GRID_RATE_HZ
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    extract_snapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.run import DEFAULT_CHECKPOINT, open_run
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.lewm import Trajectories, lejepa_forward

ROLLOUT_SPLIT = "validation"
DEFAULT_ROLLOUT_SAMPLES = 10_000

ANCHOR_LATENT = "anchor_latent"
TRUE_FUTURE_LATENT = "true_future_latent"
PRED_FUTURE_LATENT = "pred_future_latent"
ROLLOUT_ACTION_IDS = "rollout_action_ids"
ROLLOUT_TENSORS = (
    ANCHOR_LATENT,
    TRUE_FUTURE_LATENT,
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
)


def rollout_representations(
    model: JEPA,
    batch: Trajectories,
    *,
    sigreg: SIGReg,
    cfg: DictConfig,
) -> dict[str, torch.Tensor]:
    """The validation rollout of one batch, as snapshot tensors (B, ...)."""

    # The validation forward: every prediction.rollout_horizons.
    output = lejepa_forward(model, sigreg, batch, cfg)
    horizons = sorted(output.rollout_predictions)
    anchor = output.context_steps - 1

    return {
        ANCHOR_LATENT: output.latents[:, anchor],
        TRUE_FUTURE_LATENT: torch.stack(
            [output.latents[:, anchor + h] for h in horizons], dim=1
        ),
        PRED_FUTURE_LATENT: torch.stack(
            [output.rollout_predictions[h] for h in horizons], dim=1
        ),
        ROLLOUT_ACTION_IDS: batch.actions[:, : anchor + max(horizons)],
    }


def extract_rollout_snapshot(
    model: JEPA,
    cfg: DictConfig,
    batches,
    *,
    max_samples: int | None = DEFAULT_ROLLOUT_SAMPLES,
    device: torch.device | str = "cpu",
) -> RepresentationSnapshot:
    """The rollout tensors of the first `max_samples` samples of `batches`."""

    # Its losses are discarded; it is built as the training module builds it.
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    return extract_snapshot(
        model,
        batches,
        max_samples=max_samples,
        device=device,
        representations=lambda m, b: rollout_representations(
            m, b, sigreg=sigreg, cfg=cfg
        ),
    )


def rollout_provenance(cfg: DictConfig) -> dict[str, Any]:
    """How the snapshot's rollout was run, for the analysis and its report."""

    horizons = sorted(int(h) for h in cfg.prediction.rollout_horizons)
    context_steps = int(cfg.data.context_steps)
    window = int(cfg.prediction.rollout_context_size)

    return {
        "implementation": "turn_wm.training.lewm.lejepa_forward",
        "horizons_steps": horizons,
        "grid_step_s": 1 / GRID_RATE_HZ,
        "horizons_s": [h / GRID_RATE_HZ for h in horizons],
        "anchor_step": context_steps - 1,
        "rollout_context_size": window,
        "stop_gradient": bool(cfg.prediction.rollout_stop_gradient),
        "fed_back_latents": "own predictions, never ground-truth future latents",
        "persistence_baseline": ANCHOR_LATENT,
        # Trajectory steps [start, stop) of the actions conditioning z_(t + h).
        "action_steps_by_horizon": {
            str(h): [max(0, context_steps + h - 1 - window), context_steps + h - 1]
            for h in horizons
        },
        "conditioned_on_ground_truth_future_actions": {str(h): h > 1 for h in horizons},
        "future_action_tokens_by_horizon": {str(h): h - 1 for h in horizons},
        "tensors": {
            ANCHOR_LATENT: "z_t, t = anchor (last context step)",
            TRUE_FUTURE_LATENT: "z_(t + h) per horizon, axis 1 in horizons_steps order",
            PRED_FUTURE_LATENT: "rollout prediction of z_(t + h), same order",
            ROLLOUT_ACTION_IDS: (
                "action ids of trajectory steps 0 .. anchor_step + max(h) - 1"
            ),
        },
    }


def extract_rollout_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    max_samples: int | None = DEFAULT_ROLLOUT_SAMPLES,
    seed: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str = "cpu",
    mimi_cache_root: Path | None = None,
    media_roots: dict[str, Path] | None = None,
) -> Path:
    """Write the validation-rollout snapshot of one run; return the dir.

    The samples are the seeded fixed permutation of the validation split
    that `extract_run` uses, so a prefix is a uniform, deterministic sample.
    """

    opened = open_run(
        run_dir,
        checkpoint=checkpoint,
        split=ROLLOUT_SPLIT,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        mimi_cache_root=mimi_cache_root,
        media_roots=media_roots,
    )
    snapshot = extract_rollout_snapshot(
        opened.checkpoint.model,
        opened.cfg,
        opened.loader,
        max_samples=max_samples,
        device=device,
    )

    return write_snapshot(
        snapshot,
        output_dir or opened.default_output_dir("-rollout"),
        provenance={
            **opened.provenance(
                max_samples=max_samples, samples=snapshot.samples, device=device
            ),
            "rollout": rollout_provenance(opened.cfg),
        },
    )
