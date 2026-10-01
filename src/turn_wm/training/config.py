"""
Training recipe checks, and the quantities the recipe determines.

`validate_config` runs before any data is read; `training_window` is the
fixed-context window the training data must be built with, and
`rollout_horizons_for_progress` the curriculum's active horizons.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from turn_wm.data.dataset import WindowConfig
from turn_wm.models.build import observation_source


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

    if (
        cfg.model.predictor.get("position_encoding", "learned") == "learned"
        and cfg.model.predictor.num_frames < context_steps
    ):
        raise ValueError(
            f"model.predictor.num_frames ({cfg.model.predictor.num_frames}) must "
            f"cover the teacher-forced context ({context_steps} steps) when "
            "using learned positional embeddings"
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

        if cfg.prediction.curriculum.get("progress_basis", "run") not in (
            "run",
            "first_epoch",
        ):
            raise ValueError(
                "prediction.curriculum.progress_basis must be run or first_epoch"
            )


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
