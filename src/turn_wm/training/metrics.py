"""Validation metrics for the V2 world model.

The primary dynamics result is autoregressive latent rollout MSE at each
configured horizon. Persistence is reported as a separate baseline. The same
errors can be stratified by whether the supplied ego-action sequence contains
a vocal transition.

Representation health is intentionally limited to effective rank. Task
readouts, action ablations and planning metrics live in evaluation code rather
than being mixed into this training-time accumulator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from turn_wm.data.dataset import ACTION_TO_ID, TRANSITION_ACTIONS

GLOBAL = ""


def skill_score(model_mse: float, persistence_mse: float) -> float:
    """Legacy analysis helper; not part of the V2 training metric surface."""

    eps = torch.finfo(torch.float32).eps
    return 1.0 - model_mse / max(persistence_mse, eps)


def effective_rank(latents: torch.Tensor) -> float:
    """Effective rank of centred rows, from normalized singular values."""

    if latents.shape[0] < 2:
        return float("nan")

    centred = latents.double() - latents.double().mean(dim=0)
    return entropy_rank(torch.linalg.svdvals(centred))


def entropy_rank(weights: torch.Tensor) -> float:
    """exp(entropy) of non-negative weights normalized to sum to one."""

    weights = weights.double()
    total = weights.sum()

    if total <= torch.finfo(torch.float64).eps:
        return 0.0

    p = weights / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@dataclass
class _MSE:
    squared_error: float = 0.0
    elements: int = 0

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.squared_error += float((prediction - target).pow(2).sum())
        self.elements += target.numel()

    @property
    def value(self) -> float:
        return self.squared_error / self.elements


@dataclass
class _PredictionErrors:
    model: _MSE = field(default_factory=_MSE)
    persistence: _MSE = field(default_factory=_MSE)

    def add(
        self,
        prediction: torch.Tensor,
        persistence: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        self.model.add(prediction, target)
        self.persistence.add(persistence, target)


class _LatentSample:
    """Deterministic bounded sample used only for effective-rank estimation."""

    def __init__(self, size: int, seed: int = 0) -> None:
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.rows = torch.empty(0)
        self.priorities = torch.empty(0)

    def add(self, rows: torch.Tensor) -> None:
        if self.size <= 0 or rows.numel() == 0:
            return

        rows = rows.detach().float().cpu()
        priorities = torch.rand(rows.shape[0], generator=self.generator)

        if self.rows.numel():
            rows = torch.cat([self.rows, rows])
            priorities = torch.cat([self.priorities, priorities])

        if rows.shape[0] > self.size:
            keep = priorities.topk(self.size).indices
            rows, priorities = rows[keep], priorities[keep]

        self.rows, self.priorities = rows, priorities


class ValidationMetrics:
    """Epoch-level V2 metrics.

    Headline dynamics:
      - rollout_{h}_mse
      - persistence_{h}_mse

    Diagnostics:
      - tf_mse
      - transition/stable rollout MSE and their persistence baselines
      - effective_rank

    Prediction and persistence errors are accumulated over the whole epoch
    before division, so results do not depend on validation batch size.
    """

    def __init__(
        self,
        horizons: Sequence[int],
        *,
        persistence_baseline: bool = True,
        effective_rank_health: bool = True,
        latent_rank_samples: int = 8192,
        transition_metrics: bool = False,
    ) -> None:
        self.horizons = sorted(horizons)
        self.persistence_baseline = persistence_baseline
        self.effective_rank_health = effective_rank_health
        self.transition_metrics = transition_metrics

        self.tf: dict[str, _MSE] = defaultdict(_MSE)
        self.rollout: dict[tuple[str, int], _PredictionErrors] = defaultdict(
            _PredictionErrors
        )
        self.conditions: dict[tuple[str, int], _PredictionErrors] = defaultdict(
            _PredictionErrors
        )
        self.condition_counts: dict[tuple[str, int], int] = defaultdict(int)
        self.sample = _LatentSample(latent_rank_samples if effective_rank_health else 0)

    @torch.no_grad()
    def update(
        self,
        *,
        latents: torch.Tensor,
        tf_predictions: torch.Tensor,
        rollout_predictions: dict[int, torch.Tensor],
        context_steps: int,
        datasets: Sequence[str],
        mask: torch.Tensor | None = None,
        context_action: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
        future_action: torch.Tensor | None = None,
        future_valid: torch.Tensor | None = None,
    ) -> None:
        z = latents.detach().double()
        tf_pred = tf_predictions.detach().double()
        c = context_steps
        valid = (
            torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
            if mask is None
            else mask.to(device=z.device, dtype=torch.bool)
        )

        groups = [GLOBAL, *sorted(set(datasets))]
        members = {
            name: torch.tensor(
                [name == GLOBAL or dataset == name for dataset in datasets],
                device=z.device,
            )
            for name in groups
        }

        tf_valid = valid[:, :c] & valid[:, 1 : c + 1]
        tf_target = z[:, 1 : c + 1]

        for name in groups:
            rows = tf_valid & members[name][:, None]
            self.tf[name].add(tf_pred[rows], tf_target[rows])

        last = z[:, c - 1]

        if self.transition_metrics and any(
            value is None
            for value in (context_action, context_valid, future_action, future_valid)
        ):
            raise ValueError(
                "Transition stratification requires context/future actions and validity"
            )

        for h in self.horizons:
            if h not in rollout_predictions:
                continue

            pred = rollout_predictions[h].detach().double()
            target = z[:, c + h - 1]
            ok = valid[:, c - 1] & valid[:, c + h - 1]

            for name in groups:
                rows = ok & members[name]
                self.rollout[(name, h)].add(pred[rows], last[rows], target[rows])

            if not self.transition_metrics:
                continue

            assert context_action is not None
            assert context_valid is not None
            assert future_action is not None
            assert future_valid is not None

            # z[C+h-1] is reached with a[C-1] followed by h-1 future actions.
            transition_actions = torch.cat(
                [context_action[:, -1:], future_action[:, : h - 1]], dim=1
            ).to(z.device)
            transition_valid = torch.cat(
                [context_valid[:, -1:], future_valid[:, : h - 1]], dim=1
            ).to(z.device)

            evaluable = transition_valid.bool().all(dim=1) & ok
            transition_ids = torch.tensor(
                [ACTION_TO_ID[name] for name in TRANSITION_ACTIONS],
                device=transition_actions.device,
            )
            has_transition = torch.isin(transition_actions, transition_ids).any(dim=1)

            for condition, rows in (
                ("transition", evaluable & has_transition),
                ("stable", evaluable & ~has_transition),
            ):
                self.conditions[(condition, h)].add(
                    pred[rows], last[rows], target[rows]
                )
                self.condition_counts[(condition, h)] += int(rows.sum())

        if self.effective_rank_health:
            self.sample.add(z[valid])

    def compute(self) -> dict[str, float]:
        """Return metric names without the val/ stage prefix."""

        metrics: dict[str, float] = {}

        for name, mse in sorted(self.tf.items()):
            if not mse.elements:
                continue
            prefix = f"{name}/" if name else ""
            metrics[f"{prefix}tf_mse"] = mse.value

        for (name, h), errors in sorted(self.rollout.items()):
            if not errors.model.elements:
                continue

            prefix = f"{name}/" if name else ""
            metrics[f"{prefix}rollout_{h}_mse"] = errors.model.value

            if self.persistence_baseline:
                metrics[f"{prefix}persistence_{h}_mse"] = errors.persistence.value

        if self.transition_metrics:
            for (condition, h), errors in sorted(self.conditions.items()):
                if not errors.model.elements:
                    continue

                metrics[f"{condition}_{h}_mse"] = errors.model.value
                if self.persistence_baseline:
                    metrics[f"{condition}_{h}_persistence_mse"] = (
                        errors.persistence.value
                    )
                metrics[f"{condition}_{h}_n"] = self.condition_counts[(condition, h)]

        if self.effective_rank_health and self.sample.rows.numel():
            metrics["effective_rank"] = effective_rank(self.sample.rows)

        return {name: value for name, value in metrics.items() if not math.isnan(value)}
