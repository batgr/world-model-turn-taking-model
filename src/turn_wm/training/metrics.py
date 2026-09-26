"""
Validation metrics of the LeWM objective, aggregated over a whole epoch.

Everything is accumulated as sums (float64) and turned into metrics once, at
the end of the epoch, so results do not depend on the batch size. In
particular a skill score is the ratio of epoch-level MSEs,

    skill = 1 - MSE_model / MSE_persistence,

never an average of per-batch skills. Global metrics are computed from all
elements together, never as an average of per-corpus metrics.

Baselines ("persistence"):
    teacher forcing   z_hat[t + 1] = z[t]           (the step's own input)
    rollout, horizon  z_hat[C + h - 1] = z[C - 1]   (last ground-truth
                                                     context latent)

The metrics reuse the latents and predictions `lejepa_forward` computed for
the losses: the predictor never runs again.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

GLOBAL = ""
EPS = torch.finfo(torch.float32).eps


def skill_score(model_mse: float, persistence_mse: float) -> float:
    """`1 - model / persistence`, unclipped; eps guards a zero baseline."""

    return 1.0 - model_mse / max(persistence_mse, EPS)


def effective_rank(latents: torch.Tensor) -> float:
    """exp(entropy) of the normalized singular values of centred `latents` [N, D].

    1 for a rank-1 cloud, up to D for variance spread evenly over D directions.
    """

    if latents.shape[0] < 2:
        return float("nan")

    centred = latents.double() - latents.double().mean(dim=0)

    return entropy_rank(torch.linalg.svdvals(centred))


def entropy_rank(weights: torch.Tensor) -> float:
    """exp(entropy) of non-negative `weights` normalized to sum to 1.

    Scale-free: `weights` and `c * weights` give the same rank. On singular
    values it is `effective_rank`; on covariance eigenvalues, a variance-based
    rank. 0 when every weight is (numerically) zero.
    """

    weights = weights.double()
    total = weights.sum()

    if total <= EPS:
        return 0.0

    p = weights / total
    p = p[p > 0]

    return float(torch.exp(-(p * p.log()).sum()))


@dataclass
class _ErrorSums:
    """Squared errors of a model and its persistence baseline on one set."""

    model: float = 0.0
    persistence: float = 0.0
    elements: int = 0

    def add(self, prediction, persistence, target) -> None:
        self.model += float((prediction - target).pow(2).sum())
        self.persistence += float((persistence - target).pow(2).sum())
        self.elements += target.numel()

    def metrics(self, prefix: str) -> dict[str, float]:
        if not self.elements:
            return {}

        model = self.model / self.elements
        persistence = self.persistence / self.elements

        return {
            f"{prefix}mse": model,
            f"{prefix}persistence_mse": persistence,
            f"{prefix}skill": skill_score(model, persistence),
        }


@dataclass
class _HorizonSums:
    errors: _ErrorSums = field(default_factory=_ErrorSums)
    cosine: float = 0.0
    target_delta_norm: float = 0.0
    prediction_delta_norm: float = 0.0
    positions: int = 0


class _LatentSample:
    """Uniform random sample of at most `size` rows, deterministic per epoch.

    Every row gets a random priority from a fixed-seed generator and the
    `size` highest priorities are kept, so only `size` rows are stored.
    """

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
    """Epoch accumulator for the validation metrics; one per validation epoch.

    `update` takes one batch: trajectory latents [B, T, D], teacher-forcing
    predictions [B, C, D], rollout predictions {h: [B, D]}, the context length
    C and each sample's corpus name. `mask` [B, T] (optional, all valid by
    default) marks the positions the losses supervise; model and baseline
    are always measured on the same positions.
    """

    def __init__(
        self,
        horizons: Sequence[int],
        *,
        persistence_baseline: bool = True,
        cosine_similarity: bool = True,
        latent_health: bool = True,
        latent_rank_samples: int = 8192,
    ) -> None:
        self.horizons = sorted(horizons)
        self.persistence_baseline = persistence_baseline
        self.cosine_similarity = cosine_similarity
        self.latent_health = latent_health
        self.tf: dict[str, _ErrorSums] = defaultdict(_ErrorSums)
        self.rollout: dict[tuple[str, int], _HorizonSums] = defaultdict(_HorizonSums)
        self.latent_sum: torch.Tensor | None = None
        self.latent_square_sum: torch.Tensor | None = None
        self.latent_norm_sum = 0.0
        self.latent_count = 0
        self.prediction_norm_sum = 0.0
        self.prediction_count = 0
        self.sample = _LatentSample(latent_rank_samples if latent_health else 0)

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

        # Teacher forcing: z[:, :C] -> z[:, 1 : C + 1]; baseline z[t].
        tf_valid = valid[:, :c] & valid[:, 1 : c + 1]

        for name in groups:
            rows = tf_valid & members[name][:, None]
            self.tf[name].add(tf_pred[rows], z[:, :c][rows], z[:, 1 : c + 1][rows])

        # Rollout: z[:, C + h - 1] from the context ending at z[:, C - 1].
        last = z[:, c - 1]

        for h in self.horizons:
            if h not in rollout_predictions:
                continue

            pred = rollout_predictions[h].detach().double()
            target = z[:, c + h - 1]
            ok = valid[:, c - 1] & valid[:, c + h - 1]

            for name in groups:
                rows = ok & members[name]
                sums = self.rollout[(name, h)]
                sums.errors.add(pred[rows], last[rows], target[rows])
                sums.positions += int(rows.sum())

                if self.cosine_similarity:
                    sums.cosine += float(
                        F.cosine_similarity(
                            pred[rows], target[rows], dim=-1, eps=EPS
                        ).sum()
                    )

                sums.target_delta_norm += float(
                    (target[rows] - last[rows]).norm(dim=-1).sum()
                )
                sums.prediction_delta_norm += float(
                    (pred[rows] - last[rows]).norm(dim=-1).sum()
                )

            if self.latent_health:
                kept = pred[ok]
                self.prediction_norm_sum += float(kept.norm(dim=-1).sum())
                self.prediction_count += kept.shape[0]

        if self.latent_health:
            kept = z[valid]

            if self.latent_sum is None:
                self.latent_sum = torch.zeros(z.shape[-1], dtype=torch.float64)
                self.latent_square_sum = torch.zeros(z.shape[-1], dtype=torch.float64)

            assert self.latent_square_sum is not None
            self.latent_sum += kept.sum(dim=0).cpu()
            self.latent_square_sum += kept.pow(2).sum(dim=0).cpu()
            self.latent_norm_sum += float(kept.norm(dim=-1).sum())
            self.latent_count += kept.shape[0]
            self.sample.add(kept)

    def compute(self) -> dict[str, float]:
        """Metric name (without the `val/` stage) -> value, for this epoch."""

        metrics: dict[str, float] = {}

        for name, sums in sorted(self.tf.items()):
            prefix = f"{name}/" if name else ""
            values = sums.metrics(f"{prefix}tf_")

            if not self.persistence_baseline:
                values = {k: v for k, v in values.items() if k.endswith("tf_mse")}

            metrics.update(values)

        skills = []

        for (name, h), sums in sorted(self.rollout.items()):
            prefix = f"{name}/" if name else ""

            if not sums.positions:
                continue

            model = sums.errors.model / sums.errors.elements
            metrics[f"{prefix}rollout_{h}_mse"] = model

            if self.persistence_baseline:
                persistence = sums.errors.persistence / sums.errors.elements
                metrics[f"{prefix}persistence_{h}_mse"] = persistence
                metrics[f"{prefix}skill_{h}"] = skill_score(model, persistence)

                if not name:
                    skills.append(metrics[f"skill_{h}"])

            if not name:
                if self.cosine_similarity:
                    metrics[f"cosine_{h}"] = sums.cosine / sums.positions

                metrics[f"target_delta_norm_{h}"] = (
                    sums.target_delta_norm / sums.positions
                )
                metrics[f"prediction_delta_norm_{h}"] = (
                    sums.prediction_delta_norm / sums.positions
                )

        if skills:
            # Plain mean over horizons of the (epoch-aggregated) global skills.
            metrics["skill_mean"] = sum(skills) / len(skills)

        if self.latent_health and self.latent_count:
            assert self.latent_sum is not None and self.latent_square_sum is not None
            mean = self.latent_sum / self.latent_count
            variance = (
                self.latent_square_sum / self.latent_count - mean.pow(2)
            ).clamp_min(0)
            metrics["latent_std"] = float(variance.sqrt().mean())
            metrics["latent_norm"] = self.latent_norm_sum / self.latent_count
            metrics["effective_rank"] = effective_rank(self.sample.rows)

            if self.prediction_count:
                metrics["prediction_norm"] = (
                    self.prediction_norm_sum / self.prediction_count
                )

        return {name: value for name, value in metrics.items() if not math.isnan(value)}
