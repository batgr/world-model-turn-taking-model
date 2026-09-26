"""
Spectral geometry of representations: how their variance spreads over
directions.

Rows are only centred, never standardized per dimension, so the spectrum is
the geometry the model actually learned. The covariance

    C = Xc.T @ Xc / (N - 1),    Xc = X - mean(X)

is built in float64 and diagonalized with `eigvalsh` (D x D, not N x D); the
singular values of Xc follow as sqrt((N - 1) * eigenvalues).

Three ranks, each also as a fraction of D so that spaces of different
dimension compare:

- `effective_rank_singular`: exp(entropy) of the normalized singular values,
  the validation metric `effective_rank` of training (same formula,
  `training.metrics.entropy_rank`);
- `spectral_entropy_rank`: exp(entropy) of the normalized eigenvalues, i.e.
  of the explained variance;
- `participation_ratio`: (sum lambda)^2 / sum lambda^2.

This module only computes; reading artifacts and rendering live in
`analyze`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from turn_wm.training.metrics import entropy_rank

ALL = "all"

# Cumulative explained variance reported at these numbers of components.
CUMULATIVE_AT = (1, 5, 10, 25, 50, 100)

# Fewest components explaining at least these fractions of the variance.
VARIANCE_THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class Spectrum:
    """Covariance spectrum of one representation over one group of rows."""

    representation: str
    group: str
    samples: int
    mean_norm: float
    mean_dimension_std: float
    eigenvalues: torch.Tensor  # (D,) float64, descending, >= 0

    @property
    def dim(self) -> int:
        return len(self.eigenvalues)

    @property
    def singular_values(self) -> torch.Tensor:
        """Singular values of the centred rows."""

        return (self.eigenvalues * (self.samples - 1)).sqrt()

    @property
    def total_variance(self) -> float:
        return float(self.eigenvalues.sum())

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        total = self.eigenvalues.sum()

        if total <= 0:
            return torch.zeros_like(self.eigenvalues)

        return self.eigenvalues / total

    @property
    def cumulative_explained_variance(self) -> torch.Tensor:
        # Round-off may push the last sums a hair above 1.
        return self.explained_variance_ratio.cumsum(0).clamp_max(1.0)

    def dimensions_for(self, fraction: float) -> int:
        """Fewest leading components explaining at least `fraction`."""

        cumulative = self.cumulative_explained_variance
        # Tolerates the float error of a cumulative sum that should reach 1.
        index = int(torch.searchsorted(cumulative, fraction - 1e-12))

        return min(index + 1, self.dim)

    def summary(self) -> dict[str, Any]:
        """Aggregated metrics, JSON-ready."""

        dim = self.dim
        effective_rank_singular = entropy_rank(self.singular_values)
        spectral_entropy_rank = entropy_rank(self.eigenvalues)
        participation_ratio = _participation_ratio(self.eigenvalues)
        cumulative = self.cumulative_explained_variance

        return {
            "samples": self.samples,
            "dim": dim,
            "mean_norm": self.mean_norm,
            "mean_dimension_std": self.mean_dimension_std,
            "total_variance": self.total_variance,
            "effective_rank_singular": effective_rank_singular,
            "effective_rank_singular_fraction": effective_rank_singular / dim,
            "spectral_entropy_rank": spectral_entropy_rank,
            "spectral_entropy_rank_fraction": spectral_entropy_rank / dim,
            "participation_ratio": participation_ratio,
            "participation_ratio_fraction": participation_ratio / dim,
            "cumulative_explained_variance": {
                str(k): float(cumulative[k - 1]) for k in CUMULATIVE_AT if k <= dim
            },
            **{
                f"dimensions_for_{round(fraction * 100)}_percent": (
                    self.dimensions_for(fraction)
                )
                for fraction in VARIANCE_THRESHOLDS
            },
        }

    def rows(self) -> list[dict[str, Any]]:
        """One row per component, for a long-format table."""

        columns = zip(
            self.eigenvalues.tolist(),
            self.singular_values.tolist(),
            self.explained_variance_ratio.tolist(),
            self.cumulative_explained_variance.tolist(),
            strict=True,
        )

        return [
            {
                "representation": self.representation,
                "group": self.group,
                "component": component,
                "eigenvalue": eigenvalue,
                "singular_value": singular,
                "explained_variance_ratio": ratio,
                "cumulative_explained_variance": cumulative,
            }
            for component, (eigenvalue, singular, ratio, cumulative) in enumerate(
                columns, start=1
            )
        ]


def spectrum(
    rows: torch.Tensor,
    *,
    representation: str,
    group: str = ALL,
) -> Spectrum:
    """Spectrum of `rows` (N, D), centred on their own mean; `rows` is not modified."""

    if rows.ndim != 2:
        raise ValueError(
            f"{representation!r} must have shape (N, D), got {tuple(rows.shape)}"
        )

    samples, dim = rows.shape

    if samples < 2:
        raise ValueError(
            f"{representation!r} / {group!r} needs at least 2 rows, got {samples}"
        )

    x = rows.to(torch.float64)
    centred = x - x.mean(dim=0)
    covariance = centred.T @ centred / (samples - 1)

    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)
    # Round-off makes the null directions tiny (possibly negative) instead of
    # zero; below the usual numerical-rank tolerance they are zero.
    tolerance = eigenvalues[0].clamp_min(0) * dim * torch.finfo(torch.float64).eps
    eigenvalues = torch.where(eigenvalues > tolerance, eigenvalues, 0.0)

    return Spectrum(
        representation=representation,
        group=group,
        samples=samples,
        mean_norm=float(x.norm(dim=1).mean()),
        mean_dimension_std=float(x.std(dim=0).mean()),
        eigenvalues=eigenvalues,
    )


def analyze_spectra(
    representations: Mapping[str, torch.Tensor],
    *,
    groups: Sequence[str] | None = None,
) -> list[Spectrum]:
    """Spectra of every representation: over all rows, then per group.

    `groups` labels each row (e.g. its corpus); each group is centred on its
    own mean and keeps its natural size.
    """

    spectra = []

    for name, rows in representations.items():
        spectra.append(spectrum(rows, representation=name))

        if groups is None:
            continue

        if len(groups) != len(rows):
            raise ValueError(
                f"{len(groups)} group labels for {len(rows)} rows of {name!r}"
            )

        for group in sorted(set(groups)):
            if group == ALL:
                raise ValueError(f"{ALL!r} is reserved for the global analysis")

            mask = torch.tensor([label == group for label in groups])
            spectra.append(spectrum(rows[mask], representation=name, group=group))

    return spectra


def _participation_ratio(eigenvalues: torch.Tensor) -> float:
    squares = eigenvalues.pow(2).sum()

    if squares <= 0:
        return 0.0

    return float(eigenvalues.sum().pow(2) / squares)
