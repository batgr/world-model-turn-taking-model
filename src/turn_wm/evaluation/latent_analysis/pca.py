"""
Descriptive 2D PCA of representations, with their dataset and action structure.

For every representation, independently (coordinates of two representations
are never comparable):

- PCA on the centred rows, never standardized per dimension, fitted on
  every row; PC signs are fixed so the largest loading is positive;
- dataset structure: centroid distances, within- and between-group
  variance, and the silhouette of the dataset labels;
- action structure over all rows and within each dataset (always in the
  global projection and the full representation space), with the natural, unbalanced class counts.

Variance decomposition over the N rows of a condition, groups g of size n_g,
centroids c_g and overall centroid c (squared Euclidean norms, summed over
dimensions, so the trace of the covariance matrices):

    within_variance(g)       = mean_{i in g} ||x_i - c_g||^2
    pooled_within_variance   = sum_g n_g within_variance(g) / N
    between_variance         = sum_g n_g ||c_g - c||^2 / N
    total = pooled_within_variance + between_variance
    between_to_within_variance_ratio = between_variance / pooled_within_variance
    between_variance_fraction        = between_variance / total
    centroid_distance_over_pooled_within_rms
        = ||c_a - c_b|| / sqrt(pooled_within_variance)

Centroids and variances use every row. The silhouette (Euclidean, in the
full representation space, not the 2D projection) is O(n^2): it runs on at
most `silhouette_samples` rows of the condition, chosen by a seeded hash of
`sample_id`, so the choice does not depend on row order and keeps the
natural class distribution. It is undefined (None, with a reason) with fewer
than two classes or a class of fewer than two sampled rows.

This module only computes; files, figures and display live elsewhere.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import torch

from turn_wm.evaluation.latent_analysis.spectrum import ALL

DATASET = "dataset"
ACTION = "action"

DEFAULT_SILHOUETTE_SAMPLES = 10_000
DEFAULT_MAX_PLOT_SAMPLES = 20_000
# Every (dataset, action) group keeps at least this many plotted points
# (or all of them), so a rare group stays visible.
DEFAULT_PLOT_GROUP_FLOOR = 200

# Rows of the (n, n) distance matrix computed at once.
_SILHOUETTE_CHUNK = 512


@dataclass(frozen=True)
class Projection:
    """2D PCA of one representation, fitted on every row."""

    explained_variance_ratio: torch.Tensor  # (D,) float64, descending
    components: torch.Tensor  # (D, 2)
    coordinates: torch.Tensor  # (N, 2)

    def summary(self) -> dict[str, float]:
        ratio = self.explained_variance_ratio

        return {
            "pc1_explained_variance": float(ratio[0]),
            "pc2_explained_variance": float(ratio[1]) if len(ratio) > 1 else 0.0,
            "pc1_pc2_cumulative": float(ratio[:2].sum()),
        }


@dataclass(frozen=True)
class GroupStructure:
    """How rows grouped by one label spread around their centroids."""

    grouping: str  # the label column, e.g. "dataset" or "action"
    condition: str  # the rows considered: "all" or one dataset
    labels: list[str]
    counts: dict[str, int]
    within_variance: dict[str, float]
    pooled_within_variance: float
    between_variance: float
    centroid_distances: list[dict[str, Any]]
    silhouette: float | None
    silhouette_by_label: dict[str, float | None]
    silhouette_samples: int
    silhouette_undefined_reason: str | None

    @property
    def samples(self) -> int:
        return sum(self.counts.values())

    @property
    def fractions(self) -> dict[str, float]:
        return {label: count / self.samples for label, count in self.counts.items()}

    def summary(self) -> dict[str, Any]:
        total = self.pooled_within_variance + self.between_variance
        pair = self.centroid_distances[0] if len(self.labels) == 2 else None

        return {
            "samples": self.samples,
            "counts": self.counts,
            "fractions": self.fractions,
            "within_variance": self.within_variance,
            "pooled_within_variance": self.pooled_within_variance,
            "between_variance": self.between_variance,
            "between_to_within_variance_ratio": _ratio(
                self.between_variance, self.pooled_within_variance
            ),
            "between_variance_fraction": _ratio(self.between_variance, total),
            # With exactly two groups, their distance; every pair below.
            "centroid_distance": None if pair is None else pair["distance"],
            "centroid_distance_over_pooled_within_rms": (
                None if pair is None else pair["over_pooled_within_rms"]
            ),
            "centroid_distances": self.centroid_distances,
            "silhouette": self.silhouette,
            "silhouette_samples": self.silhouette_samples,
            "silhouette_undefined_reason": self.silhouette_undefined_reason,
        }


@dataclass(frozen=True)
class RepresentationPca:
    representation: str
    projection: Projection
    dataset: GroupStructure | None
    actions: dict[str, GroupStructure]  # condition -> structure


@dataclass(frozen=True)
class PcaAnalysis:
    """Everything computed, before any file or figure."""

    representations: list[RepresentationPca]
    metadata: Mapping[str, Sequence[Any]]
    plotted: torch.Tensor  # (N,) bool, rows drawn in the figures
    seed: int
    silhouette_samples: int
    max_plot_samples: int
    plot_group_floor: int

    @property
    def samples(self) -> int:
        return len(self.plotted)


def analyze_pca(
    representations: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Sequence[Any]],
    *,
    seed: int,
    silhouette_samples: int = DEFAULT_SILHOUETTE_SAMPLES,
    max_plot_samples: int = DEFAULT_MAX_PLOT_SAMPLES,
    plot_group_floor: int = DEFAULT_PLOT_GROUP_FLOOR,
) -> PcaAnalysis:
    """PCA and group structure of every representation.

    `metadata` needs `sample_id`; `dataset` and `action` add their
    structure when present.
    """

    if silhouette_samples < 2:
        raise ValueError("silhouette_samples must be at least 2")

    keys = sample_keys(metadata["sample_id"], seed=seed)
    datasets = _labels(metadata, DATASET)
    actions = _labels(metadata, ACTION)
    conditions = [ALL, *(sorted(set(datasets)) if datasets else [])]

    results = []

    for name, rows in representations.items():
        if len(rows) != len(keys):
            raise ValueError(
                f"{name!r} has {len(rows)} rows for {len(keys)} metadata rows"
            )

        x = rows.to(torch.float64)
        dataset_structure = None

        if datasets is not None and len(set(datasets)) > 1:
            dataset_structure = group_structure(
                x,
                datasets,
                keys=keys,
                grouping=DATASET,
                condition=ALL,
                silhouette_samples=silhouette_samples,
            )

        action_structures = {}

        if actions is not None:
            for condition in conditions:
                mask = (
                    None
                    if condition == ALL
                    else torch.tensor([d == condition for d in datasets or []])
                )
                action_structures[condition] = group_structure(
                    x if mask is None else x[mask],
                    actions if mask is None else _select(actions, mask),
                    keys=keys if mask is None else keys[mask],
                    grouping=ACTION,
                    condition=condition,
                    silhouette_samples=silhouette_samples,
                )

        results.append(
            RepresentationPca(
                representation=name,
                projection=pca_2d(x),
                dataset=dataset_structure,
                actions=action_structures,
            )
        )

    return PcaAnalysis(
        representations=results,
        metadata=metadata,
        plotted=plot_selection(
            keys,
            strata=_strata(datasets, actions, len(keys)),
            max_samples=max_plot_samples,
            group_floor=plot_group_floor,
        ),
        seed=seed,
        silhouette_samples=silhouette_samples,
        max_plot_samples=max_plot_samples,
        plot_group_floor=plot_group_floor,
    )


def pca_2d(rows: torch.Tensor) -> Projection:
    """Project centred (not standardized) rows on their first two PCs."""

    x = rows.to(torch.float64)
    samples, dim = x.shape

    if samples < 2:
        raise ValueError(f"PCA needs at least 2 rows, got {samples}")

    centred = x - x.mean(dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(centred.T @ centred / (samples - 1))
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    components = eigenvectors.flip(1)[:, : min(2, dim)]

    # An eigenvector's sign is arbitrary: make its largest loading positive.
    largest = components.abs().argmax(dim=0)
    signs = components[largest, torch.arange(components.shape[1])].sign()
    components = components * torch.where(signs == 0, 1.0, signs)

    total = eigenvalues.sum()
    ratio = eigenvalues / total if total > 0 else torch.zeros_like(eigenvalues)

    return Projection(
        explained_variance_ratio=ratio,
        components=components,
        coordinates=centred @ components,
    )


def group_structure(
    x: torch.Tensor,
    labels: Sequence[str],
    *,
    keys: torch.Tensor,
    grouping: str,
    condition: str,
    silhouette_samples: int,
) -> GroupStructure:
    """Centroids, variances and silhouette of `x` (N, D) grouped by `labels`.

    `silhouette_samples=0` skips the silhouette.
    """

    classes = sorted(set(labels))
    index = {label: i for i, label in enumerate(classes)}
    y = torch.tensor([index[label] for label in labels])
    x = x.to(torch.float64)

    counts = torch.bincount(y, minlength=len(classes))
    centroids = torch.zeros(len(classes), x.shape[1], dtype=torch.float64)
    centroids.index_add_(0, y, x)
    centroids /= counts.unsqueeze(1)

    squared = (x - centroids[y]).pow(2).sum(dim=1)
    within = torch.zeros(len(classes), dtype=torch.float64).index_add_(0, y, squared)
    pooled_within = float(within.sum()) / len(x)
    within /= counts

    overall = x.mean(dim=0)
    between = float((counts * (centroids - overall).pow(2).sum(dim=1)).sum() / len(x))

    pooled_rms = pooled_within**0.5
    distances = [
        {
            "groups": [a, b],
            "distance": (d := float((centroids[i] - centroids[j]).norm())),
            "over_pooled_within_rms": _ratio(d, pooled_rms),
        }
        for (i, a), (j, b) in combinations(enumerate(classes), 2)
    ]

    sampled = select_by_key(keys, silhouette_samples)

    if silhouette_samples == 0:
        value, by_class, reason = None, {}, "not computed"
    else:
        value, by_class, reason = silhouette(
            x[sampled], [labels[i] for i in sampled.tolist()]
        )

    return GroupStructure(
        grouping=grouping,
        condition=condition,
        labels=classes,
        counts={label: int(counts[i]) for label, i in index.items()},
        within_variance={label: float(within[i]) for label, i in index.items()},
        pooled_within_variance=pooled_within,
        between_variance=between,
        centroid_distances=distances,
        silhouette=value,
        silhouette_by_label={label: by_class.get(label) for label in classes},
        silhouette_samples=len(sampled),
        silhouette_undefined_reason=reason,
    )


def silhouette(
    x: torch.Tensor,
    labels: Sequence[str],
) -> tuple[float | None, dict[str, float], str | None]:
    """Mean Euclidean silhouette, per-class means, and why it is undefined.

    Computed in chunks of rows: never an (n, n) matrix at once.
    """

    counts = Counter(labels)

    if len(counts) < 2:
        return None, {}, "fewer than 2 classes"

    too_small = sorted(label for label, count in counts.items() if count < 2)

    if too_small:
        return None, {}, f"classes with fewer than 2 samples: {too_small}"

    classes = sorted(counts)
    y = torch.tensor([classes.index(label) for label in labels])
    x = x.to(torch.float64)
    sizes = torch.bincount(y, minlength=len(classes)).to(torch.float64)
    one_hot = torch.nn.functional.one_hot(y, len(classes)).to(torch.float64)
    scores = torch.empty(len(x), dtype=torch.float64)

    for start in range(0, len(x), _SILHOUETTE_CHUNK):
        rows = slice(start, start + _SILHOUETTE_CHUNK)
        own = y[rows]
        sums = torch.cdist(x[rows], x) @ one_hot  # (chunk, K)

        # Mean distance to the other members of the own class (self is 0).
        a = sums.gather(1, own.unsqueeze(1)).squeeze(1) / (sizes[own] - 1)
        means = sums / sizes
        means.scatter_(1, own.unsqueeze(1), float("inf"))
        b = means.min(dim=1).values

        largest = torch.maximum(a, b)
        scores[rows] = torch.where(largest > 0, (b - a) / largest, 0.0)

    by_class = {label: float(scores[y == i].mean()) for i, label in enumerate(classes)}

    return float(scores.mean()), by_class, None


def sample_keys(sample_ids: Sequence[str], *, seed: int) -> torch.Tensor:
    """A seeded pseudo-random key per sample, independent of row order."""

    return torch.tensor(
        [
            int.from_bytes(
                hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=7).digest()
            )
            for sample_id in sample_ids
        ],
        dtype=torch.int64,
    )


def select_by_key(keys: torch.Tensor, count: int) -> torch.Tensor:
    """Indices of the `count` smallest keys, in key order."""

    return keys.argsort()[:count]


def plot_selection(
    keys: torch.Tensor,
    *,
    strata: Sequence[tuple[str, ...]],
    max_samples: int,
    group_floor: int,
) -> torch.Tensor:
    """Rows drawn in the figures: a seeded uniform sample, topped up per stratum.

    Each stratum keeps at least `group_floor` rows (or all of its rows), so
    the total may slightly exceed `max_samples`.
    """

    plotted = torch.zeros(len(keys), dtype=torch.bool)
    plotted[select_by_key(keys, max_samples)] = True

    by_stratum: dict[tuple[str, ...], list[int]] = {}

    for row, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(row)

    for rows in by_stratum.values():
        members = torch.tensor(rows)
        missing = min(group_floor, len(rows)) - int(plotted[members].sum())

        if missing > 0:
            candidates = members[~plotted[members]]
            plotted[candidates[select_by_key(keys[candidates], missing)]] = True

    return plotted


def _labels(metadata: Mapping[str, Sequence[Any]], column: str) -> list[str] | None:
    if column not in metadata:
        return None

    return [str(value) for value in metadata[column]]


def _select(values: Sequence[str], mask: torch.Tensor) -> list[str]:
    return [value for value, keep in zip(values, mask.tolist(), strict=True) if keep]


def _strata(
    datasets: list[str] | None,
    actions: list[str] | None,
    rows: int,
) -> list[tuple[str, ...]]:
    columns = [column for column in (datasets, actions) if column is not None]

    return [tuple(column[i] for column in columns) for i in range(rows)]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None
