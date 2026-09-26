"""
Descriptive structure of representations conditioned on conversational labels.

For every representation, variable and condition (all rows, then each
corpus), on the rows where the variable is valid:

- categorical variables: class counts, the variance decomposition and
  centroid distances of `pca.group_structure`, the silhouette with the
  natural class distribution (`silhouette_natural`), and, as a secondary
  diagnostic, `silhouette_balanced` on the same number of rows per class
  (at most `balanced_cap`, chosen by the seeded sample keys);
- continuous variables: distribution and quantiles, Spearman correlation
  with the representation's PC1 and PC2 (the same global `pca_2d` as the
  PCA analysis, never refitted per label), and the variance decomposition
  over deterministic quantile bins. Bins are descriptive, not a probe.

Comparisons are scale-free: `between_variance_fraction` (between-group over
total variance) is the strength of a structure; `latent - features` deltas
compare the two spaces, never their coordinates. Domain structure is
classified by a documented heuristic on the per-corpus strengths:

    weak_in_both          max strength < WEAK_FRACTION
    similar               min / max strength >= SIMILAR_RATIO
    strong_but_different  otherwise

This module only computes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    LabelVariable,
)
from turn_wm.evaluation.latent_analysis.pca import (
    Projection,
    group_structure,
    pca_2d,
    sample_keys,
    silhouette,
)
from turn_wm.evaluation.latent_analysis.spectrum import ALL

DEFAULT_BALANCED_CAP = 1_000
DEFAULT_BINS = 10
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
# Figures colour continuous values clipped to these quantiles of all rows.
CLIP_QUANTILES = (0.01, 0.99)

WEAK_FRACTION = 0.01
SIMILAR_RATIO = 0.5
# latent vs features: "similar" below this absolute and relative change.
CHANGE_ABSOLUTE = 0.002
CHANGE_RELATIVE = 0.10


@dataclass(frozen=True)
class LabelAnalysis:
    """Every metric, keyed variable -> representation -> condition."""

    variables: list[LabelVariable]
    projections: dict[str, Projection]
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]]
    deltas: dict[str, dict[str, dict[str, float | None]]]  # variable -> condition
    domain: dict[str, dict[str, dict[str, Any]]]  # variable -> representation
    clip: dict[str, tuple[float, float]]  # continuous variable -> colour range
    conditions: list[str]
    settings: dict[str, Any]


def analyze_labels(
    representations: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Sequence[Any]],
    variables: Sequence[LabelVariable],
    *,
    seed: int,
    silhouette_samples: int,
    balanced_cap: int = DEFAULT_BALANCED_CAP,
    bins: int = DEFAULT_BINS,
) -> LabelAnalysis:
    keys = sample_keys(metadata["sample_id"], seed=seed)
    datasets = [str(d) for d in metadata["dataset"]]
    conditions = [ALL, *sorted(set(datasets))]
    in_condition = {
        condition: torch.tensor([condition in (ALL, d) for d in datasets])
        for condition in conditions
    }
    projections = {name: pca_2d(rows) for name, rows in representations.items()}
    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    clip = {}

    for variable in variables:
        valid = torch.tensor([v is not None for v in variable.values])

        if variable.kind != CATEGORICAL:
            values = torch.tensor(
                [float("nan") if v is None else v for v in variable.values],
                dtype=torch.float64,
            )
            if bool(valid.any()):
                clip[variable.name] = tuple(
                    torch.quantile(
                        values[valid], torch.tensor(CLIP_QUANTILES, dtype=torch.float64)
                    ).tolist()
                )

        for name, rows in representations.items():
            x = rows.to(torch.float64)

            for condition in conditions:
                mask = valid & in_condition[condition]
                index = mask.nonzero().squeeze(1)

                if variable.kind == CATEGORICAL:
                    result = categorical_metrics(
                        x[index],
                        [variable.values[i] for i in index.tolist()],
                        keys=keys[index],
                        classes=variable.classes or (),
                        silhouette_samples=silhouette_samples,
                        balanced_cap=balanced_cap,
                        name=variable.name,
                        condition=condition,
                    )
                else:
                    result = continuous_metrics(
                        x[index],
                        values[index],
                        coordinates=projections[name].coordinates[index],
                        keys=keys[index],
                        bins=bins,
                        name=variable.name,
                        condition=condition,
                    )

                metrics.setdefault(variable.name, {}).setdefault(name, {})[
                    condition
                ] = result

    return LabelAnalysis(
        variables=list(variables),
        projections=projections,
        metrics=metrics,
        deltas=_deltas(metrics, list(representations), conditions),
        domain={
            variable.name: {
                name: domain_structure(
                    {
                        c: strength(m)
                        for c, m in metrics[variable.name][name].items()
                        if c != ALL
                    }
                )
                | {
                    "direction_cosine": _direction_cosine(
                        representations[name], variable, datasets
                    )
                }
                for name in representations
            }
            for variable in variables
        },
        clip=clip,
        conditions=conditions,
        settings={
            "seed": seed,
            "silhouette_samples": silhouette_samples,
            "balanced_cap": balanced_cap,
            "bins": bins,
            "quantiles": list(QUANTILES),
            "clip_quantiles": list(CLIP_QUANTILES),
            "weak_fraction": WEAK_FRACTION,
            "similar_ratio": SIMILAR_RATIO,
            "change_absolute": CHANGE_ABSOLUTE,
            "change_relative": CHANGE_RELATIVE,
        },
    )


def categorical_metrics(
    x: torch.Tensor,
    labels: list[str],
    *,
    keys: torch.Tensor,
    classes: Sequence[str],
    silhouette_samples: int,
    balanced_cap: int,
    name: str,
    condition: str,
) -> dict[str, Any]:
    if len(labels) < 2:
        return {"samples": len(labels), "undefined_reason": "fewer than 2 valid rows"}

    structure = group_structure(
        x,
        labels,
        keys=keys,
        grouping=name,
        condition=condition,
        silhouette_samples=silhouette_samples,
    ).summary()
    balanced, per_class, reason = balanced_silhouette(
        x, labels, keys=keys, cap=balanced_cap
    )
    distances = [
        pair["over_pooled_within_rms"]
        for pair in structure["centroid_distances"]
        if pair["over_pooled_within_rms"] is not None
    ]
    order = [c for c in classes if c in structure["counts"]]
    natural = {
        "silhouette_natural": structure.pop("silhouette"),
        "silhouette_natural_samples": structure.pop("silhouette_samples"),
        "silhouette_natural_undefined_reason": structure.pop(
            "silhouette_undefined_reason"
        ),
    }

    return {
        **structure,
        "counts": {c: structure["counts"][c] for c in order},
        "fractions": {c: structure["fractions"][c] for c in order},
        "mean_centroid_distance_over_pooled_within_rms": (
            sum(distances) / len(distances) if distances else None
        ),
        **natural,
        "silhouette_balanced": balanced,
        "silhouette_balanced_per_class": per_class,
        "silhouette_balanced_undefined_reason": reason,
    }


def balanced_silhouette(
    x: torch.Tensor,
    labels: Sequence[str],
    *,
    keys: torch.Tensor,
    cap: int,
) -> tuple[float | None, int, str | None]:
    """Silhouette on min(cap, smallest class) rows of every class."""

    classes = sorted(set(labels))

    if len(classes) < 2:
        return None, 0, "fewer than 2 classes"

    members = {c: [i for i, label in enumerate(labels) if label == c] for c in classes}
    per_class = min(cap, *(len(rows) for rows in members.values()))

    if per_class < 2:
        return None, per_class, "a class has fewer than 2 rows"

    chosen = []

    for rows in members.values():
        rows_tensor = torch.tensor(rows)
        order = keys[rows_tensor].argsort()[:per_class]
        chosen += rows_tensor[order].tolist()

    chosen.sort(key=lambda i: int(keys[i]))
    value, _, reason = silhouette(x[chosen], [labels[i] for i in chosen])

    return value, per_class, reason


def continuous_metrics(
    x: torch.Tensor,
    values: torch.Tensor,
    *,
    coordinates: torch.Tensor,
    keys: torch.Tensor,
    bins: int,
    name: str,
    condition: str,
) -> dict[str, Any]:
    count = len(values)

    if count < 2:
        return {"samples": count, "undefined_reason": "fewer than 2 valid rows"}

    quantiles = torch.quantile(values, torch.tensor(QUANTILES, dtype=torch.float64))
    edges = torch.unique(
        torch.quantile(values, torch.linspace(0, 1, bins + 1, dtype=torch.float64))
    )
    # Bin b holds edges[b] <= v < edges[b + 1]; the maximum joins the last bin.
    labels = torch.bucketize(values, edges[1:-1], right=True).tolist()
    structure = (
        group_structure(
            x,
            [f"bin_{b}" for b in labels],
            keys=keys,
            grouping=name,
            condition=condition,
            silhouette_samples=0,
        ).summary()
        if len(set(labels)) > 1
        else None
    )

    return {
        "samples": count,
        "mean": float(values.mean()),
        "std": float(values.std()) if count > 1 else None,
        "quantiles": {
            f"{q:g}": float(v)
            for q, v in zip(QUANTILES, quantiles.tolist(), strict=True)
        },
        "spearman_pc1": spearman(values, coordinates[:, 0]),
        "spearman_pc2": (
            spearman(values, coordinates[:, 1]) if coordinates.shape[1] > 1 else None
        ),
        "bins": len(set(labels)),
        "bin_edges": edges.tolist(),
        "between_bin_variance_fraction": (
            None if structure is None else structure["between_variance_fraction"]
        ),
        "between_to_within_bin_variance_ratio": (
            None if structure is None else structure["between_to_within_variance_ratio"]
        ),
        "pooled_within_bin_variance": (
            None if structure is None else structure["pooled_within_variance"]
        ),
        "between_bin_variance": (
            None if structure is None else structure["between_variance"]
        ),
    }


def spearman(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Spearman rank correlation, ties given their average rank."""

    ra, rb = _ranks(a.to(torch.float64)), _ranks(b.to(torch.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = (ra.pow(2).sum() * rb.pow(2).sum()).sqrt()

    return None if denominator == 0 else float((ra * rb).sum() / denominator)


def strength(metrics: Mapping[str, Any]) -> float | None:
    """The scale-free strength of a structure: its between-variance fraction."""

    if "between_bin_variance_fraction" in metrics:
        return metrics["between_bin_variance_fraction"]

    return metrics.get("between_variance_fraction")


def domain_structure(strengths: Mapping[str, float | None]) -> dict[str, Any]:
    """Classify per-corpus strengths: weak_in_both, similar or strong_but_different."""

    defined = {c: s for c, s in strengths.items() if s is not None}

    if len(defined) < 2:
        return {"class": "undetermined", "strengths": dict(strengths)}

    high, low = max(defined.values()), min(defined.values())

    if high < WEAK_FRACTION:
        kind = "weak_in_both"
    elif low / high >= SIMILAR_RATIO:
        kind = "similar"
    else:
        kind = "strong_but_different"

    return {
        "class": kind,
        "strengths": dict(strengths),
        "strongest": max(defined, key=defined.get),
    }


def change(features: float | None, latent: float | None) -> str | None:
    """How a strength changes from features to latent."""

    if features is None or latent is None:
        return None

    difference = latent - features

    if abs(difference) < CHANGE_ABSOLUTE or abs(difference) < CHANGE_RELATIVE * max(
        abs(features), abs(latent)
    ):
        return "similar"

    return "stronger" if difference > 0 else "weaker"


def _deltas(metrics, representations: list[str], conditions: list[str]):
    """latent - features for every variable and condition (when both exist)."""

    if not {"features", "latent"} <= set(representations):
        return {}

    fields = (
        (
            "delta_normalized_centroid_distance",
            "mean_centroid_distance_over_pooled_within_rms",
        ),
        ("delta_silhouette_natural", "silhouette_natural"),
        ("delta_silhouette_balanced", "silhouette_balanced"),
        ("delta_between_variance_fraction", "between_variance_fraction"),
        ("delta_abs_spearman_pc1", "spearman_pc1"),
        ("delta_abs_spearman_pc2", "spearman_pc2"),
        ("delta_between_bin_variance_fraction", "between_bin_variance_fraction"),
    )
    deltas = {}

    for variable, by_representation in metrics.items():
        for condition in conditions:
            features = by_representation["features"][condition]
            latent = by_representation["latent"][condition]
            row: dict[str, Any] = {}

            for delta, source in fields:
                if source not in features and source not in latent:
                    continue
                a, b = features.get(source), latent.get(source)

                if delta.startswith("delta_abs_"):
                    a = None if a is None else abs(a)
                    b = None if b is None else abs(b)

                row[delta] = None if a is None or b is None else b - a

            row["change"] = change(strength(features), strength(latent))
            deltas.setdefault(variable, {})[condition] = row

    return deltas


def _direction_cosine(
    rows: torch.Tensor,
    variable: LabelVariable,
    datasets: Sequence[str],
) -> float | None:
    """Cosine between two corpora's class-mean differences, for two classes.

    Whether the direction separating the classes is shared across corpora;
    None unless the variable has two classes and exactly two corpora.
    """

    corpora = sorted(set(datasets))

    if variable.classes is None or len(variable.classes) != 2 or len(corpora) != 2:
        return None

    first, second = variable.classes
    x = rows.to(torch.float64)
    directions = []

    for corpus in corpora:
        means = []

        for label in (first, second):
            index = [
                i
                for i, (d, v) in enumerate(zip(datasets, variable.values, strict=True))
                if d == corpus and v == label
            ]
            if not index:
                return None
            means.append(x[index].mean(dim=0))

        directions.append(means[1] - means[0])

    a, b = directions
    norms = a.norm() * b.norm()

    return None if norms == 0 else float((a @ b) / norms)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    # Ties: every member of a run of equal values gets the run's mean rank.
    _, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(counts), dtype=torch.float64).index_add_(0, inverse, ranks)

    return (sums / counts)[inverse]
