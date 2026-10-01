"""
Files, figures and report of the descriptive PCA (`pca.analyze_pca`).

`write_pca` writes `summary.json`, the 2D coordinates, the group metrics,
the action distribution, the scatter figures and `report.md`: dataset and
action structure of each representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from turn_wm.evaluation.latent_analysis.artifacts import snapshot_source
from turn_wm.evaluation.latent_analysis.pca import (
    ACTION,
    DATASET,
    DEFAULT_MAX_PLOT_SAMPLES,
    DEFAULT_SILHOUETTE_SAMPLES,
    PcaAnalysis,
    RepresentationPca,
    analyze_pca,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    SECONDARY_INK,
    SURFACE,
    close,
    colors,
    limits,
    number,
    percent,
    style,
)
from turn_wm.evaluation.latent_analysis.snapshot import (
    Snapshot,
)
from turn_wm.evaluation.latent_analysis.spectrum import (
    ALL,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1
PCA = "pca"


# Below this difference, two silhouettes are reported as similar.
SILHOUETTE_DIFFERENCE = 0.01


_DATASET_NOTE = (
    "Dataset-dependent structure is not interpreted as inherently undesirable "
    "because the corpora differ substantially in interaction setting and "
    "recording conditions. Later analyses will test whether conversational "
    "and turn-taking information is represented within and across these "
    "domains."
)


_OPEN_QUESTIONS = (
    "whether the latents encode useful conversational cues;",
    "whether such cues are shared across domains;",
    "whether their structure is domain-conditioned;",
    "whether they will be useful for planning.",
)


def pca_summary(
    analysis: PcaAnalysis,
    *,
    source: dict[str, Any],
    figures: dict[str, list[str]],
) -> dict[str, Any]:
    """Every PCA metric and setting, JSON-ready."""

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": PCA,
        "source": source,
        "settings": {
            "seed": analysis.seed,
            "samples": analysis.samples,
            "centering": "global mean of each representation",
            "standardization": None,
            "pca_fit": "every row, each representation independently",
            "silhouette": {
                "metric": "euclidean",
                "space": "full representation (not the 2D projection)",
                "max_samples": analysis.silhouette_samples,
                "selection": (
                    "rows of the condition with the smallest "
                    "blake2b(seed:sample_id) keys; natural class distribution"
                ),
            },
            "plot": {
                "max_samples": analysis.max_plot_samples,
                "group_floor": analysis.plot_group_floor,
                "strata": [DATASET, ACTION],
                "plotted": int(analysis.plotted.sum()),
                "selection": (
                    "smallest blake2b(seed:sample_id) keys, topped up so each "
                    "(dataset, action) group keeps min(group_floor, its size) "
                    "rows; figures only, never metrics"
                ),
            },
            "silhouette_difference_threshold": SILHOUETTE_DIFFERENCE,
        },
        "representations": {
            result.representation: {
                "dim": int(result.projection.components.shape[0]),
                "pca": result.projection.summary(),
                DATASET: None if result.dataset is None else result.dataset.summary(),
                ACTION: {
                    condition: structure.summary()
                    for condition, structure in result.actions.items()
                },
            }
            for result in analysis.representations
        },
        "action_distribution": _action_distribution(analysis.metadata),
        "figures": figures,
    }


def pca_coordinates(analysis: PcaAnalysis) -> pa.Table:
    """PC1/PC2 of every row of every representation; `plotted` marks the figures'."""

    rows = analysis.samples
    metadata = analysis.metadata
    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "sample_id",
            "representation",
            "pc1",
            "pc2",
            DATASET,
            ACTION,
            "sample_class",
            "plotted",
        )
    }

    for result in analysis.representations:
        coordinates = result.projection.coordinates
        columns["sample_id"] += list(metadata["sample_id"])
        columns["representation"] += [result.representation] * rows
        columns["pc1"] += coordinates[:, 0].tolist()
        columns["pc2"] += (
            coordinates[:, 1].tolist() if coordinates.shape[1] > 1 else [0.0] * rows
        )

        for name in (DATASET, ACTION, "sample_class"):
            values = metadata.get(name)
            columns[name] += [None] * rows if values is None else list(values)

        columns["plotted"] += analysis.plotted.tolist()

    return pa.table(columns)


def pca_group_metrics(analysis: PcaAnalysis) -> pa.Table:
    """One row per (representation, grouping, condition, group)."""

    rows = []

    for result in analysis.representations:
        structures = [result.dataset, *result.actions.values()]

        for structure in structures:
            if structure is None:
                continue

            fractions = structure.fractions

            for label in structure.labels:
                rows.append(
                    {
                        "representation": result.representation,
                        "grouping": structure.grouping,
                        "condition": structure.condition,
                        "group": label,
                        "count": structure.counts[label],
                        "fraction": fractions[label],
                        "within_variance": structure.within_variance[label],
                        "silhouette": structure.silhouette_by_label[label],
                    }
                )

    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("representation", pa.string()),
                ("grouping", pa.string()),
                ("condition", pa.string()),
                ("group", pa.string()),
                ("count", pa.int64()),
                ("fraction", pa.float64()),
                ("within_variance", pa.float64()),
                ("silhouette", pa.float64()),
            ]
        ),
    )


def pca_action_distribution(summary: dict[str, Any]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "condition": condition,
                ACTION: action,
                "count": distribution["counts"][action],
                "fraction": distribution["fractions"][action],
            }
            for condition, distribution in summary["action_distribution"].items()
            for action in distribution["counts"]
        ],
        schema=pa.schema(
            [
                ("condition", pa.string()),
                (ACTION, pa.string()),
                ("count", pa.int64()),
                ("fraction", pa.float64()),
            ]
        ),
    )


def pca_report(summary: dict[str, Any]) -> str:
    """Descriptive prose generated from `summary`; no interpretation."""

    representations = summary["representations"]
    names = list(representations)
    provenance = summary["source"].get("snapshot_provenance") or {}
    run = provenance.get("run") or {}
    checkpoint = provenance.get("checkpoint") or {}
    settings = summary["settings"]

    lines = [
        "# PCA analysis",
        "",
        (
            f"Run `{run.get('run_id', 'unknown')}`, checkpoint "
            f"`{checkpoint.get('filename', 'unknown')}` (step "
            f"{checkpoint.get('global_step', 'unknown')}), split "
            f"`{(provenance.get('data') or {}).get('split', 'unknown')}`. "
            f"{settings['samples']:,} rows; seed {settings['seed']}; figures draw "
            f"{settings['plot']['plotted']:,} rows; silhouettes use at most "
            f"{settings['silhouette']['max_samples']:,} rows per condition. "
            f"Snapshot SHA-256 `{summary['source']['representations_sha256'][:16]}…`."
        ),
        "",
        "## Representation geometry",
        "",
    ]

    def pcs(name: str) -> str:
        pca = representations[name]["pca"]
        return (
            f"{percent(pca['pc1_explained_variance'])} and "
            f"{percent(pca['pc2_explained_variance'])}"
        )

    first, others = names[0], names[1:]
    sentence = (
        f"In the `{first}` representation (D={representations[first]['dim']}), "
        f"PC1 and PC2 explain {pcs(first)} of the variance respectively"
    )

    if others:
        sentence += ", compared with " + "; ".join(
            f"{pcs(name)} for `{name}` (D={representations[name]['dim']})"
            for name in others
        )

    lines += [
        sentence + ". Each PCA is fitted independently; coordinates of "
        "different representations are not comparable.",
        "",
        "## Dataset structure",
        "",
    ]

    separations = {}

    for name in names:
        dataset = representations[name][DATASET]

        if dataset is None:
            lines.append(f"`{name}` has a single dataset: no dataset structure.")
            continue

        pair = max(dataset["centroid_distances"], key=lambda p: p["distance"])
        a, b = pair["groups"]
        separation = pair["over_pooled_within_rms"]
        separations[name] = separation
        lines.append(
            f"In `{name}`, the {a} and {b} centroids are separated by "
            f"{pair['distance']:.4g} units, corresponding to "
            f"{_times(separation)} the pooled within-dataset RMS dispersion. "
            f"Between-dataset variance is {percent(dataset['between_variance_fraction'])} "
            f"of the total variance (between-to-within variance ratio "
            f"{number(dataset['between_to_within_variance_ratio'])}); the dataset "
            f"silhouette is {_silhouette(dataset)}."
        )
        lines.append("")

    defined = {name: value for name, value in separations.items() if value is not None}

    if len(defined) > 1:
        ordered = sorted(defined, key=lambda name: defined[name], reverse=True)
        lines.append(
            "Relative to within-dataset dispersion, the centroid separation is "
            + ", then ".join(f"`{n}` ({_times(defined[n])})" for n in ordered)
            + ", from largest to smallest."
        )
        lines.append("")

    lines += [_DATASET_NOTE, "", "## Action structure", ""]

    for condition, distribution in summary["action_distribution"].items():
        counts = ", ".join(
            f"{action} {count:,} ({percent(distribution['fractions'][action])})"
            for action, count in distribution["counts"].items()
        )
        lines.append(f"- Action counts, {condition}: {counts}.")

    lines.append("")

    for name in names:
        actions = representations[name][ACTION]

        if not actions:
            continue

        lines.append(
            f"In `{name}`, the action silhouette is "
            + ", ".join(
                f"{_silhouette(structure)} "
                + ("over all rows" if condition == ALL else f"within {condition}")
                for condition, structure in actions.items()
            )
            + "."
        )
        comparison = _silhouette_comparison(
            {c: s["silhouette"] for c, s in actions.items() if c != ALL}
        )

        if comparison:
            lines.append(comparison)

        lines.append("")

    lines += [
        (
            "The silhouette ranges from -1 to 1; values near 0 indicate overlapping "
            "classes. Classes are not rebalanced, so the counts above weight the "
            "global values. Per-dataset figures use each representation's global "
            "PCA."
        ),
        "",
        "## Open questions",
        "",
        "This analysis does not yet determine:",
        "",
        *(f"- {question}" for question in _OPEN_QUESTIONS),
        "",
    ]

    return "\n".join(lines)


def _silhouette_comparison(values: dict[str, float | None]) -> str | None:
    defined = {c: v for c, v in values.items() if v is not None}

    if len(defined) < 2:
        return None

    high = max(defined, key=lambda name: defined[name])
    low = min(defined, key=lambda name: defined[name])
    difference = defined[high] - defined[low]

    if difference < SILHOUETTE_DIFFERENCE:
        return (
            f"Action separation is similar within {_and(list(defined))} according "
            f"to the silhouette coefficient (differences below "
            f"{SILHOUETTE_DIFFERENCE})."
        )

    return (
        f"Action separation is stronger within {high} than within {low} "
        f"according to the silhouette coefficient ({defined[high]:.3f} vs "
        f"{defined[low]:.3f})."
    )


def _action_distribution(metadata) -> dict[str, dict[str, Any]]:
    if ACTION not in metadata:
        return {}

    actions = [str(a) for a in metadata[ACTION]]
    datasets = [str(d) for d in metadata.get(DATASET, [ALL] * len(actions))]
    order = _action_order(metadata)
    conditions = {ALL: actions}

    if DATASET in metadata:
        for condition in sorted(set(datasets)):
            conditions[condition] = [
                a for a, d in zip(actions, datasets, strict=True) if d == condition
            ]

    distribution = {}

    for condition, values in conditions.items():
        counts = {a: values.count(a) for a in order if a in values}
        distribution[condition] = {
            "counts": counts,
            "fractions": {a: c / len(values) for a, c in counts.items()},
        }

    return distribution


def _action_order(metadata) -> list[str]:
    """Actions by id when ids are recorded, else by name."""

    actions = [str(a) for a in metadata[ACTION]]

    if "action_id" not in metadata:
        return sorted(set(actions))

    ids = {
        action: int(action_id)
        for action, action_id in zip(actions, metadata["action_id"], strict=True)
    }

    return sorted(ids, key=lambda action: ids[action])


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _silhouette(structure: dict[str, Any]) -> str:
    if structure["silhouette"] is None:
        return f"undefined ({structure['silhouette_undefined_reason']})"

    return (
        f"{structure['silhouette']:.3f} ({structure['silhouette_samples']:,} "
        "sampled rows)"
    )


def _times(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} times"


def pca_figures(analysis: PcaAnalysis) -> dict[str, dict[str, Figure]]:
    """Scatter plots by section: dataset, action, action_by_dataset.

    Each representation's figures share its axis limits, from every plotted
    row; the per-dataset figures show that dataset's rows in the global PCA.
    """

    metadata = analysis.metadata
    datasets = [str(d) for d in metadata[DATASET]] if DATASET in metadata else None
    actions = [str(a) for a in metadata[ACTION]] if ACTION in metadata else None
    dataset_colors = colors(sorted(set(datasets or [])))
    action_colors = colors(_action_order(metadata) if actions else [])
    plotted = analysis.plotted
    sections: dict[str, dict[str, Figure]] = {
        DATASET: {},
        ACTION: {},
        "action_by_dataset": {},
    }

    for result in analysis.representations:
        name = result.representation
        bounds = limits(result.projection.coordinates[plotted])
        every = plotted.nonzero().squeeze(1)

        if datasets is not None:
            sections[DATASET][f"{name}_dataset.png"] = _projection_scatter(
                result,
                every,
                datasets,
                colors=dataset_colors,
                counts=_counts(datasets),
                title="colored by dataset",
                limits=bounds,
            )

        if actions is None:
            continue

        sections[ACTION][f"{name}_action.png"] = _projection_scatter(
            result,
            every,
            actions,
            colors=action_colors,
            counts=_counts(actions),
            title="colored by action",
            limits=bounds,
        )

        for dataset in sorted(set(datasets or [])):
            inside = torch.tensor([d == dataset for d in datasets or []])
            key = f"{name}_action_{dataset}.png"
            sections["action_by_dataset"][key] = _projection_scatter(
                result,
                (plotted & inside).nonzero().squeeze(1),
                actions,
                colors=action_colors,
                counts=_counts(
                    [
                        a
                        for a, keep in zip(actions, inside.tolist(), strict=True)
                        if keep
                    ]
                ),
                title=f"colored by action · {dataset} only",
                limits=bounds,
            )

    return sections


def _projection_scatter(
    result: RepresentationPca,
    rows: torch.Tensor,
    labels: list[str],
    *,
    colors: dict[str, str],
    counts: dict[str, int],
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> Figure:
    """`rows` of one representation's projection, colored by `labels`."""

    pcs = result.projection.summary()

    return _scatter(
        result.projection.coordinates[rows],
        [labels[i] for i in rows.tolist()],
        colors=colors,
        counts=counts,
        title=f"{result.representation} · {title}",
        subtitle=(
            f"N = {len(rows):,} plotted of {sum(counts.values()):,} · "
            f"PC1 {percent(pcs['pc1_explained_variance'])} · "
            f"PC2 {percent(pcs['pc2_explained_variance'])}"
        ),
        limits=limits,
    )


def _scatter(
    coordinates: torch.Tensor,
    labels: list[str],
    *,
    colors: dict[str, str],
    counts: dict[str, int],
    title: str,
    subtitle: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> Figure:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    figure = Figure(figsize=(6.4, 5.6), facecolor=SURFACE)
    ax = figure.subplots()
    total = sum(counts.values())

    # The largest groups first, so the rare ones stay on top.
    for label in sorted(colors, key=lambda label: -counts.get(label, 0)):
        rows = [i for i, value in enumerate(labels) if value == label]

        if rows:
            points = coordinates[rows]
            ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                s=2,
                # Denser groups get more transparent points.
                alpha=min(0.6, max(0.08, 2_000 / len(rows))),
                color=colors[label],
                linewidths=0,
                rasterized=True,
            )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            color=colors[label],
            label=f"{label}  {counts[label]:,} ({percent(counts[label] / total)})",
        )
        for label in colors
        if counts.get(label)
    ]

    ax.legend(handles=handles, frameon=False, labelcolor=INK, fontsize=9)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel("PC1", color=SECONDARY_INK)
    ax.set_ylabel("PC2", color=SECONDARY_INK)
    ax.set_title(subtitle, color=SECONDARY_INK, fontsize=9)
    figure.suptitle(title, color=INK, fontsize=12)
    style(ax)
    figure.tight_layout()

    return figure


def _counts(labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    return counts


def write_pca(
    snapshot: Snapshot,
    output_dir: Path,
    *,
    silhouette_samples: int = DEFAULT_SILHOUETTE_SAMPLES,
    max_plot_samples: int = DEFAULT_MAX_PLOT_SAMPLES,
) -> None:
    """Compute the PCA analysis and write its tables, report and figures."""

    if "sample_id" not in snapshot.metadata:
        raise ValueError("PCA needs a sample_id column in metadata.parquet")

    analysis = analyze_pca(
        snapshot.representations,
        snapshot.metadata,
        seed=snapshot.seed,
        silhouette_samples=silhouette_samples,
        max_plot_samples=max_plot_samples,
    )
    figures = pca_figures(analysis)
    summary = pca_summary(
        analysis,
        source=snapshot_source(snapshot),
        figures={
            section: [f"figures/{name}" for name in items]
            for section, items in figures.items()
        },
    )

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(pca_report(summary), encoding="utf-8")
    pq.write_table(pca_coordinates(analysis), output_dir / "coordinates.parquet")
    pq.write_table(pca_group_metrics(analysis), output_dir / "group_metrics.parquet")
    pq.write_table(
        pca_action_distribution(summary), output_dir / "action_distribution.parquet"
    )

    for items in figures.values():
        for name, figure in items.items():
            figure.savefig(output_dir / "figures" / name, dpi=150, facecolor=SURFACE)
            close(figure)
