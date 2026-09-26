"""
Analyze an extracted representation snapshot (`extract.write_snapshot`).

The snapshot directory is the only input: no checkpoint, dataset or feature
cache is loaded. Each analysis runs in three stages, so its results can be
logged elsewhere (e.g. W&B) without recomputing them:

    computation (spectrum.py)  ->  structured results (summary, long table)
                               ->  local rendering (JSON, Parquet, figures)

Results go to `<snapshot>/analysis/<analysis>/` unless another output root
is given. Figures need matplotlib (`uv sync --extra analysis`); nothing else
does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from turn_wm.evaluation.latent_analysis.label_report import write_labels
from turn_wm.evaluation.latent_analysis.label_source import (
    CorpusLabelSource,
    hub_label_sources,
)
from turn_wm.evaluation.latent_analysis.label_structure import DEFAULT_BALANCED_CAP
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
    SERIES,
    SURFACE,
    close,
    colors,
    limits,
    percent,
    style,
)
from turn_wm.evaluation.latent_analysis.spectrum import (
    ALL,
    CUMULATIVE_AT,
    VARIANCE_THRESHOLDS,
    Spectrum,
    analyze_spectra,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1

SPECTRUM = "spectrum"
PCA = "pca"
LABELS = "labels"
ANALYSES = (SPECTRUM, PCA, LABELS)
# Run without --analysis: those reading the snapshot alone. The label
# analysis also reads the Hub, so it runs only when asked for.
DEFAULT_ANALYSES = (SPECTRUM, PCA)

# Metadata column whose values define the per-group analyses.
GROUP_COLUMN = "dataset"


@dataclass(frozen=True)
class Snapshot:
    """The contents of a snapshot directory."""

    path: Path
    representations: dict[str, torch.Tensor]
    metadata: dict[str, list[Any]]  # column -> one value per row
    manifest: dict[str, Any]

    @property
    def groups(self) -> list[str] | None:
        """The per-group label of each row (its corpus), if recorded."""

        if GROUP_COLUMN not in self.metadata:
            return None

        return [str(value) for value in self.metadata[GROUP_COLUMN]]

    @property
    def seed(self) -> int:
        """The seed the snapshot's samples were drawn with (0 if unrecorded)."""

        sampling = (self.manifest.get("provenance") or {}).get("sampling") or {}
        seed = sampling.get("seed")

        return 0 if seed is None else int(seed)


def read_snapshot(path: Path) -> Snapshot:
    """Read representations, metadata and manifest; nothing is written."""

    path = Path(path).expanduser()
    files = {
        name: path / name
        for name in (
            "representations.safetensors",
            "metadata.parquet",
            "manifest.json",
        )
    }

    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(f"Not a representation snapshot, missing {file}")

    return Snapshot(
        path=path,
        representations=load_file(files["representations.safetensors"]),
        metadata=pq.read_table(files["metadata.parquet"]).to_pydict(),
        manifest=json.loads(files["manifest.json"].read_text(encoding="utf-8")),
    )


def analyze_snapshot(
    path: Path,
    *,
    analyses: Sequence[str] = DEFAULT_ANALYSES,
    output_root: Path | None = None,
    silhouette_samples: int = DEFAULT_SILHOUETTE_SAMPLES,
    max_plot_samples: int = DEFAULT_MAX_PLOT_SAMPLES,
    balanced_cap: int = DEFAULT_BALANCED_CAP,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
) -> dict[str, Path]:
    """Run `analyses` on a snapshot; return each analysis's output directory.

    The snapshot itself is only read. The label analysis reads the label
    sidecars of the snapshot's dataset revision from the Hub (or
    `labels_revision`, explicitly), unless `label_sources` are given.
    """

    unknown = [name for name in analyses if name not in ANALYSES]

    if unknown:
        raise ValueError(f"Unknown analyses {unknown}; expected some of {ANALYSES}")

    # Before any computation: figures are part of every analysis's output.
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    snapshot = read_snapshot(path)

    if "rollout" in (snapshot.manifest.get("provenance") or {}):
        raise ValueError(
            f"{snapshot.path} is a rollout snapshot; analyze it with "
            "`turn-wm analyze-rollouts`"
        )

    output_root = (
        snapshot.path / "analysis" if output_root is None else Path(output_root)
    )
    outputs = {name: output_root / name for name in dict.fromkeys(analyses)}

    for output_dir in outputs.values():
        _require_empty(output_dir)

    for name, output_dir in outputs.items():
        if name == SPECTRUM:
            write_spectrum(snapshot, output_dir)
        elif name == PCA:
            write_pca(
                snapshot,
                output_dir,
                silhouette_samples=silhouette_samples,
                max_plot_samples=max_plot_samples,
            )
        elif name == LABELS:
            write_labels(
                snapshot.representations,
                snapshot.metadata,
                output_dir,
                source=_source(snapshot),
                sources=(
                    label_sources
                    if label_sources is not None
                    else hub_label_sources(
                        snapshot.manifest.get("provenance") or {},
                        labels_revision=labels_revision,
                    )
                ),
                seed=snapshot.seed,
                silhouette_samples=silhouette_samples,
                balanced_cap=balanced_cap,
                max_plot_samples=max_plot_samples,
            )

    return outputs


# ---------------------------------------------------------------------------
# Spectrum: structured results
# ---------------------------------------------------------------------------


def spectrum_summary(spectra: Sequence[Spectrum]) -> dict[str, Any]:
    """Aggregated metrics, keyed representation -> group."""

    summary: dict[str, dict[str, Any]] = {}

    for spectrum in spectra:
        summary.setdefault(spectrum.representation, {})[spectrum.group] = (
            spectrum.summary()
        )

    return summary


def spectrum_table(spectra: Sequence[Spectrum]) -> pa.Table:
    """One row per (representation, group, component)."""

    return pa.Table.from_pylist(
        [row for spectrum in spectra for row in spectrum.rows()],
        schema=pa.schema(
            [
                ("representation", pa.string()),
                ("group", pa.string()),
                ("component", pa.int64()),
                ("eigenvalue", pa.float64()),
                ("singular_value", pa.float64()),
                ("explained_variance_ratio", pa.float64()),
                ("cumulative_explained_variance", pa.float64()),
            ]
        ),
    )


def write_spectrum(snapshot: Snapshot, output_dir: Path) -> None:
    """Compute the spectra of every representation and write the results."""

    spectra = analyze_spectra(snapshot.representations, groups=snapshot.groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": SPECTRUM,
        "source": _source(snapshot),
        "settings": {
            "centering": "per group (global mean for 'all')",
            "standardization": None,
            "dtype": "float64",
            "group_column": GROUP_COLUMN if snapshot.groups is not None else None,
            "cumulative_at": list(CUMULATIVE_AT),
            "variance_thresholds": list(VARIANCE_THRESHOLDS),
        },
        "representations": spectrum_summary(spectra),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pq.write_table(spectrum_table(spectra), output_dir / "spectrum.parquet")

    for name, figure in spectrum_figures(spectra).items():
        figure.savefig(output_dir / f"{name}.png", dpi=150, facecolor=SURFACE)
        close(figure)


# ---------------------------------------------------------------------------
# Spectrum: local rendering
# ---------------------------------------------------------------------------


def spectrum_figures(spectra: Sequence[Spectrum]) -> dict[str, Figure]:
    """The cumulative explained variance and the eigenvalue spectrum.

    One panel per group, one line per representation (colored by its fixed
    slot), components on a log axis so spaces of different D compare.
    """

    return {
        "cumulative_variance": _spectrum_figure(
            spectra,
            values=lambda s: s.cumulative_explained_variance,
            ylabel="Cumulative explained variance",
            title="Cumulative explained variance",
            log_y=False,
        ),
        "eigenvalue_spectrum": _spectrum_figure(
            spectra,
            values=lambda s: s.explained_variance_ratio,
            ylabel="Eigenvalue / total variance",
            title="Covariance eigenvalue spectrum (normalized)",
            log_y=True,
        ),
    }


def _spectrum_figure(
    spectra: Sequence[Spectrum],
    *,
    values,
    ylabel: str,
    title: str,
    log_y: bool,
) -> Figure:
    from matplotlib.figure import Figure

    representations = list(dict.fromkeys(s.representation for s in spectra))
    groups = list(dict.fromkeys(s.group for s in spectra))

    if len(representations) > len(SERIES):
        raise ValueError(
            f"At most {len(SERIES)} representations per figure, "
            f"got {len(representations)}"
        )

    colors = dict(zip(representations, SERIES, strict=False))
    # Every representation has the same rows, so a group has one N.
    samples = {s.group: s.samples for s in spectra}

    figure = Figure(figsize=(4.2 * len(groups), 3.6), facecolor=SURFACE)
    axes = figure.subplots(1, len(groups), sharey=True, squeeze=False)[0]

    for ax, group in zip(axes, groups, strict=True):
        for spectrum in spectra:
            if spectrum.group != group:
                continue

            y = values(spectrum)
            x = torch.arange(1, spectrum.dim + 1)

            if log_y:
                keep = y > 0
                x, y = x[keep], y[keep]

            ax.plot(
                x.numpy(),
                y.numpy(),
                color=colors[spectrum.representation],
                linewidth=1.5,
                label=f"{spectrum.representation} (D={spectrum.dim})",
            )

        ax.set_xscale("log")

        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)

        ax.set_title(
            f"{group} (N={samples[group]:,})",
            color=SECONDARY_INK,
            fontsize=10,
        )
        ax.set_xlabel("Component", color=SECONDARY_INK)
        style(ax)

    axes[0].set_ylabel(ylabel, color=SECONDARY_INK)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9)
    figure.suptitle(title, color=INK, fontsize=12)
    figure.tight_layout()

    return figure


# ---------------------------------------------------------------------------
# PCA: structured results
# ---------------------------------------------------------------------------

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
            f"{_number(dataset['between_to_within_variance_ratio'])}); the dataset "
            f"silhouette is {_silhouette(dataset)}."
        )
        lines.append("")

    defined = {name: value for name, value in separations.items() if value is not None}

    if len(defined) > 1:
        ordered = sorted(defined, key=defined.get, reverse=True)
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

    high = max(defined, key=defined.get)
    low = min(defined, key=defined.get)
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

    ids = dict(zip(actions, metadata["action_id"], strict=True))

    return sorted(ids, key=ids.get)


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


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# PCA: local rendering
# ---------------------------------------------------------------------------


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
        source=_source(snapshot),
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


# ---------------------------------------------------------------------------
# Provenance and output
# ---------------------------------------------------------------------------


def _source(snapshot: Snapshot) -> dict[str, Any]:
    representations_file = snapshot.path / "representations.safetensors"

    return {
        "snapshot": str(snapshot.path),
        "representations_sha256": _sha256(representations_file),
        "samples": snapshot.manifest.get("samples"),
        "snapshot_provenance": snapshot.manifest.get("provenance"),
    }


def _require_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
