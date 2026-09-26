"""
Write the label-conditioned analysis of a snapshot: inventory, tables,
joined labels, an English report and one multi-panel figure per variable
and representation.

    label_source (audit, exact join)  ->  label_structure (metrics)
        ->  structured results (summary, long tables)  ->  files, figures

The report describes; it never selects a metric or draws a conclusion that
later probes must test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CONVERSATIONAL_STATE,
    FUTURE,
    NUISANCE,
    SECTIONS,
    SELECTION,
    TEMPORAL_STATE,
    CorpusLabelSource,
    JoinedLabels,
    audit_corpus,
    join_labels,
    label_inventory,
)
from turn_wm.evaluation.latent_analysis.label_structure import (
    DEFAULT_BALANCED_CAP,
    LabelAnalysis,
    analyze_labels,
    strength,
)
from turn_wm.evaluation.latent_analysis.pca import (
    ACTION,
    DATASET,
    DEFAULT_MAX_PLOT_SAMPLES,
    DEFAULT_PLOT_GROUP_FLOOR,
    DEFAULT_SILHOUETTE_SAMPLES,
    plot_selection,
    sample_keys,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    SECONDARY_INK,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    close,
    limits,
    percent,
    style,
)
from turn_wm.evaluation.latent_analysis.spectrum import ALL

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1

_TITLES = {
    CONVERSATIONAL_STATE: "Conversational state",
    TEMPORAL_STATE: "Temporal state",
    FUTURE: "Future structure",
    NUISANCE: "Nuisance controls",
}

_DOMAIN_NOTE = (
    "Dataset-dependent variation may reflect genuine differences in interaction "
    "settings. The relevant question is whether conversational structure remains "
    "usable within and across those settings."
)

_HYPOTHESES = (
    (
        "whether rollout skill is a better checkpoint criterion than total "
        "validation loss;"
    ),
    (
        "whether representation-health metrics (effective rank, spectrum) should be "
        "guardrails rather than optimization targets;"
    ),
    (
        "whether a metric related to future conversational-state information should "
        "be monitored during training;"
    ),
    "whether horizon-specific skill should influence early stopping.",
)

_LITERATURE = (
    "What checkpoint-selection metrics are used in predictive latent world models?",
    "How do JEPA and world-model papers monitor latent collapse or anisotropy?",
    "Which latent-state properties correlate with downstream planning performance?",
    "How is domain-conditioned conversational representation evaluated?",
    (
        "Are predictive state representations expected to make action or cue labels "
        "linearly decodable?"
    ),
    "How are multi-horizon predictive objectives used for early stopping?",
    (
        "How do turn-taking models (e.g. voice activity projection) evaluate "
        "representations across corpora with different interaction settings?"
    ),
)

_OPEN_QUESTIONS = (
    "Are conversational cues linearly decodable?",
    "Are they shared across domains?",
    "Are they domain-conditioned?",
    "Are weakly represented cues still useful for prediction?",
    "Which properties matter for planning?",
)


def write_labels(
    representations: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Sequence[Any]],
    output_dir: Path,
    *,
    source: Mapping[str, Any],
    sources: Mapping[str, CorpusLabelSource],
    seed: int,
    silhouette_samples: int = DEFAULT_SILHOUETTE_SAMPLES,
    balanced_cap: int = DEFAULT_BALANCED_CAP,
    max_plot_samples: int = DEFAULT_MAX_PLOT_SAMPLES,
) -> None:
    """Audit, join and analyze the labels of a snapshot; write every result."""

    provenance = source.get("snapshot_provenance") or {}

    if (provenance.get("data") or {}).get("split") == "test":
        raise ValueError("Label analysis never reads the test split")

    audits = {corpus: audit_corpus(s) for corpus, s in sources.items()}
    joined = join_labels(metadata, audits, sources)
    inventory = label_inventory(audits, joined.coverage)
    analysis = analyze_labels(
        representations,
        metadata,
        joined.variables,
        seed=seed,
        silhouette_samples=silhouette_samples,
        balanced_cap=balanced_cap,
    )
    # The PCA analysis's plotting sample, exactly.
    plotted = plot_selection(
        sample_keys(metadata["sample_id"], seed=seed),
        strata=[
            (str(d), str(a))
            for d, a in zip(metadata[DATASET], metadata[ACTION], strict=True)
        ],
        max_samples=max_plot_samples,
        group_floor=DEFAULT_PLOT_GROUP_FLOOR,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = output_dir / "joined_labels.parquet"
    pq.write_table(joined_table(metadata, joined), joined_path)

    figures = label_figures(analysis, metadata, plotted)
    summary = label_summary(
        analysis,
        joined,
        audits={c: a.provenance for c, a in audits.items()},
        source=source,
        plotted=int(plotted.sum()),
        joined_sha256=_sha256(joined_path),
        figures={
            section: [f"figures/{section}/{name}" for name in items]
            for section, items in figures.items()
        },
    )

    _write_json(output_dir / "label_inventory.json", inventory)
    (output_dir / "label_inventory.md").write_text(
        inventory_markdown(inventory, audits), encoding="utf-8"
    )
    _write_json(output_dir / "summary.json", summary)
    pq.write_table(metrics_table(analysis), output_dir / "metrics.parquet")
    pq.write_table(
        pa.Table.from_pylist(joined.coverage), output_dir / "coverage.parquet"
    )
    (output_dir / "report.md").write_text(label_report(summary), encoding="utf-8")

    for section, items in figures.items():
        (output_dir / "figures" / section).mkdir(parents=True, exist_ok=True)

        for name, figure in items.items():
            figure.savefig(
                output_dir / "figures" / section / name, dpi=130, facecolor=SURFACE
            )
            close(figure)


# ---------------------------------------------------------------------------
# Structured results
# ---------------------------------------------------------------------------


def label_summary(
    analysis: LabelAnalysis,
    joined: JoinedLabels,
    *,
    audits: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
    plotted: int,
    joined_sha256: str,
    figures: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "labels",
        "source": dict(source),
        "labels": {
            "corpora": dict(audits),
            "alignment": joined.alignment,
            "selection": [{"section": s, "label": label} for s, label in SELECTION],
            "excluded": joined.excluded,
            "joined_labels_sha256": joined_sha256,
        },
        "settings": {**analysis.settings, "plotted": plotted},
        "coverage": joined.coverage,
        "variables": {
            variable.name: {
                "label": variable.label,
                "section": variable.section,
                "kind": variable.kind,
                "classes": variable.classes,
                "horizon_s": variable.horizon_s,
                "representations": analysis.metrics[variable.name],
                "features_to_latent": analysis.deltas.get(variable.name),
                "domain_structure": analysis.domain[variable.name],
                "figure_clip": analysis.clip.get(variable.name),
            }
            for variable in analysis.variables
        },
        "figures": dict(figures),
    }


def joined_table(
    metadata: Mapping[str, Sequence[Any]], joined: JoinedLabels
) -> pa.Table:
    """The snapshot keys and every analysed variable, one row per snapshot row."""

    columns: dict[str, Any] = {
        name: list(metadata[name])
        for name in ("sample_id", "dataset", "recording_id", "anchor_idx")
    }

    for variable in joined.variables:
        columns[variable.name] = pa.array(
            variable.values,
            type=pa.string() if variable.kind == CATEGORICAL else pa.float64(),
        )

    return pa.table(columns)


def metrics_table(analysis: LabelAnalysis) -> pa.Table:
    """Every numeric metric in long format."""

    rows = []
    kinds = {v.name: v for v in analysis.variables}

    for name, by_representation in analysis.metrics.items():
        variable = kinds[name]

        for representation, by_condition in by_representation.items():
            for condition, metrics in by_condition.items():
                for metric, value in _flatten(metrics):
                    rows.append(
                        {
                            "representation": representation,
                            "variable": name,
                            "label": variable.label,
                            "section": variable.section,
                            "kind": variable.kind,
                            "condition": condition,
                            "metric": metric,
                            "value": value,
                        }
                    )

    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("representation", pa.string()),
                ("variable", pa.string()),
                ("label", pa.string()),
                ("section", pa.string()),
                ("kind", pa.string()),
                ("condition", pa.string()),
                ("metric", pa.string()),
                ("value", pa.float64()),
            ]
        ),
    )


def inventory_markdown(inventory, audits) -> str:
    corpora = sorted(audits)
    by_label: dict[str, dict[str, Any]] = {}

    for row in inventory:
        by_label.setdefault(row["label"], {})[row["dataset"]] = row

    lines = ["# Label inventory", ""]

    for corpus in corpora:
        audit = audits[corpus]
        provenance = audit.provenance
        lines.append(
            f"- **{corpus}**: `{provenance.get('repo_id')}` at "
            f"`{provenance.get('labels_revision')}` "
            f"(snapshot revision `{provenance.get('snapshot_revision')}`); action "
            f"grid SHA-256 `{str(provenance.get('action_grid_sha256'))[:16]}…`."
        )
        if audit.registry is None:
            lines.append(f"  - no labels: {audit.missing_reason}")
        for extractor, manifest in audit.manifests.items():
            state = (
                "not published"
                if manifest is None
                else audit.rejected.get(extractor)
                or f"{len(manifest['materialized_labels'])} materialized, "
                f"{manifest.get('extractor_version')}"
            )
            lines.append(f"  - extractor `{extractor}`: {state}")

    header = "| label | role | level | source | modalities | " + " | ".join(corpora)
    lines += ["", "Materialized: ✓; registered but not usable: ✗ (reason).", ""]
    family = None

    for label, per_corpus in by_label.items():
        first = next(iter(per_corpus.values()))
        if first["family"] != family:
            family = first["family"]
            lines += [
                "",
                f"## {family}",
                "",
                header + " |",
                "|" + "---|" * (5 + len(corpora)),
            ]

        cells = []
        for corpus in corpora:
            row = per_corpus.get(corpus)
            if row is None:
                cells.append("–")
            elif row["materialized"]:
                cover = row["coverage"]
                cells.append(
                    "✓" + ("" if cover is None else f" ({percent(cover)} valid)")
                )
            else:
                cells.append(f"✗ ({row['unavailable_reason']})")

        lines.append(
            f"| `{label}` | {first['role']} | {first['level']} | "
            f"{first['source_kind']} | {', '.join(first['modalities'])} | "
            + " | ".join(cells)
            + " |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def label_report(summary: Mapping[str, Any]) -> str:
    variables = summary["variables"]
    provenance = summary["source"].get("snapshot_provenance") or {}
    run = provenance.get("run") or {}
    checkpoint = provenance.get("checkpoint") or {}
    labels = summary["labels"]
    alignment = labels["alignment"]
    settings = summary["settings"]
    representations = _representations(variables)

    lines = [
        "# Label-conditioned representation analysis",
        "",
        "## Data and label provenance",
        "",
        (
            f"Snapshot of run `{run.get('run_id', 'unknown')}`, checkpoint "
            f"`{checkpoint.get('filename', 'unknown')}` (step "
            f"{checkpoint.get('global_step', 'unknown')}), "
            f"`{(provenance.get('data') or {}).get('split', 'unknown')}` split, "
            f"representations SHA-256 "
            f"`{summary['source']['representations_sha256'][:16]}…`."
        ),
        "",
    ]

    for corpus, info in labels["corpora"].items():
        same = info.get("labels_revision") == info.get("snapshot_revision")
        lines.append(
            f"- {corpus}: labels from `{info.get('repo_id')}` at revision "
            f"`{info.get('labels_revision')}`"
            + (
                ", the snapshot's own dataset revision"
                if same
                else f", explicitly chosen; the snapshot used "
                f"`{info.get('snapshot_revision')}` with a byte-identical action grid"
            )
            + f". Every extractor used records the action grid SHA-256 "
            f"`{str(info.get('action_grid_sha256'))[:16]}…` it was built from, "
            "which is the snapshot's."
        )

    lines += [
        "",
        (
            f"Labels are joined on {alignment['key']}, exactly; "
            f"{alignment['joined_rows_checked']:,} joined rows also agree on time "
            f"(largest difference {alignment['max_abs_time_difference_s']:.2g} s, "
            f"tolerance {alignment['time_tolerance_s']:g} s). No nearest-neighbour "
            "matching is used. Label reference instant: the end of the anchor's "
            "100 ms cell, i.e. the last instant the representation has observed."
        ),
        "",
        "| variable | corpus | snapshot rows | joined | valid | coverage |",
        "|---|---|---|---|---|---|",
    ]

    for row in summary["coverage"]:
        lines.append(
            f"| `{row['variable']}` | {row['dataset']} | {row['n_snapshot']:,} | "
            f"{row['n_joined']:,} | {row['n_valid']:,} | "
            f"{percent(row['coverage_fraction'])} |"
        )

    if labels["excluded"]:
        lines += ["", "Selected labels not analysed:", ""]
        for label, reasons in labels["excluded"].items():
            detail = "; ".join(f"{c}: {r}" for c, r in reasons.items())
            lines.append(f"- `{label}`: {detail}.")

    lines += [
        "",
        (
            "Strength of a structure below is its between-group variance fraction "
            "(between-class, or between quantile bins for continuous variables), "
            "computed on every valid row of the full representation space. "
            f"Silhouettes use at most {settings['silhouette_samples']:,} rows "
            "(natural class distribution); the balanced silhouette, a secondary "
            f"diagnostic, uses up to {settings['balanced_cap']:,} rows per class."
        ),
    ]

    for section in SECTIONS:
        names = [n for n, v in variables.items() if v["section"] == section]
        lines += ["", f"## {_TITLES[section]}", ""]

        if not names:
            reasons = {
                label: r
                for s, label in SELECTION
                if s == section
                for r in [labels["excluded"].get(label)]
                if r
            }
            lines.append(
                "No label of this group could be analysed in this release: "
                + "; ".join(
                    f"`{label}` ({'; '.join(f'{c}: {x}' for c, x in r.items())})"
                    for label, r in reasons.items()
                )
                + "."
            )
            if section == NUISANCE:
                lines.append(
                    "Whether the representations also encode recording-level "
                    "acoustic conditions therefore cannot be assessed here, and "
                    "the conversational structures above cannot yet be separated "
                    "from such nuisance variables."
                )
            lines.append("")
            continue

        for name in names:
            lines.append(_describe(name, variables[name], representations))
            lines.append("")

    lines += ["## Feature-to-latent changes", ""]

    if {"features", "latent"} <= set(representations):
        grouped: dict[str, list[str]] = {"stronger": [], "weaker": [], "similar": []}

        for name, variable in variables.items():
            delta = (variable["features_to_latent"] or {}).get(ALL) or {}
            kind = delta.get("change")
            if kind in grouped:
                a = strength(variable["representations"]["features"][ALL])
                b = strength(variable["representations"]["latent"][ALL])
                grouped[kind].append(f"`{name}` ({percent(a)} → {percent(b)})")

        lines.append(
            "Between-group variance fraction, all rows, features → latent "
            f"(similar: change below {settings['change_absolute']:g} or "
            f"{percent(settings['change_relative'])} of the larger value):"
        )
        lines.append("")
        for kind, items in grouped.items():
            lines.append(
                f"- {kind.capitalize()} in the latent: "
                + (", ".join(items) or "none")
                + "."
            )
    else:
        lines.append("Needs both `features` and `latent` in the snapshot.")

    lines += ["", "## Domain-conditioned structure", "", _DOMAIN_NOTE, ""]
    lines.append(
        f"Heuristic classes per representation: weak in both corpora (strength "
        f"below {percent(settings['weak_fraction'])} in each), similar (weaker "
        f"corpus at least {settings['similar_ratio']:g} of the stronger), or strong "
        "but different. For two-class variables, the direction cosine compares "
        "the class-mean difference of the two corpora (1: same direction)."
    )
    lines.append("")

    for name, variable in variables.items():
        parts = []
        for representation, domain in variable["domain_structure"].items():
            strengths = ", ".join(
                f"{c} {percent(s)}" for c, s in domain["strengths"].items()
            )
            cosine = domain.get("direction_cosine")
            parts.append(
                f"{representation}: {domain['class'].replace('_', ' ')} ({strengths}"
                + ("" if cosine is None else f"; direction cosine {cosine:.2f}")
                + ")"
            )
        lines.append(f"- `{name}` — " + "; ".join(parts) + ".")

    lines += [
        "",
        "## Metric hypotheses",
        "",
        "These are hypotheses raised by the results, not a choice of the V2 metric.",
        "",
        *(f"- Hypothesis: {h}" for h in _HYPOTHESES),
        *_conditional_hypotheses(variables),
        "",
        "## Literature questions",
        "",
        "Research directions, not conclusions:",
        "",
        *(f"- {q}" for q in _LITERATURE),
        "",
        "## Open questions",
        "",
        *(f"- {q}" for q in _OPEN_QUESTIONS),
        "",
    ]

    return "\n".join(lines)


def _describe(name: str, variable: Mapping[str, Any], representations) -> str:
    sentences = []

    for representation in representations:
        by_condition = variable["representations"][representation]
        overall = by_condition[ALL]

        if "undefined_reason" in overall:
            sentences.append(f"in `{representation}`: {overall['undefined_reason']}")
            continue

        corpora = ", ".join(
            f"{c} {percent(strength(m))}" for c, m in by_condition.items() if c != ALL
        )

        if variable["kind"] == CATEGORICAL:
            classes = ", ".join(
                f"{c} {percent(f)}" for c, f in overall["fractions"].items()
            )
            sentences.append(
                f"in `{representation}`, the classes ({classes}) account for "
                f"{percent(overall['between_variance_fraction'])} of the variance "
                f"(between-to-within ratio "
                f"{_number(overall['between_to_within_variance_ratio'])}; "
                f"silhouette natural {_number(overall['silhouette_natural'])}, "
                f"balanced {_number(overall['silhouette_balanced'])}); within "
                f"corpora: {corpora}"
            )
        else:
            median = overall["quantiles"]["0.5"]
            sentences.append(
                f"in `{representation}`, {overall['samples']:,} valid rows "
                f"(median {median:.3g}) have Spearman ρ "
                f"{_number(overall['spearman_pc1'])} with PC1 and "
                f"{_number(overall['spearman_pc2'])} with PC2; quantile bins account "
                f"for {percent(overall['between_bin_variance_fraction'])} of the "
                f"variance; within corpora: {corpora}"
            )

    return f"`{name}`: " + "; ".join(sentences) + "."


def _conditional_hypotheses(variables) -> list[str]:
    future = [
        v
        for v in variables.values()
        if v["section"] == FUTURE and "latent" in v["representations"]
    ]
    strengths = [strength(v["representations"]["latent"][ALL]) for v in future]
    strengths = [s for s in strengths if s is not None]

    if strengths and max(strengths) < 0.01:
        return [
            (
                "- Hypothesis: future conversational-state labels show little "
                "variance structure in the latent at every horizon; whether that "
                "information is absent, nonlinear or simply low-variance is untested."
            )
        ]

    return []


def _representations(variables) -> list[str]:
    for variable in variables.values():
        return list(variable["representations"])

    return []


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def label_figures(
    analysis: LabelAnalysis,
    metadata: Mapping[str, Sequence[Any]],
    plotted: torch.Tensor,
) -> dict[str, dict[str, Figure]]:
    """One figure per variable and representation, a panel per condition."""

    datasets = [str(d) for d in metadata[DATASET]]
    sections: dict[str, dict[str, Figure]] = {s: {} for s in SECTIONS}

    for variable in analysis.variables:
        for name, projection in analysis.projections.items():
            file = f"{variable.name.replace(':', '_').replace('@', '_')}_{name}.png"
            sections[variable.section][file] = _variable_figure(
                analysis, variable, name, projection, datasets, plotted
            )

    return {s: items for s, items in sections.items() if items}


def _variable_figure(analysis, variable, name, projection, datasets, plotted):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    conditions = analysis.conditions
    coordinates = projection.coordinates
    bounds = limits(coordinates[plotted])
    pcs = projection.summary()
    valid = torch.tensor([v is not None for v in variable.values])
    figure = Figure(figsize=(4.6 * len(conditions), 4.6), facecolor=SURFACE)
    axes = figure.subplots(1, len(conditions), sharex=True, sharey=True, squeeze=False)[
        0
    ]
    colors = dict(zip(variable.classes or (), SERIES, strict=False))
    ramp = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    clip = analysis.clip.get(variable.name)
    image = None

    for ax, condition in zip(axes, conditions, strict=True):
        inside = torch.tensor([condition in (ALL, d) for d in datasets])
        rows = (plotted & valid & inside).nonzero().squeeze(1).tolist()
        metrics = analysis.metrics[variable.name][name][condition]
        n_valid = metrics.get("samples", 0)
        points = coordinates[rows]

        if variable.kind == CATEGORICAL:
            counts = metrics.get("counts", {})
            # The largest classes first, so the rare ones stay on top.
            for label in sorted(colors, key=lambda c: -counts.get(c, 0)):
                chosen = [k for k, i in enumerate(rows) if variable.values[i] == label]
                if chosen:
                    ax.scatter(
                        points[chosen, 0].numpy(),
                        points[chosen, 1].numpy(),
                        s=2,
                        alpha=min(0.6, max(0.08, 2_000 / len(chosen))),
                        color=colors[label],
                        linewidths=0,
                        rasterized=True,
                    )
            fractions = metrics.get("fractions", {})
            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        markersize=6,
                        color=colors[label],
                        label=f"{label}  {counts[label]:,} ({percent(fractions[label])})",
                    )
                    for label in colors
                    if counts.get(label)
                ],
                frameon=False,
                labelcolor=INK,
                fontsize=8,
            )
        elif rows:
            values = torch.tensor([variable.values[i] for i in rows])
            image = ax.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                c=values.clamp(*clip).numpy() if clip else values.numpy(),
                cmap=ramp,
                vmin=clip[0] if clip else None,
                vmax=clip[1] if clip else None,
                s=2,
                alpha=min(0.7, max(0.1, 2_000 / len(rows))),
                linewidths=0,
                rasterized=True,
            )

        ax.set_xlim(*bounds[0])
        ax.set_ylim(*bounds[1])
        ax.set_title(
            f"{condition} · N = {len(rows):,} plotted of {n_valid:,} valid",
            color=SECONDARY_INK,
            fontsize=9,
        )
        ax.set_xlabel("PC1", color=SECONDARY_INK)
        style(ax)

    axes[0].set_ylabel("PC2", color=SECONDARY_INK)

    if image is not None:
        bar = figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.02)
        bar.set_label(
            variable.name
            + (f" (clipped to [{clip[0]:.3g}, {clip[1]:.3g}])" if clip else ""),
            color=SECONDARY_INK,
            fontsize=8,
        )
        bar.ax.tick_params(labelsize=7, colors=SECONDARY_INK)

    figure.suptitle(
        f"{name} · {variable.name} · global PCA of {name}: "
        f"PC1 {percent(pcs['pc1_explained_variance'])}, "
        f"PC2 {percent(pcs['pc2_explained_variance'])}",
        color=INK,
        fontsize=11,
    )

    if image is None:
        figure.tight_layout()

    return figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten(metrics: Mapping[str, Any], prefix: str = ""):
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            yield name, float(value)
        elif isinstance(value, Mapping):
            yield from _flatten(value, f"{name}.")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
