"""
Files, figure and report of the concept probes (`concepts.analyze_concepts`).

`write_concepts` runs the probes and writes `summary.json`, the scores
table, the figure and `report.md`, with each concept's documented rule and
the concepts that are not evaluable.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from turn_wm.evaluation.latent_analysis.artifacts import snapshot_source
from turn_wm.evaluation.latent_analysis.concept_labels import AXES, AXIS_TITLES
from turn_wm.evaluation.latent_analysis.concepts import (
    ANALYSIS,
    SCHEMA_VERSION,
    SETTING_NAMES,
    analyze_concepts,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    CONTINUOUS,
    CorpusLabelSource,
    hub_label_sources,
)
from turn_wm.evaluation.latent_analysis.probes import (
    DEFAULT_BOOTSTRAP,
    FEATURES,
    LATENT,
    REPRESENTATIONS,
    check_snapshots,
)
from turn_wm.evaluation.latent_analysis.probes_report import (
    comparison_panel,
    format_score,
    projector_effect,
    score_table_lines,
    versus_reference,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    SECONDARY_INK,
    SURFACE,
    close,
    new_figure,
)
from turn_wm.evaluation.latent_analysis.snapshot import describe_snapshot, read_snapshot
from turn_wm.progress import log

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def write_concepts(
    train_snapshot: Path,
    validation_snapshot: Path,
    *,
    output_dir: Path | None = None,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> Path:
    """Probe every concept; write summary, scores, figure and report."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    train = read_snapshot(train_snapshot)
    validation = read_snapshot(validation_snapshot)
    check_snapshots(train, validation)
    output_dir = (
        validation.path / "analysis" / ANALYSIS if output_dir is None else output_dir
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    start = time.perf_counter()
    log(f"concepts: probe-train {describe_snapshot(train)}")
    log(f"concepts: validation {describe_snapshot(validation)}")
    log(f"concepts: {bootstrap} bootstrap resamples; output {output_dir}")
    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(
            validation.manifest.get("provenance") or {},
            labels_revision=labels_revision,
        )
    )
    results = analyze_concepts(train, validation, sources, bootstrap=bootstrap)
    log("concepts: writing figure, tables and report")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    figure = concept_figure(results)
    figure.savefig(output_dir / "figures" / "concepts.png", dpi=150, facecolor=SURFACE)
    close(figure)
    pq.write_table(scores_table(results["scores"]), output_dir / "scores.parquet")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "source": {
            "probe_train": snapshot_source(train),
            "probe_validation": snapshot_source(validation),
        },
        **results,
        "figures": ["figures/concepts.png"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(concept_report(summary), encoding="utf-8")
    log(f"concepts: done in {time.perf_counter() - start:.0f}s")

    return output_dir


def scores_table(scores: Sequence[Mapping[str, Any]]) -> pa.Table:
    rows = []

    for s in scores:
        row = {
            k: s.get(k)
            for k in (
                "task",
                "axis",
                "protocol",
                "kind",
                "reference",
                "n_train",
                "n_eval",
                "n_eval_recordings",
                "skipped",
            )
        }

        for name in (*REPRESENTATIONS, "delta"):
            interval = s.get(f"{name}_ci") or [None, None]
            row |= {
                f"{name}_score": s.get(f"{name}_score"),
                f"{name}_ci_low": interval[0],
                f"{name}_ci_high": interval[1],
            }

        rows.append(row)

    return pa.Table.from_pylist(rows)


def concept_figure(results: Mapping[str, Any]) -> Figure:
    """One panel per axis: features and latent per concept, with the reference."""

    scores = results["scores"]
    figure = new_figure(15, 4.2)
    axes = figure.subplots(1, len(AXES), squeeze=False)[0]

    for ax, axis in zip(axes, AXES, strict=True):
        rows = [
            s | {"task": s["task"] + (" (R²)" if s["kind"] == CONTINUOUS else "")}
            for s in scores
            if s["axis"] == axis
        ]
        comparison_panel(ax, rows, AXIS_TITLES[axis])

    axes[0].set_ylabel(
        "Balanced accuracy (R² where marked)", color=SECONDARY_INK, fontsize=9
    )
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Concept probes (dashed: trivial reference; bars: 95% bootstrap over "
        "validation recordings or, for grouped CV, conversations)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


LIMITATIONS = """\
## Limitations

- The latent is a per-frame projection of the Mimi features: a concept can
  only be as accessible as the features allow; context (e.g. the 10 s party
  size) reaches both only through Mimi's own streaming state.
- wearer_native and wearer_host concern few recurring people: a probe can
  succeed by recognizing their voices rather than the trait itself.
- Grouped-CV concepts use train-split recordings for evaluation too (out of
  fold, never in the fold that fits the probe); the world model saw that
  audio during training, though never these labels.
- participant_count counts people annotated in the scene, speaking or not.
- wearer_speech_rate is a property of the whole turn, so it includes words
  after the anchor.
- Values are derived from the release's tables by the rules in each
  concept's definition; they are not labels defined by the data repository.
"""


def concept_report(summary: Mapping[str, Any]) -> str:
    provenance = summary["source"]["probe_validation"]["snapshot_provenance"] or {}
    checkpoint = provenance.get("checkpoint") or {}
    settings = summary["settings"]
    lines = [
        "# Concept probes",
        "",
        "## Purpose",
        "",
        (
            "Which information, beyond the current conversational state, do the "
            "Mimi features and the WM latent keep **linearly accessible**? Four "
            "axes: vocal activity, multi-party structure, social signals, and "
            "information unrelated to the conversation. A successful probe does "
            "not show that the world model uses the information."
        ),
        "",
        (
            f"Checkpoint `{checkpoint.get('filename')}` (step "
            f"{checkpoint.get('global_step')}, sha256 "
            f"`{str(checkpoint.get('sha256'))[:12]}…`). Test split never read."
        ),
        "",
        "Protocols: "
        + "; ".join(
            f"**{SETTING_NAMES[k]}**: {v}" for k, v in settings["protocols"].items()
        )
        + f". Intervals: {int(100 * settings['confidence'])}% percentile bootstrap "
        f"({settings['bootstrap_resamples']} resamples, seeded); deltas paired.",
        "",
        "## Concepts",
        "",
        "| concept | axis | protocol | question | definition | not evaluable in |",
        "|---|---|---|---|---|---|",
    ]

    for name, info in summary["concepts"].items():
        missing = "; ".join(f"{c}: {r}" for c, r in info["unavailable"].items()) or "–"
        lines.append(
            f"| {name} | {info['axis']} | {SETTING_NAMES[info['protocol']]} | "
            f"{info['question']} | {info['definition']} | {missing} |"
        )

    for axis in AXES:
        rows = [s for s in summary["scores"] if s["axis"] == axis]
        lines += ["", f"## {AXIS_TITLES[axis]}", "", *score_table_lines(rows), ""]

        for s in rows:
            if s["skipped"]:
                lines.append(f"- **{s['task']}**: not evaluable ({s['skipped']}).")
                continue

            reference = s["reference"]
            lines.append(
                f"- **{s['task']}**: Mimi features {_verdict(s, FEATURES, reference)}; "
                f"WM latent {_verdict(s, LATENT, reference)}; the projector "
                f"{projector_effect(s)} linear accessibility (Δ {format_score(s, 'delta')})."
            )

    lines += [
        "",
        LIMITATIONS,
        "## Figure",
        "",
        "- `figures/concepts.png`",
        "",
    ]

    return "\n".join(lines)


def _verdict(score, name, reference) -> str:
    return {
        "above": "above the reference",
        "includes": "not distinguishable from the reference",
        "below": "below the reference",
        "undetermined": "undetermined",
    }[versus_reference(score, name, reference)]
