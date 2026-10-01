"""
Files, figures and report of the linear probes (`probes.analyze_probes`).

`write_probes` runs the probes and writes `summary.json`, the scores table,
the comparison figures and `report.md`: which variables are decodable from
the Mimi features and from the WM latent, the projector's effect (latent -
features) and cross-domain transfer.
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
from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CorpusLabelSource,
    hub_label_sources,
)
from turn_wm.evaluation.latent_analysis.probes import (
    ANALYSIS,
    CURRENT,
    DEFAULT_BOOTSTRAP,
    FEATURES,
    FUTURE_STATE,
    LATENT,
    POOLED,
    REPRESENTATION_NAMES,
    REPRESENTATIONS,
    SCHEMA_VERSION,
    TEMPORAL,
    analyze_probes,
    check_snapshots,
    task_label,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    MUTED,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    close,
    new_figure,
    style,
)
from turn_wm.evaluation.latent_analysis.snapshot import describe_snapshot, read_snapshot
from turn_wm.progress import log

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def write_probes(
    train_snapshot: Path,
    validation_snapshot: Path,
    *,
    output_dir: Path | None = None,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> Path:
    """Probe the two snapshots; write summary, scores, figures and report."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    train = read_snapshot(train_snapshot)
    validation = read_snapshot(validation_snapshot)
    # Before any download or fit.
    check_snapshots(train, validation)
    output_dir = (
        validation.path / "analysis" / ANALYSIS if output_dir is None else output_dir
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    start = time.perf_counter()
    log(f"probes: probe-train {describe_snapshot(train)}")
    log(f"probes: validation {describe_snapshot(validation)}")
    log(f"probes: {bootstrap} bootstrap resamples; output {output_dir}")
    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(
            validation.manifest.get("provenance") or {},
            labels_revision=labels_revision,
        )
    )
    results = analyze_probes(train, validation, sources, bootstrap=bootstrap)
    log("probes: writing figures, tables and report")
    figures = probe_figures(results)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    for name, figure in figures.items():
        figure.savefig(output_dir / "figures" / name, dpi=150, facecolor=SURFACE)
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
        "figures": [f"figures/{name}" for name in figures],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(probe_report(summary), encoding="utf-8")
    log(f"probes: done in {time.perf_counter() - start:.0f}s")

    return output_dir


def scores_table(scores: Sequence[Mapping[str, Any]]) -> pa.Table:
    rows = []

    for score in scores:
        row = {
            key: score[key]
            for key in (
                "task",
                "group",
                "kind",
                "setting",
                "setting_kind",
                "reference",
                "n_train",
                "n_eval",
                "n_eval_recordings",
                "skipped",
            )
        }
        row["classes"] = ",".join(score.get("classes") or []) or None
        row["regularization_parameter"] = score.get("regularization_parameter")
        row |= {
            f"{name}_selected_regularization": score.get(
                f"{name}_selected_regularization"
            )
            for name in REPRESENTATIONS
        }

        for name in (*REPRESENTATIONS, "delta"):
            interval = score[f"{name}_ci"] or [None, None]
            row |= {
                f"{name}_score": score[f"{name}_score"],
                f"{name}_ci_low": interval[0],
                f"{name}_ci_high": interval[1],
            }

        rows.append(row)

    return pa.Table.from_pylist(rows)


_COLORS = {FEATURES: SERIES[0], LATENT: SERIES[1]}


_TITLES = {
    CURRENT: "Current conversational state",
    TEMPORAL: "Temporal conversational state",
    FUTURE_STATE: "Future conversational state (1 s)",
}


def comparison_panel(ax, scores: Sequence[Mapping[str, Any]], title: str) -> None:
    """Features and latent per task, joined; the trivial reference per task."""

    style(ax)
    ticks = []

    for x, score in enumerate(scores):
        ticks.append(task_label(score["task"], figure=True))
        reference = score["reference"]

        if reference is not None:
            ax.hlines(
                reference, x - 0.35, x + 0.35, color=MUTED, linestyle="--", linewidth=1
            )

        if score["skipped"]:
            ax.annotate(
                "not evaluable",
                (x, reference or 0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=MUTED,
            )
            continue

        points = []

        for offset, name in ((-0.12, FEATURES), (0.12, LATENT)):
            value, interval = score[f"{name}_score"], score[f"{name}_ci"]
            if value is None:
                continue
            points.append((x + offset, value))
            ax.errorbar(
                [x + offset],
                [value],
                yerr=None
                if interval is None
                else [[value - interval[0]], [interval[1] - value]],
                color=_COLORS[name],
                marker="o",
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                capsize=3,
                linewidth=2,
                label=REPRESENTATION_NAMES[name] if x == 0 else None,
            )

        if len(points) == 2:
            ax.plot(*zip(*points, strict=True), color=MUTED, linewidth=1, zorder=0)
            delta = score["delta_score"]
            top = max(
                (score[f"{n}_ci"] or [0, score[f"{n}_score"]])[1]
                for n in REPRESENTATIONS
            )
            ax.annotate(
                f"Δ {delta:+.3f}",
                (x, top),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=SECONDARY_INK,
            )

    ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
    ax.set_xlim(-0.6, len(ticks) - 0.4)
    ax.set_title(title, color=INK, fontsize=9, loc="left")


def _metric_label(kind: str) -> str:
    return "Balanced accuracy" if kind == CATEGORICAL else "R²"


def probe_figures(results: Mapping[str, Any]) -> dict[str, Figure]:
    scores = results["scores"]
    corpora = results["corpora"]
    figures = {}
    within = [POOLED, *(f"within:{c}" for c in corpora if len(corpora) > 1)]

    for group, name in (
        (CURRENT, "current_state.png"),
        (TEMPORAL, "temporal_state.png"),
        (FUTURE_STATE, "future_state.png"),
    ):
        group_scores = [s for s in scores if s["group"] == group]
        figure = new_figure(3.2 * len(within) + 1.2, 3.8)
        axes = figure.subplots(1, len(within), squeeze=False, sharey=True)[0]

        for ax, setting in zip(axes, within, strict=True):
            comparison_panel(
                ax,
                [s for s in group_scores if s["setting"] == setting],
                setting.replace("within:", "within ").replace(POOLED, "pooled"),
            )

        kind = group_scores[0]["kind"] if group_scores else CATEGORICAL
        axes[0].set_ylabel(_metric_label(kind), color=SECONDARY_INK, fontsize=9)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
        figure.suptitle(
            f"{_TITLES[group]} — linear probe, validation "
            "(dashed: trivial reference; bars: 95% recording bootstrap)",
            color=INK,
            fontsize=10,
            x=0.02,
            ha="left",
        )
        figures[name] = figure

    cross = sorted({s["setting"] for s in scores if s["setting_kind"] == "cross"})
    figure = new_figure(3.6 * max(1, len(cross)) + 1.2, 3.8)
    axes = figure.subplots(1, max(1, len(cross)), squeeze=False, sharey=True)[0]

    for ax, setting in zip(axes, cross, strict=False):
        comparison_panel(
            ax,
            [s for s in scores if s["setting"] == setting],
            "train " + setting.replace("->", " → evaluate "),
        )

    axes[0].set_ylabel("Balanced accuracy", color=SECONDARY_INK, fontsize=9)
    if cross:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Cross-domain transfer of categorical probes (dashed: chance 1/K)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    figures["cross_domain.png"] = figure

    return figures


# Cross-domain wording: the share of the target's within-domain margin over
# the reference that the transferred probe keeps.
SHARED_RETENTION = 0.8


# Total-variation distance between class distributions flagged as a shift.
CLASS_SHIFT = 0.10


def format_score(score: Mapping[str, Any], name: str) -> str:
    value, interval = score[f"{name}_score"], score[f"{name}_ci"]

    if value is None:
        return "n/a"

    text = f"{value:+.3f}" if name == "delta" else f"{value:.3f}"

    if interval is None:
        return text

    return f"{text} [{interval[0]:.3f}, {interval[1]:.3f}]"


def versus_reference(score: Mapping[str, Any], name: str, reference: float) -> str:
    interval = score[f"{name}_ci"]

    if score[f"{name}_score"] is None or interval is None:
        return "undetermined"
    if interval[0] > reference:
        return "above"
    if interval[1] < reference:
        return "below"

    return "includes"


def score_table_lines(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        (
            "| task | setting | reference | N train / eval (rec.) | Mimi features | "
            "WM latent | delta (latent − features) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]

    for s in scores:
        if s["skipped"]:
            lines.append(
                f"| {task_label(s['task'])} | {s['setting']} | – | "
                f"{s['n_train']:,} / {s['n_eval']:,} | not evaluable: {s['skipped']} | | |"
            )
            continue

        lines.append(
            f"| {task_label(s['task'])} | {s['setting']} | {s['reference']:.3f} | "
            f"{s['n_train']:,} / {s['n_eval']:,} ({s['n_eval_recordings']}) | "
            f"{format_score(s, FEATURES)} | {format_score(s, LATENT)} | {format_score(s, 'delta')} |"
        )

    return lines


def _decodable(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    """Q1/Q2 per task (pooled): above the trivial reference or not."""

    lines = []
    words = {
        "above": "linearly decodable (interval above the reference)",
        "includes": "not distinguishable from the reference",
        "below": "below the reference",
        "undetermined": "undetermined",
    }

    for s in scores:
        if s["setting"] != POOLED or s["skipped"]:
            continue

        parts = [
            f"{REPRESENTATION_NAMES[name]}: {words[versus_reference(s, name, s['reference'])]}"
            for name in REPRESENTATIONS
        ]
        lines.append(f"- **{task_label(s['task'])}** — " + "; ".join(parts) + ".")

    return lines


def projector_effect(score: Mapping[str, Any]) -> str:
    return {
        "above": "improves",
        "below": "degrades",
        "includes": "preserves",
        "undetermined": "undetermined",
    }[versus_reference(score, "delta", 0.0)]


def _within_domain(summary: Mapping[str, Any]) -> list[str]:
    lines = []

    for corpus in summary["corpora"]:
        rows = [s for s in summary["scores"] if s["setting"] == f"within:{corpus}"]

        if not rows:
            continue

        lines += [f"### {corpus}", "", *score_table_lines(rows), ""]

    return lines


def _tv_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    total_a, total_b = sum(a.values()), sum(b.values())

    return 0.5 * sum(abs(a[c] / total_a - b[c] / total_b) for c in a)


def _cross_domain(summary: Mapping[str, Any]) -> list[str]:
    scores = summary["scores"]
    lines = []

    for s in scores:
        if s["setting_kind"] != "cross":
            continue

        target = s["eval_corpora"][0]
        within = next(
            (
                w
                for w in scores
                if w["task"] == s["task"] and w["setting"] == f"within:{target}"
            ),
            None,
        )

        if s["skipped"] or within is None or within["skipped"]:
            lines.append(
                f"- **{task_label(s['task'])}, {s['setting']}**: not probed "
                f"({s['skipped'] or 'no within-domain reference'})."
            )
            continue

        parts = []

        for name in REPRESENTATIONS:
            reference = s["reference"]
            cross_above = versus_reference(s, name, reference)
            within_above = versus_reference(within, name, reference)
            # Only meaningful when the target corpus has a margin to keep.
            retention = (
                (s[f"{name}_score"] - reference) / (within[f"{name}_score"] - reference)
                if within_above == "above"
                else None
            )

            if within_above != "above":
                verdict = "not decodable within the target corpus itself"
            elif cross_above != "above":
                verdict = (
                    "does not transfer: decodable within the target corpus but "
                    "not with the source corpus's readout (domain-specific encoding, "
                    "or shift)"
                )
            elif retention is not None and retention >= SHARED_RETENTION:
                verdict = (
                    "transfers: information present in both corpora in a shared "
                    "linear form"
                )
            else:
                verdict = (
                    "transfers partly: present in both corpora, but part of it is "
                    "encoded in domain-specific ways"
                )

            kept = (
                ""
                if retention is None
                else f", {100 * retention:.0f}% of the within-domain margin kept"
            )
            parts.append(
                f"{REPRESENTATION_NAMES[name]} {format_score(s, name)} vs within-{target} "
                f"{format_score(within, name)}{kept} — {verdict}"
            )

        shift = _tv_distance(s["train_class_counts"], s["eval_class_counts"])
        shift_text = (
            f" Class distribution shift between the source's training rows and the "
            f"target's rows: total variation {shift:.2f}"
            + (
                " — large enough that class/distribution shift may contribute to "
                "any gap between transfer and within-domain scores (balanced "
                "accuracy is insensitive to the target's class frequencies, not to "
                "shifted class-conditional inputs)."
                if shift >= CLASS_SHIFT
                else "."
            )
        )
        lines.append(
            f"- **{task_label(s['task'])}, train {s['train_corpora'][0]} → evaluate "
            f"{target}** (chance {s['reference']:.3f}): "
            + "; ".join(parts)
            + "."
            + shift_text
        )

    return lines


def _projector_summary(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    effects: dict[str, list[str]] = {"improves": [], "preserves": [], "degrades": []}

    for s in scores:
        if s["skipped"]:
            continue

        effect = projector_effect(s)

        if effect in effects:
            effects[effect].append(
                f"{task_label(s['task'])} ({s['setting']}, Δ {s['delta_score']:+.3f})"
            )

    return [
        f"- **{effect.capitalize()}** linear accessibility: "
        + (", ".join(items) if items else "none")
        + "."
        for effect, items in effects.items()
    ]


def _context_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| task | snapshot | corpus | N | valid coverage | class fractions (valid rows) |",
        "|---|---|---|---|---|---|",
    ]

    for task, info in summary["tasks"].items():
        if "context" not in info:
            lines.append(f"| {task_label(task)} | – | – | – | {info['missing']} | |")
            continue

        for which, by_corpus in info["context"].items():
            for corpus, entry in by_corpus.items():
                fractions = entry.get("class_fractions")
                text = (
                    ", ".join(
                        f"{c} {100 * f:.1f}%"
                        for c, f in fractions.items()
                        if f is not None
                    )
                    if fractions
                    else "continuous"
                )
                coverage = entry["coverage"]
                lines.append(
                    f"| {task_label(task)} | {which} | {corpus} | {entry['n']:,} | "
                    f"{'n/a' if coverage is None else f'{100 * coverage:.1f}%'} | {text} |"
                )

    return lines


def _grid(values: Sequence[float]) -> str:
    return "{" + ", ".join(f"{v:g}" for v in values) + "}"


def _regularization_table(fitted_probes: Sequence[Mapping[str, Any]]) -> list[str]:
    """One row per fitted probe: its CV-selected C or alpha."""

    lines = [
        (
            "| task | train domain | representation | model | selected "
            "regularization | valid folds |"
        ),
        "|---|---|---|---|---|---|",
    ]

    for entry in fitted_probes:
        cv = entry["cv"]
        categorical = entry["task_type"] == CATEGORICAL
        selected = cv["selected_c" if categorical else "selected_alpha"]
        value = (
            f"not evaluable ({cv['unsupported']})"
            if selected is None
            else f"{'C' if categorical else 'alpha'}={selected:g}"
        )
        lines.append(
            f"| {task_label(entry['task'])} | {' + '.join(entry['training_domain'])} | "
            f"{REPRESENTATION_NAMES[entry['representation']]} | "
            f"{'logistic' if categorical else 'ridge'} | {value} | "
            f"{len(cv['valid_cv_folds'])}/{cv['requested_cv_folds']} |"
        )

    return lines


def _hypotheses(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    pooled = {
        s["task"]: s for s in scores if s["setting"] == POOLED and not s["skipped"]
    }
    future = pooled.get("future.future_joint_speech_state@1s")
    lines = []

    if future is not None:
        lines.append(
            "- Future joint-speech state (1 s) from the latent: "
            f"{format_score(future, LATENT)} against chance {future['reference']:.3f} "
            f"(projector effect: {projector_effect(future)}). A probe-based future-state "
            "score could be monitored during training as a representation check, "
            "alongside rollout skill; whether it tracks model quality is untested."
        )

    degraded = [t for t, s in pooled.items() if projector_effect(s) == "degrades"]

    if degraded:
        lines.append(
            "- The projector degrades linear access to "
            + ", ".join(task_label(t) for t in degraded)
            + ": a guardrail on such probes could flag a projector that discards "
            "conversational information, if that information matters downstream."
        )

    cross = [s for s in scores if s["setting_kind"] == "cross" and not s["skipped"]]

    if cross:
        lines.append(
            "- Cross-domain probe transfer could serve as a check that representations "
            "stay reusable across interaction settings; its relation to rollout skill "
            "is unknown."
        )

    lines.append("- No metric is selected here; these are hypotheses to test.")

    return lines


def _questions(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    questions = [
        (
            "How is predictive-state representation quality evaluated with linear probes "
            "in JEPA and latent world-model literature?"
        ),
    ]
    pooled = [s for s in scores if s["setting"] == POOLED and not s["skipped"]]

    if any(projector_effect(s) == "degrades" for s in pooled):
        questions.append(
            "Do predictive projectors in JEPA-style models discard linearly accessible "
            "input information, and is this considered harmful or a form of useful "
            "abstraction?"
        )
    if any(projector_effect(s) == "improves" for s in pooled):
        questions.append(
            "Which training signals make a learned latent more linearly accessible "
            "than its frozen input features?"
        )
    if any(s["group"] == FUTURE_STATE for s in pooled):
        questions.append(
            "Should future-state linear decodability correlate with rollout skill or "
            "planning utility?"
        )
    if any(
        s["setting_kind"] == "cross"
        and not s["skipped"]
        and versus_reference(s, LATENT, s["reference"]) != "above"
        for s in scores
    ):
        questions.append(
            "How are domain-conditioned representations evaluated in multi-domain "
            "predictive models?"
        )
    if any(s["group"] == TEMPORAL for s in pooled):
        questions.append(
            "How do turn-taking models (e.g. voice activity projection) evaluate "
            "access to timing information such as time to the next speaker onset?"
        )

    return [f"- {q}" for q in questions]


def probe_report(summary: Mapping[str, Any]) -> str:
    scores = summary["scores"]
    settings = summary["settings"]
    train = summary["source"]["probe_train"]
    validation = summary["source"]["probe_validation"]
    provenance = validation["snapshot_provenance"] or {}
    checkpoint = provenance.get("checkpoint") or {}

    def group(name, settings_kinds=(POOLED,)):
        return [
            s
            for s in scores
            if s["group"] == name and s["setting_kind"] in settings_kinds
        ]

    return "\n".join(
        [
            "# Linear probe evaluation",
            "",
            "## Purpose",
            "",
            (
                "These probes test **linear accessibility**: whether a linear readout "
                "fitted on one set of recordings can recover a conversational variable "
                "from a representation on other recordings. They do not test whether the "
                "world model causally uses that information."
            ),
            "",
            (
                f"Checkpoint `{checkpoint.get('filename')}` (step "
                f"{checkpoint.get('global_step')}, sha256 "
                f"`{str(checkpoint.get('sha256'))[:12]}…`). Probes are fitted on a "
                f"**train-split** snapshot ({train['samples']:,} anchors) and evaluated "
                f"on a **validation-split** snapshot ({validation['samples']:,} anchors); "
                "no recording occurs in both, and the test split is never read. "
                "Representations: Mimi features (the encoder baseline) and the WM latent "
                "(V1 projector output)."
            ),
            "",
            (
                f"Probes: {settings['categorical_probe']}, C grid "
                f"{_grid(settings['logistic_c_grid'])}; {settings['continuous_probe']}, "
                f"alpha grid {_grid(settings['ridge_alpha_grid'])}. Regularization: "
                f"{settings['regularization']}. Standardization: "
                f"{settings['standardization']}. Sampling: {settings['sampling']}. "
                f"Scores: {settings['categorical_score']}; "
                f"{settings['continuous_score']}. Every canonical class needs ≥ "
                f"{settings['min_class_support']} rows in probe-train and in the "
                "evaluated rows of a setting; otherwise the setting is reported as "
                "unsupported. "
                f"Intervals: {int(100 * settings['confidence'])}% "
                f"{settings['interval']} ({settings['bootstrap_resamples']} resamples, "
                "seeded)."
            ),
            "",
            "### Data context (not performance)",
            "",
            *_context_table(summary),
            "",
            "### Selected regularization (probe-train CV, frozen before validation)",
            "",
            (
                "Logistic regression uses C, where smaller values mean stronger L2 "
                "regularization. Ridge uses alpha, where larger values mean "
                "stronger regularization. Ties are resolved in favor of stronger "
                "regularization (smaller C for logistic regression, larger alpha "
                "for ridge)."
            ),
            "",
            *_regularization_table(summary["fitted_probes"]),
            "",
            "## Current conversational state",
            "",
            *score_table_lines(group(CURRENT)),
            "",
            *_decodable(group(CURRENT)),
            "",
            "## Temporal state",
            "",
            "R² on raw seconds; 0 is predicting the evaluated rows' mean.",
            "",
            *score_table_lines(group(TEMPORAL)),
            "",
            *_decodable(group(TEMPORAL)),
            "",
            "## Future conversational state",
            "",
            *score_table_lines(group(FUTURE_STATE)),
            "",
            *_decodable(group(FUTURE_STATE)),
            "",
            "## Within-domain representation",
            "",
            *_within_domain(summary),
            "## Cross-domain transfer",
            "",
            (
                "Does a linear readout trained on one corpus retain useful performance on "
                "the other? Each transfer is compared with a probe trained within the "
                "target corpus: the share of its margin over chance that the "
                f"transferred probe keeps (≥ {int(100 * SHARED_RETENTION)}%: shared "
                "linear form). Weak transfer is not by itself a bad representation: "
                "domain-conditioned cues are a legitimate hypothesis. Continuous timing "
                "labels are not transferred across corpora in this block."
            ),
            "",
            *score_table_lines([s for s in scores if s["setting_kind"] == "cross"]),
            "",
            *_cross_domain(summary),
            "",
            "## Mimi vs WM representation",
            "",
            (
                "Delta = latent score − features score, with a paired recording "
                "bootstrap; the projector improves (interval above 0), preserves "
                "(interval includes 0) or degrades (interval below 0) linear "
                "accessibility. Probe coefficients are not compared: the spaces differ "
                "in dimension and scale."
            ),
            "",
            *_projector_summary(scores),
            "",
            "## What this does NOT show",
            "",
            (
                "- Linear decodability does not prove that the predictor uses the "
                "information."
            ),
            (
                "- Poor linear decodability does not prove that the information is "
                "absent: it may be present non-linearly."
            ),
            (
                "- Cross-domain transfer between EgoCom and Ego4D is not equivalent to "
                "full real-world generalization."
            ),
            "- Probes do not establish planning usefulness.",
            "",
            "## Metric hypotheses",
            "",
            *_hypotheses(scores),
            "",
            "## Literature questions",
            "",
            *_questions(scores),
            "",
        ]
    )
