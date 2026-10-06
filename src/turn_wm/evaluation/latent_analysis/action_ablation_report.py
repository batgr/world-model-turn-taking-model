"""
Files, figures and report of the action ablation (`action_ablation_analysis`).

`write_action_ablation` runs the analysis and writes `summary.json`, the
scores table, the rollout and counterfactual figures and `report.md`: how
much the rollout relies on the recorded future ONSET/OFFSET tokens.
"""

from __future__ import annotations

import importlib.util
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from turn_wm.evaluation.latent_analysis.action_ablation import (
    ABLATION_TENSORS,
    CONDITIONS,
    OBSERVED,
    SHUFFLED,
    STATE_PRESERVING,
)
from turn_wm.evaluation.latent_analysis.action_ablation_analysis import (
    ALL,
    ANALYSIS,
    DIFFERENCES,
    EXPOSED,
    ROLES,
    SCHEMA_VERSION,
    STATES,
    SUBSETS,
    UNEXPOSED,
    counterfactual_effects,
    rollout_ablation,
)
from turn_wm.evaluation.latent_analysis.artifacts import snapshot_source
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
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    CONFIDENCE,
    DEFAULT_BOOTSTRAP,
    METRICS,
    MOVEMENT,
    SKILL,
)
from turn_wm.evaluation.latent_analysis.snapshot import read_snapshot
from turn_wm.progress import log, progress

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def write_action_ablation(
    snapshot_dir: Path,
    *,
    output_dir: Path | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> Path:
    """Analyze an action-ablation snapshot; write tables, figures, report."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    snapshot = read_snapshot(snapshot_dir)
    provenance = snapshot.manifest.get("provenance") or {}
    missing = [n for n in ABLATION_TENSORS if n not in snapshot.representations]

    if "action_ablation" not in provenance or missing:
        raise ValueError(
            f"{snapshot.path} is not an action-ablation snapshot "
            "(turn-wm extract-action-ablation)"
        )

    if (provenance.get("data") or {}).get("split") != "validation":
        raise ValueError("The action ablation is analysed on the validation split only")

    output_dir = (
        snapshot.path / "analysis" / ANALYSIS if output_dir is None else output_dir
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    start = time.perf_counter()
    corpora = [str(d) for d in snapshot.metadata["dataset"]]
    recordings = [
        f"{d}/{r}"
        for d, r in zip(corpora, snapshot.metadata["recording_id"], strict=True)
    ]
    horizons = provenance["rollout"]["horizons_steps"]
    checkpoint = provenance.get("checkpoint") or {}
    log(
        f"action ablation: {snapshot.path} ({len(recordings):,} anchors, "
        f"{len(set(recordings)):,} recordings, checkpoint "
        f"{checkpoint.get('filename')} step {checkpoint.get('global_step')})"
    )
    log(
        f"action ablation: {len(horizons)} horizons x {len(SUBSETS)} subsets + "
        f"{len(STATES)} focal states, {bootstrap} bootstrap resamples each; "
        f"output {output_dir}"
    )

    with progress(
        total=len(horizons) * len(SUBSETS) + len(STATES),
        desc="action ablation",
        unit="bootstrap",
    ) as bar:
        bar.set_description("action ablation | rollout conditions")
        rollout = rollout_ablation(
            snapshot.representations,
            provenance["rollout"],
            recordings=recordings,
            corpora=corpora,
            seed=snapshot.seed,
            bootstrap=bootstrap,
            tick=bar.update,
        )
        bar.set_description("action ablation | counterfactual")
        counterfactual = counterfactual_effects(
            snapshot.representations,
            recordings=recordings,
            corpora=corpora,
            seed=snapshot.seed,
            bootstrap=bootstrap,
            tick=bar.update,
        )

    for entry in rollout.values():
        for subset, values in entry["subsets"].items():
            if values["n"] == 0:
                log(
                    f"action ablation: {entry['horizon_s']:g} s, {subset}: no rows, "
                    "not evaluable"
                )

    for state in STATES:
        if counterfactual[state]["n"] == 0:
            log(f"action ablation: no anchor with focal state {state}, not evaluable")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "source": snapshot_source(snapshot),
        "ablation": provenance["action_ablation"],
        "rollout": provenance["rollout"],
        "settings": {
            "seed": snapshot.seed,
            "bootstrap_resamples": bootstrap,
            "confidence": CONFIDENCE,
            "interval": (
                "percentile bootstrap over recordings within each corpus; the "
                "same resampled recordings for every condition (paired)"
            ),
            "metric_roles": ROLES,
            "event_exposed": (
                "at least one true START/STOP among the future tokens the "
                "rollout reads before the horizon (steps t+1 .. t+h-1)"
            ),
            "alignment_rows": "rows with a displacement direction in every condition",
        },
        "rollout_ablation": rollout,
        "counterfactual": counterfactual,
        "figures": [
            "figures/rollout_ablation.png",
            "figures/counterfactual_action_effect.png",
        ],
    }

    log("action ablation: writing figures, tables and report")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("rollout_ablation.png", rollout_figure(summary)),
        ("counterfactual_action_effect.png", counterfactual_figure(summary)),
    ):
        figure.savefig(output_dir / "figures" / name, dpi=150, facecolor=SURFACE)
        close(figure)

    pq.write_table(scores_table(summary), output_dir / "scores.parquet")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(ablation_report(summary), encoding="utf-8")
    log(f"action ablation: done in {time.perf_counter() - start:.0f}s")

    return output_dir


def scores_table(summary: Mapping[str, Any]) -> pa.Table:
    """One row per (horizon, subset, condition or difference, metric)."""

    rows = []

    for h, entry in summary["rollout_ablation"].items():
        for subset, values in entry["subsets"].items():
            for kind, group in (
                ("condition", "conditions"),
                ("difference", "differences"),
            ):
                for name, metrics in values[group].items():
                    for metric in METRICS:
                        interval = metrics[f"{metric}_ci"] or [None, None]
                        rows.append(
                            {
                                "horizon_steps": int(h),
                                "horizon_s": entry["horizon_s"],
                                "subset": subset,
                                "n": values["n"],
                                "n_recordings": values["n_recordings"],
                                "kind": kind,
                                "name": name,
                                "metric": metric,
                                "role": ROLES[metric],
                                "value": metrics[metric],
                                "ci_low": interval[0],
                                "ci_high": interval[1],
                            }
                        )

    return pa.Table.from_pylist(rows)


_COLORS = dict(zip(CONDITIONS, SERIES, strict=False))


_TITLES = {
    SKILL: "Skill vs persistence (primary)",
    ALIGNMENT: "Displacement alignment (secondary)",
    MOVEMENT: "Movement ratio (diagnostic)",
}


_REFERENCES = {SKILL: 0.0, ALIGNMENT: 0.0, MOVEMENT: 1.0}


_SUBSET_TITLES = {
    ALL: "all anchors",
    EXPOSED: "event exposed",
    UNEXPOSED: "event unexposed",
}


def rollout_figure(summary: Mapping[str, Any]) -> Figure:
    """Rows: subsets; columns: the three measures; lines: conditions."""

    ablation = summary["rollout_ablation"]
    horizons = list(ablation)
    x = [ablation[h]["horizon_s"] for h in horizons]
    offsets = {OBSERVED: -0.015, STATE_PRESERVING: 0.0, SHUFFLED: 0.015}
    figure = new_figure(12.0, 9.0)
    axes = figure.subplots(len(SUBSETS), len(METRICS), squeeze=False)

    for r, subset in enumerate(SUBSETS):
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            style(ax)
            ax.axhline(_REFERENCES[metric], color=MUTED, linewidth=1, linestyle="--")

            for condition in CONDITIONS:
                points = []

                for xi, h in zip(x, horizons, strict=True):
                    values = ablation[h]["subsets"][subset]["conditions"][condition]

                    if values[metric] is not None:
                        points.append(
                            (
                                xi + offsets[condition],
                                values[metric],
                                values[f"{metric}_ci"],
                            )
                        )

                if not points:
                    continue

                ax.errorbar(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    yerr=[
                        [p[1] - (p[2] or [p[1], p[1]])[0] for p in points],
                        [(p[2] or [p[1], p[1]])[1] - p[1] for p in points],
                    ],
                    color=_COLORS[condition],
                    marker="o",
                    markersize=6,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    linewidth=2,
                    capsize=3,
                    label=condition,
                )

            ax.set_xticks(x, [f"{v:g} s" for v in x])
            if r == 0:
                ax.set_title(_TITLES[metric], color=INK, fontsize=10, loc="left")
            if c == 0:
                ax.set_ylabel(_SUBSET_TITLES[subset], color=SECONDARY_INK, fontsize=9)

    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Rollout under observed, state-preserving and shuffled future actions "
        "(95% paired recording bootstrap; at one step no future token is read)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


def counterfactual_figure(summary: Mapping[str, Any]) -> Figure:
    """Rows: distance and delta cosine; columns: focal state; valid actions only."""

    counterfactual = summary["counterfactual"]
    figure = new_figure(10.0, 7.0)
    axes = figure.subplots(2, len(STATES), squeeze=False)
    rows = (
        ("prediction_distance", "‖ẑ(a1) − ẑ(a2)‖"),
        ("delta_cosine", "cos(Δz(a1), Δz(a2))"),
    )

    for c, state in enumerate(STATES):
        pairs = counterfactual[state]["pairs"]

        for r, (key, label) in enumerate(rows):
            ax = axes[r][c]
            style(ax)
            ticks = []

            for x, (pair, values) in enumerate(pairs.items()):
                ticks.append(pair)
                stat = values[key]

                if stat["mean"] is None:
                    continue

                interval = stat["ci"] or [stat["mean"], stat["mean"]]
                ax.errorbar(
                    [x],
                    [stat["mean"]],
                    yerr=[[stat["mean"] - interval[0]], [interval[1] - stat["mean"]]],
                    color=SERIES[0],
                    marker="o",
                    markersize=8,
                    markerfacecolor=SERIES[0],
                    markeredgewidth=2,
                    capsize=3,
                    linewidth=2,
                )

            if key == "prediction_distance":
                scale = counterfactual[state]["mean_state_preserving_step"]["mean"]
                if scale is not None:
                    ax.axhline(scale, color=MUTED, linestyle="--", linewidth=1)
                    ax.annotate(
                        "mean ‖Δz(state-preserving action)‖",
                        (len(pairs) - 0.5, scale),
                        ha="right",
                        va="bottom",
                        fontsize=7,
                        color=MUTED,
                    )

            ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
            ax.set_xlim(-0.5, len(ticks) - 0.5)

            if key == "delta_cosine":
                # Cosines live in [-1, 1]; a fixed axis keeps them comparable.
                ax.set_ylim(-1.05, 1.05)
                ax.axhline(0.0, color=MUTED, linestyle="--", linewidth=1)

            if not counterfactual[state]["n"]:
                ax.text(
                    0.5,
                    0.65,
                    "no anchors in this focal state",
                    transform=ax.transAxes,
                    ha="center",
                    color=MUTED,
                    fontsize=9,
                )
            if r == 0:
                ax.set_title(
                    f"focal state {state} (n={counterfactual[state]['n']:,})",
                    color=INK,
                    fontsize=10,
                    loc="left",
                )
            if c == 0:
                ax.set_ylabel(label, color=SECONDARY_INK, fontsize=9)

    figure.suptitle(
        "One-step action effect from the same state and context "
        "(only state-valid controllable actions)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


def _fmt(values: Mapping[str, Any], metric: str, *, signed: bool = False) -> str:
    value, interval = values[metric], values[f"{metric}_ci"]

    if value is None:
        return "n/a"

    text = f"{value:+.3f}" if signed else f"{value:.3f}"

    return (
        text if interval is None else f"{text} [{interval[0]:.3f}, {interval[1]:.3f}]"
    )


def _verdict(values: Mapping[str, Any], metric: str) -> str:
    interval = values[f"{metric}_ci"]

    if values[metric] is None or interval is None:
        return "undetermined"
    if interval[0] > 0:
        return "observed higher"
    if interval[1] < 0:
        return "observed lower"

    return "no difference within the interval"


def _ablation_table(summary: Mapping[str, Any], metric: str) -> list[str]:
    lines = [
        (
            "| horizon | subset | n (rec.) | observed | state_preserving | shuffled | "
            "observed − state_preserving | observed − shuffled |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    for entry in summary["rollout_ablation"].values():
        for subset in SUBSETS:
            values = entry["subsets"][subset]
            conditions, differences = values["conditions"], values["differences"]
            lines.append(
                f"| {entry['horizon_s']:g} s | {subset} | {values['n']:,} "
                f"({values['n_recordings']}) | "
                + " | ".join(_fmt(conditions[c], metric) for c in CONDITIONS)
                + " | "
                + " | ".join(
                    _fmt(differences[f"{a}-{b}"], metric, signed=True)
                    for a, b in DIFFERENCES
                )
                + " |"
            )

    return lines


def _skill_findings(summary: Mapping[str, Any]) -> list[str]:
    lines = []

    for entry in summary["rollout_ablation"].values():
        if entry["future_tokens_read"] == 0:
            continue

        for subset in SUBSETS:
            differences = entry["subsets"][subset]["differences"]
            parts = [
                f"observed − {b}: {_fmt(differences[f'{a}-{b}'], SKILL, signed=True)} "
                f"({_verdict(differences[f'{a}-{b}'], SKILL)})"
                for a, b in DIFFERENCES
            ]
            lines.append(
                f"- {entry['horizon_s']:g} s, {subset}: " + "; ".join(parts) + "."
            )

    return lines


def _counterfactual_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| focal state | valid action pair | ‖ẑ(a1) − ẑ(a2)‖ | cos(Δz(a1), Δz(a2)) |",
        "|---|---|---|---|",
    ]

    for state in STATES:
        entry = summary["counterfactual"][state]

        for pair, values in entry["pairs"].items():
            distance, cosine = values["prediction_distance"], values["delta_cosine"]
            lines.append(
                f"| {state} (n={entry['n']:,}) | {pair} | "
                f"{_mean(distance)} | {_mean(cosine)} |"
            )

    return lines


def _mean(stat: Mapping[str, Any]) -> str:
    if stat["mean"] is None:
        return "n/a"

    interval = stat["ci"]

    return (
        f"{stat['mean']:.3f}"
        if interval is None
        else f"{stat['mean']:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"
    )


def ablation_report(summary: Mapping[str, Any]) -> str:
    provenance = summary["source"]["snapshot_provenance"] or {}
    checkpoint = provenance.get("checkpoint") or {}
    ablation = summary["ablation"]
    shuffle = ablation["shuffle"]
    first = next(iter(summary["rollout_ablation"].values()))
    integrity = first.get("integrity") or {}
    embeddings = ablation["action_embeddings"]
    scales = "; ".join(
        f"{state}: {_mean(summary['counterfactual'][state]['mean_state_preserving_step'])}"
        for state in STATES
    )

    return "\n".join(
        [
            "# Ego-action ablation and forced-action effects",
            "",
            "## Purpose",
            "",
            (
                "1. Does the predictor materially use its future ego-action "
                "conditioning?"
            ),
            (
                "2. Does changing only the action token change the predicted latent "
                "transition from the same state?"
            ),
            "",
            (
                f"Checkpoint `{checkpoint.get('filename')}` (step "
                f"{checkpoint.get('global_step')}); **validation split only**; "
                f"{summary['source']['samples']:,} anchors (seeded fixed permutation). "
                "Every prediction comes from the existing validation rollout "
                "(`lejepa_forward`); only the action tensor changes."
            ),
            "",
            "## Conditions",
            "",
            (
                "- **observed**: the real future WAIT/START/HOLD/STOP sequence, as in "
                "the rollout-dynamics analysis."
            ),
            "- **state_preserving**: future tokens replaced by the valid state-preserving action (WAIT or HOLD).",
            (
                "- **shuffled**: the future sequence replaced by another sample's "
                f"complete future sequence, same corpus and focal state at the anchor "
                f"(seed {shuffle['seed']}, {shuffle['groups']} group(s); "
                f"{shuffle['rows_in_singleton_groups']} rows in single-member groups "
                "keep their own sequence)."
            ),
            "",
            f"Unchanged in every condition: {ablation['unchanged']}.",
            "",
            (
                f"**Integrity check (one step, no future token read):** maximum absolute "
                f"difference between conditions = "
                f"{integrity.get('max_abs_difference_between_conditions')} — "
                f"{'passed' if integrity.get('passed') else 'FAILED'}."
            ),
            "",
            (
                "Subsets: **event_exposed** = at least one true START/STOP among the "
                "future tokens read before the horizon; **event_unexposed** = none."
            ),
            "",
            "## Rollout ablation",
            "",
            "### Skill vs persistence (primary)",
            "",
            *_ablation_table(summary, SKILL),
            "",
            *_skill_findings(summary),
            "",
            "### Displacement alignment (secondary)",
            "",
            (
                "Rows with a displacement direction in every condition, so the "
                "comparison is paired."
            ),
            "",
            *_ablation_table(summary, ALIGNMENT),
            "",
            "### Movement ratio (diagnostic)",
            "",
            *_ablation_table(summary, MOVEMENT),
            "",
            "## Forced one-step action effect",
            "",
            (
                "From the same state and context, only the anchor's own action is "
                "forced to WAIT, START, HOLD or STOP, and the one-step prediction "
                "ẑ(a) = ẑ_(t+1)(a) is compared: ‖ẑ(a1) − ẑ(a2)‖ and the cosine between "
                "Δz(a) = ẑ(a) − z_t. Only state-valid alternatives are reported: "
                "WAIT vs START from SILENT, and HOLD vs STOP from SPEAKING."
            ),
            "",
            *_counterfactual_table(summary),
            "",
            (
                f"Scale context, mean ‖Δz(state-preserving action)‖ per focal state: {scales}. "
                f"Anchors with another focal state are not stratified: "
                f"{summary['counterfactual']['unstratified_rows']}."
            ),
            "",
            "## Action embedding diagnostic",
            "",
            "Descriptive only; not a scientific conclusion.",
            "",
            "- Norms: "
            + ", ".join(f"{a} {v:.3f}" for a, v in embeddings["norm"].items()),
            "- Cosines: "
            + ", ".join(f"{p} {v:.3f}" for p, v in embeddings["cosine"].items()),
            "- Euclidean distances: "
            + ", ".join(f"{p} {v:.3f}" for p, v in embeddings["euclidean"].items()),
            "",
            "## Interpretation",
            "",
            (
                "Where observed outperforms both ablations (intervals of the paired "
                "differences above 0), this supports: *the V1 predictor makes useful "
                "use of the observed ego-action conditioning channel.* Where an "
                "interval includes 0, this analysis finds no measurable benefit of "
                "the observed tokens at that horizon and subset, which is not proof "
                "that the channel is unused. Where observed is lower than an "
                "ablation, the observed tokens degrade the rollout there."
            ),
            "",
            "## What this does NOT show",
            "",
            (
                "- It does not establish causal intervention semantics: the conditioning "
                "tokens are observed conversation events, not interventions."
            ),
            (
                "- It does not establish planner controllability: forced-action comparisons "
                "are out of distribution, and no agent chose any action."
            ),
            "- It does not establish planning utility.",
            "",
        ]
    )
