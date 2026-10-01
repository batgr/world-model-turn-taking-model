"""
Files, figures and report of the rollout dynamics (`rollout_dynamics`).

`write_rollout_dynamics` runs the analysis and writes `summary.json`, the
metric tables, the skill, displacement and trajectory figures and
`report.md`: skill against persistence, direction and magnitude of the
predicted motion, on stable and transition rows.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from turn_wm.evaluation.latent_analysis.artifacts import sha256, snapshot_source
from turn_wm.evaluation.latent_analysis.label_source import (
    CorpusLabelSource,
    audit_corpus,
    hub_label_sources,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    GRID,
    INK,
    MUTED,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    close,
    new_figure,
    style,
)
from turn_wm.evaluation.latent_analysis.rollout import (
    ROLLOUT_TENSORS,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    ALL,
    ANALYSIS,
    CONDITIONS,
    CURRENT_STATE,
    DEFAULT_BOOTSTRAP,
    DEFAULT_TRAJECTORIES,
    FUTURE_STATE,
    METRICS,
    MOVEMENT,
    SCHEMA_VERSION,
    SELECTION,
    SKILL,
    STABLE,
    TRANSITION,
    RolloutDynamics,
    TrajectoryProjection,
    analyze_rollout_dynamics,
    rollout_conditions,
    trajectory_selection,
    trajectory_table,
    true_latent_projection,
)
from turn_wm.evaluation.latent_analysis.snapshot import describe_snapshot, read_snapshot
from turn_wm.progress import log, progress

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def write_rollout_dynamics(
    snapshot_dir: Path,
    *,
    output_dir: Path | None = None,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    trajectories: int = DEFAULT_TRAJECTORIES,
) -> Path:
    """Analyze a rollout snapshot; write tables, figures and the report."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    snapshot = read_snapshot(snapshot_dir)
    provenance = snapshot.manifest.get("provenance") or {}
    rollout = provenance.get("rollout")
    missing = [n for n in ROLLOUT_TENSORS if n not in snapshot.representations]

    if rollout is None or missing:
        raise ValueError(
            f"{snapshot.path} is not a rollout snapshot (turn-wm extract-rollouts)"
        )

    if (provenance.get("data") or {}).get("split") != "validation":
        raise ValueError("Rollout dynamics are analysed on the validation split only")

    output_dir = (
        snapshot.path / "analysis" / ANALYSIS if output_dir is None else output_dir
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    start = time.perf_counter()
    horizons_steps = [int(h) for h in rollout["horizons_steps"]]
    horizons_s = dict(
        zip(horizons_steps, map(float, rollout["horizons_s"]), strict=True)
    )
    log(f"analyze-rollouts: {describe_snapshot(snapshot)}; output {output_dir}")

    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(provenance, labels_revision=labels_revision)
    )
    log("analyze-rollouts: joining labels")
    audits = {corpus: audit_corpus(source) for corpus, source in sources.items()}
    joined = join_labels(snapshot.metadata, audits, sources, selection=SELECTION)
    conditions = rollout_conditions(joined.variables, horizons_s)
    log(
        f"analyze-rollouts: measuring {len(horizons_steps)} horizons x "
        f"{len(CONDITIONS)} conditions ({bootstrap} bootstrap resamples each)"
    )
    corpora = [str(d) for d in snapshot.metadata["dataset"]]

    with progress(
        total=len(horizons_steps) * len(CONDITIONS),
        desc="analyze-rollouts",
        unit="bootstrap",
    ) as bar:
        dynamics = analyze_rollout_dynamics(
            snapshot.representations,
            conditions,
            rollout,
            # Recording ids are unique within a corpus only.
            recordings=[
                f"{d}/{r}"
                for d, r in zip(corpora, snapshot.metadata["recording_id"], strict=True)
            ],
            corpora=corpora,
            seed=snapshot.seed,
            bootstrap=bootstrap,
            tick=bar.update,
        )

    log("analyze-rollouts: writing figures, tables and report")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "skill_vs_horizon.png": skill_figure(dynamics),
        "displacement_vs_horizon.png": displacement_figure(dynamics),
    }
    trajectory = None

    if trajectories > 0:
        rows = trajectory_selection(
            snapshot.metadata["sample_id"],
            conditions,
            max(horizons_steps),
            count=trajectories,
            seed=snapshot.seed,
        )

        if rows:
            projection = true_latent_projection(snapshot.representations)
            table = trajectory_table(
                snapshot.representations,
                snapshot.metadata,
                dynamics,
                projection,
                rows,
            )
            pq.write_table(table, output_dir / "trajectories.parquet")
            figures["transition_trajectories.png"] = trajectory_figure(
                table, projection
            )
            trajectory = {
                "samples": [str(snapshot.metadata["sample_id"][i]) for i in rows],
                "selected_at_horizon_steps": max(horizons_steps),
                "selection": (
                    "transition rows at the largest horizon, in seeded sample-key "
                    "order, one per distinct (current, future) change first"
                ),
                "pca_fit": "true latents only (anchor_latent and true_future_latent)",
                "pca_fit_rows": projection.fit_rows,
                "explained_variance_ratio": projection.explained_variance_ratio,
            }

    for name, figure in figures.items():
        figure.savefig(figures_dir / name, dpi=150, facecolor=SURFACE)
        close(figure)

    conditions_path = output_dir / "conditions.parquet"
    pq.write_table(conditions_table(snapshot.metadata, dynamics), conditions_path)
    pq.write_table(metrics_table(dynamics), output_dir / "metrics.parquet")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS,
        "source": snapshot_source(snapshot),
        "rollout": rollout,
        "labels": {
            "current_state": f"{CURRENT_STATE}:dominant",
            "future_state": f"{FUTURE_STATE}@h",
            "corpora": {c: a.provenance for c, a in audits.items()},
            "alignment": joined.alignment,
            "excluded": joined.excluded,
            "conditions_sha256": sha256(conditions_path),
        },
        "settings": dynamics.settings,
        "horizons_steps": horizons_steps,
        "horizons_s": [horizons_s[h] for h in horizons_steps],
        "metrics": {str(h): dynamics.metrics[h] for h in horizons_steps},
        "trajectories": trajectory,
        "figures": [f"figures/{name}" for name in figures],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(rollout_report(summary), encoding="utf-8")
    log(f"analyze-rollouts: done in {time.perf_counter() - start:.0f}s")

    return output_dir


def metrics_table(dynamics: RolloutDynamics) -> pa.Table:
    """One row per (horizon, condition)."""

    rows = []

    for h in dynamics.horizons_steps:
        for condition in CONDITIONS:
            values = dynamics.metrics[h][condition]
            row = {
                "horizon_steps": h,
                "horizon_s": dynamics.horizons_s[h],
                "condition": condition,
                "n": values["n"],
                "n_recordings": values["n_recordings"],
                "n_direction_defined": values["n_direction_defined"],
                "future_event_fraction": values["future_event_fraction"],
            }

            for name in METRICS:
                interval = values[f"{name}_ci"] or [None, None]
                row |= {
                    name: values[name],
                    f"{name}_ci_low": interval[0],
                    f"{name}_ci_high": interval[1],
                }

            rows.append(row)

    return pa.Table.from_pylist(rows)


def conditions_table(
    metadata: Mapping[str, Sequence[Any]], dynamics: RolloutDynamics
) -> pa.Table:
    """The states and condition of every snapshot row, per horizon."""

    columns: dict[str, Any] = {
        name: list(metadata[name])
        for name in ("sample_id", "dataset", "recording_id", "anchor_idx")
    }
    columns["current_state"] = dynamics.conditions.current

    for h in dynamics.horizons_steps:
        tag = f"{dynamics.horizons_s[h]:g}s"
        columns[f"future_state@{tag}"] = dynamics.conditions.future[h]
        columns[f"condition@{tag}"] = dynamics.conditions.condition[h]

    return pa.table(columns)


# Fixed slots: the condition, never its rank, picks the color.
_CONDITION_COLORS = dict(zip(CONDITIONS, SERIES, strict=False))


_TITLES = {
    SKILL: "Skill vs persistence",
    ALIGNMENT: "Displacement alignment",
    MOVEMENT: "Movement ratio",
}


_REFERENCES = {SKILL: 0.0, ALIGNMENT: 0.0, MOVEMENT: 1.0}


def _metric_panel(ax, dynamics: RolloutDynamics, name: str) -> None:
    x = [dynamics.horizons_s[h] for h in dynamics.horizons_steps]
    offsets = {ALL: -0.012, STABLE: 0.0, TRANSITION: 0.012}

    style(ax)
    ax.axhline(_REFERENCES[name], color=MUTED, linewidth=1.0, linestyle="--")

    for condition in CONDITIONS:
        values = [dynamics.metrics[h][condition] for h in dynamics.horizons_steps]
        points = [
            (xi + offsets[condition], v[name], v[f"{name}_ci"])
            for xi, v in zip(x, values, strict=True)
            if v[name] is not None
        ]

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        low = [p[1] - (p[2][0] if p[2] else p[1]) for p in points]
        high = [(p[2][1] if p[2] else p[1]) - p[1] for p in points]
        color = _CONDITION_COLORS[condition]
        ax.errorbar(
            xs,
            ys,
            yerr=[low, high],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            capsize=3,
            label=f"{condition} (n={values[-1]['n']:,} at {x[-1]:g} s)",
        )
        ax.annotate(
            condition,
            (xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=SECONDARY_INK,
        )

    ax.set_xticks(x, [f"{v:g} s" for v in x])
    ax.set_xlim(min(x) - 0.08, max(x) + 0.2)
    ax.set_xlabel("Rollout horizon", color=SECONDARY_INK, fontsize=9)
    ax.set_title(_TITLES[name], color=INK, fontsize=10, loc="left")


def skill_figure(dynamics: RolloutDynamics) -> Figure:
    figure = new_figure(6.0, 4.0)
    ax = figure.add_subplot()
    _metric_panel(ax, dynamics, SKILL)
    ax.set_ylabel("1 − MSE model / MSE persistence", color=SECONDARY_INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Q1 · Does the rollout beat persistence during state transitions?",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


def displacement_figure(dynamics: RolloutDynamics) -> Figure:
    figure = new_figure(10.0, 4.0)
    left, right = figure.subplots(1, 2)
    _metric_panel(left, dynamics, ALIGNMENT)
    left.set_ylabel(
        "mean cos(pred − anchor, true − anchor)", color=SECONDARY_INK, fontsize=9
    )
    _metric_panel(right, dynamics, MOVEMENT)
    right.set_ylabel(
        "Σ‖pred − anchor‖ / Σ‖true − anchor‖", color=SECONDARY_INK, fontsize=9
    )
    left.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Q2 · Direction and Q3 · magnitude of predicted latent motion "
        "(95% bootstrap intervals)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


def trajectory_figure(table: pa.Table, projection: TrajectoryProjection) -> Figure:
    rows = table.to_pylist()
    samples = list(dict.fromkeys(row["sample_id"] for row in rows))
    columns = min(3, len(samples))
    lines = math.ceil(len(samples) / columns)
    figure = new_figure(3.6 * columns, 3.3 * lines + 0.5)
    axes = figure.subplots(lines, columns, squeeze=False).flatten()
    paths = {"true": SERIES[0], "pred": SERIES[1]}

    for ax, sample in zip(axes, samples, strict=False):
        style(ax)
        points = [row for row in rows if row["sample_id"] == sample]

        for path, color in paths.items():
            line = sorted(
                (p for p in points if p["path"] == path),
                key=lambda p: p["horizon_steps"],
            )
            ax.plot(
                [p["pc1"] for p in line],
                [p["pc2"] for p in line],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=path,
            )

            for p in line[1:]:
                ax.annotate(
                    f"{p['horizon_s']:g}s",
                    (p["pc1"], p["pc2"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=SECONDARY_INK,
                )

        start = next(p for p in points if p["horizon_steps"] == 0)
        ax.scatter(
            [start["pc1"]], [start["pc2"]], s=60, color=INK, zorder=3, label="anchor"
        )
        end = max(
            (p for p in points if p["path"] == "true"),
            key=lambda p: p["horizon_steps"],
        )
        ax.set_title(
            f"{start['state']} → {end['state']} @ {end['horizon_s']:g} s\n{sample}",
            color=INK,
            fontsize=8,
            loc="left",
        )

    for ax in axes[len(samples) :]:
        ax.set_visible(False)

    axes[0].legend(frameon=False, fontsize=7, labelcolor=SECONDARY_INK)
    ratio = projection.explained_variance_ratio
    figure.suptitle(
        "True vs predicted latent paths of conversational transitions — PCA fitted "
        f"on true latents only (PC1 {100 * ratio[0]:.1f}%, PC2 "
        f"{100 * ratio[1]:.1f}% of variance)",
        color=INK,
        fontsize=9,
        x=0.02,
        ha="left",
    )

    for ax in axes[: len(samples)]:
        ax.tick_params(labelsize=7)
        ax.grid(True, color=GRID, linewidth=0.6)

    return figure


def _interval(values: Mapping[str, Any], name: str) -> str:
    value, interval = values[name], values[f"{name}_ci"]

    if value is None:
        return "n/a"

    if interval is None:
        return f"{value:.3f}"

    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


def _versus(values: Mapping[str, Any], name: str, reference: float) -> str:
    """Where the interval lies relative to `reference`."""

    interval = values[f"{name}_ci"]

    if values[name] is None or interval is None:
        return "undetermined"
    if interval[0] > reference:
        return "above"
    if interval[1] < reference:
        return "below"

    return "includes"


def _table(summary: Mapping[str, Any], name: str) -> list[str]:
    lines = [
        "| horizon | " + " | ".join(CONDITIONS) + " |",
        "|---|" + "---|" * len(CONDITIONS),
    ]

    for h, seconds in zip(
        summary["horizons_steps"], summary["horizons_s"], strict=True
    ):
        cells = []

        for c in CONDITIONS:
            values = summary["metrics"][str(h)][c]
            count = f"n={values['n']:,}, {values['n_recordings']:,} rec."

            if name == ALIGNMENT:
                count = (
                    f"{values['n_direction_defined']:,} of {values['n']:,} rows "
                    f"valid, {values['n_recordings']:,} rec."
                )

            cells.append(f"{_interval(values, name)} ({count})")

        lines.append(f"| {seconds:g} s | " + " | ".join(cells) + " |")

    return lines


def _answers(summary: Mapping[str, Any], name: str) -> list[str]:
    reference = _REFERENCES[name]
    words = {
        SKILL: {
            "above": "beats persistence",
            "below": "is worse than persistence",
            "includes": "is not distinguishable from persistence",
        },
        ALIGNMENT: {
            "above": "points, on average, toward the true displacement",
            "below": "points, on average, away from the true displacement",
            "includes": "has no measurable average alignment with the true displacement",
        },
        MOVEMENT: {
            "above": "moves farther than the true latent (overshoots its magnitude)",
            "below": "moves less than the true latent (undershoots its magnitude)",
            "includes": "moves as far as the true latent, within the interval",
        },
    }[name]
    lines = []

    for h, seconds in zip(
        summary["horizons_steps"], summary["horizons_s"], strict=True
    ):
        parts = []

        for condition in (TRANSITION, STABLE):
            values = summary["metrics"][str(h)][condition]
            where = _versus(values, name, reference)
            text = words.get(where, "cannot be assessed (too few rows)")
            parts.append(
                f"on **{condition}** rows the rollout {text} ({_interval(values, name)})"
            )

        lines.append(f"- At {seconds:g} s, " + "; ".join(parts) + ".")

    return lines


def _event_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| horizon | future tokens read | " + " | ".join(CONDITIONS) + " |",
        "|---|---|" + "---|" * len(CONDITIONS),
    ]
    tokens = summary["rollout"]["future_action_tokens_by_horizon"]

    for h, seconds in zip(
        summary["horizons_steps"], summary["horizons_s"], strict=True
    ):
        cells = []

        for c in CONDITIONS:
            values = summary["metrics"][str(h)][c]
            fraction = values["future_event_fraction"]
            cells.append(
                "n/a"
                if fraction is None
                else f"{100 * fraction:.1f}% ({values['n_future_event']:,}/"
                f"{values['n']:,})"
            )

        lines.append(
            f"| {seconds:g} s | {tokens[str(h)]} | " + " | ".join(cells) + " |"
        )

    return lines


def rollout_report(summary: Mapping[str, Any]) -> str:
    rollout = summary["rollout"]
    provenance = summary["source"]["snapshot_provenance"] or {}
    checkpoint = provenance.get("checkpoint") or {}
    data = provenance.get("data") or {}
    settings = summary["settings"]
    horizons = list(zip(summary["horizons_steps"], summary["horizons_s"], strict=True))
    conditioned = [
        f"{seconds:g} s: {rollout['future_action_tokens_by_horizon'][str(h)]} "
        "ground-truth future action token(s)"
        for h, seconds in horizons
    ]
    first = summary["metrics"][str(horizons[0][0])][ALL]["n"]

    lines = [
        "# Rollout dynamics under conversational state transitions",
        "",
        (
            f"Run `{(provenance.get('run') or {}).get('run_id')}`, checkpoint "
            f"`{checkpoint.get('filename')}` (step {checkpoint.get('global_step')}), "
            f"{data.get('dataset')} @ `{data.get('dataset_revision')}`, "
            f"**{data.get('split')} split only**; {first:,} deterministic anchors "
            f"(seeded fixed permutation, seed {settings['seed']})."
        ),
        "",
        "## How the rollout was run",
        "",
        (
            f"The rollout is the validation rollout itself (`{rollout['implementation']}`): "
            "from the ground-truth context ending at the anchor t, the predictor feeds "
            f"back {rollout['fed_back_latents']}, keeping the latest "
            f"{rollout['rollout_context_size']} states and actions before each step. "
            "Persistence predicts z_(t+h) = z_t. The extraction ran in "
            f"{(provenance.get('extraction') or {}).get('precision')}, not the "
            "training-time bf16 autocast, so values can differ slightly from the "
            "logged val/skill_h."
        ),
        "",
        "**Is the rollout conditioned on ground-truth future ego-action tokens? "
        + (
            "Yes.**"
            if any(rollout["conditioned_on_ground_truth_future_actions"].values())
            else "No.**"
        )
        + " To predict z_(t+h) the predictor receives the model's ground-truth "
        "ego-action tokens (WAIT, START, HOLD, STOP, MASKED), deterministically "
        "derived from the audited action grid, for future steps t+1 … t+h−1, "
        "as in training and validation: " + "; ".join(conditioned) + ".",
        "",
        (
            "The question answered is therefore not whether the model can "
            "anticipate a conversational transition, but: **given the future "
            "event/action sequence used by the V1 formulation, can the model "
            "evolve the latent state through the corresponding conversational "
            "transition?** These tokens are conversation events e_t, not "
            "agent-controllable actions a_t: the results describe V1's "
            "event-conditioned dynamics, not an autonomous planner."
        ),
        "",
        "### Confounding diagnostic: events among the future tokens read",
        "",
        (
            "Fraction of rows whose future action tokens read by the rollout "
            "(steps t+1 … t+h−1) contain at least one START or STOP. Where it is "
            "high on transition rows, transition skill can partly come from the "
            "announced event. This is a diagnostic of the conditioning, not a "
            "performance measure."
        ),
        "",
        *_event_table(summary),
        "",
        "## Conditions",
        "",
        (
            f"Current state: `{summary['labels']['current_state']}` at the anchor; future "
            f"state: `{summary['labels']['future_state']}` at the same horizon as the "
            "rollout target. **stable**: same state, **transition**: another state, "
            "**all**: every anchor (rows without both labels count only in all). The "
            "current state is the dominant occupancy of the anchor's grid cell; the "
            "future label follows the data release's definition, so a mismatch between "
            "the two definitions can itself register as a transition."
        ),
        "",
        (
            f"Intervals: {int(100 * settings['confidence'])}% percentile cluster "
            "bootstrap: anchors of one recording are temporally dependent, so "
            "recordings, not anchors, are resampled with replacement, within each "
            f"corpus ({settings['bootstrap_resamples']} resamples, seeded). "
            "`rec.` is the number of recordings behind each value."
        ),
        "",
        "## Q1 · Does the model beat persistence during transitions?",
        "",
        (
            "Skill = 1 − Σ‖pred − true‖² / Σ‖anchor − true‖² (pooled over the "
            "condition's rows; > 0 beats persistence)."
        ),
        "",
        *_table(summary, SKILL),
        "",
        *_answers(summary, SKILL),
        "",
        "## Q2 · Does predicted latent motion point in the right direction?",
        "",
        (
            "Displacement alignment = mean over rows of cos(pred − anchor, true − "
            "anchor); 0 is no alignment, 1 perfect direction. A row has a direction "
            "only if ‖true − anchor‖ ≥ "
            f"{settings['min_true_motion_fraction']:g} × the horizon's median "
            "‖true − anchor‖ (over all anchors) and the prediction moves; other "
            "rows are excluded from the mean (never counted as 0), and the valid "
            "rows are listed."
        ),
        "",
        *_table(summary, ALIGNMENT),
        "",
        *_answers(summary, ALIGNMENT),
        "",
        "## Q3 · Does the predictor reproduce the magnitude of true motion?",
        "",
        (
            "Movement ratio = Σ‖pred − anchor‖ / Σ‖true − anchor‖; 1 reproduces the "
            "mean magnitude, < 1 stays closer to persistence than the data moves."
        ),
        "",
        *_table(summary, MOVEMENT),
        "",
        *_answers(summary, MOVEMENT),
        "",
        "## Figures",
        "",
        *[f"- `{name}`" for name in summary["figures"]],
    ]

    trajectory = summary.get("trajectories")

    if trajectory:
        ratio = trajectory["explained_variance_ratio"]
        lines += [
            "",
            (
                f"The trajectory figure shows {len(trajectory['samples'])} transitions "
                f"({trajectory['selection']}). Its 2D PCA is fitted on "
                f"{trajectory['pca_fit']} ({trajectory['pca_fit_rows']:,} rows) and then "
                "applied to the predictions; PC1 and PC2 explain "
                f"{100 * ratio[0]:.1f}% and {100 * ratio[1]:.1f}% of the true-latent "
                "variance, so the panels are illustrations, not evidence: the metrics "
                "above are computed in the full latent space."
            ),
        ]

    return "\n".join(lines) + "\n"
