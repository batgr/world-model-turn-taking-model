"""
Analyze an action-ablation snapshot (`action_ablation.py`).

A. Rollout ablation, per horizon and subset (ALL, EVENT_EXPOSED: at least
   one true ONSET/OFFSET among the future tokens read before the horizon,
   EVENT_UNEXPOSED: none), with the rollout-dynamics measures:
   skill vs persistence (primary), displacement alignment (secondary) and
   movement ratio (diagnostic), for OBSERVED, NO_EVENT and SHUFFLED, and
   the paired differences observed - no_event and observed - shuffled.
   Paired means the same rows and the same resampled recordings for every
   condition; the alignment counts only rows with a direction in every
   condition.

B. Counterfactual one-step effect, per focal state at the anchor (SILENT,
   SPEAKING): ||z_hat(a1) - z_hat(a2)|| and cos(delta_z(a1), delta_z(a2)),
   delta_z(a) = z_hat_next(a) - z_t, for ONSET vs NO_EVENT, OFFSET vs
   NO_EVENT and ONSET vs OFFSET. SILENT + ONSET and SPEAKING + OFFSET are
   the natural interventions; SILENT + OFFSET and SPEAKING + ONSET are
   out-of-distribution stress tests.

Intervals: seeded 95% percentile bootstrap over recordings within each
corpus, as in the rollout-dynamics analysis. Nothing is refitted.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from turn_wm.data.dataset import STATE_TO_ID
from turn_wm.evaluation.latent_analysis.action_ablation import (
    ABLATION_TENSORS,
    CONDITIONS,
    COUNTERFACTUAL_NEXT,
    FOCAL_STATE,
    FORCED_ACTIONS,
    NO_EVENT,
    OBSERVED,
    PRED,
    SHUFFLED,
)
from turn_wm.evaluation.latent_analysis.analyze import Snapshot, read_snapshot
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    MUTED,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    close,
    style,
)
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    CONFIDENCE,
    DEFAULT_BOOTSTRAP,
    METRICS,
    MOVEMENT,
    SKILL,
    RowTerms,
    _measures,
    _row_sums,
    cluster_bootstrap_weights,
    future_event_rows,
    row_terms,
)
from turn_wm.progress import log, progress

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1
ANALYSIS = "action_ablation"

ALL, EXPOSED, UNEXPOSED = "all", "event_exposed", "event_unexposed"
SUBSETS = (ALL, EXPOSED, UNEXPOSED)
DIFFERENCES = ((OBSERVED, NO_EVENT), (OBSERVED, SHUFFLED))
ROLES = {SKILL: "primary", ALIGNMENT: "secondary", MOVEMENT: "diagnostic"}

PAIRS = (("ONSET", "NO_EVENT"), ("OFFSET", "NO_EVENT"), ("ONSET", "OFFSET"))
STATES = ("SILENT", "SPEAKING")
NATURAL = {("SILENT", "ONSET"), ("SPEAKING", "OFFSET")}
STRESS = {("SILENT", "OFFSET"), ("SPEAKING", "ONSET")}


# ---------------------------------------------------------------------------
# A. Rollout ablation
# ---------------------------------------------------------------------------


def _paired_terms(terms: Mapping[str, RowTerms]) -> dict[str, RowTerms]:
    """The same alignment rows for every condition: defined in all of them."""

    common = torch.stack([t.direction_defined for t in terms.values()]).all(0)

    return {
        name: RowTerms(
            model_error=t.model_error,
            persistence_error=t.persistence_error,
            cosine=torch.where(common, t.cosine, torch.nan),
            true_motion=t.true_motion,
            pred_motion=t.pred_motion,
            min_true_motion=t.min_true_motion,
        )
        for name, t in terms.items()
    }


def paired_metrics(
    terms: Mapping[str, RowTerms],
    members: torch.Tensor,
    *,
    recordings: Sequence[str],
    corpora: Sequence[str],
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Every condition's measures and the paired differences, one subset."""

    index = [int(i) for i in members.nonzero().flatten()]
    terms = _paired_terms(terms)
    first = next(iter(terms.values()))
    defined = int(first.direction_defined[index].sum()) if index else 0
    clusters = sorted({recordings[i] for i in index})
    result: dict[str, Any] = {
        "n": len(index),
        "n_recordings": len(clusters),
        "n_direction_defined_all_conditions": defined,
    }
    empty = {name: None for name in METRICS} | {f"{n}_ci": None for n in METRICS}

    if not index:
        result["conditions"] = {c: dict(empty) for c in terms}
        result["differences"] = {f"{a}-{b}": dict(empty) for a, b in DIFFERENCES}
        return result

    position = {c: g for g, c in enumerate(clusters)}
    rows = torch.tensor([position[recordings[i]] for i in index])
    sums = {}

    for name, t in terms.items():
        per_cluster = torch.zeros(len(clusters), 6, dtype=torch.float64)
        sums[name] = per_cluster.index_add_(0, rows, _row_sums(t)[index])

    stratum = {recordings[i]: corpora[i] for i in index}
    # One set of resampled recordings for every condition: paired.
    weights = cluster_bootstrap_weights(
        clusters,
        [stratum[c] for c in clusters],
        resamples=bootstrap,
        generator=generator,
    )
    point = {name: _measures(s.sum(0)) for name, s in sums.items()}
    resampled = {name: _measures(weights @ s) for name, s in sums.items()}

    def summarize(value: torch.Tensor, samples: torch.Tensor) -> tuple:
        value_ = float(value)

        if math.isnan(value_):
            return None, None

        tail = (1 - CONFIDENCE) / 2
        quantiles = torch.tensor([tail, 1 - tail], dtype=torch.float64)
        interval = (
            torch.nanquantile(samples, quantiles).tolist()
            if len(clusters) > 1
            else None
        )

        return value_, interval

    result["conditions"] = {}

    for name in terms:
        entry = {}

        for metric in METRICS:
            entry[metric], entry[f"{metric}_ci"] = summarize(
                point[name][metric], resampled[name][metric]
            )

        result["conditions"][name] = entry

    result["differences"] = {}

    for a, b in DIFFERENCES:
        entry = {}

        for metric in METRICS:
            entry[metric], entry[f"{metric}_ci"] = summarize(
                point[a][metric] - point[b][metric],
                resampled[a][metric] - resampled[b][metric],
            )

        result["differences"][f"{a}-{b}"] = entry

    return result


def rollout_ablation(
    representations: Mapping[str, torch.Tensor],
    rollout: Mapping[str, Any],
    *,
    recordings: Sequence[str],
    corpora: Sequence[str],
    seed: int,
    bootstrap: int,
    tick: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Per horizon: integrity at h = 1, and every subset's paired metrics.

    `tick` is called after every subset's bootstrap (progress display only).
    """

    horizons = [int(h) for h in rollout["horizons_steps"]]
    anchor = representations[ANCHOR_LATENT]
    true = representations[TRUE_FUTURE_LATENT]
    actions = representations[ROLLOUT_ACTION_IDS]
    results: dict[str, Any] = {}

    for k, h in enumerate(horizons):
        predictions = {c: representations[PRED[c]][:, k] for c in CONDITIONS}
        exposed = future_event_rows(actions, rollout, h)
        entry: dict[str, Any] = {
            "horizon_s": float(rollout["horizons_s"][k]),
            "future_tokens_read": int(
                rollout["future_action_tokens_by_horizon"][str(h)]
            ),
        }

        if entry["future_tokens_read"] == 0:
            difference = max(
                float((predictions[c] - predictions[OBSERVED]).abs().max())
                for c in CONDITIONS
            )
            entry["integrity"] = {
                "max_abs_difference_between_conditions": difference,
                "passed": difference == 0.0,
            }

            if difference != 0.0:
                raise ValueError(
                    f"Integrity check failed at h = {h}: no future action is read, "
                    f"yet conditions differ by up to {difference}"
                )

        terms = {c: row_terms(anchor, true[:, k], p) for c, p in predictions.items()}
        subsets = {
            ALL: torch.ones(len(anchor), dtype=torch.bool),
            EXPOSED: exposed,
            UNEXPOSED: ~exposed,
        }
        entry["subsets"] = {}

        for name, members in subsets.items():
            entry["subsets"][name] = paired_metrics(
                terms,
                members,
                recordings=recordings,
                corpora=corpora,
                bootstrap=bootstrap,
                generator=torch.Generator().manual_seed(_seed(seed, h, name)),
            )

            if tick is not None:
                tick()

        results[str(h)] = entry

    return results


# ---------------------------------------------------------------------------
# B. Counterfactual one-step action effect
# ---------------------------------------------------------------------------


NATURAL_INTERVENTION = "natural"
STRESS_INTERVENTION = "stress test (out of distribution)"
MIXED_INTERVENTION = "mixed: one natural, one stress-test action"


def intervention(state: str, a: str, b: str) -> str:
    """How natural the pair's forced events are for the focal state.

    NO_EVENT is neutral; ONSET and OFFSET are natural or stress tests
    depending on the state (SILENT + ONSET, SPEAKING + OFFSET are natural).
    """

    kinds = {(state, action) in NATURAL for action in (a, b) if action != "NO_EVENT"}

    if kinds == {True}:
        return NATURAL_INTERVENTION
    if kinds == {False}:
        return STRESS_INTERVENTION

    return MIXED_INTERVENTION


def _recording_mean(
    values: torch.Tensor,
    *,
    rows: list[int],
    cluster_of: torch.Tensor,
    clusters: int,
    weights: torch.Tensor | None,
) -> dict[str, Any]:
    """Mean of the defined values of `rows`, with a recording-bootstrap CI."""

    valued = values[rows]
    defined = ~valued.isnan()

    if not rows or weights is None or not bool(defined.any()):
        return {"mean": None, "ci": None, "n": int(defined.sum())}

    # Per-recording sums of the defined values, and their count.
    per_cluster = torch.zeros(clusters, 2, dtype=torch.float64)
    per_cluster.index_add_(
        0,
        cluster_of,
        torch.stack([torch.where(defined, valued, 0.0), defined.double()], -1),
    )
    total = per_cluster.sum(0)
    resampled = weights @ per_cluster
    tail = (1 - CONFIDENCE) / 2
    interval = (
        torch.nanquantile(
            resampled[:, 0] / resampled[:, 1],
            torch.tensor([tail, 1 - tail], dtype=torch.float64),
        ).tolist()
        if clusters > 1
        else None
    )

    return {"mean": float(total[0] / total[1]), "ci": interval, "n": int(defined.sum())}


def counterfactual_effects(
    representations: Mapping[str, torch.Tensor],
    *,
    recordings: Sequence[str],
    corpora: Sequence[str],
    seed: int,
    bootstrap: int,
    tick: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Per focal state and action pair: mean distance and mean delta cosine.

    `tick` is called after every focal state (progress display only).
    """

    anchor = representations[ANCHOR_LATENT].double()
    nxt = representations[COUNTERFACTUAL_NEXT].double()
    focal = representations[FOCAL_STATE]
    index = {a: FORCED_ACTIONS.index(a) for a in FORCED_ACTIONS}
    delta = nxt - anchor[:, None]
    results: dict[str, Any] = {
        "unstratified_rows": {
            name: int((focal == value).sum())
            for name, value in STATE_TO_ID.items()
            if name not in STATES
        }
    }

    for state in STATES:
        members = focal == STATE_TO_ID[state]
        rows = [int(i) for i in members.nonzero().flatten()]
        clusters = sorted({recordings[i] for i in rows})
        position = {c: g for g, c in enumerate(clusters)}
        cluster_of = torch.tensor(
            [position[recordings[i]] for i in rows], dtype=torch.long
        )
        stratum = {recordings[i]: corpora[i] for i in rows}
        weights = (
            cluster_bootstrap_weights(
                clusters,
                [stratum[c] for c in clusters],
                resamples=bootstrap,
                generator=torch.Generator().manual_seed(_seed(seed, "cf", state)),
            )
            if clusters
            else None
        )

        mean = partial(
            _recording_mean,
            rows=rows,
            cluster_of=cluster_of,
            clusters=len(clusters),
            weights=weights,
        )

        pairs = {}

        for a, b in PAIRS:
            da, db = delta[:, index[a]], delta[:, index[b]]
            both = (da.norm(dim=-1) > 0) & (db.norm(dim=-1) > 0)
            cosine = torch.where(both, F.cosine_similarity(da, db, dim=-1), torch.nan)
            pairs[f"{a} vs {b}"] = {
                "intervention": intervention(state, a, b),
                "prediction_distance": mean(
                    (nxt[:, index[a]] - nxt[:, index[b]]).norm(dim=-1)
                ),
                "delta_cosine": mean(cosine),
            }

        results[state] = {
            "n": len(rows),
            "n_recordings": len(clusters),
            # Context for the distances' scale, not a separate measure.
            "mean_no_event_step": mean(delta[:, index["NO_EVENT"]].norm(dim=-1)),
            "pairs": pairs,
        }

        if tick is not None:
            tick()

    return results


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


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
        "source": _source(snapshot),
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
                "at least one true ONSET/OFFSET among the future tokens the "
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_COLORS = dict(zip(CONDITIONS, SERIES, strict=False))
_TITLES = {
    SKILL: "Skill vs persistence (primary)",
    ALIGNMENT: "Displacement alignment (secondary)",
    MOVEMENT: "Movement ratio (diagnostic)",
}
_REFERENCES = {SKILL: 0.0, ALIGNMENT: 0.0, MOVEMENT: 1.0}
_TICK_SUFFIX = {
    NATURAL_INTERVENTION: "",
    STRESS_INTERVENTION: "\n(stress test)",
    MIXED_INTERVENTION: "\n(mixed)",
}
_SUBSET_TITLES = {
    ALL: "all anchors",
    EXPOSED: "event exposed",
    UNEXPOSED: "event unexposed",
}


def _new_figure(width: float, height: float) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def rollout_figure(summary: Mapping[str, Any]) -> Figure:
    """Rows: subsets; columns: the three measures; lines: conditions."""

    ablation = summary["rollout_ablation"]
    horizons = list(ablation)
    x = [ablation[h]["horizon_s"] for h in horizons]
    offsets = {OBSERVED: -0.015, NO_EVENT: 0.0, SHUFFLED: 0.015}
    figure = _new_figure(12.0, 9.0)
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
        "Rollout under observed, NO_EVENT and shuffled future actions "
        "(95% paired recording bootstrap; at 0.1 s no future token is read)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


def counterfactual_figure(summary: Mapping[str, Any]) -> Figure:
    """Rows: distance and delta cosine; columns: focal state; hollow: stress test."""

    counterfactual = summary["counterfactual"]
    figure = _new_figure(10.0, 7.0)
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
                natural = values["intervention"] == NATURAL_INTERVENTION
                ticks.append(pair + _TICK_SUFFIX[values["intervention"]])
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
                    markerfacecolor=SERIES[0] if natural else SURFACE,
                    markeredgewidth=2,
                    capsize=3,
                    linewidth=2,
                )

            if key == "prediction_distance":
                scale = counterfactual[state]["mean_no_event_step"]["mean"]
                if scale is not None:
                    ax.axhline(scale, color=MUTED, linestyle="--", linewidth=1)
                    ax.annotate(
                        "mean ‖Δz(NO_EVENT)‖",
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
        "Counterfactual one-step action effect from the same state and context "
        "(filled: natural intervention; hollow: stress test or mixed)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


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
            "| horizon | subset | n (rec.) | observed | no_event | shuffled | "
            "observed − no_event | observed − shuffled |"
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
        "| focal state | pair | intervention | ‖ẑ(a1) − ẑ(a2)‖ | cos(Δz(a1), Δz(a2)) |",
        "|---|---|---|---|---|",
    ]

    for state in STATES:
        entry = summary["counterfactual"][state]

        for pair, values in entry["pairs"].items():
            distance, cosine = values["prediction_distance"], values["delta_cosine"]
            lines.append(
                f"| {state} (n={entry['n']:,}) | {pair} | {values['intervention']} | "
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
        f"{state}: {_mean(summary['counterfactual'][state]['mean_no_event_step'])}"
        for state in STATES
    )

    return "\n".join(
        [
            "# Action/event ablation and counterfactual action effects",
            "",
            "## Purpose",
            "",
            (
                "1. Does the V1 predictor materially use its future action/event "
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
                "- **observed**: the real future NO_EVENT/ONSET/OFFSET sequence, as in "
                "the rollout-dynamics analysis."
            ),
            "- **no_event**: every future token read by the rollout set to NO_EVENT.",
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
                f"**Integrity check (0.1 s, no future token read):** maximum absolute "
                f"difference between conditions = "
                f"{integrity.get('max_abs_difference_between_conditions')} — "
                f"{'passed' if integrity.get('passed') else 'FAILED'}."
            ),
            "",
            (
                "Subsets: **event_exposed** = at least one true ONSET/OFFSET among the "
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
            "## Counterfactual one-step action effect",
            "",
            (
                "From the same state and context, only the anchor's own action is "
                "forced to NO_EVENT, ONSET or OFFSET, and the one-step prediction "
                "ẑ(a) = ẑ_(t+1)(a) is compared: ‖ẑ(a1) − ẑ(a2)‖ and the cosine between "
                "Δz(a) = ẑ(a) − z_t. SILENT + ONSET and SPEAKING + OFFSET are natural "
                "interventions; SILENT + OFFSET and SPEAKING + ONSET are "
                "out-of-distribution stress tests."
            ),
            "",
            *_counterfactual_table(summary),
            "",
            (
                f"Scale context, mean ‖Δz(NO_EVENT)‖ per focal state: {scales}. "
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
                "use of the observed vocal-action conditioning channel.* Where an "
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
                "- It does not establish planner controllability: stress-test actions "
                "are out of distribution, and no agent chose any action."
            ),
            "- It does not establish planning utility.",
            "",
        ]
    )


def _seed(seed: int, *parts: Any) -> int:
    text = ":".join(map(str, (seed, *parts)))

    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=7).digest())


def _source(snapshot: Snapshot) -> dict[str, Any]:
    digest = hashlib.sha256()

    with (snapshot.path / "representations.safetensors").open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return {
        "snapshot": str(snapshot.path),
        "representations_sha256": digest.hexdigest(),
        "samples": snapshot.manifest.get("samples"),
        "snapshot_provenance": snapshot.manifest.get("provenance"),
    }
