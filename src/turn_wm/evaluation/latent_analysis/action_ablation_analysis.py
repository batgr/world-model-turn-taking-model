"""
Analyze an action-ablation snapshot (`action_ablation.py`).

A. Rollout ablation, per horizon and subset (ALL, EVENT_EXPOSED: at least
   one true START/STOP among the future tokens read before the horizon,
   EVENT_UNEXPOSED: none), with the rollout-dynamics measures:
   skill vs persistence (primary), displacement alignment (secondary) and
   movement ratio (diagnostic), for OBSERVED, STATE_PRESERVING and SHUFFLED, and
   the paired differences observed - state_preserving and observed - shuffled.
   Paired means the same rows and the same resampled recordings for every
   condition; the alignment counts only rows with a direction in every
   condition.

B. Forced one-step action effect, per focal state at the anchor (SILENT,
   SPEAKING): ||z_hat(a1) - z_hat(a2)|| and cos(delta_z(a1), delta_z(a2)),
   delta_z(a) = z_hat_next(a) - z_t, using only the valid pair for the current
   focal state: WAIT vs START from SILENT, HOLD vs STOP from SPEAKING.
   Invalid state/action combinations are not reported.

Intervals: seeded 95% percentile bootstrap over recordings within each
corpus, as in the rollout-dynamics analysis. Nothing is refitted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from turn_wm.data.dataset import STATE_TO_ID
from turn_wm.evaluation.latent_analysis.action_ablation import (
    CONDITIONS,
    COUNTERFACTUAL_NEXT,
    FOCAL_STATE,
    FORCED_ACTIONS,
    OBSERVED,
    PRED,
    SHUFFLED,
    STATE_PRESERVING,
)
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    CONFIDENCE,
    METRICS,
    MOVEMENT,
    SKILL,
    RowTerms,
    cluster_bootstrap_weights,
    future_event_rows,
    measures_of_sums,
    row_sums,
    row_terms,
)
from turn_wm.evaluation.latent_analysis.seeding import derived_seed

SCHEMA_VERSION = 1
ANALYSIS = "action_ablation"

ALL, EXPOSED, UNEXPOSED = "all", "event_exposed", "event_unexposed"
SUBSETS = (ALL, EXPOSED, UNEXPOSED)
DIFFERENCES = ((OBSERVED, STATE_PRESERVING), (OBSERVED, SHUFFLED))
ROLES = {SKILL: "primary", ALIGNMENT: "secondary", MOVEMENT: "diagnostic"}

STATES = ("SILENT", "SPEAKING")
PAIRS_BY_STATE = {
    "SILENT": (("START", "WAIT"),),
    "SPEAKING": (("STOP", "HOLD"),),
}


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
        sums[name] = per_cluster.index_add_(0, rows, row_sums(t)[index])

    stratum = {recordings[i]: corpora[i] for i in index}
    # One set of resampled recordings for every condition: paired.
    weights = cluster_bootstrap_weights(
        clusters,
        [stratum[c] for c in clusters],
        resamples=bootstrap,
        generator=generator,
    )
    point = {name: measures_of_sums(s.sum(0)) for name, s in sums.items()}
    resampled = {name: measures_of_sums(weights @ s) for name, s in sums.items()}

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
                generator=torch.Generator().manual_seed(derived_seed(seed, h, name)),
            )

            if tick is not None:
                tick()

        results[str(h)] = entry

    return results


# ---------------------------------------------------------------------------
# B. Counterfactual one-step action effect
# ---------------------------------------------------------------------------


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
                generator=torch.Generator().manual_seed(
                    derived_seed(seed, "cf", state)
                ),
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

        for a, b in PAIRS_BY_STATE[state]:
            da, db = delta[:, index[a]], delta[:, index[b]]
            both = (da.norm(dim=-1) > 0) & (db.norm(dim=-1) > 0)
            cosine = torch.where(both, F.cosine_similarity(da, db, dim=-1), torch.nan)
            pairs[f"{a} vs {b}"] = {
                "prediction_distance": mean(
                    (nxt[:, index[a]] - nxt[:, index[b]]).norm(dim=-1)
                ),
                "delta_cosine": mean(cosine),
            }

        results[state] = {
            "n": len(rows),
            "n_recordings": len(clusters),
            # Context for the distances' scale, not a separate measure.
            "mean_state_preserving_step": mean(
                delta[:, index["WAIT" if state == "SILENT" else "HOLD"]].norm(dim=-1)
            ),
            "pairs": pairs,
        }

        if tick is not None:
            tick()

    return results
