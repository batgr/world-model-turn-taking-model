"""
Rollout dynamics of a validation-rollout snapshot (`rollout.py`), conditioned
on whether the conversational state changes.

Three questions, one measure each, at every rollout horizon h:

1. Does the world model beat persistence during state transitions?
   skill = 1 - sum ||pred - true||^2 / sum ||anchor - true||^2
   (pooled over the condition's rows, as the validation skill).
2. Does predicted latent motion point in the right direction?
   displacement alignment = mean over rows of cos(pred - anchor, true - anchor).
3. Does the predictor reproduce the magnitude of true latent motion?
   movement ratio = sum ||pred - anchor|| / sum ||true - anchor||
   (a ratio of means, as val/prediction_delta_norm / val/target_delta_norm).

Conditions, per horizon, from the data release's labels (joined exactly by
`label_source`): the current state is
`instantaneous.joint_speech_state_occupancy:dominant` at the anchor, the
future state `future.future_joint_speech_state@h`. `stable` rows keep the
state, `transition` rows change it; `all` is every row of the snapshot.

Each value has a seeded 95% percentile interval from a cluster bootstrap:
recordings (not anchors, which are temporally dependent) are resampled with
replacement within each corpus. The alignment mean only counts rows whose
true displacement is at least `MIN_TRUE_MOTION_FRACTION` of the horizon's
median; excluded rows are counted. As a confounding diagnostic (not a
performance measure), each (horizon, condition) also reports the fraction of
rows whose future action tokens read by the rollout hold an START or STOP.
The optional trajectory figure projects a few transitions on the first two
principal components of the true latents only (anchor and true futures),
then applies that projection to the predictions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import torch

from turn_wm.evaluation.latent_analysis.label_source import (
    CONVERSATIONAL_STATE,
    FUTURE,
)
from turn_wm.evaluation.latent_analysis.pca import pca_2d, sample_keys
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
)
from turn_wm.training.metrics import skill_score

SCHEMA_VERSION = 1
ANALYSIS = "rollout_dynamics"

CURRENT_STATE = "instantaneous.joint_speech_state_occupancy"
FUTURE_STATE = "future.future_joint_speech_state"
SELECTION = ((CONVERSATIONAL_STATE, CURRENT_STATE), (FUTURE, FUTURE_STATE))

ALL, STABLE, TRANSITION = "all", "stable", "transition"
CONDITIONS = (ALL, STABLE, TRANSITION)

SKILL = "skill"
ALIGNMENT = "displacement_alignment"
MOVEMENT = "movement_ratio"
METRICS = (SKILL, ALIGNMENT, MOVEMENT)

DEFAULT_BOOTSTRAP = 1_000
DEFAULT_TRAJECTORIES = 6
CONFIDENCE = 0.95
# A row has a displacement direction only if its true displacement is at least
# this fraction of the horizon's median true displacement (over every anchor);
# rows below it are excluded from the alignment mean, and counted.
MIN_TRUE_MOTION_FRACTION = 0.01
HORIZON_TOLERANCE_S = 1e-6

# Action ids announcing a speech event (turn_wm.data.dataset.ACTION_TO_ID).
EVENT_ACTIONS = ("START", "STOP")


@dataclass(frozen=True)
class RolloutConditions:
    """Per-row current state and, per horizon, future state and condition."""

    current: list[str | None]
    future: dict[int, list[str | None]]  # horizon steps -> per row
    condition: dict[int, list[str | None]]  # STABLE, TRANSITION or None


def rollout_conditions(
    joined_variables: Sequence[Any],
    horizons_s: Mapping[int, float],
) -> RolloutConditions:
    """Stable / transition per row and horizon, from the joined labels."""

    current = next(
        (v for v in joined_variables if v.name == f"{CURRENT_STATE}:dominant"),
        None,
    )

    if current is None:
        raise ValueError(f"No {CURRENT_STATE} labels for the snapshot")

    future: dict[int, list[str | None]] = {}
    condition: dict[int, list[str | None]] = {}

    for h, seconds in horizons_s.items():
        match = [
            v
            for v in joined_variables
            if v.label == FUTURE_STATE
            and v.horizon_s is not None
            and abs(v.horizon_s - seconds) <= HORIZON_TOLERANCE_S
        ]

        if not match:
            available = sorted(
                v.horizon_s for v in joined_variables if v.label == FUTURE_STATE
            )
            raise ValueError(
                f"{FUTURE_STATE} has no {seconds:g} s horizon (rollout horizon "
                f"{h}); the labels provide {available}"
            )

        future[h] = list(match[0].values)
        condition[h] = [
            None
            if now is None or later is None
            else STABLE
            if now == later
            else TRANSITION
            for now, later in zip(current.values, future[h], strict=True)
        ]

    return RolloutConditions(
        current=list(current.values), future=future, condition=condition
    )


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowTerms:
    """Per-row terms of the three measures at one horizon, float64 (N,)."""

    model_error: torch.Tensor  # ||pred - true||^2
    persistence_error: torch.Tensor  # ||anchor - true||^2
    cosine: torch.Tensor  # cos(pred - anchor, true - anchor); nan if excluded
    true_motion: torch.Tensor  # ||true - anchor||
    pred_motion: torch.Tensor  # ||pred - anchor||
    min_true_motion: float  # below it, a row has no direction

    @property
    def direction_defined(self) -> torch.Tensor:
        return ~self.cosine.isnan()


def row_terms(anchor: torch.Tensor, true: torch.Tensor, pred: torch.Tensor) -> RowTerms:
    anchor, true, pred = (x.double() for x in (anchor, true, pred))
    true_delta = true - anchor
    pred_delta = pred - anchor
    true_motion = true_delta.norm(dim=-1)
    pred_motion = pred_delta.norm(dim=-1)
    min_true_motion = MIN_TRUE_MOTION_FRACTION * float(true_motion.median())
    # A prediction that does not move has no direction either.
    defined = (true_motion >= min_true_motion) & (pred_motion > 0)
    cosine = (true_delta * pred_delta).sum(-1) / (true_motion * pred_motion)

    return RowTerms(
        model_error=(pred - true).pow(2).sum(-1),
        persistence_error=true_delta.pow(2).sum(-1),
        cosine=torch.where(defined, cosine, torch.full_like(cosine, math.nan)),
        true_motion=true_motion,
        pred_motion=pred_motion,
        min_true_motion=min_true_motion,
    )


def row_sums(terms: RowTerms) -> torch.Tensor:
    """The additive terms of the three measures, (N, 6)."""

    defined = terms.direction_defined

    return torch.stack(
        [
            terms.model_error,
            terms.persistence_error,
            torch.where(defined, terms.cosine, 0.0),
            defined.double(),
            terms.pred_motion,
            terms.true_motion,
        ],
        dim=-1,
    )


def measures_of_sums(sums: torch.Tensor) -> dict[str, torch.Tensor]:
    """The three measures from summed terms (..., 6); nan where undefined."""

    def ratio(numerator, denominator):
        return torch.where(
            denominator > 0, numerator / denominator.clamp_min(1e-300), math.nan
        )

    return {
        SKILL: 1.0 - ratio(sums[..., 0], sums[..., 1]),
        ALIGNMENT: ratio(sums[..., 2], sums[..., 3]),
        MOVEMENT: ratio(sums[..., 4], sums[..., 5]),
    }


def cluster_bootstrap_weights(
    clusters: Sequence[str],
    strata: Sequence[str],
    *,
    resamples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """(resamples, G) draw counts of G clusters, resampled within each stratum.

    Each stratum (corpus) keeps its number of clusters (recordings); its
    clusters are drawn with replacement. `clusters[g]` lies in `strata[g]`.
    """

    weights = torch.zeros(resamples, len(clusters), dtype=torch.float64)

    for stratum in sorted(set(strata)):
        members = torch.tensor([g for g, s in enumerate(strata) if s == stratum])
        draws = members[
            torch.randint(len(members), (resamples, len(members)), generator=generator)
        ]
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))

    return weights


def condition_metrics(
    terms: RowTerms,
    members: torch.Tensor,
    *,
    recordings: Sequence[str],
    corpora: Sequence[str],
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Values, cluster-bootstrap intervals and counts of one (horizon, condition).

    `recordings` and `corpora` give each row's cluster and stratum.
    """

    index = [int(i) for i in members.nonzero().flatten().tolist()]
    n = len(index)
    defined = int(terms.direction_defined[index].sum())
    clusters = sorted({recordings[i] for i in index})
    result: dict[str, Any] = {
        "n": n,
        "n_recordings": len(clusters),
        "n_direction_defined": defined,
        "direction_defined_fraction": defined / n if n else None,
        "mean_true_motion": float(terms.true_motion[index].mean()) if n else None,
        "mean_pred_motion": float(terms.pred_motion[index].mean()) if n else None,
    }

    if n == 0:
        return (
            result
            | {name: None for name in METRICS}
            | {f"{name}_ci": None for name in METRICS}
        )

    # Per-recording sums: the bootstrap resamples recordings, not anchors.
    position = {cluster: g for g, cluster in enumerate(clusters)}
    sums = torch.zeros(len(clusters), 6, dtype=torch.float64).index_add_(
        0,
        torch.tensor([position[recordings[i]] for i in index]),
        row_sums(terms)[index],
    )
    stratum = {recordings[i]: corpora[i] for i in index}
    point = measures_of_sums(sums.sum(dim=0))
    # The validation skill, from the same pooled errors.
    point[SKILL] = torch.tensor(
        skill_score(float(sums[:, 0].sum()), float(sums[:, 1].sum()))
    )
    resampled = measures_of_sums(
        cluster_bootstrap_weights(
            clusters,
            [stratum[c] for c in clusters],
            resamples=bootstrap,
            generator=generator,
        )
        @ sums
    )
    tail = (1 - CONFIDENCE) / 2
    quantiles = torch.tensor([tail, 1 - tail], dtype=torch.float64)

    for name in METRICS:
        value = float(point[name])

        if math.isnan(value):
            result[name] = result[f"{name}_ci"] = None
            continue

        result[name] = value
        result[f"{name}_ci"] = (
            torch.nanquantile(resampled[name], quantiles).tolist()
            if len(clusters) > 1
            else None
        )

    return result


def future_event_rows(
    action_ids: torch.Tensor,
    rollout: Mapping[str, Any],
    horizon: int,
) -> torch.Tensor:
    """Rows whose future action tokens read for `horizon` hold START/STOP.

    The future tokens read are those of the trajectory steps after the
    anchor within the rollout's action window for that horizon.
    """

    from turn_wm.data.dataset import ACTION_TO_ID

    start, stop = rollout["action_steps_by_horizon"][str(horizon)]
    start = max(start, int(rollout["anchor_step"]) + 1)
    read = action_ids[:, start:stop]
    events = torch.tensor([ACTION_TO_ID[name] for name in EVENT_ACTIONS])

    return torch.isin(read, events).any(dim=1)


@dataclass(frozen=True)
class RolloutDynamics:
    horizons_steps: list[int]
    horizons_s: dict[int, float]
    conditions: RolloutConditions
    metrics: dict[int, dict[str, dict[str, Any]]]  # h -> condition -> values
    settings: dict[str, Any]


def analyze_rollout_dynamics(
    representations: Mapping[str, torch.Tensor],
    conditions: RolloutConditions,
    rollout: Mapping[str, Any],
    *,
    recordings: Sequence[str],
    corpora: Sequence[str],
    seed: int,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    tick: Callable[[], Any] | None = None,
) -> RolloutDynamics:
    """The three measures per horizon and condition, with the event diagnostic.

    `rollout` is the snapshot's rollout provenance; `recordings` (unique per
    corpus) and `corpora` give each row's bootstrap cluster and stratum.
    `tick` is called after every condition's bootstrap (progress display only).
    """

    horizons_steps = [int(h) for h in rollout["horizons_steps"]]
    horizons_s = dict(
        zip(horizons_steps, map(float, rollout["horizons_s"]), strict=True)
    )
    anchor = representations[ANCHOR_LATENT]
    true = representations[TRUE_FUTURE_LATENT]
    pred = representations[PRED_FUTURE_LATENT]
    actions = representations[ROLLOUT_ACTION_IDS]
    metrics: dict[int, dict[str, dict[str, Any]]] = {}
    thresholds = {}

    for k, h in enumerate(horizons_steps):
        terms = row_terms(anchor, true[:, k], pred[:, k])
        thresholds[str(h)] = terms.min_true_motion
        events = future_event_rows(actions, rollout, h)
        labels = conditions.condition[h]
        # One generator per horizon: each horizon's intervals are reproducible
        # on their own.
        generator = torch.Generator().manual_seed(seed * 1_000 + h)
        metrics[h] = {}

        for name in CONDITIONS:
            members = torch.tensor(
                [name == ALL or label == name for label in labels], dtype=torch.bool
            )
            values = condition_metrics(
                terms,
                members,
                recordings=recordings,
                corpora=corpora,
                bootstrap=bootstrap,
                generator=generator,
            )
            with_event = int((events & members).sum())
            values["n_future_event"] = with_event
            values["future_event_fraction"] = (
                with_event / values["n"] if values["n"] else None
            )
            metrics[h][name] = values

            if tick is not None:
                tick()

    return RolloutDynamics(
        horizons_steps=horizons_steps,
        horizons_s=horizons_s,
        conditions=conditions,
        metrics=metrics,
        settings={
            "seed": seed,
            "bootstrap_resamples": bootstrap,
            "confidence": CONFIDENCE,
            "interval": (
                "percentile cluster bootstrap: recordings resampled with "
                "replacement within each corpus, per horizon and condition"
            ),
            "cluster": "(dataset, recording_id)",
            "strata": "dataset",
            "min_true_motion_fraction": MIN_TRUE_MOTION_FRACTION,
            "min_true_motion": thresholds,
            "direction_rule": (
                "cosine only where ||true - anchor|| >= min_true_motion_fraction "
                "x the horizon's median ||true - anchor|| (all anchors) and "
                "||pred - anchor|| > 0; other rows are excluded and counted"
            ),
            "future_event_actions": list(EVENT_ACTIONS),
            "future_event_rule": (
                "at least one START/STOP among the future action tokens the "
                "rollout reads for the horizon (steps t+1 .. t+h-1)"
            ),
            "dtype": "float64",
        },
    )


# ---------------------------------------------------------------------------
# Trajectories (optional figure)
# ---------------------------------------------------------------------------


def trajectory_selection(
    sample_ids: Sequence[str],
    conditions: RolloutConditions,
    horizon: int,
    *,
    count: int,
    seed: int,
) -> list[int]:
    """Transition rows at `horizon`, distinct state changes first, in key order."""

    keys = sample_keys(sample_ids, seed=seed)
    rows = [
        i
        for i in keys.argsort().tolist()
        if conditions.condition[horizon][i] == TRANSITION
    ]
    chosen: list[int] = []
    seen: set[tuple[Any, Any]] = set()

    for i in rows:
        change = (conditions.current[i], conditions.future[horizon][i])

        if change not in seen:
            seen.add(change)
            chosen.append(i)

    chosen += [i for i in rows if i not in chosen]

    return chosen[:count]


@dataclass(frozen=True)
class TrajectoryProjection:
    mean: torch.Tensor  # (D,)
    components: torch.Tensor  # (D, 2)
    explained_variance_ratio: list[float]  # PC1, PC2
    fit_rows: int

    def transform(self, rows: torch.Tensor) -> torch.Tensor:
        return (rows.double() - self.mean) @ self.components


def true_latent_projection(
    representations: Mapping[str, torch.Tensor],
) -> TrajectoryProjection:
    """2D PCA fitted on the true latents only: anchors and true futures."""

    true = torch.cat(
        [
            representations[ANCHOR_LATENT],
            representations[TRUE_FUTURE_LATENT].flatten(0, 1),
        ]
    ).double()
    projection = pca_2d(true)

    return TrajectoryProjection(
        mean=true.mean(dim=0),
        components=projection.components,
        explained_variance_ratio=projection.explained_variance_ratio[:2].tolist(),
        fit_rows=len(true),
    )


def trajectory_table(
    representations: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Sequence[Any]],
    dynamics: RolloutDynamics,
    projection: TrajectoryProjection,
    rows: Sequence[int],
) -> pa.Table:
    """One row per (sample, path, point): true and predicted 2D paths."""

    records = []

    for i in rows:
        for path, name in (("true", TRUE_FUTURE_LATENT), ("pred", PRED_FUTURE_LATENT)):
            points = torch.cat(
                [representations[ANCHOR_LATENT][i][None], representations[name][i]]
            )
            coordinates = projection.transform(points)

            for step, (x, y) in zip(
                [0, *dynamics.horizons_steps], coordinates.tolist(), strict=True
            ):
                records.append(
                    {
                        "sample_id": str(metadata["sample_id"][i]),
                        "dataset": str(metadata["dataset"][i]),
                        "path": path,
                        "horizon_steps": step,
                        "horizon_s": dynamics.horizons_s.get(step, 0.0),
                        "state": (
                            dynamics.conditions.current[i]
                            if step == 0
                            else dynamics.conditions.future[step][i]
                        ),
                        "pc1": x,
                        "pc2": y,
                    }
                )

    return pa.Table.from_pylist(records)
