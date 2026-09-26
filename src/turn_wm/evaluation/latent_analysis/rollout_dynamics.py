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
rows whose future action tokens read by the rollout hold an ONSET or OFFSET.
The optional trajectory figure projects a few transitions on the first two
principal components of the true latents only (anchor and true futures),
then applies that projection to the predictions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from turn_wm.evaluation.latent_analysis.analyze import Snapshot, read_snapshot
from turn_wm.evaluation.latent_analysis.label_source import (
    CONVERSATIONAL_STATE,
    FUTURE,
    CorpusLabelSource,
    audit_corpus,
    hub_label_sources,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.pca import pca_2d, sample_keys
from turn_wm.evaluation.latent_analysis.rendering import (
    GRID,
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
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    ROLLOUT_TENSORS,
    TRUE_FUTURE_LATENT,
)
from turn_wm.training.metrics import skill_score

if TYPE_CHECKING:
    from matplotlib.figure import Figure

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
EVENT_ACTIONS = ("ONSET", "OFFSET")


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


def _row_sums(terms: RowTerms) -> torch.Tensor:
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


def _measures(sums: torch.Tensor) -> dict[str, torch.Tensor]:
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

    index = members.nonzero().flatten().tolist()
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
        _row_sums(terms)[index],
    )
    stratum = {recordings[i]: corpora[i] for i in index}
    point = _measures(sums.sum(dim=0))
    # The validation skill, from the same pooled errors.
    point[SKILL] = torch.tensor(
        skill_score(float(sums[:, 0].sum()), float(sums[:, 1].sum()))
    )
    resampled = _measures(
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
    """Rows whose future action tokens read for `horizon` hold ONSET/OFFSET.

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
) -> RolloutDynamics:
    """The three measures per horizon and condition, with the event diagnostic.

    `rollout` is the snapshot's rollout provenance; `recordings` (unique per
    corpus) and `corpora` give each row's bootstrap cluster and stratum.
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
                "at least one ONSET/OFFSET among the future action tokens the "
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


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


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

    horizons_steps = [int(h) for h in rollout["horizons_steps"]]
    horizons_s = dict(
        zip(horizons_steps, map(float, rollout["horizons_s"]), strict=True)
    )

    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(provenance, labels_revision=labels_revision)
    )
    audits = {corpus: audit_corpus(source) for corpus, source in sources.items()}
    joined = join_labels(snapshot.metadata, audits, sources, selection=SELECTION)
    conditions = rollout_conditions(joined.variables, horizons_s)
    corpora = [str(d) for d in snapshot.metadata["dataset"]]
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
    )

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
        "source": _source(snapshot),
        "rollout": rollout,
        "labels": {
            "current_state": f"{CURRENT_STATE}:dominant",
            "future_state": f"{FUTURE_STATE}@h",
            "corpora": {c: a.provenance for c, a in audits.items()},
            "alignment": joined.alignment,
            "excluded": joined.excluded,
            "conditions_sha256": _sha256(conditions_path),
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Fixed slots: the condition, never its rank, picks the color.
_CONDITION_COLORS = dict(zip(CONDITIONS, SERIES, strict=False))
_TITLES = {
    SKILL: "Skill vs persistence",
    ALIGNMENT: "Displacement alignment",
    MOVEMENT: "Movement ratio",
}
_REFERENCES = {SKILL: 0.0, ALIGNMENT: 0.0, MOVEMENT: 1.0}


def _new_figure(width: float, height: float) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


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
    figure = _new_figure(6.0, 4.0)
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
    figure = _new_figure(10.0, 4.0)
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
    figure = _new_figure(3.6 * columns, 3.3 * lines + 0.5)
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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


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
        "**Is the rollout conditioned on ground-truth future action/event tokens? "
        + (
            "Yes.**"
            if any(rollout["conditioned_on_ground_truth_future_actions"].values())
            else "No.**"
        )
        + " To predict z_(t+h) the predictor receives the action grid's ground-truth "
        "tokens (NO_EVENT, ONSET, OFFSET, MASKED) of the future steps t+1 … t+h−1, "
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
            "(steps t+1 … t+h−1) contain at least one ONSET or OFFSET. Where it is "
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


def _source(snapshot: Snapshot) -> dict[str, Any]:
    return {
        "snapshot": str(snapshot.path),
        "representations_sha256": _sha256(
            snapshot.path / "representations.safetensors"
        ),
        "samples": snapshot.manifest.get("samples"),
        "snapshot_provenance": snapshot.manifest.get("provenance"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
