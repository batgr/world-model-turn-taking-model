"""
Linear probes of a run's representations: Mimi `features` and the V1
projector's `latent`, both from `extract-latents` snapshots of one checkpoint.

Questions:

1. Which conversational variables are linearly decodable from the features?
2. Which from the latent?
3. Does the projector improve or degrade linear accessibility
   (delta = latent score - features score)?
4. Is the same readout reusable across corpora (cross-domain transfer)?

Protocol:

- probes are fitted on a TRAIN-split snapshot and evaluated on a
  VALIDATION-split snapshot of the same checkpoint; no (dataset,
  recording_id) may occur in both, and no test-split snapshot is read;
- both snapshots are seeded fixed permutations of their split, with no
  class or corpus balancing: probes see the natural distribution;
- every preprocessing step (standardization, class weights) and the
  regularization are fitted on probe-train rows only: logistic C (smaller =
  stronger L2) and ridge alpha (larger = stronger), scikit-learn semantics,
  are chosen from predeclared log-spaced grids by recording-grouped CV
  inside probe-train (mean out-of-fold primary score, ties to the stronger
  regularization), separately for each representation, task and training
  set, then frozen before validation. A categorical CV fold is valid only
  if every canonical class is in both its parts; fewer than 3 valid folds
  make the configuration unsupported;
- categorical labels: multinomial logistic regression, class-balanced
  weights; score = balanced accuracy, reference 1 / K over the task's
  canonical K classes. A setting where any of the K classes lacks support
  (train or evaluation) is marked unsupported, never reduced to K - 1;
- continuous labels: ridge regression; score = R^2 on the evaluated rows,
  reference 0 (predicting their mean);
- settings: pooled, within each corpus and, for categorical labels,
  across corpora (train on one, evaluate on the other);
- 95% intervals: seeded percentile bootstrap over validation recordings,
  resampled within each corpus; the delta is bootstrapped paired (the same
  resampled recordings for both representations). Probes are fitted once.
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
import torch.nn.functional as F

from turn_wm.evaluation.latent_analysis.analyze import Snapshot, read_snapshot
from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CONVERSATIONAL_STATE,
    FUTURE,
    TEMPORAL_STATE,
    CorpusLabelSource,
    audit_corpus,
    hub_label_sources,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    MUTED,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    close,
    style,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    cluster_bootstrap_weights,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1
ANALYSIS = "probes"

FEATURES = "features"
LATENT = "latent"
REPRESENTATIONS = (FEATURES, LATENT)

CURRENT = "current_state"
TEMPORAL = "temporal_state"
FUTURE_STATE = "future_state"
GROUPS = (CURRENT, TEMPORAL, FUTURE_STATE)


@dataclass(frozen=True)
class ProbeTask:
    variable: str  # the joined variable name (label_source)
    label: str  # the registry label
    section: str  # label_source section, for the join
    group: str


TASKS = (
    ProbeTask(
        "instantaneous.ego_speaking",
        "instantaneous.ego_speaking",
        CONVERSATIONAL_STATE,
        CURRENT,
    ),
    ProbeTask(
        "instantaneous.others_active",
        "instantaneous.others_active",
        CONVERSATIONAL_STATE,
        CURRENT,
    ),
    ProbeTask(
        "instantaneous.joint_speech_state_occupancy:dominant",
        "instantaneous.joint_speech_state_occupancy",
        CONVERSATIONAL_STATE,
        CURRENT,
    ),
    ProbeTask(
        "timing.time_to_next_speaker_onset",
        "timing.time_to_next_speaker_onset",
        TEMPORAL_STATE,
        TEMPORAL,
    ),
    ProbeTask(
        "timing.silence_duration", "timing.silence_duration", TEMPORAL_STATE, TEMPORAL
    ),
    ProbeTask(
        "future.future_joint_speech_state@1s",
        "future.future_joint_speech_state",
        FUTURE,
        FUTURE_STATE,
    ),
)
SELECTION = tuple(dict.fromkeys((task.section, task.label) for task in TASKS))

# Predeclared regularization grids, with scikit-learn's semantics, on
# standardized inputs (intercepts unpenalized):
#   logistic, C:  minimize 1/2 ||W||^2 + C * sum_i s_i CE_i  (s_i: balanced
#                 class weights); SMALLER C = STRONGER L2 regularization
#   ridge, alpha: minimize ||y - b - X w||^2 + alpha ||w||^2;
#                 LARGER alpha = STRONGER regularization
# One value is chosen by recording-grouped CV inside probe-train, ties going
# to the stronger regularization, then frozen; validation never sees it.
LOGISTIC_C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RIDGE_ALPHA_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4)
CV_FOLDS = 5
# Model selection needs at least this many valid grouped folds.
MIN_VALID_CV_FOLDS = 3
# Mean CV scores this close count as a tie.
CV_TIE_TOLERANCE = 1e-9
CV_GROUPING = "(dataset, recording_id)"
INSUFFICIENT_CV_CLASS_SUPPORT = "insufficient_grouped_cv_class_support"
INSUFFICIENT_CV_SUPPORT = "insufficient_grouped_cv_support"
LBFGS_MAX_ITER = 500
# A categorical setting is evaluable only if every canonical class has at
# least this many rows in probe-train and in the evaluated rows.
MIN_CLASS_SUPPORT = 20
# A setting is probed only with at least this many training rows.
MIN_TRAIN_ROWS = 50
DEFAULT_BOOTSTRAP = 1_000
CONFIDENCE = 0.95
STD_FLOOR = 1e-8

POOLED = "pooled"


# ---------------------------------------------------------------------------
# Snapshots and leakage
# ---------------------------------------------------------------------------


def check_snapshots(train: Snapshot, validation: Snapshot) -> None:
    """Refuse a test split, another checkpoint or run, or shared recordings."""

    train_provenance = train.manifest.get("provenance") or {}
    validation_provenance = validation.manifest.get("provenance") or {}
    splits = {
        "probe-train": (train_provenance.get("data") or {}).get("split"),
        "probe-validation": (validation_provenance.get("data") or {}).get("split"),
    }

    if "test" in splits.values():
        raise ValueError("Linear probes never read the test split")

    if splits != {"probe-train": "train", "probe-validation": "validation"}:
        raise ValueError(
            "Probes are fitted on a train-split snapshot and evaluated on a "
            f"validation-split snapshot; got {splits}"
        )

    for which, provenance in (
        ("probe-train", train_provenance),
        ("probe-validation", validation_provenance),
    ):
        order = (provenance.get("sampling") or {}).get("order")

        if order != "fixed_permutation":
            raise ValueError(
                f"The {which} snapshot's rows are not a seeded fixed permutation "
                f"of its split (sampling.order={order!r}); probes need the "
                "split's natural distribution, without class or corpus balancing"
            )

    for section, key in (
        ("checkpoint", "sha256"),
        ("run", "config_hash"),
        ("data", "dataset_revision"),
    ):
        a = (train_provenance.get(section) or {}).get(key)
        b = (validation_provenance.get(section) or {}).get(key)

        if a != b:
            raise ValueError(
                f"The snapshots differ in {section}.{key} ({a} vs {b}); probes "
                "compare one checkpoint's representations"
            )

    for name in REPRESENTATIONS:
        for which, snapshot in (
            ("probe-train", train),
            ("probe-validation", validation),
        ):
            if name not in snapshot.representations:
                raise ValueError(f"The {which} snapshot has no {name!r} representation")

        if (
            train.representations[name].shape[1:]
            != validation.representations[name].shape[1:]
        ):
            raise ValueError(f"{name!r} has another shape in the two snapshots")

    check_no_recording_leakage(train.metadata, validation.metadata)


def check_no_recording_leakage(
    train: Mapping[str, Sequence[Any]], validation: Mapping[str, Sequence[Any]]
) -> None:
    """Refuse any (dataset, recording_id) present in both snapshots."""

    def keys(metadata):
        return set(
            zip(
                map(str, metadata["dataset"]),
                map(str, metadata["recording_id"]),
                strict=True,
            )
        )

    shared = sorted(keys(train) & keys(validation))

    if shared:
        raise ValueError(
            f"{len(shared)} recording(s) occur in both probe-train and "
            f"probe-validation, e.g. {shared[:3]}; refusing to probe"
        )


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Standardizer:
    """Per-dimension mean and std of the probe-train rows."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> Standardizer:
        x = x.double()

        return cls(mean=x.mean(dim=0), std=x.std(dim=0, correction=0))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.double() - self.mean) / self.std.clamp_min(STD_FLOOR)


def balanced_class_weights(y: torch.Tensor, classes: int) -> torch.Tensor:
    """Per-row weights n / (K * n_class): every class weighs the same."""

    counts = torch.bincount(y, minlength=classes).double()

    return (len(y) / (classes * counts.clamp_min(1)))[y]


def fit_logistic(
    x: torch.Tensor, y: torch.Tensor, classes: int, *, c: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced multinomial logistic regression; (W (D, K), b (K,)).

    scikit-learn's objective, 1/2 ||W||^2 + C * sum_i s_i CE_i with
    s_i = n / (K n_class), solved divided by C * n for conditioning: the
    minimizer is the same. Smaller C = stronger regularization.
    """

    weights = balanced_class_weights(y, classes)
    n = len(y)
    w = torch.zeros(x.shape[1], classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [w, b],
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = (F.cross_entropy(x @ w + b, y, reduction="none") * weights).sum() / n
        loss = loss + w.pow(2).sum() / (2 * c * n)
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)  # pyright: ignore[reportArgumentType]

    return w.detach(), b.detach()


def fit_ridge(
    x: torch.Tensor, y: torch.Tensor, *, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """scikit-learn's ridge, ||y - b - X w||^2 + alpha ||w||^2; (w, b).

    Larger alpha = stronger regularization.
    """

    intercept = y.mean()
    centred = x - x.mean(dim=0)
    gram = centred.T @ centred + alpha * torch.eye(x.shape[1], dtype=x.dtype)
    w = torch.linalg.solve(gram, centred.T @ (y - intercept))

    return w, intercept - x.mean(dim=0) @ w


@dataclass(frozen=True)
class LinearProbe:
    """A probe frozen after selection: train-fitted scaler and weights."""

    kind: str
    standardizer: Standardizer
    weight: torch.Tensor
    bias: torch.Tensor

    @classmethod
    def fit(
        cls, kind: str, x: torch.Tensor, y: torch.Tensor, *, classes, value: float
    ) -> LinearProbe:
        standardizer = Standardizer.fit(x)
        x = standardizer.transform(x)

        if kind == CATEGORICAL:
            weight, bias = fit_logistic(x, y, classes, c=value)
        else:
            weight, bias = fit_ridge(x, y.double(), alpha=value)

        return cls(kind, standardizer, weight, bias)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        output = self.standardizer.transform(x) @ self.weight + self.bias

        return output.argmax(dim=1) if self.kind == CATEGORICAL else output


def recording_folds(
    recordings: Sequence[str], *, folds: int, seed: int
) -> tuple[torch.Tensor, int]:
    """Fold of each row: whole recordings, dealt in seeded key order.

    `recordings` are "<dataset>/<recording_id>" keys, so the corpus is part
    of the grouping.
    """

    unique = sorted(set(recordings), key=lambda r: (_seed(seed, r), r))
    count = min(folds, len(unique))
    fold_of = {r: i % count for i, r in enumerate(unique)}

    return torch.tensor([fold_of[r] for r in recordings]), count


def regularization_parameter(kind: str) -> str:
    return "C" if kind == CATEGORICAL else "alpha"


def candidates(kind: str) -> tuple[float, ...]:
    return LOGISTIC_C_GRID if kind == CATEGORICAL else RIDGE_ALPHA_GRID


def strongest_first(kind: str) -> list[float]:
    """The grid from strongest to weakest regularization."""

    # Smaller C is stronger; larger alpha is stronger.
    return sorted(candidates(kind), reverse=kind != CATEGORICAL)


def choose_regularization(
    mean_scores: Mapping[float, float], *, kind: str
) -> float | None:
    """Best mean CV score; ties (within tolerance) to the stronger value."""

    finite = {v: s for v, s in mean_scores.items() if not math.isnan(s)}

    if not finite:
        return None

    best = max(finite.values())

    return next(
        v
        for v in strongest_first(kind)
        if v in finite and finite[v] >= best - CV_TIE_TOLERANCE
    )


@dataclass(frozen=True)
class CrossValidation:
    """Recording-grouped model selection inside probe-train."""

    kind: str
    requested_folds: int
    valid_folds: list[int]
    invalid_folds: dict[int, str]
    fold_class_counts: dict[int, dict[str, list[int]]] | None
    fold_scores: dict[float, dict[int, float]]  # candidate -> fold -> score
    mean_scores: dict[float, float]
    selected: float | None
    unsupported: str | None

    def provenance(self) -> dict[str, Any]:
        name = "c" if self.kind == CATEGORICAL else "alpha"
        plural = "c" if self.kind == CATEGORICAL else "alphas"

        return {
            "grouping": CV_GROUPING,
            "regularization_parameter": regularization_parameter(self.kind),
            "direction": (
                "smaller C = stronger L2 regularization"
                if self.kind == CATEGORICAL
                else "larger alpha = stronger regularization"
            ),
            "criterion": (
                "mean out-of-fold balanced accuracy over the canonical classes"
                if self.kind == CATEGORICAL
                else "mean out-of-fold R^2"
            ),
            "tie_break": "stronger regularization",
            "requested_cv_folds": self.requested_folds,
            "valid_cv_folds": self.valid_folds,
            "invalid_cv_folds": {str(f): r for f, r in self.invalid_folds.items()},
            "fold_class_counts": (
                None
                if self.fold_class_counts is None
                else {str(f): c for f, c in self.fold_class_counts.items()}
            ),
            f"candidate_{plural}": list(candidates(self.kind)),
            f"selected_{name}": self.selected,
            f"mean_score_by_{name}": {f"{v:g}": s for v, s in self.mean_scores.items()},
            f"fold_scores_by_{name}": {
                f"{v:g}": {str(f): s for f, s in by_fold.items()}
                for v, by_fold in self.fold_scores.items()
            },
            "unsupported": self.unsupported,
        }


def _fold_checks(kind, y, fold, count, classes):
    """Valid folds, invalid ones with a reason, per-fold class counts."""

    valid, invalid = [], {}
    counts = {} if kind == CATEGORICAL else None

    for f in range(CV_FOLDS):
        if f >= count:
            invalid[f] = "no recording: fewer training recordings than folds"
            continue

        held = fold == f

        if kind == CATEGORICAL:
            train_counts = torch.bincount(y[~held], minlength=classes).tolist()
            held_counts = torch.bincount(y[held], minlength=classes).tolist()
            assert counts is not None
            counts[f] = {"train": train_counts, "held_out": held_counts}

            # Every canonical class on both sides, never a K - 1 class fold.
            if min(train_counts) == 0 or min(held_counts) == 0:
                invalid[f] = (
                    "a canonical class is absent from the fold's train or held-out part"
                )
                continue
        elif int(held.sum()) < 2 or float(y[held].double().var()) == 0:
            invalid[f] = "held-out part has no target variance"
            continue

        valid.append(f)

    return valid, invalid, counts


def select_regularization(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    recordings: Sequence[str],
    *,
    classes: int | None,
    seed: int,
) -> CrossValidation:
    """Choose C or alpha by recording-grouped CV inside probe-train.

    Each fold standardizes on its own training part. Every candidate is
    scored on exactly the same valid folds; the criterion is the mean
    out-of-fold primary score (balanced accuracy over the K canonical
    classes, or R^2). Fewer than MIN_VALID_CV_FOLDS valid folds: no choice.
    """

    fold, count = recording_folds(recordings, folds=CV_FOLDS, seed=seed)
    valid, invalid, class_counts = _fold_checks(kind, y, fold, count, classes)

    def result(fold_scores, mean_scores, selected, unsupported):
        return CrossValidation(
            kind=kind,
            requested_folds=CV_FOLDS,
            valid_folds=valid,
            invalid_folds=invalid,
            fold_class_counts=class_counts,
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            selected=selected,
            unsupported=unsupported,
        )

    if len(valid) < MIN_VALID_CV_FOLDS:
        reason = (
            INSUFFICIENT_CV_CLASS_SUPPORT
            if kind == CATEGORICAL
            else INSUFFICIENT_CV_SUPPORT
        )
        return result(
            {},
            {},
            None,
            f"{reason}: {len(valid)} of {CV_FOLDS} grouped folds valid, "
            f"{MIN_VALID_CV_FOLDS} needed",
        )

    fold_scores: dict[float, dict[int, float]] = {}

    for value in candidates(kind):
        fold_scores[value] = {}

        for f in valid:
            held = fold == f
            probe = LinearProbe.fit(
                kind, x[~held], y[~held], classes=classes, value=value
            )
            pred = probe.predict(x[held])
            fold_scores[value][f] = (
                balanced_accuracy(y[held], pred, classes)
                if classes is not None
                else r2(y[held], pred)
            )

    mean_scores = {
        v: sum(by_fold.values()) / len(by_fold) for v, by_fold in fold_scores.items()
    }
    selected = choose_regularization(mean_scores, kind=kind)

    return result(
        fold_scores,
        mean_scores,
        selected,
        None if selected is not None else "no finite CV score",
    )


@dataclass(frozen=True)
class FittedProbe:
    """Selection and, when selection succeeded, the frozen probe."""

    cv: CrossValidation
    probe: LinearProbe | None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert self.probe is not None
        return self.probe.predict(x)


def fit_probe(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    recordings: Sequence[str],
    *,
    classes: int | None = None,
    seed: int = 0,
) -> FittedProbe:
    """Select C / alpha inside probe-train, then fit on all of it and freeze."""

    cv = select_regularization(kind, x, y, recordings, classes=classes, seed=seed)

    if cv.selected is None:
        return FittedProbe(cv, None)

    return FittedProbe(
        cv, LinearProbe.fit(kind, x, y, classes=classes, value=cv.selected)
    )


def probe_predictions(
    kind: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    recordings: Sequence[str],
    classes: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Fit on train rows only (C / alpha and standardization); predict eval rows."""

    return fit_probe(
        kind, train_x, train_y, recordings, classes=classes, seed=seed
    ).predict(eval_x)


def balanced_accuracy(y: torch.Tensor, pred: torch.Tensor, classes: int) -> float:
    """Mean recall over the K canonical classes; nan if one is absent from `y`.

    The denominator is always K: a missing class never makes it K - 1.
    """

    return float(_balanced_accuracy(_class_sums(y, pred, classes).sum(0)))


def r2(y: torch.Tensor, pred: torch.Tensor) -> float:
    """1 - SS_res / SS_tot, SS_tot around the mean of `y`: 0 predicts that mean."""

    return float(_r2(_regression_sums(y, pred).sum(0)))


def _class_sums(y, pred, classes) -> torch.Tensor:
    """(..., K, 2): rows of each class, and those predicted right."""

    total = F.one_hot(y, classes).double()

    return torch.stack([total, total * (pred == y)[..., None].double()], dim=-1)


def _balanced_accuracy(sums: torch.Tensor) -> torch.Tensor:
    total, correct = sums[..., 0], sums[..., 1]
    recall = correct / total.clamp_min(1e-300)
    score = recall.mean(-1)

    # Undefined unless every canonical class is present.
    return torch.where((total > 0).all(-1), score, torch.nan)


def _regression_sums(y, pred) -> torch.Tensor:
    y, pred = y.double(), pred.double()

    return torch.stack([torch.ones_like(y), y, y.pow(2), (y - pred).pow(2)], dim=-1)


def _r2(sums: torch.Tensor) -> torch.Tensor:
    n, total, square, residual = sums.unbind(-1)
    variance = square - total.pow(2) / n

    return torch.where(
        variance > 0, 1 - residual / variance.clamp_min(1e-300), torch.nan
    )


# ---------------------------------------------------------------------------
# Settings and one probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    name: str
    kind: str  # "pooled", "within" or "cross"
    train: tuple[str, ...]  # corpora of the probe-train rows
    evaluate: tuple[str, ...]  # corpora of the probe-validation rows


def probe_settings(corpora: Sequence[str], *, cross_domain: bool) -> list[Setting]:
    corpora = sorted(corpora)
    settings = [Setting(POOLED, POOLED, tuple(corpora), tuple(corpora))]

    if len(corpora) > 1:
        settings += [Setting(f"within:{c}", "within", (c,), (c,)) for c in corpora]

    if cross_domain and len(corpora) > 1:
        settings += [
            Setting(f"{a}->{b}", "cross", (a,), (b,))
            for a in corpora
            for b in corpora
            if a != b
        ]

    return settings


@dataclass(frozen=True)
class ProbeData:
    """One snapshot's rows for one task: representations, labels, corpora."""

    representations: Mapping[str, torch.Tensor]
    values: list[Any]  # label per row, None when missing
    corpora: list[str]
    recordings: list[str]  # "<dataset>/<recording_id>"
    sample_ids: list[str]  # canonical row order: fits never see file order


def probe_data(snapshot: Snapshot, values: list[Any]) -> ProbeData:
    corpora = [str(d) for d in snapshot.metadata["dataset"]]

    return ProbeData(
        representations={n: snapshot.representations[n] for n in REPRESENTATIONS},
        values=values,
        corpora=corpora,
        recordings=[
            f"{d}/{r}"
            for d, r in zip(corpora, snapshot.metadata["recording_id"], strict=True)
        ],
        sample_ids=[str(i) for i in snapshot.metadata["sample_id"]],
    )


def run_probe(
    task_kind: str,
    classes: Sequence[str] | None,
    setting: Setting,
    train: ProbeData,
    validation: ProbeData,
    *,
    bootstrap: int,
    seed: int,
    fitted: dict[Any, FittedProbe] | None = None,
) -> dict[str, Any]:
    """Scores of both representations and their delta in one setting.

    `fitted` caches probes by (representation, training corpora): a probe
    depends only on its training rows, so e.g. within:egocom and
    egocom->ego4d share one.
    """

    fitted = {} if fitted is None else fitted

    def rows(data: ProbeData, corpora) -> list[int]:
        # In sample-id order: the snapshot's row order changes nothing.
        return sorted(
            (
                i
                for i, (value, corpus) in enumerate(
                    zip(data.values, data.corpora, strict=True)
                )
                if value is not None and corpus in corpora
            ),
            key=lambda i: data.sample_ids[i],
        )

    train_rows, eval_rows = (
        rows(train, setting.train),
        rows(validation, setting.evaluate),
    )
    result: dict[str, Any] = {
        "setting": setting.name,
        "setting_kind": setting.kind,
        "train_corpora": list(setting.train),
        "eval_corpora": list(setting.evaluate),
    }

    if task_kind == CATEGORICAL:
        assert classes is not None
        train_counts = _counts(train.values, train_rows, classes)
        eval_counts = _counts(validation.values, eval_rows, classes)
        unsupported = {
            c: {"train": train_counts[c], "eval": eval_counts[c]}
            for c in classes
            if train_counts[c] < MIN_CLASS_SUPPORT or eval_counts[c] < MIN_CLASS_SUPPORT
        }
        # The task keeps its canonical K classes and 1 / K reference; if one
        # of them cannot be evaluated here, neither can the setting.
        result |= {
            "classes": list(classes),
            "unsupported_classes": unsupported,
            "train_class_counts": train_counts,
            "eval_class_counts": eval_counts,
            "reference": 1 / len(classes),
        }

        if unsupported:
            return result | _skipped(
                "unsupported: "
                + ", ".join(
                    f"{c} (train {n['train']}, eval {n['eval']})"
                    for c, n in unsupported.items()
                )
                + f" below {MIN_CLASS_SUPPORT} rows; the {len(classes)}-class task "
                "is not evaluable in this setting",
                train_rows,
                eval_rows,
            )

        index = {c: k for k, c in enumerate(classes)}
        train_y = torch.tensor([index[train.values[i]] for i in train_rows])
        eval_y = torch.tensor([index[validation.values[i]] for i in eval_rows])
    else:
        result["reference"] = 0.0
        train_y = torch.tensor([float(train.values[i]) for i in train_rows])
        eval_y = torch.tensor([float(validation.values[i]) for i in eval_rows])

    if len(train_rows) < MIN_TRAIN_ROWS or not eval_rows:
        return result | _skipped(
            f"fewer than {MIN_TRAIN_ROWS} training rows or no evaluation rows",
            train_rows,
            eval_rows,
        )

    # Selection and fit on probe-train only, one scaler per representation.
    probes: dict[str, FittedProbe] = {}

    for name in REPRESENTATIONS:
        key = (name, setting.train)

        if key not in fitted:
            fitted[key] = fit_probe(
                task_kind,
                train.representations[name][train_rows],
                train_y,
                [train.recordings[i] for i in train_rows],
                classes=len(classes) if classes is not None else None,
                seed=_seed(seed, "cv", *setting.train),
            )

        probes[name] = fitted[key]
        result[f"{name}_selected_regularization"] = probes[name].cv.selected

    result["regularization_parameter"] = regularization_parameter(task_kind)
    unsupported = [p.cv.unsupported for p in probes.values() if p.cv.unsupported]

    if unsupported:
        # No validation score, interval or delta without a CV-selected probe.
        return result | _skipped(
            unsupported[0], train_rows, eval_rows, keep_regularization=True
        )

    recordings = [validation.recordings[i] for i in eval_rows]
    clusters = sorted(set(recordings))
    position = {c: g for g, c in enumerate(clusters)}
    members = torch.tensor([position[r] for r in recordings])
    stratum = {validation.recordings[i]: validation.corpora[i] for i in eval_rows}
    generator = torch.Generator().manual_seed(
        _seed(seed, result["setting"], task_kind, len(eval_rows))
    )
    # One set of resampled recordings for both representations: paired delta.
    weights = cluster_bootstrap_weights(
        clusters,
        [stratum[c] for c in clusters],
        resamples=bootstrap,
        generator=generator,
    )
    point: dict[str, torch.Tensor] = {}
    resampled: dict[str, torch.Tensor] = {}

    for name in REPRESENTATIONS:
        pred = probes[name].predict(validation.representations[name][eval_rows])

        if classes is not None:
            sums = _class_sums(eval_y, pred, len(classes))
            per_cluster = torch.zeros(
                len(clusters), *sums.shape[1:], dtype=torch.float64
            )
            per_cluster.index_add_(0, members, sums)
            point[name] = _balanced_accuracy(per_cluster.sum(0))
            resampled[name] = _balanced_accuracy(
                torch.einsum("bg,gkc->bkc", weights, per_cluster)
            )
        else:
            sums = _regression_sums(eval_y, pred)
            per_cluster = torch.zeros(len(clusters), 4, dtype=torch.float64)
            per_cluster.index_add_(0, members, sums)
            point[name] = _r2(per_cluster.sum(0))
            resampled[name] = _r2(weights @ per_cluster)

    point["delta"] = point[LATENT] - point[FEATURES]
    resampled["delta"] = resampled[LATENT] - resampled[FEATURES]
    tail = (1 - CONFIDENCE) / 2
    quantiles = torch.tensor([tail, 1 - tail], dtype=torch.float64)

    for name in (*REPRESENTATIONS, "delta"):
        value = float(point[name])
        result[f"{name}_score"] = None if math.isnan(value) else value
        result[f"{name}_ci"] = (
            torch.nanquantile(resampled[name], quantiles).tolist()
            if len(clusters) > 1 and not math.isnan(value)
            else None
        )

    return result | {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_eval_recordings": len(clusters),
        "skipped": None,
    }


def _skipped(
    reason, train_rows, eval_rows, *, keep_regularization: bool = False
) -> dict[str, Any]:
    regularization = (
        {}
        if keep_regularization
        else {f"{name}_selected_regularization": None for name in REPRESENTATIONS}
    )

    return {
        "skipped": reason,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "n_eval_recordings": 0,
        **{
            key: None
            for name in (*REPRESENTATIONS, "delta")
            for key in (f"{name}_score", f"{name}_ci")
        },
        **regularization,
    }


def _counts(values, rows, classes) -> dict[str, int]:
    counts = dict.fromkeys(classes, 0)

    for i in rows:
        counts[values[i]] += 1

    return counts


def _seed(seed: int, *parts: Any) -> int:
    text = ":".join(map(str, (seed, *parts)))

    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=7).digest())


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------


def _variables(snapshot: Snapshot, sources):
    audits = {corpus: audit_corpus(source) for corpus, source in sources.items()}
    joined = join_labels(snapshot.metadata, audits, sources, selection=SELECTION)

    return {v.name: v for v in joined.variables}, joined, audits


def analyze_probes(
    train: Snapshot,
    validation: Snapshot,
    sources: Mapping[str, CorpusLabelSource],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """Every task in every setting; the context (N, classes, coverage)."""

    check_snapshots(train, validation)
    train_variables, _, _ = _variables(train, sources)
    validation_variables, joined, audits = _variables(validation, sources)
    seed = validation.seed
    corpora = sorted(set(map(str, validation.metadata["dataset"])))
    tasks: dict[str, Any] = {}
    scores: list[dict[str, Any]] = []
    fitted_probes: list[dict[str, Any]] = []

    for task in TASKS:
        variable = validation_variables.get(task.variable)
        train_variable = train_variables.get(task.variable)

        if variable is None or train_variable is None:
            tasks[task.variable] = {
                "group": task.group,
                "missing": "label not available for these snapshots",
                "excluded": joined.excluded.get(task.label),
            }
            continue

        categorical = variable.kind == CATEGORICAL
        train_data = probe_data(train, train_variable.values)
        validation_data = probe_data(validation, variable.values)
        tasks[task.variable] = {
            "group": task.group,
            "kind": variable.kind,
            "classes": list(variable.classes or ()) if categorical else None,
            "context": _context(
                variable.classes if categorical else None,
                {"probe-train": train_data, "probe-validation": validation_data},
            ),
        }

        fitted: dict[Any, FittedProbe] = {}

        for setting in probe_settings(corpora, cross_domain=categorical):
            scores.append(
                {
                    "task": task.variable,
                    "group": task.group,
                    "kind": variable.kind,
                    **run_probe(
                        variable.kind,
                        variable.classes if categorical else None,
                        setting,
                        train_data,
                        validation_data,
                        bootstrap=bootstrap,
                        seed=_seed(seed, task.variable),
                        fitted=fitted,
                    ),
                }
            )

        # Model-selection provenance, once per fitted probe.
        fitted_probes += [
            {
                "task": task.variable,
                "task_type": variable.kind,
                "representation": name,
                "training_domain": list(train_corpora),
                "canonical_classes": (
                    list(variable.classes or ()) if categorical else None
                ),
                "cv": probe.cv.provenance(),
            }
            for (name, train_corpora), probe in fitted.items()
        ]

    return {
        "tasks": tasks,
        "scores": scores,
        "fitted_probes": fitted_probes,
        "corpora": corpora,
        "labels": {
            "corpora": {c: a.provenance for c, a in audits.items()},
            "alignment": joined.alignment,
        },
        "settings": {
            "seed": seed,
            "standardization": "per dimension, mean and std of the probe-train rows",
            "categorical_probe": (
                "multinomial logistic regression, class-balanced weights "
                "n / (K n_class), L-BFGS, float64"
            ),
            "logistic_c_grid": list(LOGISTIC_C_GRID),
            "continuous_probe": "ridge regression on centred targets, float64",
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "regularization": (
                "scikit-learn semantics on standardized inputs, intercepts "
                "unpenalized. Logistic regression uses C, where smaller values "
                "mean stronger L2 regularization. Ridge uses alpha, where larger "
                "values mean stronger regularization. The value is chosen from the "
                f"predeclared grid by {CV_FOLDS}-fold CV grouped by "
                f"{CV_GROUPING} inside probe-train (criterion: mean out-of-fold "
                "primary score over the same valid folds for every candidate; at "
                f"least {MIN_VALID_CV_FOLDS} valid folds; a categorical fold is "
                "valid only if every canonical class is in both its train and "
                "held-out parts). Ties are resolved in favor of stronger "
                "regularization (smaller C for logistic regression, larger alpha "
                "for ridge). Selected separately per representation, task and "
                "training set, then frozen before validation"
            ),
            "sampling": (
                "both snapshots: seeded fixed permutation of the split, no class "
                "or corpus balancing"
            ),
            "categorical_score": (
                "balanced accuracy; reference 1 / K over the task's canonical K "
                "classes; a setting where a class lacks support is unsupported"
            ),
            "continuous_score": "R^2 on the evaluated rows; reference 0",
            "min_class_support": MIN_CLASS_SUPPORT,
            "min_train_rows": MIN_TRAIN_ROWS,
            "bootstrap_resamples": bootstrap,
            "confidence": CONFIDENCE,
            "interval": (
                "percentile bootstrap over validation recordings within each "
                "corpus; delta paired; probes fitted once"
            ),
            "cross_domain": "categorical labels only",
        },
    }


def _context(classes, snapshots: Mapping[str, ProbeData]) -> dict[str, Any]:
    """N, valid coverage and class fractions per snapshot and corpus."""

    context: dict[str, Any] = {}

    for which, data in snapshots.items():
        for corpus in sorted(set(data.corpora)):
            rows = [i for i, c in enumerate(data.corpora) if c == corpus]
            valid = [data.values[i] for i in rows if data.values[i] is not None]
            entry: dict[str, Any] = {
                "n": len(rows),
                "n_valid": len(valid),
                "coverage": len(valid) / len(rows) if rows else None,
            }

            if classes is not None:
                entry["class_counts"] = {c: valid.count(c) for c in classes}
                entry["class_fractions"] = {
                    c: valid.count(c) / len(valid) if valid else None for c in classes
                }

            context.setdefault(which, {})[corpus] = entry

    return context


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

    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(
            validation.manifest.get("provenance") or {},
            labels_revision=labels_revision,
        )
    )
    results = analyze_probes(train, validation, sources, bootstrap=bootstrap)
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
            "probe_train": _source(train),
            "probe_validation": _source(validation),
        },
        **results,
        "figures": [f"figures/{name}" for name in figures],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(probe_report(summary), encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_COLORS = {FEATURES: SERIES[0], LATENT: SERIES[1]}
_NAMES = {FEATURES: "Mimi features", LATENT: "WM latent"}
_TITLES = {
    CURRENT: "Current conversational state",
    TEMPORAL: "Temporal conversational state",
    FUTURE_STATE: "Future conversational state (1 s)",
}


_SHORT = {
    "instantaneous.ego_speaking": "ego speaking",
    "instantaneous.others_active": "others active",
    "instantaneous.joint_speech_state_occupancy:dominant": "joint state",
    "timing.time_to_next_speaker_onset": "time to next\nspeaker onset",
    "timing.silence_duration": "silence duration",
    "future.future_joint_speech_state@1s": "joint state\nin 1 s",
}


def _short(task: str, *, figure: bool = False) -> str:
    name = _SHORT.get(task, task)

    return name if figure else name.replace("\n", " ")


def _new_figure(width: float, height: float) -> Figure:
    from matplotlib.figure import Figure

    return Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")


def _comparison_panel(ax, scores: Sequence[Mapping[str, Any]], title: str) -> None:
    """Features and latent per task, joined; the trivial reference per task."""

    style(ax)
    ticks = []

    for x, score in enumerate(scores):
        ticks.append(_short(score["task"], figure=True))
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
                label=_NAMES[name] if x == 0 else None,
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
        figure = _new_figure(3.2 * len(within) + 1.2, 3.8)
        axes = figure.subplots(1, len(within), squeeze=False, sharey=True)[0]

        for ax, setting in zip(axes, within, strict=True):
            _comparison_panel(
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
    figure = _new_figure(3.6 * max(1, len(cross)) + 1.2, 3.8)
    axes = figure.subplots(1, max(1, len(cross)), squeeze=False, sharey=True)[0]

    for ax, setting in zip(axes, cross, strict=False):
        _comparison_panel(
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


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Cross-domain wording: the share of the target's within-domain margin over
# the reference that the transferred probe keeps.
SHARED_RETENTION = 0.8
# Total-variation distance between class distributions flagged as a shift.
CLASS_SHIFT = 0.10


def _fmt(score: Mapping[str, Any], name: str) -> str:
    value, interval = score[f"{name}_score"], score[f"{name}_ci"]

    if value is None:
        return "n/a"

    text = f"{value:+.3f}" if name == "delta" else f"{value:.3f}"

    if interval is None:
        return text

    return f"{text} [{interval[0]:.3f}, {interval[1]:.3f}]"


def _above(score: Mapping[str, Any], name: str, reference: float) -> str:
    interval = score[f"{name}_ci"]

    if score[f"{name}_score"] is None or interval is None:
        return "undetermined"
    if interval[0] > reference:
        return "above"
    if interval[1] < reference:
        return "below"

    return "includes"


def _score_table(scores: Sequence[Mapping[str, Any]]) -> list[str]:
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
                f"| {_short(s['task'])} | {s['setting']} | – | "
                f"{s['n_train']:,} / {s['n_eval']:,} | not evaluable: {s['skipped']} | | |"
            )
            continue

        lines.append(
            f"| {_short(s['task'])} | {s['setting']} | {s['reference']:.3f} | "
            f"{s['n_train']:,} / {s['n_eval']:,} ({s['n_eval_recordings']}) | "
            f"{_fmt(s, FEATURES)} | {_fmt(s, LATENT)} | {_fmt(s, 'delta')} |"
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
            f"{_NAMES[name]}: {words[_above(s, name, s['reference'])]}"
            for name in REPRESENTATIONS
        ]
        lines.append(f"- **{_short(s['task'])}** — " + "; ".join(parts) + ".")

    return lines


def _projector_effect(score: Mapping[str, Any]) -> str:
    return {
        "above": "improves",
        "below": "degrades",
        "includes": "preserves",
        "undetermined": "undetermined",
    }[_above(score, "delta", 0.0)]


def _within_domain(summary: Mapping[str, Any]) -> list[str]:
    lines = []

    for corpus in summary["corpora"]:
        rows = [s for s in summary["scores"] if s["setting"] == f"within:{corpus}"]

        if not rows:
            continue

        lines += [f"### {corpus}", "", *_score_table(rows), ""]

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
                f"- **{_short(s['task'])}, {s['setting']}**: not probed "
                f"({s['skipped'] or 'no within-domain reference'})."
            )
            continue

        parts = []

        for name in REPRESENTATIONS:
            reference = s["reference"]
            cross_above = _above(s, name, reference)
            within_above = _above(within, name, reference)
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
                f"{_NAMES[name]} {_fmt(s, name)} vs within-{target} "
                f"{_fmt(within, name)}{kept} — {verdict}"
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
            f"- **{_short(s['task'])}, train {s['train_corpora'][0]} → evaluate "
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

        effect = _projector_effect(s)

        if effect in effects:
            effects[effect].append(
                f"{_short(s['task'])} ({s['setting']}, Δ {s['delta_score']:+.3f})"
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
            lines.append(f"| {_short(task)} | – | – | – | {info['missing']} | |")
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
                    f"| {_short(task)} | {which} | {corpus} | {entry['n']:,} | "
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
            f"| {_short(entry['task'])} | {' + '.join(entry['training_domain'])} | "
            f"{_NAMES[entry['representation']]} | "
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
            f"{_fmt(future, LATENT)} against chance {future['reference']:.3f} "
            f"(projector effect: {_projector_effect(future)}). A probe-based future-state "
            "score could be monitored during training as a representation check, "
            "alongside rollout skill; whether it tracks model quality is untested."
        )

    degraded = [t for t, s in pooled.items() if _projector_effect(s) == "degrades"]

    if degraded:
        lines.append(
            "- The projector degrades linear access to "
            + ", ".join(_short(t) for t in degraded)
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

    if any(_projector_effect(s) == "degrades" for s in pooled):
        questions.append(
            "Do predictive projectors in JEPA-style models discard linearly accessible "
            "input information, and is this considered harmful or a form of useful "
            "abstraction?"
        )
    if any(_projector_effect(s) == "improves" for s in pooled):
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
        and _above(s, LATENT, s["reference"]) != "above"
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
            *_score_table(group(CURRENT)),
            "",
            *_decodable(group(CURRENT)),
            "",
            "## Temporal state",
            "",
            "R² on raw seconds; 0 is predicting the evaluated rows' mean.",
            "",
            *_score_table(group(TEMPORAL)),
            "",
            *_decodable(group(TEMPORAL)),
            "",
            "## Future conversational state",
            "",
            *_score_table(group(FUTURE_STATE)),
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
            *_score_table([s for s in scores if s["setting_kind"] == "cross"]),
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
