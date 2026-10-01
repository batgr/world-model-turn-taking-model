"""
Regularized linear probes: the estimator behind `probes` and `concepts`.

Standardization and class weights fitted on probe-train rows only;
multinomial logistic regression (categorical labels) or ridge regression
(continuous labels); the regularization chosen from predeclared grids by
recording-grouped cross-validation inside probe-train, ties going to the
stronger regularization, then frozen. Scores: balanced accuracy and R^2,
also from accumulated sums so bootstrap resamples reuse one prediction.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
)
from turn_wm.evaluation.latent_analysis.seeding import derived_seed

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


STD_FLOOR = 1e-8


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

    unique = sorted(set(recordings), key=lambda r: (derived_seed(seed, r), r))
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
    tick: Callable[[], Any] | None = None,
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

            if tick is not None:
                tick()

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
    seconds: float = 0.0  # progress display only; never written

    @property
    def fits(self) -> int:
        """Fits done: every candidate on every valid fold, then the final one."""

        if not self.cv.fold_scores:
            return 0

        return len(self.cv.fold_scores) * len(self.cv.valid_folds) + (
            self.probe is not None
        )

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
    tick: Callable[[], Any] | None = None,
) -> FittedProbe:
    """Select C / alpha inside probe-train, then fit on all of it and freeze.

    `tick` is called after every fit (progress display only).
    """

    start = time.perf_counter()
    cv = select_regularization(
        kind, x, y, recordings, classes=classes, seed=seed, tick=tick
    )

    if cv.selected is None:
        return FittedProbe(cv, None, time.perf_counter() - start)

    probe = LinearProbe.fit(kind, x, y, classes=classes, value=cv.selected)

    if tick is not None:
        tick()

    return FittedProbe(cv, probe, time.perf_counter() - start)


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

    return float(balanced_accuracy_of_sums(class_sums(y, pred, classes).sum(0)))


def r2(y: torch.Tensor, pred: torch.Tensor) -> float:
    """1 - SS_res / SS_tot, SS_tot around the mean of `y`: 0 predicts that mean."""

    return float(r2_of_sums(regression_sums(y, pred).sum(0)))


def class_sums(y, pred, classes) -> torch.Tensor:
    """(..., K, 2): rows of each class, and those predicted right."""

    total = F.one_hot(y, classes).double()

    return torch.stack([total, total * (pred == y)[..., None].double()], dim=-1)


def balanced_accuracy_of_sums(sums: torch.Tensor) -> torch.Tensor:
    total, correct = sums[..., 0], sums[..., 1]
    recall = correct / total.clamp_min(1e-300)
    score = recall.mean(-1)

    # Undefined unless every canonical class is present.
    return torch.where((total > 0).all(-1), score, torch.nan)


def regression_sums(y, pred) -> torch.Tensor:
    y, pred = y.double(), pred.double()

    return torch.stack([torch.ones_like(y), y, y.pow(2), (y - pred).pow(2)], dim=-1)


def r2_of_sums(sums: torch.Tensor) -> torch.Tensor:
    n, total, square, residual = sums.unbind(-1)
    variance = square - total.pow(2) / n

    return torch.where(
        variance > 0, 1 - residual / variance.clamp_min(1e-300), torch.nan
    )
