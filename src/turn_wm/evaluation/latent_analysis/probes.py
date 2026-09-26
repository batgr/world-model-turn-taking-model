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
- every preprocessing step (standardization, class weights, class
  selection) is fitted on probe-train rows only;
- categorical labels: multinomial logistic regression, class-balanced
  weights, fixed L2 (`LOGISTIC_L2`); score = balanced accuracy, reference
  1 / K for the K classes probed;
- continuous labels: ridge regression, fixed penalty (`RIDGE_ALPHA`);
  score = R^2 on the evaluated rows, reference 0 (predicting their mean);
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

# Fixed, documented regularization; never tuned on validation. Both act on
# standardized representations, per sample:
#   logistic: weighted mean cross-entropy + LOGISTIC_L2 / 2 * ||W||^2
#   ridge:    mean squared error + RIDGE_ALPHA * ||w||^2
LOGISTIC_L2 = 1e-3
RIDGE_ALPHA = 1e-2
LBFGS_MAX_ITER = 500
# A class is probed only with at least this many rows on both sides.
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
    x: torch.Tensor, y: torch.Tensor, classes: int, *, l2: float = LOGISTIC_L2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression, class-balanced; (W (D, K), b (K,))."""

    weights = balanced_class_weights(y, classes)
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
        loss = (F.cross_entropy(x @ w + b, y, reduction="none") * weights).sum()
        loss = loss / weights.sum() + 0.5 * l2 * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        optimizer.step(closure)

    return w.detach(), b.detach()


def fit_ridge(
    x: torch.Tensor, y: torch.Tensor, *, alpha: float = RIDGE_ALPHA
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression on centred targets; (w (D,), intercept)."""

    intercept = y.mean()
    gram = x.T @ x + len(x) * alpha * torch.eye(x.shape[1], dtype=x.dtype)

    return torch.linalg.solve(gram, x.T @ (y - intercept)), intercept


def probe_predictions(
    kind: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    *,
    classes: int | None = None,
) -> torch.Tensor:
    """Fit on train rows only (standardization included); predict eval rows."""

    standardizer = Standardizer.fit(train_x)
    train_x = standardizer.transform(train_x)
    eval_x = standardizer.transform(eval_x)

    if kind == CATEGORICAL:
        assert classes is not None
        w, b = fit_logistic(train_x, train_y, classes)
        return (eval_x @ w + b).argmax(dim=1)

    w, intercept = fit_ridge(train_x, train_y.double())
    return eval_x @ w + intercept


def balanced_accuracy(y: torch.Tensor, pred: torch.Tensor, classes: int) -> float:
    """Mean recall over the classes present in `y`."""

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
    present = total > 0
    recall = torch.where(present, correct / total.clamp_min(1e-300), 0.0)

    return recall.sum(-1) / present.sum(-1)


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
) -> dict[str, Any]:
    """Scores of both representations and their delta in one setting."""

    def rows(data: ProbeData, corpora) -> list[int]:
        return [
            i
            for i, (value, corpus) in enumerate(
                zip(data.values, data.corpora, strict=True)
            )
            if value is not None and corpus in corpora
        ]

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
        kept = [
            c
            for c in classes
            if train_counts[c] >= MIN_CLASS_SUPPORT
            and eval_counts[c] >= MIN_CLASS_SUPPORT
        ]
        # Explicit, never silent: a class too rare on either side is not
        # probed, and its rows are counted as excluded.
        result |= {
            "classes": kept,
            "excluded_classes": {
                c: {"train": train_counts[c], "eval": eval_counts[c]}
                for c in classes
                if c not in kept
            },
            "train_class_counts": {c: train_counts[c] for c in kept},
            "eval_class_counts": {c: eval_counts[c] for c in kept},
            "reference": 1 / len(kept) if kept else None,
        }
        train_rows = [i for i in train_rows if train.values[i] in kept]
        eval_rows = [i for i in eval_rows if validation.values[i] in kept]

        if len(kept) < 2:
            return result | _skipped(
                f"fewer than 2 classes with >= {MIN_CLASS_SUPPORT} rows on both sides",
                train_rows,
                eval_rows,
            )

        index = {c: k for k, c in enumerate(kept)}
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
        pred = probe_predictions(
            task_kind,
            train.representations[name][train_rows],
            train_y,
            validation.representations[name][eval_rows],
            classes=len(result.get("classes") or []) or None,
        )

        if task_kind == CATEGORICAL:
            sums = _class_sums(eval_y, pred, len(result["classes"]))
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


def _skipped(reason, train_rows, eval_rows) -> dict[str, Any]:
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
            "classes": list(variable.classes) if categorical else None,
            "context": _context(
                variable.classes if categorical else None,
                {"probe-train": train_data, "probe-validation": validation_data},
            ),
        }

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
                    ),
                }
            )

    return {
        "tasks": tasks,
        "scores": scores,
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
            "logistic_l2": LOGISTIC_L2,
            "continuous_probe": "ridge regression on centred targets, float64",
            "ridge_alpha": RIDGE_ALPHA,
            "regularization": (
                "fixed, per sample on standardized inputs; not tuned on validation"
            ),
            "categorical_score": "balanced accuracy; reference 1 / K classes probed",
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
                "skipped", (x, reference or 0), ha="center", fontsize=7, color=MUTED
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
                f"{s['n_train']:,} / {s['n_eval']:,} | skipped: {s['skipped']} | | |"
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
                f"Probes: {settings['categorical_probe']}, L2 {settings['logistic_l2']:g}; "
                f"{settings['continuous_probe']}, alpha {settings['ridge_alpha']:g}; "
                f"{settings['regularization']}. Standardization: "
                f"{settings['standardization']}. Scores: {settings['categorical_score']}; "
                f"{settings['continuous_score']}. Classes need ≥ "
                f"{settings['min_class_support']} rows in both probe-train and the "
                "evaluated rows of a setting, otherwise they are excluded and reported. "
                f"Intervals: {int(100 * settings['confidence'])}% "
                f"{settings['interval']} ({settings['bootstrap_resamples']} resamples, "
                "seeded)."
            ),
            "",
            "### Data context (not performance)",
            "",
            *_context_table(summary),
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
