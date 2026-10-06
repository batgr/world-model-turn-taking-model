"""
Linear probes of a run's representations: encoder `features` and the V1
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

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CONVERSATIONAL_STATE,
    FUTURE,
    TEMPORAL_STATE,
    CorpusLabelSource,
    audit_corpus,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.linear_probe import (
    CV_FOLDS,
    CV_GROUPING,
    LOGISTIC_C_GRID,
    MIN_VALID_CV_FOLDS,
    RIDGE_ALPHA_GRID,
    FittedProbe,
    balanced_accuracy_of_sums,
    candidates,
    class_sums,
    fit_probe,
    r2_of_sums,
    regression_sums,
    regularization_parameter,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    cluster_bootstrap_weights,
)
from turn_wm.evaluation.latent_analysis.seeding import derived_seed
from turn_wm.evaluation.latent_analysis.snapshot import Snapshot
from turn_wm.progress import log, progress

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

# A categorical setting is evaluable only if every canonical class has at
# least this many rows in probe-train and in the evaluated rows.
MIN_CLASS_SUPPORT = 20
# A setting is probed only with at least this many training rows.
MIN_TRAIN_ROWS = 50
DEFAULT_BOOTSTRAP = 1_000
CONFIDENCE = 0.95

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
    tick: Callable[[], Any] | None = None,
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
                seed=derived_seed(seed, "cv", *setting.train),
                tick=tick,
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
        derived_seed(seed, result["setting"], task_kind, len(eval_rows))
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
            sums = class_sums(eval_y, pred, len(classes))
            per_cluster = torch.zeros(
                len(clusters), *sums.shape[1:], dtype=torch.float64
            )
            per_cluster.index_add_(0, members, sums)
            point[name] = balanced_accuracy_of_sums(per_cluster.sum(0))
            resampled[name] = balanced_accuracy_of_sums(
                torch.einsum("bg,gkc->bkc", weights, per_cluster)
            )
        else:
            sums = regression_sums(eval_y, pred)
            per_cluster = torch.zeros(len(clusters), 4, dtype=torch.float64)
            per_cluster.index_add_(0, members, sums)
            point[name] = r2_of_sums(per_cluster.sum(0))
            resampled[name] = r2_of_sums(weights @ per_cluster)

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


def _progress_line(task, name, train_corpora, probe: FittedProbe) -> str:
    cv = probe.cv
    head = f"  {task_label(task)} | trained on {'+'.join(train_corpora)} | {REPRESENTATION_NAMES[name]}"

    if cv.selected is None:
        return f"{head}: not evaluable ({cv.unsupported})"

    return (
        f"{head}: {regularization_parameter(cv.kind)}={cv.selected:g} "
        f"({len(cv.valid_folds)}/{cv.requested_folds} folds) in {probe.seconds:.0f}s"
    )


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
    log("probes: joining labels to the probe-train and validation snapshots")
    train_variables, _, _ = _variables(train, sources)
    validation_variables, joined, audits = _variables(validation, sources)
    seed = validation.seed
    corpora = sorted(set(map(str, validation.metadata["dataset"])))
    tasks: dict[str, Any] = {}
    scores: list[dict[str, Any]] = []
    fitted_probes: list[dict[str, Any]] = []
    training_sets = len({s.train for s in probe_settings(corpora, cross_domain=True)})

    def planned(kind: str) -> int:
        # Upper bound per probe: every candidate on every fold, then the final.
        return len(candidates(kind)) * CV_FOLDS + 1

    available = [
        validation_variables[t.variable]
        for t in TASKS
        if t.variable in validation_variables and t.variable in train_variables
    ]
    bar = progress(
        total=sum(
            planned(v.kind) * training_sets * len(REPRESENTATIONS) for v in available
        ),
        desc="probes",
        unit="fit",
    )

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
            bar.set_description(
                f"probes | {task_label(task.variable)} | {setting.name}"
            )
            before = set(fitted)
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
                        seed=derived_seed(seed, task.variable),
                        fitted=fitted,
                        tick=bar.update,
                    ),
                }
            )

            for name, train_corpora in sorted(set(fitted) - before):
                probe = fitted[(name, train_corpora)]
                # Folds that were not valid are fits that never happen.
                bar.total -= planned(variable.kind) - probe.fits
                bar.refresh()
                log(_progress_line(task.variable, name, train_corpora, probe))

        # Training sets never fitted (the class support excluded them).
        for train_corpora in {
            s.train for s in probe_settings(corpora, cross_domain=True)
        }:
            for name in REPRESENTATIONS:
                if (name, train_corpora) not in fitted:
                    bar.total -= planned(variable.kind)
                    bar.refresh()

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

    bar.close()

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


# ---------------------------------------------------------------------------
# Display names (progress lines, figures and report)
# ---------------------------------------------------------------------------

REPRESENTATION_NAMES = {FEATURES: "encoder features", LATENT: "WM latent"}


TASK_LABELS = {
    "instantaneous.ego_speaking": "ego speaking",
    "instantaneous.others_active": "others active",
    "instantaneous.joint_speech_state_occupancy:dominant": "joint state",
    "timing.time_to_next_speaker_onset": "time to next\nspeaker onset",
    "timing.silence_duration": "silence duration",
    "future.future_joint_speech_state@1s": "joint state\nin 1 s",
}


def task_label(task: str, *, figure: bool = False) -> str:
    name = TASK_LABELS.get(task, task)

    return name if figure else name.replace("\n", " ")
