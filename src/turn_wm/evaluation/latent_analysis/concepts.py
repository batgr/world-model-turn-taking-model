"""
Concept probes: what the frozen representations keep linearly accessible
beyond the current conversational state.

Question: besides who speaks now (`probes.py`), which information do the
encoder features and the WM latent keep in a linearly accessible form, along
four axes?

- vocal activity: how many voices are active now, whether another
  participant starts speaking in the current cell;
- multi-party structure: how many distinct speakers were active in the last
  10 s, how many participants the recording has, dyad or group;
- social signals: whether someone talks to the wearer (Ego4D Talking-To-Me),
  whether the wearer is a native speaker or the host (EgoCom);
- information unrelated to the conversation: background fan or music
  (EgoCom), the wearer's speech rate in words per second (EgoCom).

The latent is a per-frame projection of the encoder features: it cannot hold
information the features lack. A score difference measures what the
projector keeps linearly accessible, not what it adds. A successful probe
supports "the information is linearly accessible", not "the world model
uses it".

Two protocols, fixed per concept before any fit:

- `train_to_validation` (concepts that vary within a recording): the
  `probes.py` protocol unchanged. Fitted on the train-split snapshot,
  evaluated on the validation-split snapshot, 95% bootstrap over
  validation recordings, paired delta.
- `grouped_cv` (concepts constant within a recording, which the validation
  split barely varies): out-of-fold predictions over the union of both
  snapshots, folds grouped by conversation (EgoCom films one conversation
  from three wearers: those views never straddle a fold). At most
  `MAX_ROWS_PER_RECORDING` seeded rows per recording, since the information
  is per recording; the regularization is chosen inside each outer training
  part by conversation-grouped CV. 95% bootstrap over conversations.

Every value is derived from the data release's label tables, as documented
per concept (`definition`); nothing is approximated. A concept whose labels
are not published, or whose classes lack support, is reported as not
evaluable, never dropped. The test split is never read.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from turn_wm.evaluation.latent_analysis.concept_labels import (
    CONCEPTS,
    GROUPED_CV,
    LOCAL_WINDOW_CELLS,
    SPLIT,
    Concept,
    CorpusTables,
    concept_values,
    load_tables,
    unavailable,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CorpusLabelSource,
    audit_corpus,
)
from turn_wm.evaluation.latent_analysis.linear_probe import (
    CV_FOLDS,
    balanced_accuracy_of_sums,
    class_sums,
    fit_probe,
    r2_of_sums,
    recording_folds,
    regression_sums,
)
from turn_wm.evaluation.latent_analysis.probes import (
    CONFIDENCE,
    DEFAULT_BOOTSTRAP,
    FEATURES,
    LATENT,
    MIN_CLASS_SUPPORT,
    POOLED,
    REPRESENTATIONS,
    Setting,
    check_snapshots,
    probe_data,
    run_probe,
)
from turn_wm.evaluation.latent_analysis.probes_report import (
    format_score,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    cluster_bootstrap_weights,
)
from turn_wm.evaluation.latent_analysis.seeding import derived_seed
from turn_wm.evaluation.latent_analysis.snapshot import Snapshot
from turn_wm.progress import log, progress

SCHEMA_VERSION = 1
ANALYSIS = "concepts"


SETTING_NAMES = {SPLIT: "train → validation", GROUPED_CV: "grouped CV (conversations)"}

MAX_ROWS_PER_RECORDING = 25


# ---------------------------------------------------------------------------
# Conversation-grouped cross-validation
# ---------------------------------------------------------------------------


def grouped_cv(
    concept: Concept,
    rows: Mapping[str, Any],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Out-of-fold scores of both representations, folds = whole conversations.

    `rows` holds, for the selected rows: `values`, `groups` (conversation
    keys), `corpora` and `representations` (name -> tensor).
    """

    values, groups = rows["values"], rows["groups"]
    classes = concept.classes
    result: dict[str, Any] = {"n_train": len(values), "n_eval": len(values)}

    if concept.kind == CATEGORICAL:
        assert classes is not None
        index = {c: k for k, c in enumerate(classes)}
        y = torch.tensor([index[v] for v in values], dtype=torch.long)
        per_class = {
            c: len({g for g, v in zip(groups, values, strict=True) if v == c})
            for c in classes
        }
        result |= {
            "reference": 1 / len(classes),
            "classes": list(classes),
            "class_counts": {c: values.count(c) for c in classes},
            "conversations_per_class": per_class,
        }
        short = {c: n for c, n in per_class.items() if n < CV_FOLDS}

        if short or min(result["class_counts"].values()) < MIN_CLASS_SUPPORT:
            return result | _not_evaluable(
                "unsupported: "
                + ", ".join(f"{c} in {n} conversation(s)" for c, n in per_class.items())
                + f"; every class needs {CV_FOLDS} conversations and "
                f"{MIN_CLASS_SUPPORT} rows"
            )
    else:
        result["reference"] = 0.0
        y = torch.tensor([float(v) for v in values], dtype=torch.float64)

        if len(set(values)) < 2:
            return result | _not_evaluable("no target variance")

    fold, count = recording_folds(
        groups, folds=CV_FOLDS, seed=derived_seed(seed, "outer")
    )
    predictions = {name: torch.zeros_like(y) for name in REPRESENTATIONS}
    selected: dict[str, list[float | None]] = {name: [] for name in REPRESENTATIONS}

    for f in range(count):
        held = fold == f
        train_groups = [g for g, h in zip(groups, held.tolist(), strict=True) if not h]

        for name in REPRESENTATIONS:
            x = rows["representations"][name]
            probe = fit_probe(
                concept.kind,
                x[~held],
                y[~held],
                train_groups,
                classes=None if classes is None else len(classes),
                seed=derived_seed(seed, "inner", f),
            )
            selected[name].append(probe.cv.selected)

            if probe.probe is None:
                return (
                    result
                    | _not_evaluable(f"outer fold {f}: {probe.cv.unsupported}")
                    | {"selected_regularization": selected}
                )

            predictions[name][held] = probe.predict(x[held]).to(y.dtype)

    clusters = sorted(set(groups))
    position = {g: k for k, g in enumerate(clusters)}
    members = torch.tensor([position[g] for g in groups])
    stratum = dict(zip(groups, rows["corpora"], strict=True))
    weights = cluster_bootstrap_weights(
        clusters,
        [stratum[g] for g in clusters],
        resamples=bootstrap,
        generator=torch.Generator().manual_seed(derived_seed(seed, "bootstrap")),
    )
    point, resampled = {}, {}

    for name in REPRESENTATIONS:
        if classes is not None:
            sums = class_sums(y, predictions[name], len(classes))
            per_cluster = torch.zeros(
                len(clusters), *sums.shape[1:], dtype=torch.float64
            )
            per_cluster.index_add_(0, members, sums)
            point[name] = balanced_accuracy_of_sums(per_cluster.sum(0))
            resampled[name] = balanced_accuracy_of_sums(
                torch.einsum("bg,gkc->bkc", weights, per_cluster)
            )
        else:
            sums = regression_sums(y, predictions[name])
            per_cluster = torch.zeros(len(clusters), 4, dtype=torch.float64)
            per_cluster.index_add_(0, members, sums)
            point[name] = r2_of_sums(per_cluster.sum(0))
            resampled[name] = r2_of_sums(weights @ per_cluster)

    point["delta"] = point[LATENT] - point[FEATURES]
    resampled["delta"] = resampled[LATENT] - resampled[FEATURES]

    return (
        result
        | _intervals(point, resampled)
        | {
            "n_eval_recordings": len(clusters),
            "outer_folds": count,
            "selected_regularization": selected,
            "skipped": None,
        }
    )


def _intervals(point, resampled) -> dict[str, Any]:
    tail = (1 - CONFIDENCE) / 2
    quantiles = torch.tensor([tail, 1 - tail], dtype=torch.float64)
    result = {}

    for name in (*REPRESENTATIONS, "delta"):
        value = float(point[name])
        result[f"{name}_score"] = None if math.isnan(value) else value
        result[f"{name}_ci"] = (
            None
            if math.isnan(value)
            else torch.nanquantile(resampled[name], quantiles).tolist()
        )

    return result


def _not_evaluable(reason: str) -> dict[str, Any]:
    return {
        "skipped": reason,
        "n_eval_recordings": 0,
        **{
            key: None
            for name in (*REPRESENTATIONS, "delta")
            for key in (f"{name}_score", f"{name}_ci")
        },
    }


def grouped_rows(
    concept: Concept,
    snapshots: Sequence[tuple[Snapshot, list[Any]]],
    tables: Mapping[str, CorpusTables],
    *,
    seed: int,
) -> dict[str, Any]:
    """At most MAX_ROWS_PER_RECORDING seeded valid rows per recording, from
    every snapshot; the group of a row is its conversation."""

    picked = []

    for s, (snapshot, values) in enumerate(snapshots):
        metadata = snapshot.metadata
        by_recording: dict[tuple[str, str], list[int]] = {}

        for i, value in enumerate(values):
            if value is not None:
                key = (str(metadata["dataset"][i]), str(metadata["recording_id"][i]))
                by_recording.setdefault(key, []).append(i)

        for (corpus, recording), members in sorted(by_recording.items()):
            members.sort(key=lambda i: derived_seed(seed, metadata["sample_id"][i]))
            conversation = tables[corpus].recordings[recording]["conversation_id"]
            picked += [
                (s, i, corpus, f"{corpus}/{conversation}")
                for i in members[:MAX_ROWS_PER_RECORDING]
            ]

    return {
        "values": [snapshots[s][1][i] for s, i, _, _ in picked],
        "groups": [g for _, _, _, g in picked],
        "corpora": [c for _, _, c, _ in picked],
        "representations": {
            name: torch.stack(
                [snapshots[s][0].representations[name][i] for s, i, _, _ in picked]
            )
            if picked
            else torch.empty(0)
            for name in REPRESENTATIONS
        },
    }


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------


def analyze_concepts(
    train: Snapshot,
    validation: Snapshot,
    sources: Mapping[str, CorpusLabelSource],
    *,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """Every concept under its protocol; availability and counts."""

    check_snapshots(train, validation)
    corpora = sorted(
        set(map(str, train.metadata["dataset"]))
        | set(map(str, validation.metadata["dataset"]))
    )
    unknown = sorted(set(corpora) - set(sources))

    if unknown:
        raise ValueError(f"No label source for corpora {unknown}")

    audits = {c: audit_corpus(sources[c]) for c in corpora}
    reasons = {
        concept.name: {c: unavailable(audits[c], concept) for c in corpora}
        for concept in CONCEPTS
    }
    tables = {}

    for corpus in corpora:
        recordings = {
            str(r)
            for snapshot in (train, validation)
            for d, r in zip(
                snapshot.metadata["dataset"],
                snapshot.metadata["recording_id"],
                strict=True,
            )
            if str(d) == corpus
        }
        log(f"concepts: reading {corpus} label tables ({len(recordings)} recordings)")
        tables[corpus] = load_tables(sources[corpus], audits[corpus], recordings)

    values = {
        "train": concept_values(train, tables, reasons),
        "validation": concept_values(validation, tables, reasons),
    }
    seed = validation.seed
    scores, concepts = [], {}

    for concept in progress(CONCEPTS, desc="concepts", unit="concept"):
        where = [
            c
            for c in concept.corpora
            if c in corpora and reasons[concept.name][c] is None
        ]
        info = asdict(concept) | {
            "unavailable": {c: r for c, r in reasons[concept.name].items() if r},
            "coverage": {
                which: {
                    c: _coverage(
                        snapshot, values[which][concept.name], c, concept.classes
                    )
                    for c in corpora
                }
                for which, snapshot in (("train", train), ("validation", validation))
            },
        }
        concepts[concept.name] = info
        base = {
            "task": concept.name,
            "axis": concept.axis,
            "protocol": concept.protocol,
            "kind": concept.kind,
            "setting": SETTING_NAMES[concept.protocol],
            "setting_kind": POOLED,
        }
        start = time.perf_counter()

        if not where:
            scores.append(
                base
                | {"reference": None, "n_train": 0, "n_eval": 0}
                | _not_evaluable(
                    "; ".join(f"{c}: {r}" for c, r in info["unavailable"].items())
                    or "no corpus"
                )
            )
            log(f"  {concept.name}: not evaluable ({scores[-1]['skipped']})")
            continue

        concept_seed = derived_seed(seed, concept.name)

        if concept.protocol == SPLIT:
            score = run_probe(
                concept.kind,
                concept.classes,
                Setting(POOLED, POOLED, tuple(where), tuple(where)),
                probe_data(train, values["train"][concept.name]),
                probe_data(validation, values["validation"][concept.name]),
                bootstrap=bootstrap,
                seed=concept_seed,
            )
        else:
            rows = grouped_rows(
                concept,
                [
                    (train, values["train"][concept.name]),
                    (validation, values["validation"][concept.name]),
                ],
                tables,
                seed=concept_seed,
            )
            score = grouped_cv(concept, rows, bootstrap=bootstrap, seed=concept_seed)

        # The protocol names the setting; run_probe's own name is "pooled".
        scores.append(base | score | {"setting": base["setting"], "corpora": where})
        log(
            f"  {concept.name}: "
            + (
                f"not evaluable ({score['skipped']})"
                if score.get("skipped")
                else f"features {format_score(score, FEATURES)}, latent {format_score(score, LATENT)}"
            )
            + f" in {time.perf_counter() - start:.0f}s"
        )

    return {
        "concepts": concepts,
        "scores": scores,
        "corpora": corpora,
        "labels": {c: a.provenance for c, a in audits.items()},
        "settings": {
            "seed": seed,
            "protocols": {
                SPLIT: (
                    "fitted on the train-split snapshot, evaluated on the "
                    "validation-split snapshot (the probes.py protocol: "
                    "recording-grouped CV inside probe-train selects the "
                    "regularization; bootstrap over validation recordings)"
                ),
                GROUPED_CV: (
                    f"{CV_FOLDS}-fold out-of-fold predictions over both snapshots, "
                    "folds grouped by conversation; regularization chosen by "
                    "conversation-grouped CV inside each outer training part; at "
                    f"most {MAX_ROWS_PER_RECORDING} seeded rows per recording; "
                    "bootstrap over conversations within each corpus"
                ),
            },
            "min_class_support": MIN_CLASS_SUPPORT,
            "min_conversations_per_class": CV_FOLDS,
            "local_window_cells": LOCAL_WINDOW_CELLS,
            "bootstrap_resamples": bootstrap,
            "confidence": CONFIDENCE,
        },
    }


def _coverage(snapshot, values, corpus, classes) -> dict[str, Any]:
    rows = [i for i, d in enumerate(snapshot.metadata["dataset"]) if str(d) == corpus]
    valid = [values[i] for i in rows if values[i] is not None]
    entry: dict[str, Any] = {"n": len(rows), "n_valid": len(valid)}

    if classes is not None:
        entry["class_counts"] = {c: valid.count(c) for c in classes}

    return entry
