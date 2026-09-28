"""
Concept probes: what the frozen representations keep linearly accessible
beyond the current conversational state.

Question: besides who speaks now (`probes.py`), which information do the
Mimi features and the WM latent keep in a linearly accessible form, along
four axes?

- vocal activity: how many voices are active now, whether another
  participant starts speaking in the current cell;
- multi-party structure: how many distinct speakers were active in the last
  10 s, how many participants the recording has, dyad or group;
- social signals: whether someone talks to the wearer (Ego4D Talking-To-Me),
  whether the wearer is a native speaker or the host (EgoCom);
- information unrelated to the conversation: background fan or music
  (EgoCom), the wearer's speech rate in words per second (EgoCom).

The latent is a per-frame projection of the Mimi features: it cannot hold
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

import bisect
import importlib.util
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from turn_wm.evaluation.latent_analysis.analyze import (
    Snapshot,
    describe_snapshot,
    read_snapshot,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    BOOLEAN_CLASSES,
    CATEGORICAL,
    CONTINUOUS,
    TIME_TOLERANCE_S,
    CorpusAudit,
    CorpusLabelSource,
    audit_corpus,
    hub_label_sources,
)
from turn_wm.evaluation.latent_analysis.probes import (
    CONFIDENCE,
    CV_FOLDS,
    DEFAULT_BOOTSTRAP,
    FEATURES,
    LATENT,
    MIN_CLASS_SUPPORT,
    POOLED,
    REPRESENTATIONS,
    Setting,
    _above,
    _balanced_accuracy,
    _class_sums,
    _comparison_panel,
    _fmt,
    _new_figure,
    _projector_effect,
    _r2,
    _regression_sums,
    _score_table,
    _seed,
    _source,
    check_snapshots,
    fit_probe,
    probe_data,
    recording_folds,
    run_probe,
)
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    SECONDARY_INK,
    SURFACE,
    close,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    cluster_bootstrap_weights,
)
from turn_wm.progress import log, progress

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1
ANALYSIS = "concepts"

VOCAL, MULTI_PARTY, SOCIAL, UNRELATED = (
    "vocal_activity",
    "multi_party",
    "social",
    "non_conversational",
)
AXES = (VOCAL, MULTI_PARTY, SOCIAL, UNRELATED)
AXIS_TITLES = {
    VOCAL: "Vocal activity",
    MULTI_PARTY: "Multi-party structure",
    SOCIAL: "Social signals",
    UNRELATED: "Unrelated to the conversation",
}

SPLIT, GROUPED_CV = "train_to_validation", "grouped_cv"
SETTING_NAMES = {SPLIT: "train → validation", GROUPED_CV: "grouped CV (conversations)"}

GRID_STEP_S = 0.1
LOCAL_WINDOW_CELLS = 100  # 10 s of 100 ms cells, ending at the anchor's cell
MAX_ROWS_PER_RECORDING = 25

VOICES = ("0", "1", "2+")
LOCAL_SPEAKERS = ("0-1", "2", "3+")


@dataclass(frozen=True)
class Concept:
    name: str
    axis: str
    kind: str  # CATEGORICAL or CONTINUOUS
    classes: tuple[str, ...] | None
    protocol: str  # SPLIT or GROUPED_CV
    corpora: tuple[str, ...]  # where the release defines it
    labels: tuple[str, ...]  # registry labels it is derived from
    question: str
    definition: str


CONCEPTS = (
    Concept(
        "voices_now",
        VOCAL,
        CATEGORICAL,
        VOICES,
        SPLIT,
        ("egocom", "ego4d"),
        ("instantaneous.active_speaker_count_subframes",),
        "Is the number of simultaneous voices (silence, one, overlap) accessible?",
        "max over the cell's subframes of active_speaker_count; null if any "
        "subframe is unknown",
    ),
    Concept(
        "other_onset_now",
        VOCAL,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        SPLIT,
        ("egocom", "ego4d"),
        ("events.other_onset_subframes",),
        "Is another participant's onset accessible? It is not in the action "
        "channel, which carries the wearer's events only.",
        "true if another participant starts speaking in any subframe of the "
        "cell; false only if every subframe is known false",
    ),
    Concept(
        "local_speakers_10s",
        MULTI_PARTY,
        CATEGORICAL,
        LOCAL_SPEAKERS,
        SPLIT,
        ("egocom", "ego4d"),
        ("instantaneous.speaker_activity",),
        "Is the local party size (distinct speakers in the last 10 s) accessible?",
        f"participants known to speak in the {LOCAL_WINDOW_CELLS} cells ending "
        "at the anchor (0-1, 2, 3+); null if the window starts before the "
        "recording or more than half of its cells have an unknown participant",
    ),
    Concept(
        "participant_count",
        MULTI_PARTY,
        CONTINUOUS,
        None,
        GROUPED_CV,
        ("ego4d",),
        ("metadata.participants",),
        "Is the recording's exact number of participants accessible?",
        "participant_count of the recording (people annotated in the scene, "
        "speaking or not); EgoCom is excluded: 53 of its 54 conversations "
        "have 3",
    ),
    Concept(
        "dyadic",
        MULTI_PARTY,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        GROUPED_CV,
        ("ego4d",),
        ("metadata.participants",),
        "Is a dyad (2 participants) distinguishable from a group?",
        "participant_count == 2",
    ),
    Concept(
        "addressed_to_wearer",
        SOCIAL,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        SPLIT,
        ("ego4d",),
        ("social_native.anyone_talking_to_wearer_subframes",),
        "Is speech addressed to the wearer (Talking-To-Me) accessible?",
        "true if any valid subframe of the cell is true; false only if every "
        "subframe is valid and false",
    ),
    Concept(
        "wearer_native",
        SOCIAL,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        GROUPED_CV,
        ("egocom",),
        ("metadata.participant_native_speaker", "instantaneous.ego_speaking"),
        "Is the wearer's native-speaker status accessible while they speak?",
        "the wearer's native_speaker, on cells where the wearer speaks",
    ),
    Concept(
        "wearer_host",
        SOCIAL,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        GROUPED_CV,
        ("egocom",),
        ("metadata.participant_is_host", "instantaneous.ego_speaking"),
        "Is the wearer's role (host or guest) accessible while they speak?",
        "the wearer's is_host, on cells where the wearer speaks",
    ),
    Concept(
        "background_fan",
        UNRELATED,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        GROUPED_CV,
        ("egocom",),
        ("metadata.background_conditions",),
        "Is a background fan, a nuisance, accessible?",
        "the recording's background_fan",
    ),
    Concept(
        "background_music",
        UNRELATED,
        CATEGORICAL,
        BOOLEAN_CLASSES,
        GROUPED_CV,
        ("egocom",),
        ("metadata.background_conditions",),
        "Is background music, a nuisance, accessible?",
        "the recording's background_music",
    ),
    Concept(
        "wearer_speech_rate",
        UNRELATED,
        CONTINUOUS,
        None,
        SPLIT,
        ("egocom",),
        ("text.speech_rate",),
        "Is the wearer's speaking rate, a paralinguistic trait, accessible?",
        "words_per_second of the wearer's transcribed turn containing the "
        "anchor (a property of the whole turn, so it includes words after the "
        "anchor); null outside a turn or when the rate is invalid",
    ),
)


# ---------------------------------------------------------------------------
# Derivations (pure; one value per row, None when missing)
# ---------------------------------------------------------------------------


def voices(counts: Sequence[int | None] | None) -> str | None:
    known = [c for c in counts or () if c is not None]

    if counts is None or len(known) != len(counts):
        return None

    return VOICES[min(max(known), 2)]


def any_event(
    values: Sequence[bool | None] | None, valid: Sequence[bool] | None = None
) -> str | None:
    """Tri-state "any": true as soon as one known value is true; false only
    when every value is known false."""

    if values is None:
        return None

    known = [v if valid is None or valid[i] else None for i, v in enumerate(values)]

    if any(v is True for v in known):
        return "true"
    if all(v is False for v in known):
        return "false"

    return None


def local_speakers(window: Sequence[Sequence[bool | None] | None]) -> str | None:
    """Distinct participants known to speak in a full window of cells."""

    if len(window) < LOCAL_WINDOW_CELLS:
        return None

    unknown = sum(cell is None or any(v is None for v in cell) for cell in window)

    if unknown > len(window) / 2:
        return None

    speakers = {
        p for cell in window if cell is not None for p, v in enumerate(cell) if v
    }

    return LOCAL_SPEAKERS[min(max(len(speakers) - 1, 0), 2)]


def turn_rate(
    turns: Sequence[tuple[float, float, float | None]], time_s: float
) -> float | None:
    """Rate of the turn [start, end) containing `time_s`; turns sorted by start."""

    k = bisect.bisect_right([start for start, _, _ in turns], time_s) - 1

    if k < 0:
        return None

    _, end, rate = turns[k]

    return rate if time_s < end else None


# ---------------------------------------------------------------------------
# Label tables
# ---------------------------------------------------------------------------


@dataclass
class CorpusTables:
    """One corpus's label tables, restricted to the snapshots' recordings."""

    grid: dict[str, dict[str, list[Any]]]  # recording -> column -> per cell
    social: dict[str, dict[str, list[Any]]]
    recordings: dict[str, dict[str, Any]]
    wearers: dict[str, dict[str, Any]]  # recording -> the wearer's participant row
    turns: dict[str, list[tuple[float, float, float | None]]]  # wearer's turns


GRID_COLUMNS = (
    "active_speaker_count_subframes",
    "other_onset_subframes",
    "speaker_activity",
    "ego_speaking",
)
SOCIAL_COLUMNS = (
    "anyone_talking_to_wearer_subframes",
    "anyone_talking_to_wearer_valid_subframes",
)


def unavailable(audit: CorpusAudit, concept: Concept) -> str | None:
    """Why `concept` cannot be derived for this corpus, or None."""

    if audit.corpus not in concept.corpora:
        return f"not defined for {audit.corpus}"

    for label in concept.labels:
        entry = audit.entry(label)

        if entry is not None and entry.get("extractor") in audit.rejected:
            raise ValueError(
                f"{audit.corpus}: {label} comes from extractor {entry['extractor']}, "
                f"{audit.rejected[entry['extractor']]}"
            )

        reason = audit.unavailable_reason(label)

        if reason is not None:
            return f"{label}: {reason}"

    return None


def load_tables(
    source: CorpusLabelSource, audit: CorpusAudit, recordings: set[str]
) -> CorpusTables:
    """Read the tables the concepts need, for `recordings` only."""

    def table(extractor: str, kind: str, columns: Sequence[str]) -> pa.Table | None:
        manifest = audit.manifests.get(extractor)

        if manifest is None or kind not in (manifest.get("tables") or {}):
            return None

        path = source.fetch(f"{extractor}/{manifest['tables'][kind]['file']}")
        present = set(pq.read_schema(path).names)

        return pq.read_table(
            path,
            columns=[c for c in columns if c in present],
            filters=[("recording_id", "in", sorted(recordings))],
        )

    keys = ["recording_id", "decision_index", "decision_time_s"]
    grid = table("speech", "grid", [*keys, *GRID_COLUMNS])
    social = table("social", "grid", [*keys, *SOCIAL_COLUMNS])
    recording_rows = table(
        "speech",
        "recordings",
        [
            "recording_id",
            "conversation_id",
            "wearer_index",
            "participant_count",
            "background_fan",
            "background_music",
        ],
    )
    participants = table(
        "speech",
        "participants",
        ["recording_id", "participant_index", "is_ego", "native_speaker", "is_host"],
    )
    segments = table(
        "text",
        "segments",
        [
            "recording_id",
            "participant_index",
            "start_s",
            "end_s",
            "words_per_second",
            "speech_rate_valid",
        ],
    )

    if recording_rows is None:
        raise ValueError(f"{source.corpus}: no speech recordings table")

    recording_info = {r["recording_id"]: r for r in recording_rows.to_pylist()}
    missing = sorted(recordings - set(recording_info))

    if missing:
        raise ValueError(
            f"{source.corpus}: recordings {missing[:3]} are not in the label release"
        )

    wearers = {}

    for row in [] if participants is None else participants.to_pylist():
        if row["is_ego"]:
            if row["recording_id"] in wearers:
                raise ValueError(f"{row['recording_id']}: several wearers")
            wearers[row["recording_id"]] = row

    turns: dict[str, list[tuple[float, float, float | None]]] = {}

    for row in [] if segments is None else segments.to_pylist():
        info = recording_info[row["recording_id"]]

        if row["participant_index"] != info["wearer_index"]:
            continue

        rate = row["words_per_second"] if row["speech_rate_valid"] else None
        turns.setdefault(row["recording_id"], []).append(
            (row["start_s"], row["end_s"], rate)
        )

    for rows in turns.values():
        rows.sort()

    return CorpusTables(
        grid=_by_recording(grid),
        social=_by_recording(social),
        recordings=recording_info,
        wearers=wearers,
        turns=turns,
    )


def _by_recording(table: pa.Table | None) -> dict[str, dict[str, list[Any]]]:
    """recording -> column -> values indexed by decision_index (checked contiguous)."""

    if table is None:
        return {}

    data = table.to_pydict()
    rows: dict[str, list[int]] = {}

    for k, recording in enumerate(data["recording_id"]):
        rows.setdefault(recording, []).append(k)

    result = {}

    for recording, members in rows.items():
        members.sort(key=data["decision_index"].__getitem__)
        part = {column: [values[k] for k in members] for column, values in data.items()}

        if part["decision_index"] != list(range(len(members))):
            raise ValueError(
                f"{recording}: the grid's decision_index is not contiguous"
            )

        result[recording] = part

    return result


# ---------------------------------------------------------------------------
# Values per snapshot row
# ---------------------------------------------------------------------------


def concept_values(
    snapshot: Snapshot,
    tables: Mapping[str, CorpusTables],
    reasons: Mapping[str, Mapping[str, str | None]],
) -> dict[str, list[Any]]:
    """concept -> one value per snapshot row (None: missing or not derivable).

    Grid rows are joined on (recording_id, decision_index = anchor_idx), and
    their time must equal the snapshot's anchor_time.
    """

    metadata = snapshot.metadata
    values = {c.name: [None] * len(metadata["sample_id"]) for c in CONCEPTS}

    for i, (corpus, recording, anchor, time_s) in enumerate(
        zip(
            map(str, metadata["dataset"]),
            map(str, metadata["recording_id"]),
            map(int, metadata["anchor_idx"]),
            map(float, metadata["anchor_time"]),
            strict=True,
        )
    ):
        corpus_tables = tables[corpus]
        grid = corpus_tables.grid.get(recording)
        cell = None

        if grid is not None and anchor < len(grid["decision_index"]):
            if abs(grid["decision_time_s"][anchor] - time_s) > TIME_TOLERANCE_S:
                raise ValueError(
                    f"{corpus}/{recording} decision_index {anchor}: grid time "
                    f"{grid['decision_time_s'][anchor]} differs from the snapshot's "
                    f"anchor_time {time_s}"
                )
            cell = {c: grid[c][anchor] for c in GRID_COLUMNS if c in grid}

        info = corpus_tables.recordings[recording]
        wearer = corpus_tables.wearers.get(recording) or {}
        wearer_speaks = cell is not None and cell.get("ego_speaking") is True
        derived: dict[str, Any] = {}

        if grid is not None and cell is not None:
            derived["voices_now"] = voices(cell.get("active_speaker_count_subframes"))
            derived["other_onset_now"] = any_event(cell.get("other_onset_subframes"))
            start = anchor - LOCAL_WINDOW_CELLS + 1
            derived["local_speakers_10s"] = (
                None
                if start < 0
                else local_speakers(grid["speaker_activity"][start : anchor + 1])
            )

        social = corpus_tables.social.get(recording)

        if social is not None and anchor < len(social["decision_index"]):
            derived["addressed_to_wearer"] = any_event(
                social["anyone_talking_to_wearer_subframes"][anchor],
                social["anyone_talking_to_wearer_valid_subframes"][anchor],
            )

        count = info.get("participant_count")
        derived["participant_count"] = None if count is None else float(count)
        derived["dyadic"] = None if count is None else _bool(count == 2)
        derived["wearer_native"] = (
            _bool(wearer.get("native_speaker")) if wearer_speaks else None
        )
        derived["wearer_host"] = _bool(wearer.get("is_host")) if wearer_speaks else None
        derived["background_fan"] = _bool(info.get("background_fan"))
        derived["background_music"] = _bool(info.get("background_music"))
        turns = corpus_tables.turns.get(recording)
        derived["wearer_speech_rate"] = None if not turns else turn_rate(turns, time_s)

        for concept in CONCEPTS:
            if reasons[concept.name].get(corpus) is None:
                values[concept.name][i] = derived.get(concept.name)

    return values


def _bool(value: bool | None) -> str | None:
    return None if value is None else BOOLEAN_CLASSES[int(bool(value))]


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

    fold, count = recording_folds(groups, folds=CV_FOLDS, seed=_seed(seed, "outer"))
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
                seed=_seed(seed, "inner", f),
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
        generator=torch.Generator().manual_seed(_seed(seed, "bootstrap")),
    )
    point, resampled = {}, {}

    for name in REPRESENTATIONS:
        if classes is not None:
            sums = _class_sums(y, predictions[name], len(classes))
            per_cluster = torch.zeros(
                len(clusters), *sums.shape[1:], dtype=torch.float64
            )
            per_cluster.index_add_(0, members, sums)
            point[name] = _balanced_accuracy(per_cluster.sum(0))
            resampled[name] = _balanced_accuracy(
                torch.einsum("bg,gkc->bkc", weights, per_cluster)
            )
        else:
            sums = _regression_sums(y, predictions[name])
            per_cluster = torch.zeros(len(clusters), 4, dtype=torch.float64)
            per_cluster.index_add_(0, members, sums)
            point[name] = _r2(per_cluster.sum(0))
            resampled[name] = _r2(weights @ per_cluster)

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
            members.sort(key=lambda i: _seed(seed, metadata["sample_id"][i]))
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

        concept_seed = _seed(seed, concept.name)

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
                else f"features {_fmt(score, FEATURES)}, latent {_fmt(score, LATENT)}"
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


def write_concepts(
    train_snapshot: Path,
    validation_snapshot: Path,
    *,
    output_dir: Path | None = None,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
) -> Path:
    """Probe every concept; write summary, scores, figure and report."""

    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    train = read_snapshot(train_snapshot)
    validation = read_snapshot(validation_snapshot)
    check_snapshots(train, validation)
    output_dir = (
        validation.path / "analysis" / ANALYSIS if output_dir is None else output_dir
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    start = time.perf_counter()
    log(f"concepts: probe-train {describe_snapshot(train)}")
    log(f"concepts: validation {describe_snapshot(validation)}")
    log(f"concepts: {bootstrap} bootstrap resamples; output {output_dir}")
    sources = (
        label_sources
        if label_sources is not None
        else hub_label_sources(
            validation.manifest.get("provenance") or {},
            labels_revision=labels_revision,
        )
    )
    results = analyze_concepts(train, validation, sources, bootstrap=bootstrap)
    log("concepts: writing figure, tables and report")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    figure = concept_figure(results)
    figure.savefig(output_dir / "figures" / "concepts.png", dpi=150, facecolor=SURFACE)
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
        "figures": ["figures/concepts.png"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(concept_report(summary), encoding="utf-8")
    log(f"concepts: done in {time.perf_counter() - start:.0f}s")

    return output_dir


def scores_table(scores: Sequence[Mapping[str, Any]]) -> pa.Table:
    rows = []

    for s in scores:
        row = {
            k: s.get(k)
            for k in (
                "task",
                "axis",
                "protocol",
                "kind",
                "reference",
                "n_train",
                "n_eval",
                "n_eval_recordings",
                "skipped",
            )
        }

        for name in (*REPRESENTATIONS, "delta"):
            interval = s.get(f"{name}_ci") or [None, None]
            row |= {
                f"{name}_score": s.get(f"{name}_score"),
                f"{name}_ci_low": interval[0],
                f"{name}_ci_high": interval[1],
            }

        rows.append(row)

    return pa.Table.from_pylist(rows)


# ---------------------------------------------------------------------------
# Figure and report
# ---------------------------------------------------------------------------


def concept_figure(results: Mapping[str, Any]) -> Figure:
    """One panel per axis: features and latent per concept, with the reference."""

    scores = results["scores"]
    figure = _new_figure(15, 4.2)
    axes = figure.subplots(1, len(AXES), squeeze=False)[0]

    for ax, axis in zip(axes, AXES, strict=True):
        rows = [
            s | {"task": s["task"] + (" (R²)" if s["kind"] == CONTINUOUS else "")}
            for s in scores
            if s["axis"] == axis
        ]
        _comparison_panel(ax, rows, AXIS_TITLES[axis])

    axes[0].set_ylabel(
        "Balanced accuracy (R² where marked)", color=SECONDARY_INK, fontsize=9
    )
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.suptitle(
        "Concept probes (dashed: trivial reference; bars: 95% bootstrap over "
        "validation recordings or, for grouped CV, conversations)",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )

    return figure


LIMITATIONS = """\
## Limitations

- The latent is a per-frame projection of the Mimi features: a concept can
  only be as accessible as the features allow; context (e.g. the 10 s party
  size) reaches both only through Mimi's own streaming state.
- wearer_native and wearer_host concern few recurring people: a probe can
  succeed by recognizing their voices rather than the trait itself.
- Grouped-CV concepts use train-split recordings for evaluation too (out of
  fold, never in the fold that fits the probe); the world model saw that
  audio during training, though never these labels.
- participant_count counts people annotated in the scene, speaking or not.
- wearer_speech_rate is a property of the whole turn, so it includes words
  after the anchor.
- Values are derived from the release's tables by the rules in each
  concept's definition; they are not labels defined by the data repository.
"""


def concept_report(summary: Mapping[str, Any]) -> str:
    provenance = summary["source"]["probe_validation"]["snapshot_provenance"] or {}
    checkpoint = provenance.get("checkpoint") or {}
    settings = summary["settings"]
    lines = [
        "# Concept probes",
        "",
        "## Purpose",
        "",
        (
            "Which information, beyond the current conversational state, do the "
            "Mimi features and the WM latent keep **linearly accessible**? Four "
            "axes: vocal activity, multi-party structure, social signals, and "
            "information unrelated to the conversation. A successful probe does "
            "not show that the world model uses the information."
        ),
        "",
        (
            f"Checkpoint `{checkpoint.get('filename')}` (step "
            f"{checkpoint.get('global_step')}, sha256 "
            f"`{str(checkpoint.get('sha256'))[:12]}…`). Test split never read."
        ),
        "",
        "Protocols: "
        + "; ".join(
            f"**{SETTING_NAMES[k]}**: {v}" for k, v in settings["protocols"].items()
        )
        + f". Intervals: {int(100 * settings['confidence'])}% percentile bootstrap "
        f"({settings['bootstrap_resamples']} resamples, seeded); deltas paired.",
        "",
        "## Concepts",
        "",
        "| concept | axis | protocol | question | definition | not evaluable in |",
        "|---|---|---|---|---|---|",
    ]

    for name, info in summary["concepts"].items():
        missing = "; ".join(f"{c}: {r}" for c, r in info["unavailable"].items()) or "–"
        lines.append(
            f"| {name} | {info['axis']} | {SETTING_NAMES[info['protocol']]} | "
            f"{info['question']} | {info['definition']} | {missing} |"
        )

    for axis in AXES:
        rows = [s for s in summary["scores"] if s["axis"] == axis]
        lines += ["", f"## {AXIS_TITLES[axis]}", "", *_score_table(rows), ""]

        for s in rows:
            if s["skipped"]:
                lines.append(f"- **{s['task']}**: not evaluable ({s['skipped']}).")
                continue

            reference = s["reference"]
            lines.append(
                f"- **{s['task']}**: Mimi features {_verdict(s, FEATURES, reference)}; "
                f"WM latent {_verdict(s, LATENT, reference)}; the projector "
                f"{_projector_effect(s)} linear accessibility (Δ {_fmt(s, 'delta')})."
            )

    lines += [
        "",
        LIMITATIONS,
        "## Figure",
        "",
        "- `figures/concepts.png`",
        "",
    ]

    return "\n".join(lines)


def _verdict(score, name, reference) -> str:
    return {
        "above": "above the reference",
        "includes": "not distinguishable from the reference",
        "below": "below the reference",
        "undetermined": "undetermined",
    }[_above(score, name, reference)]
