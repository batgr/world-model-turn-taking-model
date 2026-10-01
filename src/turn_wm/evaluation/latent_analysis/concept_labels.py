"""
The concepts probed by `concepts`, and how their values come from the labels.

Each `Concept` names its axis (vocal activity, multi-party structure,
social signals, unrelated information), its label kind, its protocol
(`train_to_validation` or `grouped_cv`) and its documented `definition`.
`load_tables` reads the release's label tables of a corpus;
`concept_values` derives one value per snapshot row from them, or None
where the rule does not apply. Nothing is approximated.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from turn_wm.evaluation.latent_analysis.label_source import (
    BOOLEAN_CLASSES,
    CATEGORICAL,
    CONTINUOUS,
    TIME_TOLERANCE_S,
    CorpusAudit,
    CorpusLabelSource,
)
from turn_wm.evaluation.latent_analysis.snapshot import Snapshot

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


GRID_STEP_S = 0.1


LOCAL_WINDOW_CELLS = 100  # 10 s of 100 ms cells, ending at the anchor's cell


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
