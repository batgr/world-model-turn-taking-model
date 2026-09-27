"""
TRAIN supervision for the TurnBench heads, with TurnBench's own semantics.

The official gold (`turnbench.gold`) is built in two stages: a 2-of-3
annotator consensus, then floor construction (EOT and interruption event
sets). TRAIN (otoSpeech-full-duplex-turn-104h) has a single annotator track
(`annotation_a`), so there is no consensus stage: annotator A's events are
the consensus views as they are, with no excluded (disputed) interval, and
stage 2 is the official `build_conversation_events`, unchanged. The fine ->
canonical mapping is the official `CANONICAL` / `TURN_CANONICAL`. So:

- EOT positive: the end of a floor-claiming segment when the floor leaves
  the speaker (the other takes over first, is already speaking, or nobody
  speaks again); a segment end the same speaker resumes from first is a
  mid-turn pause (an EOT negative span), never a positive;
- INT positive: the onset of a floor-taking interruption, on the interrupter;
- non-floor-taking interruptions are INT-excluded (neither positive nor
  negative), backchannels and non-content are INT negatives.

TRAIN supervision is therefore single-annotator, whereas the DEV gold the
heads are scored against is the official 2-of-3 consensus.

Frame targets, per output (EOT / INT x speaker 1 / 2), on the extraction's
frames (row k available at `available_s[k]`):

- positive: for each anchor t, the first causal frame available at or after
  t and the frame right after it, when they exist (POSITIVE_FRAMES = 2: V1's
  200 ms positive target window);
- ignored: available in [t - TAU_PRE_S, t) (the scorer's matching
  tolerance), or inside the task's excluded intervals for that speaker;
- negative: every other frame, including those after the two positives.

Annotations are read here only: representation extraction never sees them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch
from turnbench.gold import (
    CANONICAL,
    TAU_PRE_S,
    TURN_CANONICAL,
    AnchorEvent,
    ConsensusEvent,
    ConsensusViews,
    ConversationEvents,
    Interval,
    build_conversation_events,
    map_events,
)

from turn_wm.evaluation.turnbench.data import TRAIN_REPO, TRAIN_REVISION

ANNOTATION_FILES = {
    1: "speaker_1_annotation_a.srt",
    2: "speaker_2_annotation_a.srt",
}
OUTPUTS = ("eot_speaker_1", "eot_speaker_2", "int_speaker_1", "int_speaker_2")
POSITIVE_FRAMES = 2  # 200 ms at 10 Hz

Annotation = tuple[float, float, str, str]

_TIME = r"(\d+):(\d{2}):(\d{2})[,.](\d{3})"
_TIMING = re.compile(rf"^{_TIME}\s*-->\s*{_TIME}\s*$")
_LABEL = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


def parse_srt(text: str) -> list[Annotation]:
    """(start_s, end_s, fine label, transcript) per entry; fails on any
    malformed entry or label outside the TurnBench taxonomy."""

    annotations = []

    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [line.strip() for line in block.split("\n") if line.strip()]

        if lines and lines[0].isdigit():
            lines = lines[1:]

        timing = _TIMING.match(lines[0]) if lines else None
        label = _LABEL.match(lines[1]) if len(lines) > 1 else None

        if timing is None or label is None:
            raise ValueError(f"Malformed SRT entry: {block!r}")

        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in timing.groups())
        fine = label.group(1)

        if fine not in CANONICAL:
            raise ValueError(f"Label {fine!r} is not in the TurnBench taxonomy")

        annotations.append(
            (
                h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                fine,
                " ".join([label.group(2), *lines[2:]]).strip(),
            )
        )

    return annotations


def single_annotator_events(
    tracks: Mapping[int, list[Annotation]],
) -> ConversationEvents:
    """The official EOT / INT event sets from one annotator's two tracks."""

    def view(canonical: dict[str, str]) -> list[ConsensusEvent]:
        return [
            ConsensusEvent(speaker, round(start, 4), round(end, 4), label)
            for speaker in (1, 2)
            for start, end, label in map_events(
                [(start, end, fine) for start, end, fine, _ in tracks[speaker]],
                canonical,
            )
        ]

    # One annotator: nothing is disputed, so no excluded interval.
    return build_conversation_events(
        ConsensusViews(view(TURN_CANONICAL), [], view(CANONICAL), [])
    )


def frame_targets(
    events: ConversationEvents, available_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(K, 4) targets and (K, 4) bool supervision mask, columns = OUTPUTS."""

    times = available_s.double()

    if not bool((times.diff() > 0).all()):
        raise ValueError("available_s must be strictly increasing")

    targets = torch.zeros(len(times), len(OUTPUTS))
    supervised = torch.ones(len(times), len(OUTPUTS), dtype=torch.bool)
    tasks = {
        "eot": (events.eot_positive_events, events.eot_excluded),
        "int": (events.int_positive_events, events.int_excluded),
    }

    for column, name in enumerate(OUTPUTS):
        task, speaker = name.split("_speaker_")
        anchors, excluded = tasks[task]

        for anchor in _for_speaker(anchors, int(speaker)):
            t = anchor.time_s
            supervised[(times >= t - TAU_PRE_S) & (times < t), column] = False
            first = int(torch.searchsorted(times, t))  # first frame available >= t
            targets[first : first + POSITIVE_FRAMES, column] = 1.0

        for interval in _for_speaker(excluded, int(speaker)):
            supervised[(times >= interval.start) & (times <= interval.end), column] = (
                False
            )

    return targets, supervised


def download_annotations(
    conversation_id: str, destination: Path, *, token: str | None = None
) -> dict[int, Path]:
    """Annotator A's two SRT files of one TRAIN conversation, at the pin."""

    from huggingface_hub import hf_hub_download

    return {
        speaker: Path(
            hf_hub_download(
                TRAIN_REPO,
                f"{conversation_id}/{name}",
                repo_type="dataset",
                revision=TRAIN_REVISION,
                local_dir=destination,
                token=token,
            )
        )
        for speaker, name in ANNOTATION_FILES.items()
    }


def _for_speaker[T: (AnchorEvent, Interval)](items: list[T], speaker: int) -> list[T]:
    return [item for item in items if item.speaker == speaker]
