"""
TurnBench conversations as V1 inference inputs: two isolated speaker
channels and the global mono scene built from them.

Both TurnBench splits reach the same `Conversation`, from their audio only:

- DEV (`mundo-ai/turn-benchmark-dev`): the two FLAC audio columns and
  `conversation_id`; the annotation columns are not even selected;
- TRAIN (`otoearth/otoSpeech-full-duplex-turn-104h`): per conversation
  directory, only `speaker_1_audio.wav` and `speaker_2_audio.wav` are
  downloaded; never `combined_audio.wav`, the `.srt` annotations or
  `metadata.json`.

Channels are decoded at their native rate as float audio in [-1, 1] and must
share their sample rate and length: an inconsistent conversation fails
instead of being repaired.

Scene policy: scene = speaker_1 + speaker_2, sample by sample, with no
normalization and no statistic of the conversation, so each scene sample
depends on the two channels' samples at that instant only.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import turnbench.data as official

from turn_wm.evaluation.turnbench.activity import RMS_THRESHOLD, WINDOW_S
from turn_wm.evaluation.turnbench.timing import CONTROL_RATE_HZ

# DEV is read at the official scorer's pinned revision: the audio and the gold
# it scores against are then the same release.
DEV_REPO = official.DEV_DATASET
DEV_REVISION = official.DEV_REVISION
DEV_SPLIT = "dev"
TRAIN_REPO = "otoearth/otoSpeech-full-duplex-turn-104h"
TRAIN_REVISION = "46f520297f434edf804389f82f9075a59d2f8268"
REFERENCE_REPO = "https://github.com/SesameAILabs/turnbench"
REFERENCE_REVISION = "76ccd045f121ccfa921abac2ad3107027e736911"

SPEAKERS = ("speaker_1", "speaker_2")
AUDIO_COLUMNS = {speaker: f"{speaker}_audio" for speaker in SPEAKERS}
INFERENCE_COLUMNS = ("conversation_id", *AUDIO_COLUMNS.values())
TRAIN_AUDIO_FILES = {speaker: f"{speaker}_audio.wav" for speaker in SPEAKERS}
SOURCES = {
    "dev": {
        "repo": DEV_REPO,
        "revision": DEV_REVISION,
        "read": f"split {DEV_SPLIT!r}, columns {list(INFERENCE_COLUMNS)}",
    },
    "train": {
        "repo": TRAIN_REPO,
        "revision": TRAIN_REVISION,
        "read": f"<conversation_id>/{{{', '.join(TRAIN_AUDIO_FILES.values())}}}",
    },
}

SCENE_MIX_POLICY = "sum"
# kyutai/mimi feature-extractor sampling rate (its config.json).
MIMI_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class Conversation:
    """One TurnBench conversation: two synchronized mono channels."""

    conversation_id: str
    sample_rate: int
    speaker_1: torch.Tensor  # (N,) float, [-1, 1]
    speaker_2: torch.Tensor  # (N,) float, [-1, 1]

    def __post_init__(self) -> None:
        for speaker in SPEAKERS:
            channel = self.channel(speaker)

            if channel.ndim != 1 or not channel.is_floating_point():
                raise ValueError(
                    f"{self.conversation_id}: {speaker} must be a 1-D float "
                    f"tensor, got {channel.dtype} {tuple(channel.shape)}"
                )

        if len(self.speaker_1) != len(self.speaker_2):
            raise ValueError(
                f"{self.conversation_id}: speaker channels differ in length "
                f"({len(self.speaker_1)} vs {len(self.speaker_2)} samples); "
                "they must share one clock"
            )

        if self.sample_rate <= 0:
            raise ValueError(f"{self.conversation_id}: sample_rate must be positive")

    @property
    def duration_s(self) -> float:
        return len(self.speaker_1) / self.sample_rate

    def channel(self, speaker: str) -> torch.Tensor:
        """One speaker's isolated channel."""

        if speaker not in SPEAKERS:
            raise KeyError(f"Unknown speaker {speaker!r}; expected one of {SPEAKERS}")

        return self.speaker_1 if speaker == "speaker_1" else self.speaker_2

    def scene(self) -> torch.Tensor:
        """The global mono scene: the plain sum of the two channels."""

        return self.speaker_1 + self.speaker_2


def conversation_from_row(row: Mapping[str, Any]) -> Conversation:
    """A conversation from a dataset row, reading its inference columns only.

    Each audio column holds undecoded FLAC (`{"bytes": ...}`, as
    `datasets.Audio(decode=False)` returns it).
    """

    return conversation_from_audio(
        str(row["conversation_id"]),
        {speaker: row[column]["bytes"] for speaker, column in AUDIO_COLUMNS.items()},
    )


def conversation_from_audio(
    conversation_id: str, audio: Mapping[str, bytes | Path]
) -> Conversation:
    """A conversation from each speaker's encoded audio (bytes or a file)."""

    decoded = {speaker: decode_audio(audio[speaker]) for speaker in SPEAKERS}
    rates = {speaker: rate for speaker, (_, rate) in decoded.items()}

    if len(set(rates.values())) != 1:
        raise ValueError(f"{conversation_id}: speaker sample rates differ: {rates}")

    return Conversation(
        conversation_id=conversation_id,
        sample_rate=rates["speaker_1"],
        speaker_1=decoded["speaker_1"][0],
        speaker_2=decoded["speaker_2"][0],
    )


def decode_audio(source: bytes | Path) -> tuple[torch.Tensor, int]:
    """Mono encoded audio as (N,) float32 in [-1, 1] at its native rate."""

    file = io.BytesIO(source) if isinstance(source, bytes) else str(source)

    with av.open(file, mode="r") as container:
        stream = container.streams.audio[0]
        rate = stream.rate

        if stream.channels != 1:
            raise ValueError(f"Expected a mono channel, got {stream.channels}")

        # Sample-format conversion only: same rate and layout, no resampling.
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        chunks = [
            converted.to_ndarray().reshape(-1)
            for frame in container.decode(stream)
            for converted in resampler.resample(frame)
        ]
        chunks += [
            converted.to_ndarray().reshape(-1) for converted in resampler.resample(None)
        ]

    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    return torch.from_numpy(samples.astype(np.float32, copy=False)), rate


def scorer_revision() -> str:
    """The installed official TurnBench commit; refuses any other than the pin."""

    from importlib.metadata import distribution

    direct = json.loads(distribution("turnbench").read_text("direct_url.json") or "{}")
    commit = (direct.get("vcs_info") or {}).get("commit_id")

    if commit != REFERENCE_REVISION:
        raise RuntimeError(
            f"Installed turnbench is at {commit}, not the pinned {REFERENCE_REVISION}"
        )

    return commit


def load_dev(
    *,
    revision: str = DEV_REVISION,
    token: str | None = None,
    skip: Collection[str] = (),
) -> Iterator[Conversation]:
    """Stream the pinned DEV split, inference columns only (gated: needs a login).

    Conversations in `skip` (already extracted) are not decoded.
    """

    from datasets import Audio, load_dataset

    rows = load_dataset(
        DEV_REPO, split=DEV_SPLIT, revision=revision, streaming=True, token=token
    ).select_columns(list(INFERENCE_COLUMNS))

    for column in AUDIO_COLUMNS.values():
        rows = rows.cast_column(column, Audio(decode=False))

    for row in rows:
        if str(row["conversation_id"]) not in skip:
            yield conversation_from_row(row)


def load_train(
    *,
    revision: str = TRAIN_REVISION,
    scratch_dir: Path | None = None,
    token: str | None = None,
    skip: Collection[str] = (),
) -> Iterator[Conversation]:
    """The pinned TRAIN conversations in id order, speaker audio files only.

    One conversation at a time: its two WAV files are downloaded into a
    temporary directory (under `scratch_dir`), decoded, and deleted before the
    conversation is yielded, so the ~140 GB of audio never accumulates.
    Conversations in `skip` (already extracted) are not downloaded.
    """

    from huggingface_hub import hf_hub_download

    for conversation_id in train_conversation_ids(revision=revision, token=token):
        if conversation_id in skip:
            continue

        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
            conversation = conversation_from_audio(
                conversation_id,
                {
                    speaker: Path(
                        hf_hub_download(
                            TRAIN_REPO,
                            f"{conversation_id}/{name}",
                            repo_type="dataset",
                            revision=revision,
                            local_dir=tmp,
                            token=token,
                        )
                    )
                    for speaker, name in TRAIN_AUDIO_FILES.items()
                },
            )

        yield conversation


def train_conversation_ids(
    *, revision: str = TRAIN_REVISION, token: str | None = None
) -> list[str]:
    """TRAIN conversations holding both speaker files, in id order."""

    from huggingface_hub import HfApi

    files = set(
        HfApi(token=token).list_repo_files(
            TRAIN_REPO, repo_type="dataset", revision=revision
        )
    )
    ids = {
        name.split("/")[0]
        for name in files
        if name.count("/") == 1
        and all(
            f"{name.split('/')[0]}/{f}" in files for f in TRAIN_AUDIO_FILES.values()
        )
    }

    return sorted(
        ids,
        key=lambda name: [
            int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)
        ],
    )


def split_size(split: str, *, token: str | None = None) -> int | None:
    """Conversations in a pinned split, for progress display; None if unknown."""

    if split == "train":
        return len(train_conversation_ids(token=token))

    from huggingface_hub import HfApi

    card = HfApi(token=token).dataset_info(DEV_REPO, revision=DEV_REVISION).card_data
    info = (card.to_dict() if card is not None else {}).get("dataset_info") or {}

    for entry in info.get("splits", []) if isinstance(info, dict) else []:
        if entry.get("name") == DEV_SPLIT:
            return entry.get("num_examples")

    return None


def input_provenance(*, split: str, source_sample_rate: int) -> dict[str, Any]:
    """What produced the scene and the actions, and when they are available."""

    if split not in SOURCES:
        raise ValueError(f"Unknown TurnBench split {split!r}; expected {list(SOURCES)}")

    return {
        "turnbench_dataset": {"split": split, **SOURCES[split]},
        "turnbench_reference": {
            "repo": REFERENCE_REPO,
            "revision": REFERENCE_REVISION,
            "activity_rule": "baselines/rms_vad/predict.py (reimplemented)",
        },
        "scene_mix_policy": (
            f"{SCENE_MIX_POLICY}: speaker_1 + speaker_2, no normalization"
        ),
        "source_sample_rate": source_sample_rate,
        "mimi_target_sample_rate": MIMI_SAMPLE_RATE,
        "activity": {
            "rule": "RMS > threshold per window, on each speaker's own channel",
            "window_s": WINDOW_S,
            "hop_s": WINDOW_S,
            "threshold": RMS_THRESHOLD,
            "sample_rate": "native (no resampling)",
            "tuned_on_turnbench_labels": False,
        },
        "control_rate_hz": CONTROL_RATE_HZ,
        "preprocessing_lookahead_s": {
            "actions": 0.0,
            # The Mimi resampler reads past the slot end: measured for the
            # actual rates by the extraction (manifest timing), never 0 here.
            "mimi_scene": "measured at extraction: timing.resample_lookahead_s",
        },
        "commit_time_convention": (
            "(z_k, a_k) -> zhat_(k+1) is committed at t_k + 0.1 s + "
            "preprocessing lookahead (the slot end), never at t_k"
        ),
    }
