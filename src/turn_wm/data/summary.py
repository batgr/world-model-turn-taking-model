"""
The text summary `turn-wm inspect-data` prints: sources, corpora, one batch.

`format_summary` describes what the training data path produced: the
loaded corpora and splits, the window, the tensor shapes of one batch, the
vocabularies, and the media of the first sample when media is decoded.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    PAD_ACTION_ID,
    PAD_STATE_ID,
    STATE_TO_ID,
    WindowConfig,
)
from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaIndex,
    MediaModality,
    MediaPaths,
)
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.reader import MediaWindow
from turn_wm.data.source import (
    HuggingFaceSource,
    LoadedCorpus,
    LoadedData,
)

_CONTEXT_KEYS = ("context_state", "context_action", "context_valid", "context_mask")
_FUTURE_KEYS = ("future_state", "future_action", "future_valid")


def format_summary(
    *,
    source: HuggingFaceSource,
    split: str,
    data: LoadedData,
    dataset: MultiCorpusDataset,
    window: WindowConfig,
    batch: dict[str, Any],
    media_index: MediaIndex | None = None,
    modalities: tuple[MediaModality, ...] = MEDIA_MODALITIES,
) -> str:
    """Render a concise structural summary of one inspected batch."""

    lengths = batch["context_lengths"]
    usable = dataset.corpus_sizes()
    not_in_split = [name for name in data.names if name not in usable]

    lines = [
        *_section("Dataset"),
        f"source: {source.repo_id}",
        f"revision: {source.revision or 'default branch (unpinned)'}",
        f"resolved commit: {data.revision or 'unknown'}",
        f"split: {split}",
        "corpora:",
        *[f"  - {name}" for name in dataset.corpora],
        *([f"not in this split: {', '.join(not_in_split)}"] if not_in_split else []),
        f"usable samples: {len(dataset):,}",
        "",
    ]

    for name in dataset.corpora:
        lines += _corpus_lines(data.corpus(name), split=split, usable=usable[name])

    lines += [
        *_section("Window"),
        f"context: {window.min_context_steps}–{window.max_context_steps} steps",
        f"future: {window.future_steps} steps",
        "",
        *_section("Batch"),
        f"batch size: {batch['context_state'].shape[0]}",
        "datasets:",
        *[
            f"  {name}: {count}"
            for name, count in Counter(batch["dataset"]).most_common()
        ],
        *_shape_lines(batch, _CONTEXT_KEYS),
        "",
        *_shape_lines(batch, _FUTURE_KEYS),
        "",
        "context lengths:",
        f"min: {int(lengths.min())}",
        f"max: {int(lengths.max())}",
        "",
        *_section("First sample"),
        f"dataset: {batch['dataset'][0]}",
        f"sample_id: {batch['sample_id'][0]}",
        f"recording_id: {batch['recording_id'][0]}",
        f"anchor_idx: {int(batch['anchor_idx'][0])}",
        f"sample_class: {batch['sample_class'][0]}",
        f"context_length: {int(lengths[0])}",
        "",
        *_media_lines(batch, media_index, modalities),
        *_section("States"),
        *_vocabulary_lines({**STATE_TO_ID, "PAD": PAD_STATE_ID}),
        "",
        *_section("Actions"),
        *_vocabulary_lines(
            {**ACTION_TO_ID, "MASKED": MASKED_ACTION_ID, "PAD": PAD_ACTION_ID}
        ),
    ]

    return "\n".join(lines)


def _corpus_lines(corpus: LoadedCorpus, *, split: str, usable: int) -> list[str]:
    metadata = corpus.metadata

    return [
        *_section(f"Corpus: {corpus.name}"),
        f"samples: {len(corpus.model_ready[split]):,}",
        f"usable (is_trainable): {usable:,}",
        f"recordings: {_count(_metadata_value(metadata, 'splits', 'recordings', split))}",
        f"action grid rows: {len(corpus.action_grid):,}",
        f"grid: {_frequency(_metadata_value(metadata, 'grid', 'frequency_hz'))}",
        "",
    ]


def _media_lines(
    batch: dict[str, Any],
    media_index: MediaIndex | None,
    modalities: tuple[MediaModality, ...],
) -> list[str]:
    if media_index is None or "context_media" not in batch:
        return []

    media = media_index.get(
        dataset=batch["dataset"][0],
        recording_id=batch["recording_id"][0],
    )

    return [
        *_section("Media"),
        f"dataset: {media.dataset}",
        f"recording: {media.recording_id}",
        f"modalities: {', '.join(modalities)}",
        f"video file: {media.video_path or 'none'}",
        f"audio file: {media.audio_path or 'none'}",
        f"media offset: {media.media_offset_s:g} s",
        "",
        "context:",
        *_window_lines(batch["context_media"][0], media),
        "",
        "future:",
        *_window_lines(batch["future_media"][0], media),
        "",
    ]


def _window_lines(window: MediaWindow, media: MediaPaths) -> list[str]:
    lines = [
        (
            "  canonical time: "
            f"{_seconds(window.canonical_start_time_s)} → "
            f"{_seconds(window.canonical_end_time_s)}"
        ),
        (
            "  physical media time: "
            f"{_seconds(window.start_time_s)} → {_seconds(window.end_time_s)}"
        ),
        "  audio:",
    ]

    audio = window.audio

    if audio is None:
        lines.append("    present: no")
    else:
        lines += [
            "    present: yes",
            f"    source: {_audio_source(media)}",
            f"    sample rate: {audio.sample_rate} Hz",
            f"    shape (channels, samples): {tuple(audio.waveform.shape)}",
        ]

    lines.append("  video:")

    video = window.video

    if video is None:
        lines.append("    present: no")
    else:
        timestamps = video.timestamps_s
        lines += [
            "    present: yes",
            f"    frames: {video.frames.shape[0]}",
            f"    shape (T, C, H, W): {tuple(video.frames.shape)}",
            f"    first timestamp: {_seconds(float(timestamps[0]))}",
            f"    last timestamp: {_seconds(float(timestamps[-1]))}",
        ]

    return lines


def _audio_source(media: MediaPaths) -> str:
    if media.audio_path is not None:
        return "dedicated audio file"

    if media.video_has_audio is None:
        return "embedded video (unprobed)"

    return "embedded video"


def _seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f} s"


def _section(title: str) -> list[str]:
    return [title, "-" * len(title)]


def _shape_lines(batch: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    width = max(len(key) for key in _CONTEXT_KEYS + _FUTURE_KEYS) + 1

    return [f"{key + ':':<{width}} {tuple(batch[key].shape)}" for key in keys]


def _vocabulary_lines(vocabulary: dict[str, int]) -> list[str]:
    return [
        f"{index} {name}"
        for name, index in sorted(vocabulary.items(), key=lambda item: item[1])
    ]


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    value: Any = metadata

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None

        value = value[key]

    return value


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "unknown (not in metadata)"


def _frequency(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:g} Hz"

    return "unknown (not in metadata)"
