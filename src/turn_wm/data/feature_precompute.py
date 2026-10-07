"""
Precompute a frozen encoder's features for whole recordings, on the action grid.

For every recording of the action grid:

    recording audio (canonical span, exact duration, its own sample rate)
          ↓ the encoder (Mimi: resampled once to 24 kHz, then streamed with
          ↓ persistent caches)
    one feature row per grid step: frame k is step start_index + k

The encoder runs at the grid's rate, so no feature is resampled or realigned;
no projector or other learned transform is applied.

Every recording's grid and media are checked before the first one is encoded,
so an inconsistent input fails in seconds rather than hours into a run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import torch
import torch.nn.functional as F

from turn_wm.data.feature_cache import (
    CachedAudioGap,
    CacheExclusion,
    FeatureRecord,
    write_features,
    write_manifest,
)
from turn_wm.data.media import MediaIndex, MediaPaths
from turn_wm.data.reader import MediaReader
from turn_wm.data.source import LoadedCorpus, LoadedData
from turn_wm.models.encoders.base import Encoder

# Audio missing before a recording's first grid step that may be filled with
# silence: 100 ms (a manifest offset slightly before the media start).
MAX_AUDIO_PREFIX_GAP_S = 0.1

# Audio missing after the last grid step that may be filled with silence.
# Published grids run to the next whole second past the media (EgoCom: up to
# 0.998 s, in 144 of 175 recordings), and label those steps UNKNOWN; larger
# gaps mean the span and the media disagree.
MAX_AUDIO_TAIL_GAP_S = 1.0

# V1 keeps true local gaps as silence, but these three Ego4D recordings also
# have a separate, slowly accumulated decoded-sample/PTS discrepancy spanning
# several 100 ms action-grid cells. The values are maxima measured from decoded
# frames after true local gaps (>= 100 ms) are classified separately. No clock
# correction is inferred; the affected recordings are omitted explicitly.
EGO4D_V1_EXCLUSIONS = (
    CacheExclusion(
        dataset="ego4d",
        recording_id="85506322-449a-45c0-b77a-48a8077b4bbd",
        reason="audio_annotation_clock_drift",
        max_drift_s=0.44265625,
    ),
    CacheExclusion(
        dataset="ego4d",
        recording_id="b3ef3563-ecc0-4a15-9a7b-4feb4558d953",
        reason="audio_annotation_clock_drift",
        max_drift_s=0.476,
    ),
    CacheExclusion(
        dataset="ego4d",
        recording_id="ba5b1882-c9d7-48e7-85fe-2c7b10494fac",
        reason="audio_annotation_clock_drift",
        max_drift_s=0.52534375,
    ),
)

type Progress = Callable[[int, int, "RecordingSpan"], None]


@dataclass(frozen=True)
class PreparedRecordingAudio:
    """Exact-duration encoder input and its canonical true-gap metadata."""

    waveform: torch.Tensor  # (channels, samples) at sample_rate
    sample_rate: int
    audio_gaps: tuple[CachedAudioGap, ...]


@dataclass(frozen=True)
class RecordingSpan:
    """Contiguous action-grid steps of one recording."""

    dataset: str
    recording_id: str
    start_index: int
    start_time_s: float
    steps: int
    grid_rate_hz: float

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.steps / self.grid_rate_hz


def _recording_spans(corpus: LoadedCorpus, *, grid_rate: float) -> list[RecordingSpan]:
    """One span per recording; refuses grids with gaps or irregular timing."""

    # with_format("arrow")[:] honours any filter applied to the grid, unlike
    # the underlying .data.table.
    table = cast(
        pa.Table,
        corpus.action_grid.select_columns(
            ["dataset", "recording_id", "decision_index", "decision_time_s"]
        ).with_format("arrow")[:],
    )

    grouped = table.group_by(["dataset", "recording_id"]).aggregate(
        [
            ("decision_index", "min"),
            ("decision_index", "max"),
            ("decision_index", "count"),
            ("decision_time_s", "min"),
            ("decision_time_s", "max"),
        ]
    )

    spans = []

    for row in grouped.to_pylist():
        start_index = int(row["decision_index_min"])
        end_index = int(row["decision_index_max"])
        steps = int(row["decision_index_count"])

        if end_index - start_index + 1 != steps:
            raise ValueError(f"Non-contiguous action grid for {row['recording_id']!r}")

        start_time = float(row["decision_time_s_min"])
        last_time = float(row["decision_time_s_max"])

        expected_last = start_time + (steps - 1) / grid_rate

        if not math.isclose(last_time, expected_last, abs_tol=1e-5):
            raise ValueError(
                f"Action-grid timing mismatch for {row['recording_id']!r}: "
                f"{last_time} != {expected_last}"
            )

        spans.append(
            RecordingSpan(
                dataset=str(row["dataset"]),
                recording_id=str(row["recording_id"]),
                start_index=start_index,
                start_time_s=start_time,
                steps=steps,
                grid_rate_hz=grid_rate,
            )
        )

    return sorted(spans, key=lambda span: (span.dataset, span.recording_id))


def _load_recording_audio(
    *, reader: MediaReader, media: MediaPaths, span: RecordingSpan, grid_rate: float
) -> PreparedRecordingAudio:
    """Exact-duration encoder input and true gaps on the canonical timeline."""

    canonical_start = span.start_time_s
    canonical_end = canonical_start + span.steps / grid_rate

    media_start = media.to_media_time(canonical_start)
    media_end = media.to_media_time(canonical_end)

    # A manifest may place canonical t=0 slightly before media t=0; that
    # part has no audio and is filled with silence. The reader never gets a
    # negative time.
    prefix_seconds = max(0.0, -media_start)
    read_start = max(0.0, media_start)

    if prefix_seconds > MAX_AUDIO_PREFIX_GAP_S:
        raise ValueError(
            f"{media.key!r} starts {prefix_seconds:.3f} s before its media; "
            f"at most {MAX_AUDIO_PREFIX_GAP_S:g} s can be filled with silence"
        )

    if media_end <= read_start:
        raise ValueError(f"No usable audio interval for {media.key!r}")

    window = reader.read_window(
        media, start_time_s=read_start, end_time_s=media_end, modalities=("audio",)
    )

    if window.audio is None:
        raise ValueError(f"No audio for {media.key!r}")

    waveform = window.audio.waveform
    sample_rate = window.audio.sample_rate

    audio_gaps = tuple(
        CachedAudioGap(
            start_time_s=max(canonical_start, gap.start_time_s - media.media_offset_s),
            end_time_s=min(canonical_end, gap.end_time_s - media.media_offset_s),
        )
        for gap in window.audio.audio_gaps
        if gap.end_time_s - media.media_offset_s > canonical_start
        and gap.start_time_s - media.media_offset_s < canonical_end
    )

    if prefix_seconds > 0:
        waveform = F.pad(waveform, (round(prefix_seconds * sample_rate), 0))

    # Exact canonical duration: a small shortfall at the end is silence.
    expected_samples = round(span.steps / grid_rate * sample_rate)
    current_samples = waveform.shape[-1]
    missing_s = (expected_samples - current_samples) / sample_rate

    if missing_s > MAX_AUDIO_TAIL_GAP_S:
        raise ValueError(
            f"{media.key!r} has {missing_s:.3f} s less audio than its "
            f"{span.steps} grid steps; at most {MAX_AUDIO_TAIL_GAP_S:g} s can be "
            "filled with silence"
        )

    if current_samples < expected_samples:
        waveform = F.pad(waveform, (0, expected_samples - current_samples))
    elif current_samples > expected_samples:
        waveform = waveform[..., :expected_samples]

    return PreparedRecordingAudio(
        waveform=waveform, sample_rate=sample_rate, audio_gaps=audio_gaps
    )


def _encode_recording(
    *, encoder: Encoder, audio: PreparedRecordingAudio, steps: int, chunk_seconds: float
) -> torch.Tensor:
    """`(steps, output_dim)` features: frame k of the encoder is grid step k."""

    features = encoder.encode_recording(
        audio.waveform, audio.sample_rate, chunk_seconds=chunk_seconds
    )

    # A partial last frame (audio not a whole number of frames) is dropped.
    if features.ndim != 2 or features.shape[0] < steps:
        raise ValueError(
            f"Expected at least {steps} frames, got shape {tuple(features.shape)}"
        )

    return features[:steps]


def precompute_features(
    loaded: LoadedData,
    *,
    encoder: Encoder,
    media_roots: dict[str, Path],
    output_root: Path,
    chunk_seconds: float = 20.0,
    device: str = "cpu",
    progress: Progress | None = None,
    excluded_recordings: tuple[CacheExclusion, ...] | None = None,
) -> Path:
    """Write one feature file per recording and the manifest; return its path.

    `output_root` must be absent or empty. `progress(index, total, span)` is
    called before each recording is encoded. By default, the evidence-backed
    Ego4D V1 exclusions are applied when Ego4D is present; callers may pass an
    explicit tuple for another release or a synthetic test.

    The encoder must read audio and run at the loaded data's decision grid
    rate: its frame `k` is the recording's grid step `start_index + k`.
    """

    grid_rate = loaded.grid_rate_hz

    if encoder.modality != "audio":
        raise ValueError(f"{encoder.name} reads {encoder.modality}, not audio")

    if not math.isclose(encoder.frame_rate, grid_rate):
        raise ValueError(
            f"{encoder.name} gives {encoder.frame_rate:g} Hz frames; the loaded "
            f"data's decision grid is {grid_rate:g} Hz: an encoder runs on a "
            "grid at its own frame rate"
        )

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    output_root = Path(output_root)

    if output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    ):
        raise ValueError(
            f"Output {output_root} already exists and is not an empty directory; "
            "choose a new cache directory"
        )

    corpus_spans: list[tuple[LoadedCorpus, list[RecordingSpan]]] = []
    for corpus in loaded.corpora:
        if corpus.media_manifest is None:
            raise ValueError(f"{corpus.name!r} has no media manifest")

        corpus_spans.append((corpus, _recording_spans(corpus, grid_rate=grid_rate)))

    canonical_keys = {
        (span.dataset, span.recording_id) for _, spans in corpus_spans for span in spans
    }
    loaded_datasets = {dataset for dataset, _ in canonical_keys}
    configured_exclusions = (
        tuple(
            exclusion
            for exclusion in EGO4D_V1_EXCLUSIONS
            if exclusion.dataset in loaded_datasets
        )
        if excluded_recordings is None
        else excluded_recordings
    )
    exclusions_by_key: dict[tuple[str, str], CacheExclusion] = {}

    for exclusion in configured_exclusions:
        key = (exclusion.dataset, exclusion.recording_id)
        if key in exclusions_by_key:
            raise ValueError(f"Duplicate feature cache exclusion for {key!r}")
        exclusions_by_key[key] = exclusion

    unknown_exclusions = sorted(set(exclusions_by_key) - canonical_keys)
    if unknown_exclusions:
        raise ValueError(
            f"feature cache exclusions are not in the canonical action grid: "
            f"{unknown_exclusions!r}"
        )

    # Check every retained recording's media before any encoding.
    jobs: list[tuple[RecordingSpan, MediaPaths]] = []

    for corpus, spans in corpus_spans:
        assert corpus.media_manifest is not None
        media_index = MediaIndex.from_manifest(corpus.media_manifest, media_roots)

        for span in spans:
            if (span.dataset, span.recording_id) in exclusions_by_key:
                continue

            media = media_index.get(
                dataset=span.dataset, recording_id=span.recording_id
            )

            if media.audio_source is None:
                raise ValueError(f"No audio source for {media.key!r}")

            jobs.append((span, media))

    encoder.to(device)
    encoder.eval()

    reader = MediaReader()
    records: list[FeatureRecord] = []

    for index, (span, media) in enumerate(jobs, start=1):
        if progress is not None:
            progress(index, len(jobs), span)

        prepared = _load_recording_audio(
            reader=reader, media=media, span=span, grid_rate=grid_rate
        )

        features = _encode_recording(
            encoder=encoder,
            audio=prepared,
            steps=span.steps,
            chunk_seconds=chunk_seconds,
        )

        path = write_features(
            output_root,
            dataset=span.dataset,
            recording_id=span.recording_id,
            features=features,
        )

        records.append(
            FeatureRecord(
                dataset=span.dataset,
                recording_id=span.recording_id,
                path=str(path.relative_to(output_root)),
                steps=span.steps,
                start_index=span.start_index,
                start_time_s=span.start_time_s,
                audio_gaps=prepared.audio_gaps,
            )
        )

    return write_manifest(
        output_root,
        recordings=records,
        model_name=encoder.name,
        model_revision=encoder.revision,
        model_resolved_revision=encoder.resolved_revision,
        source_dataset_revision=loaded.revision,
        feature_rate_hz=grid_rate,
        feature_dim=encoder.output_dim,
        excluded_recordings=tuple(
            exclusions_by_key[key] for key in sorted(exclusions_by_key)
        ),
    )
