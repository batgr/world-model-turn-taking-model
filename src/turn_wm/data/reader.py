from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaModality,
    MediaPaths,
    validate_modalities,
)

# A decoded jump of at least TRUE_AUDIO_GAP_S between consecutive audio frames
# is missing audio, filled with silence; shorter ones are timestamp jitter.

TRUE_AUDIO_GAP_S = 0.100
AUDIO_GAP_NUMERICAL_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class AudioGap:
    """A decoded local gap on the media file's own timeline."""

    start_time_s: float
    end_time_s: float

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded audio window at the source sample rate."""

    waveform: torch.Tensor  # [channels, samples]
    sample_rate: int
    audio_gaps: tuple[AudioGap, ...] = ()


@dataclass(frozen=True)
class DecodedVideo:
    """Decoded RGB video window."""

    frames: torch.Tensor  # [time, channels, height, width]
    timestamps_s: torch.Tensor  # [time]


@dataclass(frozen=True)
class MediaWindow:
    """Decoded multimodal observation for a temporal interval.

    `start_time_s`/`end_time_s` are the requested bounds on the media file's
    own timeline. When the window comes from a canonical sample, the
    corresponding grid-time bounds are kept in `canonical_*_time_s`.
    """

    start_time_s: float
    end_time_s: float
    audio: DecodedAudio | None
    video: DecodedVideo | None
    canonical_start_time_s: float | None = None
    canonical_end_time_s: float | None = None


class MediaReader:
    """Decode timestamp-aligned audio/video windows using PyAV.

    Times are on the media file's own timeline (relative to each stream's
    start); callers convert canonical grid times beforehand. Only the
    selected modalities are decoded; the others are also skipped by the
    demuxer, so embedded audio can be read without touching the video
    packets of its container, and vice versa.
    """

    def read_window(
        self,
        media: MediaPaths,
        *,
        start_time_s: float,
        end_time_s: float,
        modalities: Iterable[MediaModality] = MEDIA_MODALITIES,
    ) -> MediaWindow:
        """Decode the selected modalities of `media` over a time interval.

        A selected modality the media does not provide is None, as is a
        modality that was not selected.
        """

        selected = validate_modalities(modalities)

        if start_time_s < 0:
            raise ValueError("start_time_s must be non-negative")

        if end_time_s <= start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")

        video = audio = None

        if "video" in selected and media.video_path is not None:
            video = _read_video(media.video_path, start_time_s, end_time_s)

        if "audio" in selected and media.audio_source is not None:
            audio = _read_audio(media.audio_source, start_time_s, end_time_s)

        return MediaWindow(
            start_time_s=start_time_s, end_time_s=end_time_s, audio=audio, video=video
        )


def _read_video(path: Path, start_s: float, end_s: float) -> DecodedVideo | None:
    with av.open(str(path)) as container:
        if not container.streams.video:
            return None

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        origin_s = _open_stream(container, stream, start_s)
        frames: list[torch.Tensor] = []
        timestamps: list[float] = []

        for frame in container.decode(stream):
            time_s = _frame_time_s(frame, origin_s)

            if time_s is None or time_s < start_s:
                continue

            if time_s >= end_s:
                break

            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            timestamps.append(time_s)

    if not frames:
        return None

    return DecodedVideo(
        frames=torch.stack(frames),
        timestamps_s=torch.tensor(timestamps, dtype=torch.float64),
    )


def _read_audio(path: Path, start_s: float, end_s: float) -> DecodedAudio | None:
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None

        stream = container.streams.audio[0]
        origin_s = _open_stream(container, stream, start_s)
        chunks: list[torch.Tensor] = []
        sample_rate: int | None = None
        audio_gaps: list[AudioGap] = []
        previous_end_s: float | None = None

        for frame in container.decode(stream):
            time_s = _frame_time_s(frame, origin_s)

            if time_s is None:
                continue

            rate = frame.sample_rate

            if rate is None:
                raise ValueError(f"Audio frame in {path} has no sample rate")

            if sample_rate is None:
                sample_rate = rate
            elif sample_rate != rate:
                raise ValueError(f"Audio sample rate changed inside {path}")

            frame_end_s = time_s + frame.samples / rate

            if frame_end_s <= start_s:
                # Keep the immediately preceding frame end so a window
                # beginning inside a true gap starts with silence.
                previous_end_s = frame_end_s
                continue

            if (
                previous_end_s is not None
                and time_s - previous_end_s + AUDIO_GAP_NUMERICAL_TOLERANCE_S
                >= TRUE_AUDIO_GAP_S
            ):
                gap = AudioGap(start_time_s=previous_end_s, end_time_s=time_s)
                audio_gaps.append(gap)
                silence = round(
                    max(
                        0.0, min(gap.end_time_s, end_s) - max(gap.start_time_s, start_s)
                    )
                    * rate
                )

                if silence:
                    chunks.append(torch.zeros(len(frame.layout.channels), silence))

            if time_s >= end_s:
                break

            first = max(0, math.ceil((start_s - time_s) * rate))
            last = min(frame.samples, math.ceil((end_s - time_s) * rate))
            previous_end_s = frame_end_s

            if last > first:
                chunks.append(torch.from_numpy(_audio_to_float32(frame)[:, first:last]))

    if not chunks or sample_rate is None:
        return None

    return DecodedAudio(
        waveform=torch.cat(chunks, dim=1),
        sample_rate=sample_rate,
        audio_gaps=tuple(audio_gaps),
    )


def _audio_to_float32(frame: av.AudioFrame) -> np.ndarray:
    """`(channels, samples)` float32, integer samples scaled to [-1, 1]."""

    array = frame.to_ndarray()
    channels, samples = len(frame.layout.channels), frame.samples

    if array.shape != (channels, samples):
        if array.size != channels * samples:
            raise ValueError(f"Unexpected decoded audio shape: {array.shape}")

        array = array.reshape(samples, channels).transpose()

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype.name)
        array = array.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
    else:
        array = array.astype(np.float32, copy=False)

    return np.ascontiguousarray(array)


def _open_stream(container: Any, stream: Any, start_s: float) -> float:
    """Skip every other stream, seek to `start_s`; return the stream's origin (s)."""

    for other in container.streams:
        if other.index != stream.index:
            other.discard = vars(av)["stream"].Discard.all

    start = stream.start_time if stream.start_time is not None else 0
    container.seek(
        start + int(start_s / float(stream.time_base)),
        stream=stream,
        backward=True,
        any_frame=False,
    )

    return float(start * stream.time_base)


def _frame_time_s(
    frame: av.AudioFrame | av.VideoFrame, origin_s: float
) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None

    return float(frame.pts * frame.time_base) - origin_s
