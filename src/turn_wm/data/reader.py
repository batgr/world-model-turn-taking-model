from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import torch

from turn_wm.data.media import MediaPaths


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded audio window at the source sample rate."""

    waveform: torch.Tensor  # [channels, samples]
    sample_rate: int


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

    Times are on the media file's own timeline; callers convert canonical
    grid times beforehand.
    """

    def read_window(
        self,
        media: MediaPaths,
        *,
        start_time_s: float,
        end_time_s: float,
    ) -> MediaWindow:
        self._validate_window(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )

        video = None
        audio = None

        if media.video_path is not None:
            video = self._read_video(
                media.video_path,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )

        audio_source = media.audio_source

        if audio_source is not None:
            audio = self._read_audio(
                audio_source,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )

        return MediaWindow(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            audio=audio,
            video=video,
        )

    @staticmethod
    def _validate_window(
        *,
        start_time_s: float,
        end_time_s: float,
    ) -> None:
        if start_time_s < 0:
            raise ValueError("start_time_s must be non-negative")

        if end_time_s <= start_time_s:
            raise ValueError("end_time_s must be greater than start_time_s")

    def _read_video(
        self,
        path: Path,
        *,
        start_time_s: float,
        end_time_s: float,
    ) -> DecodedVideo | None:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None

            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            origin_s = self._stream_origin_s(stream)

            self._seek(
                container,
                stream,
                start_time_s=start_time_s,
            )

            frames: list[torch.Tensor] = []
            timestamps: list[float] = []

            for frame in container.decode(stream):
                frame_time_s = self._frame_time_s(
                    frame,
                    origin_s=origin_s,
                )

                if frame_time_s is None:
                    continue

                if frame_time_s < start_time_s:
                    continue

                if frame_time_s >= end_time_s:
                    break

                array = frame.to_ndarray(format="rgb24")

                tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

                frames.append(tensor)
                timestamps.append(frame_time_s)

        if not frames:
            return None

        return DecodedVideo(
            frames=torch.stack(frames),
            timestamps_s=torch.tensor(
                timestamps,
                dtype=torch.float64,
            ),
        )

    def _read_audio(
        self,
        path: Path,
        *,
        start_time_s: float,
        end_time_s: float,
    ) -> DecodedAudio | None:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return None

            stream = container.streams.audio[0]
            origin_s = self._stream_origin_s(stream)

            self._seek(
                container,
                stream,
                start_time_s=start_time_s,
            )

            chunks: list[torch.Tensor] = []
            sample_rate: int | None = None

            for frame in container.decode(stream):
                frame_time_s = self._frame_time_s(
                    frame,
                    origin_s=origin_s,
                )

                if frame_time_s is None:
                    continue

                rate = frame.sample_rate

                if rate is None:
                    raise ValueError(f"Audio frame in {path} has no sample rate")

                if sample_rate is None:
                    sample_rate = rate
                elif sample_rate != rate:
                    raise ValueError(f"Audio sample rate changed inside {path}")

                frame_end_s = frame_time_s + frame.samples / rate

                if frame_end_s <= start_time_s:
                    continue

                if frame_time_s >= end_time_s:
                    break

                array = self._audio_to_float32(frame)

                first_sample = max(
                    0,
                    math.ceil((start_time_s - frame_time_s) * rate),
                )

                last_sample = min(
                    frame.samples,
                    math.ceil((end_time_s - frame_time_s) * rate),
                )

                if last_sample <= first_sample:
                    continue

                chunks.append(
                    torch.from_numpy(
                        array[
                            :,
                            first_sample:last_sample,
                        ]
                    )
                )

        if not chunks or sample_rate is None:
            return None

        return DecodedAudio(
            waveform=torch.cat(chunks, dim=1),
            sample_rate=sample_rate,
        )

    @staticmethod
    def _audio_to_float32(
        frame: av.AudioFrame,
    ) -> np.ndarray:
        array = frame.to_ndarray()

        channels = len(frame.layout.channels)
        samples = frame.samples

        if array.shape == (channels, samples):
            pass

        elif array.size == channels * samples:
            array = array.reshape(samples, channels).transpose()

        else:
            raise ValueError(f"Unexpected decoded audio shape: {array.shape}")

        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)

            scale = float(
                max(
                    abs(info.min),
                    abs(info.max),
                )
            )

            array = array.astype(np.float32) / scale

        else:
            array = array.astype(
                np.float32,
                copy=False,
            )

        return np.ascontiguousarray(array)

    @staticmethod
    def _stream_origin_s(stream: av.Stream) -> float:
        if stream.start_time is None:
            return 0.0

        return float(stream.start_time * stream.time_base)

    @staticmethod
    def _frame_time_s(
        frame: av.AudioFrame | av.VideoFrame,
        *,
        origin_s: float,
    ) -> float | None:
        if frame.pts is None:
            return None

        return float(frame.pts * frame.time_base) - origin_s

    @staticmethod
    def _seek(
        container: av.container.InputContainer,
        stream: av.Stream,
        *,
        start_time_s: float,
    ) -> None:
        stream_start = stream.start_time if stream.start_time is not None else 0

        offset = stream_start + int(start_time_s / float(stream.time_base))

        container.seek(
            offset,
            stream=stream,
            backward=True,
            any_frame=False,
        )
