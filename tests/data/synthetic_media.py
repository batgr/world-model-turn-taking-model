"""Tiny synthetic media files shared by reader and dataset tests."""

import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def make_audio(
    path: Path,
    *,
    sample_rate: int = 16_000,
    duration_s: float = 1.0,
) -> None:
    samples = int(sample_rate * duration_s)

    signal = np.zeros(
        samples,
        dtype=np.int16,
    )

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(signal.tobytes())


def make_video(
    path: Path,
    *,
    fps: int = 10,
    frames: int = 10,
) -> None:
    container = av.open(
        str(path),
        mode="w",
    )

    stream = container.add_stream(
        "mpeg4",
        rate=fps,
    )

    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"

    for index in range(frames):
        array = np.full(
            (32, 32, 3),
            index,
            dtype=np.uint8,
        )

        frame = av.VideoFrame.from_ndarray(
            array,
            format="rgb24",
        )

        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def make_video_with_audio(
    path: Path,
    *,
    duration_s: float = 1.0,
    fps: int = 10,
    sample_rate: int = 16_000,
) -> None:
    container = av.open(str(path), mode="w")

    video = container.add_stream("mpeg4", rate=fps)
    video.width = 32
    video.height = 32
    video.pix_fmt = "yuv420p"

    audio = container.add_stream("aac", rate=sample_rate, layout="mono")

    for index in range(int(duration_s * fps)):
        frame = av.VideoFrame.from_ndarray(
            np.full((32, 32, 3), index, dtype=np.uint8),
            format="rgb24",
        )
        for packet in video.encode(frame):
            container.mux(packet)

    chunk = 1024
    total = int(duration_s * sample_rate)

    for start in range(0, total, chunk):
        samples = np.zeros((1, min(chunk, total - start)), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = start
        for packet in audio.encode(frame):
            container.mux(packet)

    for stream in (video, audio):
        for packet in stream.encode():
            container.mux(packet)

    container.close()


def make_audio_with_timestamps(
    path: Path,
    frame_starts: list[int],
    *,
    sample_rate: int = 1_000,
    frame_samples: int = 100,
) -> None:
    """PCM audio whose frames start at `frame_starts` (in samples).

    Every sample stores its own intended position (value = 10 * index), so a
    reader that shifts audio in time is caught. Missing starts are gaps in the
    stream; starts closer than `frame_samples` overlap.
    """

    container = av.open(str(path), mode="w")
    stream = container.add_stream("pcm_s16le", rate=sample_rate, layout="mono")

    for start in frame_starts:
        positions = np.arange(start, start + frame_samples)
        frame = av.AudioFrame.from_ndarray(
            (positions[None, :] * 10).astype(np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        frame.pts = start
        frame.time_base = Fraction(1, sample_rate)

        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
