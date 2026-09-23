from pathlib import Path
import wave

import av
import numpy as np
import pytest

from turn_wm.data.media import MediaPaths
from turn_wm.data.reader import MediaReader


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


def test_invalid_negative_start_raises():
    reader = MediaReader()

    media = MediaPaths(
        recording_id="r1",
        audio_path=Path("audio.wav"),
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        reader.read_window(
            media,
            start_time_s=-1.0,
            end_time_s=1.0,
        )


def test_invalid_window_order_raises():
    reader = MediaReader()

    media = MediaPaths(
        recording_id="r1",
        audio_path=Path("audio.wav"),
    )

    with pytest.raises(
        ValueError,
        match="greater than",
    ):
        reader.read_window(
            media,
            start_time_s=1.0,
            end_time_s=1.0,
        )


def test_audio_window_is_decoded(
    tmp_path: Path,
):
    path = tmp_path / "audio.wav"

    make_audio(path)

    reader = MediaReader()

    window = reader.read_window(
        MediaPaths(
            recording_id="r1",
            audio_path=path,
        ),
        start_time_s=0.25,
        end_time_s=0.50,
    )

    assert window.audio is not None
    assert window.video is None

    assert window.audio.sample_rate == 16_000
    assert window.audio.waveform.shape == (
        1,
        4_000,
    )


def test_audio_is_float32(
    tmp_path: Path,
):
    path = tmp_path / "audio.wav"

    make_audio(path)

    window = MediaReader().read_window(
        MediaPaths(
            recording_id="r1",
            audio_path=path,
        ),
        start_time_s=0.0,
        end_time_s=0.2,
    )

    assert window.audio is not None

    assert window.audio.waveform.dtype.is_floating_point


def test_video_window_is_decoded(
    tmp_path: Path,
):
    path = tmp_path / "video.mp4"

    make_video(path)

    window = MediaReader().read_window(
        MediaPaths(
            recording_id="r1",
            video_path=path,
        ),
        start_time_s=0.2,
        end_time_s=0.6,
    )

    assert window.video is not None

    assert window.video.frames.ndim == 4
    assert window.video.frames.shape[1:] == (
        3,
        32,
        32,
    )

    assert window.video.frames.shape[0] == window.video.timestamps_s.shape[0]


def test_video_timestamps_are_inside_window(
    tmp_path: Path,
):
    path = tmp_path / "video.mp4"

    make_video(path)

    window = MediaReader().read_window(
        MediaPaths(
            recording_id="r1",
            video_path=path,
        ),
        start_time_s=0.2,
        end_time_s=0.6,
    )

    assert window.video is not None

    timestamps = window.video.timestamps_s

    assert (timestamps >= 0.2).all()
    assert (timestamps < 0.6).all()


def test_video_without_audio_returns_none(
    tmp_path: Path,
):
    path = tmp_path / "video.mp4"

    make_video(path)

    window = MediaReader().read_window(
        MediaPaths(
            recording_id="r1",
            video_path=path,
        ),
        start_time_s=0.0,
        end_time_s=0.5,
    )

    assert window.video is not None
    assert window.audio is None


def test_separate_audio_and_video_are_loaded(
    tmp_path: Path,
):
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"

    make_video(video_path)
    make_audio(audio_path)

    window = MediaReader().read_window(
        MediaPaths(
            recording_id="r1",
            video_path=video_path,
            audio_path=audio_path,
        ),
        start_time_s=0.1,
        end_time_s=0.4,
    )

    assert window.audio is not None
    assert window.video is not None
