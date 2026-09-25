from pathlib import Path

import pytest
from synthetic_media import (
    make_audio,
    make_audio_with_timestamps,
    make_video,
    make_video_with_audio,
)

from turn_wm.data.media import MediaPaths
from turn_wm.data.reader import MediaReader, MediaWindow


def test_invalid_negative_start_raises():
    reader = MediaReader()

    media = MediaPaths(
        dataset="synthetic",
        recording_id="r1",
        audio_path=Path("audio.wav"),
    )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        reader.read_window(
            media,
            start_time_s=-0.0013,
            end_time_s=1.0,
        )


def test_invalid_window_order_raises():
    reader = MediaReader()

    media = MediaPaths(
        dataset="synthetic",
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
            dataset="synthetic",
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
            dataset="synthetic",
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
            dataset="synthetic",
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
            dataset="synthetic",
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
            dataset="synthetic",
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
            dataset="synthetic",
            recording_id="r1",
            video_path=video_path,
            audio_path=audio_path,
        ),
        start_time_s=0.1,
        end_time_s=0.4,
    )

    assert window.audio is not None
    assert window.video is not None


def test_embedded_audio_is_decoded_when_video_has_audio(tmp_path: Path):
    path = tmp_path / "video.mp4"
    make_video_with_audio(path)

    window = MediaReader().read_window(
        MediaPaths(
            dataset="synthetic",
            recording_id="r1",
            video_path=path,
            video_has_audio=True,
        ),
        start_time_s=0.2,
        end_time_s=0.6,
    )

    assert window.video is not None
    assert window.audio is not None
    assert window.audio.sample_rate == 16_000
    assert window.audio.waveform.shape[1] > 0


def test_audio_is_absent_when_video_has_no_audio(tmp_path: Path):
    path = tmp_path / "video.mp4"
    make_video_with_audio(path)

    window = MediaReader().read_window(
        MediaPaths(
            dataset="synthetic",
            recording_id="r1",
            video_path=path,
            video_has_audio=False,
        ),
        start_time_s=0.2,
        end_time_s=0.6,
    )

    # The container carries audio, but the manifest says not to use it.
    assert window.video is not None
    assert window.audio is None


@pytest.fixture
def separate_media(tmp_path: Path) -> MediaPaths:
    make_video(tmp_path / "video.mp4")
    make_audio(tmp_path / "audio.wav")

    return MediaPaths(
        dataset="synthetic",
        recording_id="r1",
        video_path=tmp_path / "video.mp4",
        audio_path=tmp_path / "audio.wav",
    )


def test_default_decodes_audio_and_video(separate_media, decode_spies):
    window = MediaReader().read_window(separate_media, start_time_s=0.1, end_time_s=0.4)

    assert window.audio is not None
    assert window.video is not None
    assert decode_spies == {
        "audio": [separate_media.audio_path],
        "video": [separate_media.video_path],
    }


def test_audio_only_never_decodes_video(separate_media, decode_spies):
    window = MediaReader().read_window(
        separate_media, start_time_s=0.1, end_time_s=0.4, modalities=("audio",)
    )

    assert window.audio is not None
    assert window.video is None
    assert decode_spies == {"audio": [separate_media.audio_path], "video": []}


def test_video_only_never_decodes_audio(separate_media, decode_spies):
    window = MediaReader().read_window(
        separate_media, start_time_s=0.1, end_time_s=0.4, modalities=("video",)
    )

    assert window.audio is None
    assert window.video is not None
    assert decode_spies == {"audio": [], "video": [separate_media.video_path]}


def embedded(path: Path) -> MediaPaths:
    return MediaPaths(
        dataset="synthetic",
        recording_id="r1",
        video_path=path,
        video_has_audio=True,
    )


def test_audio_only_reads_embedded_audio_without_video_frames(
    tmp_path: Path, decode_spies
):
    path = tmp_path / "video.mp4"
    make_video_with_audio(path)

    window = MediaReader().read_window(
        embedded(path), start_time_s=0.2, end_time_s=0.6, modalities=("audio",)
    )

    # Audio comes from the video container, but no video frame is decoded.
    assert window.audio is not None
    assert window.audio.sample_rate == 16_000
    assert window.audio.waveform.shape[1] > 0
    assert window.video is None
    assert decode_spies == {"audio": [path], "video": []}


def test_video_only_skips_embedded_audio(tmp_path: Path, decode_spies):
    path = tmp_path / "video.mp4"
    make_video_with_audio(path)

    window = MediaReader().read_window(
        embedded(path), start_time_s=0.2, end_time_s=0.6, modalities=("video",)
    )

    assert window.audio is None
    assert window.video is not None
    assert window.video.frames.shape[0] > 0
    assert decode_spies == {"audio": [], "video": [path]}


def test_single_stream_decode_matches_full_decode(tmp_path: Path):
    # The unselected stream is discarded by the demuxer; decoded content of
    # the selected one must not change.
    path = tmp_path / "video.mp4"
    make_video_with_audio(path)
    reader = MediaReader()

    both = reader.read_window(embedded(path), start_time_s=0.2, end_time_s=0.6)
    audio = reader.read_window(
        embedded(path), start_time_s=0.2, end_time_s=0.6, modalities=("audio",)
    )
    video = reader.read_window(
        embedded(path), start_time_s=0.2, end_time_s=0.6, modalities=("video",)
    )

    assert both.audio is not None and audio.audio is not None
    assert both.video is not None and video.video is not None
    assert audio.audio.waveform.equal(both.audio.waveform)
    assert video.video.frames.equal(both.video.frames)
    assert video.video.timestamps_s.equal(both.video.timestamps_s)


def test_requested_audio_missing_from_media_is_none(tmp_path: Path, decode_spies):
    path = tmp_path / "video.mp4"
    make_video(path)

    window = MediaReader().read_window(
        MediaPaths(dataset="synthetic", recording_id="r1", video_path=path),
        start_time_s=0.1,
        end_time_s=0.4,
        modalities=("audio",),
    )

    # Existing contract: unavailable media is None; video is not substituted.
    assert window.audio is None
    assert window.video is None
    assert decode_spies["video"] == []


@pytest.mark.parametrize(
    "modalities",
    [(), ("text",), ("audio", "depth"), ("audio", "audio")],
)
def test_reader_rejects_invalid_modalities(separate_media, modalities):
    with pytest.raises(ValueError, match="modalit"):
        MediaReader().read_window(
            separate_media, start_time_s=0.1, end_time_s=0.4, modalities=modalities
        )


def positions(window: MediaWindow) -> list[int]:
    """Intended sample index stored in each decoded sample (0 = silence)."""

    assert window.audio is not None
    return (window.audio.waveform[0] * 32_768 / 10).round().long().tolist()


def read_audio_window(path: Path, start: float, end: float) -> MediaWindow:
    return MediaReader().read_window(
        MediaPaths(dataset="synthetic", recording_id="r1", audio_path=path),
        start_time_s=start,
        end_time_s=end,
        modalities=("audio",),
    )


def read_positions(path: Path, start: float, end: float) -> list[int]:
    return positions(read_audio_window(path, start, end))


def test_audio_after_a_stream_gap_keeps_its_time(tmp_path: Path):
    # 3 s at 1 kHz in 0.1 s frames, frames 1.0-1.5 s missing (Ego4D-like).
    path = tmp_path / "gap.mkv"
    make_audio_with_timestamps(path, [i * 100 for i in range(30) if not 10 <= i < 15])

    window = read_audio_window(path, 0.5, 2.5)
    samples = positions(window)

    # The window keeps its duration; the gap is silence, not skipped.
    assert len(samples) == 2_000
    assert samples[:500] == list(range(500, 1_000))
    assert samples[500:1_000] == [0] * 500
    assert samples[1_000:] == list(range(1_500, 2_500))
    assert window.audio is not None
    [gap] = window.audio.audio_gaps
    assert (gap.start_time_s, gap.end_time_s, gap.duration_s) == pytest.approx(
        (1.0, 1.5, 0.5)
    )


def test_window_starting_inside_a_gap_starts_with_silence(tmp_path: Path):
    path = tmp_path / "gap.mkv"
    make_audio_with_timestamps(path, [i * 100 for i in range(30) if not 10 <= i < 15])

    samples = read_positions(path, 1.2, 1.8)

    assert samples[:300] == [0] * 300
    assert samples[300:] == list(range(1_500, 1_800))


def test_sub_threshold_overlap_is_concatenated_as_jitter(tmp_path: Path):
    path = tmp_path / "overlap.mkv"
    make_audio_with_timestamps(path, [0, 100, 150, 250, 350])

    window = read_audio_window(path, 0.0, 0.45)

    assert positions(window) == [
        *range(100),
        *range(100, 200),
        *range(150, 250),
        *range(250, 350),
        *range(350, 450),
    ]
    assert window.audio is not None
    assert window.audio.audio_gaps == ()


def test_clean_audio_is_unchanged_and_has_no_gaps(tmp_path: Path):
    path = tmp_path / "clean.mkv"
    make_audio_with_timestamps(path, [0, 100, 200])

    window = read_audio_window(path, 0.0, 0.3)

    assert positions(window) == list(range(300))
    assert window.audio is not None
    assert window.audio.audio_gaps == ()


def test_local_timestamp_jitter_is_concatenated_without_silence(tmp_path: Path):
    path = tmp_path / "jitter.mkv"
    make_audio_with_timestamps(path, [0, 103, 196, 308])

    window = read_audio_window(path, 0.0, 0.408)

    assert positions(window) == [
        *range(100),
        *range(103, 203),
        *range(196, 296),
        *range(308, 408),
    ]
    assert window.audio is not None
    assert window.audio.audio_gaps == ()


@pytest.mark.parametrize("gap_samples", [101, 250])
def test_true_local_gap_is_exact_silence_and_metadata(
    tmp_path: Path,
    gap_samples: int,
):
    path = tmp_path / f"gap-{gap_samples}.mkv"
    next_start = 200 + gap_samples
    make_audio_with_timestamps(path, [0, 100, next_start])

    window = read_audio_window(path, 0.0, (next_start + 100) / 1_000)
    samples = positions(window)

    assert samples[:200] == list(range(200))
    assert samples[200 : 200 + gap_samples] == [0] * gap_samples
    assert samples[200 + gap_samples :] == list(range(next_start, next_start + 100))
    assert window.audio is not None
    [gap] = window.audio.audio_gaps
    assert (gap.start_time_s, gap.end_time_s, gap.duration_s) == pytest.approx(
        (0.2, next_start / 1_000, gap_samples / 1_000)
    )


def test_gap_just_below_threshold_is_jitter(tmp_path: Path):
    path = tmp_path / "gap-099.mkv"
    make_audio_with_timestamps(path, [0, 100, 299])

    window = read_audio_window(path, 0.0, 0.399)

    assert len(positions(window)) == 300
    assert window.audio is not None
    assert window.audio.audio_gaps == ()


def test_large_cumulative_drift_does_not_create_a_gap(tmp_path: Path):
    path = tmp_path / "cumulative-drift.mkv"
    make_audio_with_timestamps(path, [150 * index for index in range(10)])

    window = read_audio_window(path, 0.0, 1.45)

    assert len(positions(window)) == 1_000
    assert window.audio is not None
    assert window.audio.audio_gaps == ()
