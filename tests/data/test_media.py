from pathlib import Path

import pytest

from turn_wm.data.media import MediaIndex, MediaPaths


def test_media_paths_requires_at_least_one_medium():
    with pytest.raises(
        ValueError,
        match="has no audio or video",
    ):
        MediaPaths(
            recording_id="r1",
        )


def test_media_index_resolves_recording(tmp_path: Path):
    video = tmp_path / "r1.mp4"
    video.touch()

    index = MediaIndex(
        {
            "r1": MediaPaths(
                recording_id="r1",
                video_path=video,
            )
        }
    )

    media = index.get("r1")

    assert media.recording_id == "r1"
    assert media.video_path == video
    assert media.audio_path is None


def test_media_index_supports_separate_audio(tmp_path: Path):
    video = tmp_path / "r1.mp4"
    audio = tmp_path / "r1.wav"

    video.touch()
    audio.touch()

    index = MediaIndex(
        {
            "r1": MediaPaths(
                recording_id="r1",
                video_path=video,
                audio_path=audio,
            )
        }
    )

    media = index.get("r1")

    assert media.video_path == video
    assert media.audio_path == audio


def test_missing_recording_raises(tmp_path: Path):
    video = tmp_path / "r1.mp4"
    video.touch()

    index = MediaIndex(
        {
            "r1": MediaPaths(
                recording_id="r1",
                video_path=video,
            )
        }
    )

    with pytest.raises(
        KeyError,
        match="No media found",
    ):
        index.get("unknown")


def test_missing_video_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.mp4"

    with pytest.raises(
        FileNotFoundError,
        match="Video file does not exist",
    ):
        MediaIndex(
            {
                "r1": MediaPaths(
                    recording_id="r1",
                    video_path=missing,
                )
            }
        )


def test_missing_audio_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.wav"

    with pytest.raises(
        FileNotFoundError,
        match="Audio file does not exist",
    ):
        MediaIndex(
            {
                "r1": MediaPaths(
                    recording_id="r1",
                    audio_path=missing,
                )
            }
        )


def test_path_validation_can_be_disabled():
    index = MediaIndex(
        {
            "r1": MediaPaths(
                recording_id="r1",
                video_path=Path("/does/not/exist.mp4"),
            )
        },
        validate_paths=False,
    )

    assert "r1" in index


def test_media_index_length(tmp_path: Path):
    first = tmp_path / "r1.mp4"
    second = tmp_path / "r2.mp4"

    first.touch()
    second.touch()

    index = MediaIndex(
        {
            "r1": MediaPaths(
                recording_id="r1",
                video_path=first,
            ),
            "r2": MediaPaths(
                recording_id="r2",
                video_path=second,
            ),
        }
    )

    assert len(index) == 2
