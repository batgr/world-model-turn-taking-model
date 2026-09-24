from pathlib import Path

import pytest
from datasets import Dataset

from turn_wm.data.media import MediaIndex, MediaPaths


def make_manifest(*rows: dict) -> Dataset:
    defaults = {
        "dataset": "corpus_a",
        "recording_id": "r1",
        "video_path": "videos/r1.mp4",
        "audio_path": None,
        "media_offset_s": 0.0,
        "video_has_audio": True,
    }
    records = [{**defaults, **row} for row in rows or [{}]]

    return Dataset.from_list(records)


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_media_paths_requires_at_least_one_medium():
    with pytest.raises(ValueError, match="has no audio or video"):
        MediaPaths(dataset="corpus_a", recording_id="r1")


def test_media_paths_rejects_non_finite_offset():
    with pytest.raises(ValueError, match="non-finite media offset"):
        MediaPaths(
            dataset="corpus_a",
            recording_id="r1",
            video_path=Path("r1.mp4"),
            media_offset_s=float("nan"),
        )


def test_lookup_by_dataset_and_recording(tmp_path: Path):
    video = touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(make_manifest(), {"corpus_a": tmp_path})

    media = index.get(dataset="corpus_a", recording_id="r1")

    assert media.key == ("corpus_a", "r1")
    assert media.video_path == video
    assert ("corpus_a", "r1") in index


def test_same_recording_id_in_two_datasets_does_not_collide(tmp_path: Path):
    touch(tmp_path / "a/videos/r1.mp4")
    touch(tmp_path / "b/videos/r1.mp4")

    index = MediaIndex.from_manifest(
        make_manifest(
            {"dataset": "corpus_a"},
            {"dataset": "corpus_b", "media_offset_s": 5.0},
        ),
        {"corpus_a": tmp_path / "a", "corpus_b": tmp_path / "b"},
    )

    first = index.get(dataset="corpus_a", recording_id="r1")
    second = index.get(dataset="corpus_b", recording_id="r1")

    assert len(index) == 2
    assert first.video_path == tmp_path / "a/videos/r1.mp4"
    assert second.video_path == tmp_path / "b/videos/r1.mp4"
    assert (first.media_offset_s, second.media_offset_s) == (0.0, 5.0)


def test_duplicate_key_is_rejected():
    record = MediaPaths(
        dataset="corpus_a",
        recording_id="r1",
        video_path=Path("r1.mp4"),
    )

    with pytest.raises(ValueError, match="Duplicate media record"):
        MediaIndex([record, record])


def test_relative_video_and_audio_paths_resolve_under_root(tmp_path: Path):
    touch(tmp_path / "videos/r1.mp4")
    touch(tmp_path / "audio/r1.wav")

    index = MediaIndex.from_manifest(
        make_manifest({"audio_path": "audio/r1.wav"}),
        {"corpus_a": tmp_path},
    )

    media = index.get(dataset="corpus_a", recording_id="r1")

    assert media.video_path == tmp_path / "videos/r1.mp4"
    assert media.audio_path == tmp_path / "audio/r1.wav"


def test_dedicated_audio_path_is_the_audio_source(tmp_path: Path):
    media = MediaPaths(
        dataset="corpus_a",
        recording_id="r1",
        video_path=tmp_path / "r1.mp4",
        audio_path=tmp_path / "r1.wav",
        video_has_audio=True,
    )

    assert media.audio_source == tmp_path / "r1.wav"


def test_audio_only_record(tmp_path: Path):
    touch(tmp_path / "audio/r1.wav")

    index = MediaIndex.from_manifest(
        make_manifest({"video_path": None, "audio_path": "audio/r1.wav"}),
        {"corpus_a": tmp_path},
    )

    media = index.get(dataset="corpus_a", recording_id="r1")

    assert media.video_path is None
    assert media.audio_source == tmp_path / "audio/r1.wav"


@pytest.mark.parametrize(
    ("video_has_audio", "uses_video"),
    [(True, True), (None, True), (False, False)],
)
def test_video_only_audio_source_respects_video_has_audio(
    tmp_path: Path, video_has_audio, uses_video
):
    touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(
        make_manifest({"video_has_audio": video_has_audio}),
        {"corpus_a": tmp_path},
    )

    media = index.get(dataset="corpus_a", recording_id="r1")

    assert media.video_has_audio is video_has_audio
    assert media.audio_source == (media.video_path if uses_video else None)


@pytest.mark.parametrize("offset", [0.0, 300.0, -0.5])
def test_media_offset_maps_canonical_to_media_time(tmp_path: Path, offset):
    touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(
        make_manifest({"media_offset_s": offset}),
        {"corpus_a": tmp_path},
    )

    media = index.get(dataset="corpus_a", recording_id="r1")

    assert media.media_offset_s == offset
    assert media.to_media_time(1.5) == pytest.approx(1.5 + offset)


def test_unknown_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="No media root configured.*corpus_a"):
        MediaIndex.from_manifest(make_manifest(), {"corpus_b": tmp_path})


@pytest.mark.parametrize(
    "bad_path",
    ["/abs/r1.mp4", "../outside/r1.mp4", "videos/../../r1.mp4", ""],
)
def test_malformed_canonical_path_is_rejected(tmp_path: Path, bad_path):
    with pytest.raises(ValueError, match="relative to the corpus root"):
        MediaIndex.from_manifest(
            make_manifest({"video_path": bad_path}),
            {"corpus_a": tmp_path},
        )


def test_missing_video_file_fails_on_lookup(tmp_path: Path):
    index = MediaIndex.from_manifest(make_manifest(), {"corpus_a": tmp_path})

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        index.get(dataset="corpus_a", recording_id="r1")


def test_missing_audio_file_fails_on_lookup(tmp_path: Path):
    touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(
        make_manifest({"audio_path": "audio/r1.wav"}),
        {"corpus_a": tmp_path},
    )

    with pytest.raises(FileNotFoundError, match="Audio file does not exist"):
        index.get(dataset="corpus_a", recording_id="r1")


def test_partial_local_corpus_serves_available_recordings(tmp_path: Path):
    touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(
        make_manifest({}, {"recording_id": "r2", "video_path": "videos/r2.mp4"}),
        {"corpus_a": tmp_path},
    )

    assert index.get(dataset="corpus_a", recording_id="r1").video_path.is_file()

    with pytest.raises(FileNotFoundError):
        index.get(dataset="corpus_a", recording_id="r2")


def test_path_validation_can_be_disabled():
    index = MediaIndex(
        [
            MediaPaths(
                dataset="corpus_a",
                recording_id="r1",
                video_path=Path("/does/not/exist.mp4"),
            )
        ],
        validate_paths=False,
    )

    assert index.get(dataset="corpus_a", recording_id="r1").video_path


def test_unknown_media_mapping_raises(tmp_path: Path):
    touch(tmp_path / "videos/r1.mp4")

    index = MediaIndex.from_manifest(make_manifest(), {"corpus_a": tmp_path})

    with pytest.raises(KeyError, match="No media found"):
        index.get(dataset="corpus_a", recording_id="unknown")

    # Same recording ID under another dataset is a different recording.
    with pytest.raises(KeyError, match="No media found"):
        index.get(dataset="corpus_b", recording_id="r1")


def test_empty_index_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        MediaIndex([])
