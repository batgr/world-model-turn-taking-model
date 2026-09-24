"""
Real-media smoke tests: published manifest + local raw media → DataLoader.

Raw media is never downloaded. Point these tests at local corpus roots:

    EGOCOM_MEDIA_ROOT=/path/to/EgoCom uv run pytest -m integration
    EGO4D_MEDIA_ROOT=/path/to/Ego4D uv run pytest -m integration   # private

Tests needing a root are skipped when its variable is unset.
"""

import os
from pathlib import Path

import pyarrow.compute as pc
import pytest

from turn_wm.data.dataset import TurnTakingDataset, WindowConfig
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.media import MEDIA_MODALITIES, MediaIndex
from turn_wm.data.reader import MediaWindow
from turn_wm.data.source import EGO4D, EGOCOM, HuggingFaceSource, load_data

pytestmark = pytest.mark.integration

WINDOW = WindowConfig()
GRID_STEP_S = 0.1


@pytest.fixture(scope="module")
def egocom():
    return load_data(EGOCOM).corpus("egocom")


def media_root(variable: str) -> Path:
    value = os.environ.get(variable)

    if not value:
        pytest.skip(f"{variable} is not set; local raw media unavailable")

    root = Path(value)

    if not root.is_dir():
        pytest.fail(f"{variable}={value} is not a directory")

    return root


def test_published_egocom_manifest_matches_contract(egocom):
    manifest = egocom.media_manifest

    assert manifest is not None

    table = manifest.with_format("arrow")[:]
    grid_recordings = set(pc.unique(egocom.action_grid.data.column("recording_id")))
    manifest_recordings = set(pc.unique(table["recording_id"]))

    assert set(pc.unique(table["dataset"]).to_pylist()) == {"egocom"}
    assert pc.all(pc.equal(table["media_offset_s"], 0.0)).as_py()
    assert manifest_recordings == grid_recordings


def first_batch_with_local_media(
    data, root: Path, *, offset_filter=None, modalities=MEDIA_MODALITIES
):
    """Build the real pipeline for one recording whose media exists locally."""

    manifest = data.media_manifest
    dataset_name = manifest[0]["dataset"]
    index = MediaIndex.from_manifest(manifest, {dataset_name: root})

    candidates = [
        row
        for row in manifest
        if (root / row["video_path"]).is_file()
        and (offset_filter is None or offset_filter(row["media_offset_s"]))
    ]

    if not candidates:
        pytest.skip(f"No matching {dataset_name} media found under {root}")

    recording_id = candidates[0]["recording_id"]
    anchors = data.model_ready["train"].filter(
        lambda batch: [value == recording_id for value in batch["recording_id"]],
        batched=True,
    )

    dataset = TurnTakingDataset(
        anchors=anchors,
        action_grid=data.action_grid,
        window=WINDOW,
        training=False,
        media_index=index,
        modalities=modalities,
    )
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=1, num_workers=0),
    )

    batch = next(iter(loader))
    media = index.get(dataset=dataset_name, recording_id=recording_id)

    return batch, media


def assert_window(
    window: MediaWindow,
    *,
    steps: int,
    offset: float,
    expect_audio: bool,
    expect_video: bool = True,
):
    duration = steps * GRID_STEP_S

    assert window.canonical_start_time_s is not None
    assert window.end_time_s - window.start_time_s == pytest.approx(duration)
    assert window.start_time_s == pytest.approx(window.canonical_start_time_s + offset)
    assert window.end_time_s == pytest.approx(window.canonical_end_time_s + offset)

    video = window.video

    if expect_video:
        assert video is not None
        assert video.frames.ndim == 4
        assert video.frames.shape[0] > 0
        assert video.frames.shape[0] == video.timestamps_s.shape[0]
        assert (video.timestamps_s >= window.start_time_s - 1e-6).all()
        assert (video.timestamps_s < window.end_time_s).all()
    else:
        assert video is None

    audio = window.audio

    if not expect_audio:
        assert audio is None
        return

    assert audio is not None
    assert audio.sample_rate > 0
    assert audio.waveform.ndim == 2
    assert audio.waveform.shape[1] == pytest.approx(
        duration * audio.sample_rate,
        abs=0.01 * audio.sample_rate,
    )


def assert_multimodal_batch(batch, media, modalities=MEDIA_MODALITIES):
    assert batch["dataset"] == [media.dataset]
    assert batch["recording_id"] == [media.recording_id]
    assert len(batch["context_media"]) == len(batch["future_media"]) == 1

    context = batch["context_media"][0]
    future = batch["future_media"][0]
    context_steps = int(batch["context_lengths"][0])

    # No leakage: context ends exactly where the future starts.
    assert context.end_time_s == future.start_time_s
    assert context.canonical_end_time_s == pytest.approx(
        float(batch["anchor_time"][0]) + GRID_STEP_S, abs=1e-4
    )

    expect_audio = "audio" in modalities and media.audio_source is not None
    expect_video = "video" in modalities

    assert_window(
        context,
        steps=context_steps,
        offset=media.media_offset_s,
        expect_audio=expect_audio,
        expect_video=expect_video,
    )
    assert_window(
        future,
        steps=WINDOW.future_steps,
        offset=media.media_offset_s,
        expect_audio=expect_audio,
        expect_video=expect_video,
    )


def test_egocom_raw_multimodal_batch(egocom):
    root = media_root("EGOCOM_MEDIA_ROOT")

    batch, media = first_batch_with_local_media(egocom, root)

    assert media.dataset == "egocom"
    assert media.media_offset_s == 0
    assert media.video_has_audio is True

    assert_multimodal_batch(batch, media)


def test_nonzero_offset_raw_multimodal_batch():
    root = media_root("EGO4D_MEDIA_ROOT")

    data = load_private(EGO4D).corpus("ego4d")
    batch, media = first_batch_with_local_media(
        data,
        root,
        offset_filter=lambda offset: offset > 1.0,
    )

    assert media.media_offset_s > 1.0

    context = batch["context_media"][0]

    assert context.start_time_s == pytest.approx(
        context.canonical_start_time_s + media.media_offset_s
    )

    assert_multimodal_batch(batch, media)


@pytest.mark.parametrize("modalities", [("audio",), ("video",)])
def test_egocom_single_modality_batch(egocom, modalities):
    root = media_root("EGOCOM_MEDIA_ROOT")

    batch, media = first_batch_with_local_media(egocom, root, modalities=modalities)

    # EgoCom audio is embedded in the video container.
    assert media.audio_path is None and media.video_has_audio is True

    assert_multimodal_batch(batch, media, modalities)


@pytest.mark.parametrize("modalities", [("audio",), ("video",)])
def test_ego4d_single_modality_batch(modalities):
    root = media_root("EGO4D_MEDIA_ROOT")

    data = load_private(EGO4D).corpus("ego4d")
    batch, media = first_batch_with_local_media(
        data,
        root,
        offset_filter=lambda offset: offset > 1.0,
        modalities=modalities,
    )

    assert media.audio_path is None and media.video_has_audio is True

    assert_multimodal_batch(batch, media, modalities)


def load_private(source: HuggingFaceSource):
    try:
        return load_data(source)
    except Exception as error:  # noqa: BLE001 - access depends on credentials
        pytest.skip(f"{source.repo_id} is not accessible: {error}")


def test_published_ego4d_offsets_are_whole_frames_inside_the_video():
    # Ego4D clips are placed on their source video by frame index (30 fps);
    # an offset off the frame grid or before the file start means the
    # manifest used the release's full-scale-timeline seconds instead.
    manifest = load_private(EGO4D).corpus("ego4d").media_manifest

    assert manifest is not None

    offsets = manifest.with_format("arrow")[:]["media_offset_s"]
    frames = pc.multiply(offsets, 30.0)

    assert pc.min(offsets).as_py() >= 0.0
    assert pc.max(pc.abs(pc.subtract(frames, pc.round(frames)))).as_py() < 1e-6
