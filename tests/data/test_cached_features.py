"""Cached Mimi features in TurnTakingDataset and collate (no Mimi, no media)."""

from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset

from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.dataset import ACTION_TO_ID, TurnTakingDataset, WindowConfig
from turn_wm.data.media import MediaIndex, MediaPaths
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.reader import MediaReader, MediaWindow

ACTIONS = list(ACTION_TO_ID)

# The grid starts at decision_index 10, as the cache record does: nothing
# may assume that a recording starts at 0.
FIRST_INDEX = 10
GRID_LENGTH = 40


def make_grid() -> Dataset:
    indices = list(range(FIRST_INDEX, FIRST_INDEX + GRID_LENGTH))

    return Dataset.from_dict(
        {
            "dataset": ["egocom"] * GRID_LENGTH,
            "recording_id": ["r1"] * GRID_LENGTH,
            "decision_index": indices,
            "decision_time_s": [index / 10 for index in indices],
            "focal_state_before": ["SILENT"] * GRID_LENGTH,
            # The action depends on the index, so rows can be told apart.
            "action": [ACTIONS[index % 3] for index in indices],
            "action_valid": [True] * GRID_LENGTH,
        }
    )


def make_anchors(*anchor_indices: int) -> Dataset:
    count = len(anchor_indices)

    return Dataset.from_dict(
        {
            "sample_id": [f"egocom:train#{index}" for index in anchor_indices],
            "dataset": ["egocom"] * count,
            "recording_id": ["r1"] * count,
            "anchor_idx": list(anchor_indices),
            # Position of the anchor's row in the grid table.
            "anchor_row": [index - FIRST_INDEX for index in anchor_indices],
            "anchor_time": [index / 10 for index in anchor_indices],
            "max_context_steps": [10] * count,
            "future_steps": [10] * count,
            "sample_class": ["event"] * count,
            "is_trainable": [True] * count,
        }
    )


@pytest.fixture
def store(make_mimi_cache) -> MimiFeatureStore:
    return MimiFeatureStore(
        make_mimi_cache({("egocom", "r1"): (FIRST_INDEX, GRID_LENGTH)})
    )


def dataset(store, *anchor_indices, context=4, future=3, **kwargs):
    return TurnTakingDataset(
        anchors=make_anchors(*(anchor_indices or (20,))),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=context, max_context_steps=context, future_steps=future
        ),
        training=False,
        mimi_store=store,
        modalities=kwargs.pop("modalities", ("audio",)),
        **kwargs,
    )


def test_features_are_the_rows_of_the_window_decision_indices(store):
    sample = dataset(store)[0]

    # Anchor 20, C = 4, F = 3.
    assert sample["context_features"][:, 0].tolist() == [17, 18, 19, 20]
    assert sample["future_features"][:, 0].tolist() == [21, 22, 23]
    assert sample["context_features"].shape == (4, 512)
    assert sample["future_features"].shape == (3, 512)


def test_features_line_up_with_the_state_and_action_rows(store):
    sample = dataset(store)[0]

    expected_context = [ACTION_TO_ID[ACTIONS[index % 3]] for index in range(17, 21)]
    expected_future = [ACTION_TO_ID[ACTIONS[index % 3]] for index in range(21, 24)]

    assert sample["context_action"].tolist() == expected_context
    assert sample["future_action"].tolist() == expected_future

    for features, actions in (
        (sample["context_features"], sample["context_action"]),
        (sample["future_features"], sample["future_action"]),
    ):
        indices = features[:, 0].long()
        assert [ACTION_TO_ID[ACTIONS[int(i) % 3]] for i in indices] == actions.tolist()


def test_cached_features_keep_the_cache_dtype(store):
    assert dataset(store)[0]["context_features"].dtype == torch.float16


def test_cached_audio_needs_no_media(store):
    sample = dataset(store)[0]

    assert "context_media" not in sample
    assert "future_media" not in sample


def media_index(tmp_path) -> MediaIndex:
    return MediaIndex(
        [
            MediaPaths(
                dataset="egocom",
                recording_id="r1",
                video_path=tmp_path / "r1.mp4",
            )
        ],
        validate_paths=False,
    )


def test_cached_audio_only_never_reads_media(store, tmp_path):
    reader = Mock(spec=MediaReader)

    sample = dataset(store, media_index=media_index(tmp_path), media_reader=reader)[0]

    reader.read_window.assert_not_called()
    assert "context_media" not in sample


def test_other_modalities_still_come_from_media(store, tmp_path):
    reader = Mock(spec=MediaReader)
    reader.read_window.side_effect = lambda media, **kwargs: MediaWindow(
        start_time_s=kwargs["start_time_s"],
        end_time_s=kwargs["end_time_s"],
        audio=None,
        video=None,
    )

    sample = dataset(
        store,
        media_index=media_index(tmp_path),
        media_reader=reader,
        modalities=("audio", "video"),
    )[0]

    # Audio from the cache, video from the media; audio is never decoded.
    assert [
        call.kwargs["modalities"] for call in reader.read_window.call_args_list
    ] == [
        ("video",),
        ("video",),
    ]
    assert sample["context_features"].shape == (4, 512)
    assert "context_media" in sample


def test_a_store_requires_the_audio_modality(store):
    with pytest.raises(ValueError, match="modalities must include audio"):
        dataset(store, modalities=("video",))


def test_window_outside_the_cache_is_an_error(make_mimi_cache):
    # The cache covers fewer rows than the grid.
    short = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (FIRST_INDEX, 12)}))

    with pytest.raises(IndexError, match="outside recording 'r1'"):
        dataset(short)[0]


def test_recording_missing_from_the_cache_is_filtered(make_mimi_cache):
    other = MimiFeatureStore(make_mimi_cache({("egocom", "other"): (0, 50)}))

    data = dataset(other)

    assert len(data) == 0
    assert data.canonical_anchor_count == 1
    assert data.cache_filtered_anchor_count == 1


def test_collate_batches_cached_features(store):
    data = dataset(store, 20, 25, 30)

    batch = collate_turn_taking([data[index] for index in range(3)])

    assert batch["context_features"].shape == (3, 4, 512)
    assert batch["future_features"].shape == (3, 3, 512)
    assert batch["context_features"][:, :, 0].tolist() == [
        [17, 18, 19, 20],
        [22, 23, 24, 25],
        [27, 28, 29, 30],
    ]
    assert batch["future_features"][:, :, 0].tolist() == [
        [21, 22, 23],
        [26, 27, 28],
        [31, 32, 33],
    ]
    assert "context_media" not in batch


def test_collate_pads_variable_contexts_in_time_only(store):
    short = dataset(store, 20, context=2)[0]
    long = dataset(store, 20, context=4)[0]

    batch = collate_turn_taking([short, long])

    assert batch["context_features"].shape == (2, 4, 512)
    assert batch["context_features"][0, :2, 0].tolist() == [19, 20]
    assert torch.all(batch["context_features"][0, 2:] == 0)
    assert batch["context_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_collate_rejects_mixed_cached_and_plain_samples(store):
    cached = dataset(store)[0]
    plain = {key: value for key, value in cached.items() if "features" not in key}

    with pytest.raises(ValueError, match="mixed cached-feature and plain"):
        collate_turn_taking([cached, plain])
