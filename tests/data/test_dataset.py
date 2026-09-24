from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from datasets import Dataset, concatenate_datasets

from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    STATE_TO_ID,
    TurnTakingDataset,
    WindowConfig,
)
from turn_wm.data.media import (
    MediaIndex,
    MediaPaths,
)
from turn_wm.data.reader import (
    MediaReader,
    MediaWindow,
)

STATE_ACTION_KEYS = {
    "context_state",
    "context_action",
    "context_valid",
    "future_state",
    "future_action",
    "future_valid",
    "context_length",
    "sample_id",
    "dataset",
    "recording_id",
    "anchor_idx",
    "anchor_time",
    "sample_class",
}


def test_media_reader_requires_media_index():
    reader = Mock(spec=MediaReader)

    with pytest.raises(
        ValueError,
        match="requires media_index",
    ):
        TurnTakingDataset(
            anchors=make_anchors(),
            action_grid=make_grid(),
            window=WindowConfig(),
            training=False,
            media_reader=reader,
        )


def test_dataset_without_media_keeps_original_contract():
    dataset = TurnTakingDataset(
        anchors=make_anchors(),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=10,
            max_context_steps=10,
            future_steps=10,
        ),
        training=False,
    )

    sample = dataset[0]

    assert set(sample) == STATE_ACTION_KEYS


def make_media_index(*records: MediaPaths) -> MediaIndex:
    return MediaIndex(
        records
        or [
            MediaPaths(
                dataset="synthetic",
                recording_id="r1",
                video_path=Path("/fake/r1.mp4"),
            )
        ],
        validate_paths=False,
    )


def make_reader() -> Mock:
    """Reader double that echoes the requested media-time interval."""

    reader = Mock(spec=MediaReader)

    def read_window(media, *, start_time_s, end_time_s):
        return MediaWindow(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            audio=None,
            video=None,
        )

    reader.read_window.side_effect = read_window

    return reader


def make_media_dataset(
    *,
    media_index: MediaIndex,
    reader: Mock,
    max_context_steps: int = 10,
    window: WindowConfig | None = None,
) -> TurnTakingDataset:
    return TurnTakingDataset(
        anchors=make_anchors(
            anchor_idx=19,
            anchor_row=19,
            max_context_steps=max_context_steps,
            future_steps=10,
        ),
        action_grid=make_grid(),
        window=window
        or WindowConfig(
            min_context_steps=10,
            max_context_steps=10,
            future_steps=10,
        ),
        training=False,
        media_index=media_index,
        media_reader=reader,
    )


def reader_intervals(reader: Mock) -> list[tuple[float, float]]:
    return [
        (call.kwargs["start_time_s"], call.kwargs["end_time_s"])
        for call in reader.read_window.call_args_list
    ]


def test_dataset_attaches_aligned_media_windows():
    reader = make_reader()
    dataset = make_media_dataset(media_index=make_media_index(), reader=reader)

    sample = dataset[0]

    assert set(sample) == STATE_ACTION_KEYS | {"context_media", "future_media"}
    assert reader.read_window.call_count == 2


def test_zero_offset_reads_canonical_times():
    reader = make_reader()
    dataset = make_media_dataset(media_index=make_media_index(), reader=reader)

    sample = dataset[0]

    # Anchor 19 at 10 Hz, 10 context steps: rows 10..19, future rows 20..29.
    assert reader_intervals(reader) == [
        pytest.approx((1.0, 2.0)),
        pytest.approx((2.0, 3.0)),
    ]

    context = sample["context_media"]
    future = sample["future_media"]

    assert (context.canonical_start_time_s, context.canonical_end_time_s) == (
        pytest.approx(1.0),
        pytest.approx(2.0),
    )
    assert (future.canonical_start_time_s, future.canonical_end_time_s) == (
        pytest.approx(2.0),
        pytest.approx(3.0),
    )


def test_nonzero_offset_shifts_only_media_time():
    reader = make_reader()
    media_index = make_media_index(
        MediaPaths(
            dataset="synthetic",
            recording_id="r1",
            video_path=Path("/fake/source.mp4"),
            media_offset_s=300.0,
        )
    )
    dataset = make_media_dataset(media_index=media_index, reader=reader)

    sample = dataset[0]

    assert reader_intervals(reader) == [
        pytest.approx((301.0, 302.0)),
        pytest.approx((302.0, 303.0)),
    ]

    context = sample["context_media"]
    future = sample["future_media"]

    # Physical bounds on the window, canonical bounds preserved alongside.
    assert (context.start_time_s, context.end_time_s) == (
        pytest.approx(301.0),
        pytest.approx(302.0),
    )
    assert (context.canonical_start_time_s, context.canonical_end_time_s) == (
        pytest.approx(1.0),
        pytest.approx(2.0),
    )
    assert (future.canonical_start_time_s, future.canonical_end_time_s) == (
        pytest.approx(2.0),
        pytest.approx(3.0),
    )

    # anchor_time stays on the canonical grid.
    assert sample["anchor_time"] == pytest.approx(1.9)


def test_media_lookup_uses_anchor_dataset():
    reader = make_reader()
    media_index = make_media_index(
        MediaPaths(
            dataset="synthetic",
            recording_id="r1",
            video_path=Path("/fake/a.mp4"),
            media_offset_s=10.0,
        ),
        MediaPaths(
            dataset="other",
            recording_id="r1",
            video_path=Path("/fake/b.mp4"),
            media_offset_s=500.0,
        ),
    )
    dataset = make_media_dataset(media_index=media_index, reader=reader)

    dataset[0]

    media = reader.read_window.call_args_list[0].args[0]

    assert media.key == ("synthetic", "r1")
    assert reader_intervals(reader)[0] == pytest.approx((11.0, 12.0))


def test_context_media_tracks_selected_context_length():
    starts = {}

    for context_steps in (10, 15):
        reader = make_reader()
        dataset = make_media_dataset(
            media_index=make_media_index(),
            reader=reader,
            max_context_steps=15,
            window=WindowConfig(
                min_context_steps=10,
                max_context_steps=context_steps,
                future_steps=10,
            ),
        )

        sample = dataset[0]

        assert sample["context_length"] == context_steps
        starts[context_steps] = reader_intervals(reader)[0]

    # Same anchor, longer context: earlier start, same end.
    assert starts[10] == pytest.approx((1.0, 2.0))
    assert starts[15] == pytest.approx((0.5, 2.0))


def test_context_and_future_media_share_boundary_without_leakage():
    reader = make_reader()
    media_index = make_media_index(
        MediaPaths(
            dataset="synthetic",
            recording_id="r1",
            video_path=Path("/fake/r1.mp4"),
            media_offset_s=42.0,
        )
    )
    dataset = make_media_dataset(media_index=media_index, reader=reader)

    sample = dataset[0]

    context = sample["context_media"]
    future = sample["future_media"]
    first_future_time = 2.0  # decision_time_s of the first future row

    assert context.end_time_s == future.start_time_s
    assert context.canonical_end_time_s == future.canonical_start_time_s
    assert context.canonical_end_time_s == pytest.approx(first_future_time)
    assert context.canonical_end_time_s > sample["anchor_time"]


def test_missing_media_mapping_fails():
    reader = make_reader()
    media_index = make_media_index(
        MediaPaths(
            dataset="other",
            recording_id="r1",
            video_path=Path("/fake/r1.mp4"),
        )
    )
    dataset = make_media_dataset(media_index=media_index, reader=reader)

    with pytest.raises(KeyError, match="No media found"):
        dataset[0]

    reader.read_window.assert_not_called()


def make_grid(
    *,
    recording_id: str = "r1",
    length: int = 40,
) -> Dataset:
    states = ["SILENT" if i < 20 else "SPEAKING" for i in range(length)]

    actions = ["NO_EVENT"] * length

    if length > 20:
        actions[20] = "ONSET"

    if length > 30:
        actions[30] = None

    return Dataset.from_dict(
        {
            "dataset": ["synthetic"] * length,
            "recording_id": [recording_id] * length,
            "sync_group_id": ["c1"] * length,
            "view_id": ["v1"] * length,
            "wearer_id": ["p1"] * length,
            "decision_index": list(range(length)),
            "decision_time_s": [i / 10 for i in range(length)],
            "focal_state_before": states,
            "action": actions,
            "action_valid": [action is not None for action in actions],
            "mask_reason": [
                None if action is not None else "masked" for action in actions
            ],
        }
    )


def make_anchors(
    *,
    anchor_idx: int = 19,
    anchor_row: int = 19,
    max_context_steps: int = 10,
    future_steps: int = 10,
    is_trainable: bool = True,
) -> Dataset:
    return Dataset.from_dict(
        {
            "sample_id": ["sample-0"],
            "dataset": ["synthetic"],
            "recording_id": ["r1"],
            "conversation_id": ["c1"],
            "view_id": ["v1"],
            "wearer_id": ["p1"],
            "split": ["train"],
            "split_source": ["synthetic"],
            "segment_id": [0],
            "anchor_idx": [anchor_idx],
            "anchor_time": [anchor_idx / 10],
            "anchor_row": [anchor_row],
            "max_context_steps": [max_context_steps],
            "future_steps": [future_steps],
            "context_valid_ratio": [1.0],
            "future_valid_ratio": [1.0],
            "future_event_count": [1],
            "sample_class": ["event"],
            "is_trainable": [is_trainable],
            "window_schema_version": ["1"],
            "action_schema_version": ["1"],
        }
    )


def test_eval_uses_maximum_context():
    dataset = TurnTakingDataset(
        anchors=make_anchors(
            max_context_steps=10,
        ),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=5,
            max_context_steps=10,
            future_steps=10,
        ),
        training=False,
    )

    sample = dataset[0]

    assert sample["context_length"] == 10
    assert sample["context_state"].shape == (10,)
    assert sample["future_state"].shape == (10,)


def test_context_ends_at_anchor():
    dataset = TurnTakingDataset(
        anchors=make_anchors(),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=10,
            max_context_steps=10,
            future_steps=10,
        ),
        training=False,
    )

    sample = dataset[0]

    # Anchor 19 is still SILENT.
    assert sample["context_state"][-1].item() == STATE_TO_ID["SILENT"]

    # Future starts at 20, where the synthetic sequence becomes SPEAKING.
    assert sample["future_state"][0].item() == STATE_TO_ID["SPEAKING"]


def test_future_has_requested_length():
    dataset = TurnTakingDataset(
        anchors=make_anchors(),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=5,
            max_context_steps=10,
            future_steps=7,
        ),
        training=False,
    )

    sample = dataset[0]

    assert sample["future_state"].shape == (7,)
    assert sample["future_action"].shape == (7,)


def test_training_context_stays_in_configured_range():
    torch.manual_seed(42)

    dataset = TurnTakingDataset(
        anchors=make_anchors(
            max_context_steps=10,
        ),
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=5,
            max_context_steps=10,
            future_steps=10,
        ),
        training=True,
    )

    for _ in range(20):
        length = dataset[0]["context_length"]
        assert 5 <= length <= 10


def test_masked_action_has_distinct_encoding():
    anchors = make_anchors(
        anchor_idx=29,
        anchor_row=29,
        max_context_steps=10,
    )

    dataset = TurnTakingDataset(
        anchors=anchors,
        action_grid=make_grid(),
        window=WindowConfig(
            min_context_steps=10,
            max_context_steps=10,
            future_steps=5,
        ),
        training=False,
    )

    sample = dataset[0]

    assert MASKED_ACTION_ID not in ACTION_TO_ID.values()
    assert MASKED_ACTION_ID in sample["future_action"].tolist()


def test_non_trainable_anchors_are_filtered():
    dataset = TurnTakingDataset(
        anchors=make_anchors(
            is_trainable=False,
        ),
        action_grid=make_grid(),
        window=WindowConfig(),
        training=False,
    )

    assert len(dataset) == 0


def test_window_cannot_cross_recording_boundary():
    first = make_grid(
        recording_id="r1",
        length=20,
    )
    second = make_grid(
        recording_id="r2",
        length=20,
    )

    grid = concatenate_datasets([first, second])

    anchors = make_anchors(
        anchor_idx=19,
        anchor_row=19,
        max_context_steps=10,
        future_steps=10,
    )

    dataset = TurnTakingDataset(
        anchors=anchors,
        action_grid=grid,
        window=WindowConfig(
            min_context_steps=10,
            max_context_steps=10,
            future_steps=10,
        ),
        training=False,
    )

    with pytest.raises(
        ValueError,
        match="recording boundary",
    ):
        dataset[0]
