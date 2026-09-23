import pytest
import torch
from datasets import Dataset

from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    STATE_TO_ID,
    TurnTakingDataset,
    WindowConfig,
)
from datasets import concatenate_datasets


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
