import pytest
import torch

from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.dataset import (
    PAD_ACTION_ID,
    PAD_STATE_ID,
)
from turn_wm.data.reader import MediaWindow


def test_media_windows_are_preserved_as_lists():
    first = make_sample(
        sample_id="a",
        length=2,
    )

    second = make_sample(
        sample_id="b",
        length=4,
    )

    first["context_media"] = MediaWindow(
        start_time_s=0.0,
        end_time_s=1.0,
        audio=None,
        video=None,
    )

    first["future_media"] = MediaWindow(
        start_time_s=1.0,
        end_time_s=2.0,
        audio=None,
        video=None,
    )

    second["context_media"] = MediaWindow(
        start_time_s=0.0,
        end_time_s=1.0,
        audio=None,
        video=None,
    )

    second["future_media"] = MediaWindow(
        start_time_s=1.0,
        end_time_s=2.0,
        audio=None,
        video=None,
    )

    batch = collate_turn_taking([first, second])

    assert len(batch["context_media"]) == 2

    assert len(batch["future_media"]) == 2


def test_mixed_media_batch_raises():
    first = make_sample(
        sample_id="a",
        length=2,
    )

    second = make_sample(
        sample_id="b",
        length=2,
    )

    first["context_media"] = MediaWindow(
        start_time_s=0.0,
        end_time_s=1.0,
        audio=None,
        video=None,
    )

    first["future_media"] = MediaWindow(
        start_time_s=1.0,
        end_time_s=2.0,
        audio=None,
        video=None,
    )

    with pytest.raises(
        ValueError,
        match="mixed media",
    ):
        collate_turn_taking([first, second])


def make_sample(
    *,
    sample_id: str,
    length: int,
    future_steps: int = 3,
):
    return {
        "context_state": torch.arange(length),
        "context_action": torch.arange(length),
        "context_valid": torch.ones(
            length,
            dtype=torch.bool,
        ),
        "future_state": torch.zeros(
            future_steps,
            dtype=torch.long,
        ),
        "future_action": torch.zeros(
            future_steps,
            dtype=torch.long,
        ),
        "future_valid": torch.ones(
            future_steps,
            dtype=torch.bool,
        ),
        "context_length": length,
        "sample_id": sample_id,
        "dataset": "synthetic",
        "recording_id": f"recording-{sample_id}",
        "anchor_idx": length,
        "anchor_time": float(length) / 10,
        "sample_class": "background",
    }


def test_contexts_are_padded_to_longest_sample():
    batch = collate_turn_taking(
        [
            make_sample(
                sample_id="a",
                length=2,
            ),
            make_sample(
                sample_id="b",
                length=4,
            ),
        ]
    )

    assert batch["context_state"].shape == (2, 4)
    assert batch["context_action"].shape == (2, 4)


def test_context_mask_matches_lengths():
    batch = collate_turn_taking(
        [
            make_sample(
                sample_id="a",
                length=2,
            ),
            make_sample(
                sample_id="b",
                length=4,
            ),
        ]
    )

    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )

    assert torch.equal(
        batch["context_mask"],
        expected,
    )


def test_padding_ids_are_distinct():
    batch = collate_turn_taking(
        [
            make_sample(
                sample_id="a",
                length=2,
            ),
            make_sample(
                sample_id="b",
                length=4,
            ),
        ]
    )

    assert torch.all(batch["context_state"][0, 2:] == PAD_STATE_ID)

    assert torch.all(batch["context_action"][0, 2:] == PAD_ACTION_ID)


def test_future_is_stacked_without_padding():
    batch = collate_turn_taking(
        [
            make_sample(
                sample_id="a",
                length=2,
            ),
            make_sample(
                sample_id="b",
                length=4,
            ),
        ]
    )

    assert batch["future_state"].shape == (2, 3)
    assert batch["future_action"].shape == (2, 3)
    assert batch["future_valid"].shape == (2, 3)


def test_metadata_order_is_preserved():
    batch = collate_turn_taking(
        [
            make_sample(
                sample_id="first",
                length=2,
            ),
            make_sample(
                sample_id="second",
                length=4,
            ),
        ]
    )

    assert batch["sample_id"] == [
        "first",
        "second",
    ]


def test_empty_batch_raises():
    with pytest.raises(
        ValueError,
        match="empty batch",
    ):
        collate_turn_taking([])


def test_media_batch_preserves_order_and_state_tensors():
    samples = [
        make_sample(sample_id=sample_id, length=length)
        for sample_id, length in [("a", 2), ("b", 4), ("c", 3)]
    ]

    plain = collate_turn_taking(samples)

    for index, sample in enumerate(samples):
        sample["context_media"] = MediaWindow(
            start_time_s=float(index),
            end_time_s=float(index) + 1,
            audio=None,
            video=None,
        )
        sample["future_media"] = MediaWindow(
            start_time_s=float(index) + 1,
            end_time_s=float(index) + 2,
            audio=None,
            video=None,
        )

    batch = collate_turn_taking(samples)

    assert [window.start_time_s for window in batch["context_media"]] == [0, 1, 2]
    assert [window.start_time_s for window in batch["future_media"]] == [1, 2, 3]
    assert batch["dataset"] == ["synthetic"] * 3

    for key, value in plain.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(batch[key], value), key
        else:
            assert batch[key] == value, key
