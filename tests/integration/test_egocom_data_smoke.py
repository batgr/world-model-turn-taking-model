"""
Real-data smoke test: published EgoCom → source → Dataset → DataLoader.

Downloads the public dataset from the Hugging Face Hub (cached after the first
run). Excluded from the default test run; invoke explicitly with:

    uv run pytest -m integration
"""

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    PAD_ACTION_ID,
    PAD_STATE_ID,
    STATE_TO_ID,
    TurnTakingDataset,
    WindowConfig,
)
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.sampling import SamplingConfig
from turn_wm.data.source import EGOCOM, load_data

pytestmark = pytest.mark.integration

BATCH_SIZE = 4
WINDOW = WindowConfig()


@pytest.fixture(scope="module")
def data():
    return load_data(EGOCOM)


@pytest.fixture(scope="module")
def train_anchors(data):
    return data.model_ready["train"]


def column(dataset, name: str) -> pa.ChunkedArray:
    # Only valid on unfiltered datasets (no indices mapping).
    return dataset.data.column(name)


def test_published_splits(data):
    assert set(data.model_ready) == {"train", "validation", "test"}
    assert len(data.action_grid) > 0

    for split in data.model_ready.values():
        assert len(split) > 0


def test_published_vocabularies_match_encoders(data):
    grid = data.action_grid

    states = set(pc.unique(column(grid, "focal_state_before")).to_pylist())
    actions = set(pc.unique(column(grid, "action")).to_pylist())

    assert states <= set(STATE_TO_ID)
    assert actions - {None} <= set(ACTION_TO_ID)

    # dataset.py encodes a null action as MASKED and relies on action_valid
    # to mark the same timesteps as invalid.
    null_action = pc.is_null(column(grid, "action"))
    invalid = pc.invert(column(grid, "action_valid"))

    assert pc.all(pc.equal(null_action, invalid)).as_py()


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_anchor_rows_point_at_their_grid_rows(data, split):
    """dataset.py slices the grid by anchor_row; it must match anchor_idx."""

    anchors = data.model_ready[split]
    rows = column(anchors, "anchor_row")

    grid_recording = pc.take(column(data.action_grid, "recording_id"), rows)
    grid_index = pc.take(column(data.action_grid, "decision_index"), rows)

    assert pc.all(pc.equal(grid_recording, column(anchors, "recording_id"))).as_py()
    assert pc.all(pc.equal(grid_index, column(anchors, "anchor_idx"))).as_py()


def test_trainable_anchors_support_default_window(train_anchors):
    trainable = train_anchors.data.table.filter(column(train_anchors, "is_trainable"))

    assert pc.min(trainable["max_context_steps"]).as_py() >= WINDOW.min_context_steps
    assert pc.min(trainable["future_steps"]).as_py() >= WINDOW.future_steps


def test_dataset_exposes_only_trainable_anchors(data, train_anchors):
    dataset = TurnTakingDataset(
        anchors=train_anchors,
        action_grid=data.action_grid,
        window=WINDOW,
        training=False,
    )

    trainable = pc.sum(column(train_anchors, "is_trainable")).as_py()

    # The published split contains non-trainable anchors; the Dataset drops them.
    assert trainable < len(train_anchors)
    assert len(dataset) == trainable


def test_evaluation_batch(data, train_anchors):
    dataset = TurnTakingDataset(
        anchors=train_anchors,
        action_grid=data.action_grid,
        window=WINDOW,
        training=False,
    )
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=BATCH_SIZE, num_workers=0),
    )

    batch = next(iter(loader))

    assert_batch_structure(batch)
    assert_samples_are_usable_anchors(batch, train_anchors)

    # Evaluation uses the longest context each anchor supports.
    max_context = [
        min(dataset.anchors[i]["max_context_steps"], WINDOW.max_context_steps)
        for i in range(BATCH_SIZE)
    ]
    assert batch["context_lengths"].tolist() == max_context


def test_balanced_training_batch(data, train_anchors):
    dataset = TurnTakingDataset(
        anchors=train_anchors,
        action_grid=data.action_grid,
        window=WINDOW,
        training=True,
    )
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=BATCH_SIZE, num_workers=0),
        sampling=SamplingConfig(strategy="balanced"),
    )

    batch = next(iter(loader))

    assert_batch_structure(batch)
    assert_samples_are_usable_anchors(batch, train_anchors)


def assert_batch_structure(batch):
    context_keys = ["context_state", "context_action", "context_valid", "context_mask"]
    future_keys = ["future_state", "future_action", "future_valid"]

    for key in context_keys + future_keys:
        assert batch[key].ndim == 2, key

    for key in context_keys[1:]:
        assert batch[key].shape == batch["context_state"].shape, key

    for key in future_keys[1:]:
        assert batch[key].shape == batch["future_state"].shape, key

    batch_size, context_width = batch["context_state"].shape
    lengths = batch["context_lengths"]

    assert batch_size == BATCH_SIZE
    assert batch["future_state"].shape == (BATCH_SIZE, WINDOW.future_steps)
    assert context_width <= WINDOW.max_context_steps
    assert context_width == int(lengths.max())
    assert int(lengths.min()) >= WINDOW.min_context_steps

    mask = batch["context_mask"]
    padding = ~mask

    assert mask.sum(dim=1).tolist() == lengths.tolist()

    real_states = set(STATE_TO_ID.values())
    real_actions = set(ACTION_TO_ID.values()) | {MASKED_ACTION_ID}

    assert values(batch["context_state"][mask]) <= real_states
    assert values(batch["context_action"][mask]) <= real_actions
    assert values(batch["context_state"][padding]) <= {PAD_STATE_ID}
    assert values(batch["context_action"][padding]) <= {PAD_ACTION_ID}
    assert not batch["context_valid"][padding].any()

    assert values(batch["future_state"]) <= real_states
    assert values(batch["future_action"]) <= real_actions

    for key in ("context", "future"):
        masked = batch[f"{key}_action"] == MASKED_ACTION_ID
        invalid = ~batch[f"{key}_valid"]

        if key == "context":
            invalid &= mask

        assert torch.equal(masked, invalid), key


def assert_samples_are_usable_anchors(batch, anchors):
    sample_ids = pa.array(batch["sample_id"])
    table = anchors.data.table
    matched = table.filter(pc.is_in(table["sample_id"], value_set=sample_ids))

    assert matched.num_rows == len(set(batch["sample_id"]))
    assert pc.all(matched["is_trainable"]).as_py()

    by_id = dict(
        zip(
            matched["sample_id"].to_pylist(),
            matched["recording_id"].to_pylist(),
            strict=True,
        )
    )

    assert [by_id[sample_id] for sample_id in batch["sample_id"]] == batch[
        "recording_id"
    ]


def values(tensor) -> set[int]:
    return set(tensor.unique().tolist())
