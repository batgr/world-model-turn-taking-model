import json

import pytest
from datasets import Dataset

from turn_wm.data.source import LocalSource, load_data


def make_model_ready(split: str) -> Dataset:
    return Dataset.from_dict(
        {
            "sample_id": [f"{split}-0"],
            "dataset": ["synthetic"],
            "recording_id": ["r1"],
            "conversation_id": ["c1"],
            "view_id": ["v1"],
            "wearer_id": ["p1"],
            "split": [split],
            "split_source": ["upstream"],
            "segment_id": [0],
            "anchor_idx": [10],
            "anchor_time": [1.0],
            "anchor_row": [10],
            "max_context_steps": [10],
            "future_steps": [10],
            "context_valid_ratio": [1.0],
            "future_valid_ratio": [1.0],
            "future_event_count": [1],
            "sample_class": ["event"],
            "is_trainable": [True],
            "window_schema_version": ["1"],
            "action_schema_version": ["1"],
        }
    )


def make_action_grid() -> Dataset:
    return Dataset.from_dict(
        {
            "dataset": ["synthetic"],
            "recording_id": ["r1"],
            "sync_group_id": ["c1"],
            "view_id": ["v1"],
            "wearer_id": ["p1"],
            "decision_index": [0],
            "decision_time_s": [0.0],
            "focal_state_before": ["SILENT"],
            "action": ["NO_EVENT"],
            "action_valid": [True],
            "mask_reason": [None],
        }
    )


def test_load_local_source(tmp_path):
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()

    make_model_ready("train").to_parquet(model_ready / "train.parquet")
    make_model_ready("validation").to_parquet(model_ready / "validation.parquet")

    action_grid = tmp_path / "action_grid.parquet"
    make_action_grid().to_parquet(action_grid)

    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"dataset": "synthetic"}),
        encoding="utf-8",
    )

    loaded = load_data(
        LocalSource(
            model_ready_dir=model_ready,
            action_grid_file=action_grid,
            metadata_file=metadata,
        )
    )

    assert set(loaded.model_ready) == {
        "train",
        "validation",
    }
    assert len(loaded.action_grid) == 1
    assert loaded.metadata["dataset"] == "synthetic"


def test_missing_action_grid_raises(tmp_path):
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()

    make_model_ready("train").to_parquet(model_ready / "train.parquet")

    with pytest.raises(FileNotFoundError):
        load_data(
            LocalSource(
                model_ready_dir=model_ready,
                action_grid_file=tmp_path / "missing.parquet",
            )
        )


def test_missing_model_ready_files_raise(tmp_path):
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()

    action_grid = tmp_path / "action_grid.parquet"
    make_action_grid().to_parquet(action_grid)

    with pytest.raises(FileNotFoundError):
        load_data(
            LocalSource(
                model_ready_dir=model_ready,
                action_grid_file=action_grid,
            )
        )


def test_split_column_must_match_physical_split(tmp_path):
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()

    make_model_ready("validation").to_parquet(model_ready / "train.parquet")

    action_grid = tmp_path / "action_grid.parquet"
    make_action_grid().to_parquet(action_grid)

    with pytest.raises(ValueError):
        load_data(
            LocalSource(
                model_ready_dir=model_ready,
                action_grid_file=action_grid,
            )
        )


def test_test_split_is_optional(tmp_path):
    model_ready = tmp_path / "model_ready"
    model_ready.mkdir()

    make_model_ready("train").to_parquet(model_ready / "train.parquet")
    make_model_ready("validation").to_parquet(model_ready / "validation.parquet")

    action_grid = tmp_path / "action_grid.parquet"
    make_action_grid().to_parquet(action_grid)

    loaded = load_data(
        LocalSource(
            model_ready_dir=model_ready,
            action_grid_file=action_grid,
        )
    )

    assert "test" not in loaded.model_ready
