"""
Load and validate the conversational dynamics datasets used for modelling.

This module is the boundary between storage (Hugging Face or local Parquet)
and the modelling code. It exposes the model-ready anchor index and the
underlying action grid without introducing PyTorch or training semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download

MODEL_READY_REQUIRED_COLUMNS = {
    "sample_id",
    "dataset",
    "recording_id",
    "conversation_id",
    "view_id",
    "wearer_id",
    "split",
    "split_source",
    "segment_id",
    "anchor_idx",
    "anchor_time",
    "anchor_row",
    "max_context_steps",
    "future_steps",
    "context_valid_ratio",
    "future_valid_ratio",
    "future_event_count",
    "sample_class",
    "is_trainable",
    "window_schema_version",
    "action_schema_version",
}


ACTION_GRID_REQUIRED_COLUMNS = {
    "dataset",
    "recording_id",
    "sync_group_id",
    "view_id",
    "wearer_id",
    "decision_index",
    "decision_time_s",
    "focal_state_before",
    "action",
    "action_valid",
    "mask_reason",
}


@dataclass(frozen=True)
class HuggingFaceSource:
    """Hugging Face dataset configuration."""

    repo_id: str
    model_ready_config: str
    action_grid_config: str
    metadata_file: str = "metadata.json"
    revision: str | None = None


@dataclass(frozen=True)
class LocalSource:
    """Local model-ready and action-grid artifacts."""

    model_ready_dir: Path
    action_grid_file: Path
    metadata_file: Path | None = None


DataSource: TypeAlias = HuggingFaceSource | LocalSource


@dataclass(frozen=True)
class LoadedData:
    """Canonical data exposed to the modelling repository."""

    model_ready: DatasetDict
    action_grid: Dataset
    metadata: dict[str, object]


def load_data(source: DataSource) -> LoadedData:
    """Load a dataset source and validate its modelling contract."""

    if isinstance(source, HuggingFaceSource):
        data = _load_huggingface(source)
    else:
        data = _load_local(source)

    _validate_model_ready(data.model_ready)
    _validate_action_grid(data.action_grid)

    return data


def _load_huggingface(source: HuggingFaceSource) -> LoadedData:
    model_ready = load_dataset(
        source.repo_id,
        source.model_ready_config,
        revision=source.revision,
    )

    action_grid = load_dataset(
        source.repo_id,
        source.action_grid_config,
        split="train",
        revision=source.revision,
    )

    metadata_path = hf_hub_download(
        repo_id=source.repo_id,
        filename=source.metadata_file,
        repo_type="dataset",
        revision=source.revision,
    )

    metadata = _read_json(Path(metadata_path))

    return LoadedData(
        model_ready=model_ready,
        action_grid=action_grid,
        metadata=metadata,
    )


def _load_local(source: LocalSource) -> LoadedData:
    split_files = {}

    for split in ("train", "validation", "test"):
        path = source.model_ready_dir / f"{split}.parquet"

        if path.exists():
            split_files[split] = str(path)

    if not split_files:
        raise FileNotFoundError(
            f"No model-ready split files found in {source.model_ready_dir}"
        )

    if not source.action_grid_file.exists():
        raise FileNotFoundError(f"Action grid not found: {source.action_grid_file}")

    model_ready = load_dataset(
        "parquet",
        data_files=split_files,
    )

    action_grid = load_dataset(
        "parquet",
        data_files={"train": str(source.action_grid_file)},
        split="train",
    )

    metadata = {}

    if source.metadata_file is not None:
        if not source.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {source.metadata_file}")

        metadata = _read_json(source.metadata_file)

    return LoadedData(
        model_ready=model_ready,
        action_grid=action_grid,
        metadata=metadata,
    )


def _validate_model_ready(dataset: DatasetDict) -> None:
    if "train" not in dataset:
        raise ValueError("model_ready must contain a train split")

    for split_name, split in dataset.items():
        _require_columns(
            split,
            MODEL_READY_REQUIRED_COLUMNS,
            artifact=f"model_ready/{split_name}",
        )

        declared_splits = set(split.unique("split"))

        if declared_splits != {split_name}:
            raise ValueError(
                f"model_ready/{split_name} contains split values "
                f"{sorted(declared_splits)}"
            )


def _validate_action_grid(dataset: Dataset) -> None:
    _require_columns(
        dataset,
        ACTION_GRID_REQUIRED_COLUMNS,
        artifact="action_grid",
    )


def _require_columns(
    dataset: Dataset,
    required: set[str],
    *,
    artifact: str,
) -> None:
    available = set(dataset.column_names)
    missing = required - available

    if missing:
        raise ValueError(
            f"{artifact} is missing required columns: " f"{sorted(missing)}"
        )


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)
