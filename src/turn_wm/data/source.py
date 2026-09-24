"""
Load and validate the conversational dynamics datasets used for modelling.

This module is the boundary between storage (Hugging Face or local Parquet)
and the modelling code. A source names one or more corpora; each loaded corpus
keeps its own model-ready anchor index, action grid, metadata and (when
published) media manifest together, because `anchor_row` indexes that
corpus's own grid. No PyTorch or training semantics are introduced here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download

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


MEDIA_MANIFEST_REQUIRED_COLUMNS = {
    "dataset",
    "recording_id",
    "video_path",
    "audio_path",
    "media_offset_s",
    "video_has_audio",
}


@dataclass(frozen=True)
class CorpusConfig:
    """Where one corpus's canonical artifacts live inside a Hub repository."""

    name: str
    model_ready_config: str
    action_grid_config: str
    media_manifest_config: str | None = None
    metadata_file: str = "metadata.json"


@dataclass(frozen=True)
class HuggingFaceSource:
    """One Hub repository revision providing one or more corpora."""

    repo_id: str
    corpora: tuple[CorpusConfig, ...]
    revision: str | None = None

    def __post_init__(self) -> None:
        names = [corpus.name for corpus in self.corpora]

        if not names:
            raise ValueError("A source must provide at least one corpus")

        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate corpus names in source: {names}")


@dataclass(frozen=True)
class LocalSource:
    """Local model-ready and action-grid artifacts for one corpus."""

    model_ready_dir: Path
    action_grid_file: Path
    metadata_file: Path | None = None
    media_manifest_file: Path | None = None
    name: str = "local"


type DataSource = HuggingFaceSource | LocalSource


def _private_corpus(name: str) -> CorpusConfig:
    return CorpusConfig(
        name=name,
        model_ready_config=f"{name}_model_ready",
        action_grid_config=f"{name}_action_grid",
        media_manifest_config=f"{name}_media_manifest",
        metadata_file=f"{name}/metadata.json",
    )


_PRIVATE_REPO = "batgre/conversational-dynamics-full"

EGOCOM = HuggingFaceSource(
    repo_id="batgre/conversational-dynamics-egocom",
    corpora=(
        CorpusConfig(
            name="egocom",
            model_ready_config="model_ready",
            action_grid_config="action_grid",
            media_manifest_config="media_manifest",
        ),
    ),
)

# The private release publishes each corpus as its own set of configs.
EGO4D = HuggingFaceSource(
    repo_id=_PRIVATE_REPO,
    corpora=(_private_corpus("ego4d"),),
)

FULL = HuggingFaceSource(
    repo_id=_PRIVATE_REPO,
    corpora=(_private_corpus("egocom"), _private_corpus("ego4d")),
)

# Published datasets addressable by name from the CLI and experiments.
DATASETS: dict[str, HuggingFaceSource] = {
    "egocom": EGOCOM,
    "ego4d": EGO4D,
    "full": FULL,
}


@dataclass(frozen=True)
class LoadedCorpus:
    """One corpus's canonical artifacts, kept together.

    `model_ready.anchor_row` indexes this corpus's `action_grid` only.
    """

    name: str
    model_ready: DatasetDict
    action_grid: Dataset
    metadata: dict[str, object]
    media_manifest: Dataset | None = None


@dataclass(frozen=True)
class LoadedData:
    """Canonical data exposed to the modelling repository."""

    corpora: tuple[LoadedCorpus, ...]
    # Commit every artifact was loaded from, when resolvable.
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("LoadedData requires at least one corpus")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(corpus.name for corpus in self.corpora)

    def corpus(self, name: str) -> LoadedCorpus:
        for corpus in self.corpora:
            if corpus.name == name:
                return corpus

        raise KeyError(f"No corpus {name!r}; loaded: {list(self.names)}")


def load_data(source: DataSource) -> LoadedData:
    """Load a dataset source and validate its modelling contract."""

    if isinstance(source, HuggingFaceSource):
        data = _load_huggingface(source)
    else:
        data = LoadedData(corpora=(_load_local(source),))

    for corpus in data.corpora:
        _validate_model_ready(corpus.model_ready, corpus=corpus.name)
        _validate_action_grid(corpus.action_grid, corpus=corpus.name)

        if corpus.media_manifest is not None:
            _validate_media_manifest(corpus.media_manifest, corpus=corpus.name)

    return data


def _load_huggingface(source: HuggingFaceSource) -> LoadedData:
    # Every artifact of every corpus must come from the same commit, even when
    # the configured revision is a moving branch.
    revision = _resolve_revision(source)

    return LoadedData(
        corpora=tuple(
            _load_huggingface_corpus(source.repo_id, corpus, revision=revision)
            for corpus in source.corpora
        ),
        revision=revision,
    )


def _load_huggingface_corpus(
    repo_id: str,
    corpus: CorpusConfig,
    *,
    revision: str | None,
) -> LoadedCorpus:
    model_ready = load_dataset(
        repo_id,
        corpus.model_ready_config,
        revision=revision,
    )

    action_grid = load_dataset(
        repo_id,
        corpus.action_grid_config,
        split="train",
        revision=revision,
    )

    media_manifest = None

    if corpus.media_manifest_config is not None:
        media_manifest = load_dataset(
            repo_id,
            corpus.media_manifest_config,
            split="train",
            revision=revision,
        )

    metadata_path = hf_hub_download(
        repo_id=repo_id,
        filename=corpus.metadata_file,
        repo_type="dataset",
        revision=revision,
    )

    return LoadedCorpus(
        name=corpus.name,
        model_ready=model_ready,
        action_grid=action_grid,
        metadata=_read_json(Path(metadata_path)),
        media_manifest=media_manifest,
    )


def _resolve_revision(source: HuggingFaceSource) -> str | None:
    """Pin the configured revision to a commit for the duration of one load.

    Offline, fall back to the configured revision so cached data stays usable.
    """

    try:
        info = HfApi().dataset_info(source.repo_id, revision=source.revision)
    except (ConnectionError, httpx.TransportError):
        return source.revision

    return info.sha or source.revision


def _load_local(source: LocalSource) -> LoadedCorpus:
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

    media_manifest = None

    if source.media_manifest_file is not None:
        if not source.media_manifest_file.exists():
            raise FileNotFoundError(
                f"Media manifest not found: {source.media_manifest_file}"
            )

        media_manifest = load_dataset(
            "parquet",
            data_files={"train": str(source.media_manifest_file)},
            split="train",
        )

    return LoadedCorpus(
        name=source.name,
        model_ready=model_ready,
        action_grid=action_grid,
        metadata=metadata,
        media_manifest=media_manifest,
    )


def _validate_model_ready(dataset: DatasetDict, *, corpus: str) -> None:
    if "train" not in dataset:
        raise ValueError(f"{corpus}/model_ready must contain a train split")

    for split_name, split in dataset.items():
        _require_columns(
            split,
            MODEL_READY_REQUIRED_COLUMNS,
            artifact=f"{corpus}/model_ready/{split_name}",
        )

        declared_splits = set(split.unique("split"))

        if declared_splits != {split_name}:
            raise ValueError(
                f"{corpus}/model_ready/{split_name} contains split values "
                f"{sorted(declared_splits)}"
            )


def _validate_action_grid(dataset: Dataset, *, corpus: str) -> None:
    _require_columns(
        dataset,
        ACTION_GRID_REQUIRED_COLUMNS,
        artifact=f"{corpus}/action_grid",
    )


def _validate_media_manifest(dataset: Dataset, *, corpus: str) -> None:
    _require_columns(
        dataset,
        MEDIA_MANIFEST_REQUIRED_COLUMNS,
        artifact=f"{corpus}/media_manifest",
    )

    table = dataset.with_format("arrow")[:]

    without_media = pc.and_(
        pc.is_null(table["video_path"]),
        pc.is_null(table["audio_path"]),
    )

    if pc.any(without_media).as_py():
        raise ValueError(f"{corpus}/media_manifest has records without video or audio")

    keys = table.group_by(["dataset", "recording_id"]).aggregate([])

    if keys.num_rows != table.num_rows:
        raise ValueError(
            f"{corpus}/media_manifest has duplicate (dataset, recording_id)"
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
        raise ValueError(f"{artifact} is missing required columns: {sorted(missing)}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)
