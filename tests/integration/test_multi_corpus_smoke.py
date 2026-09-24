"""
Real multi-corpus smoke test: private EgoCom + Ego4D release → one Dataset.

Uses only the public construction path (`load_data` → `build_dataset` →
`build_dataloader`). Needs access to the private release; skipped otherwise.

    uv run pytest -m integration
    EGOCOM_MEDIA_ROOT=... EGO4D_MEDIA_ROOT=... uv run pytest -m integration
"""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

from turn_wm.data.build import build_dataset
from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.dataset import TurnTakingDataset, WindowConfig
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.sampling import SamplingConfig
from turn_wm.data.source import FULL, load_data

pytestmark = pytest.mark.integration

WINDOW = WindowConfig()


@pytest.fixture(scope="module")
def full():
    try:
        return load_data(FULL)
    except Exception as error:  # noqa: BLE001 - access depends on credentials
        pytest.skip(f"{FULL.repo_id} is not accessible: {error}")


@pytest.fixture(scope="module")
def train_eval(full):
    return build_dataset(full, split="train", window=WINDOW, training=False)


def trainable_count(full, corpus: str, split: str) -> int:
    anchors = full.corpus(corpus).model_ready[split]
    return pc.sum(anchors.data.column("is_trainable")).as_py()


@pytest.mark.parametrize(
    ("split", "corpora"),
    [
        ("train", {"egocom", "ego4d"}),
        ("validation", {"egocom", "ego4d"}),
        ("test", {"egocom"}),  # Ego4D publishes no test split
    ],
)
def test_split_includes_publishing_corpora(full, split, corpora):
    dataset = build_dataset(full, split=split, window=WINDOW, training=False)

    assert set(dataset.corpora) == corpora
    assert dataset.corpus_sizes() == {
        name: trainable_count(full, name, split) for name in dataset.corpora
    }
    assert len(dataset) == sum(dataset.corpus_sizes().values())


def test_boundaries_route_to_corpus_local_samples(full, train_eval):
    """Global indices at each corpus boundary resolve against that corpus."""

    start = 0

    for name, size in train_eval.corpus_sizes().items():
        local = TurnTakingDataset(
            anchors=full.corpus(name).model_ready["train"],
            action_grid=full.corpus(name).action_grid,
            window=WINDOW,
            training=False,
        )

        for global_index, local_index in ((start, 0), (start + size - 1, -1)):
            sample = train_eval[global_index]
            expected = local[local_index]

            assert sample["dataset"] == name
            assert sample["sample_id"] == expected["sample_id"]

            for key in ("context_state", "context_action", "future_state"):
                assert torch.equal(sample[key], expected[key]), (name, key)

        start += size


def test_shuffled_training_batch_mixes_corpora(full):
    dataset = build_dataset(full, split="train", window=WINDOW, training=True)
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=64, num_workers=0, seed=0),
        sampling=SamplingConfig(strategy="natural"),
    )

    batch = next(iter(loader))

    # Seeded natural shuffle over ~2.2M anchors, roughly half from each.
    assert set(batch["dataset"]) == {"egocom", "ego4d"}
    assert batch["future_state"].shape == (64, WINDOW.future_steps)
    assert batch["context_state"].shape[0] == 64


def test_balanced_sampling_spans_both_corpora(full):
    dataset = build_dataset(full, split="train", window=WINDOW, training=True)
    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=64, num_workers=0, seed=0),
        sampling=SamplingConfig(strategy="balanced", seed=0),
    )

    assert len(loader.sampler.weights) == len(dataset)

    batch = next(iter(loader))

    assert set(batch["sample_class"]) == {"event", "background"}


def media_roots() -> dict[str, Path]:
    roots = {
        "egocom": os.environ.get("EGOCOM_MEDIA_ROOT"),
        "ego4d": os.environ.get("EGO4D_MEDIA_ROOT"),
    }

    missing = [name for name, root in roots.items() if not root]

    if missing:
        pytest.skip(f"Local media roots not set for: {missing}")

    for name, root in roots.items():
        if not Path(root).is_dir():
            pytest.fail(f"Media root for {name} is not a directory: {root}")

    return {name: Path(root) for name, root in roots.items()}


def first_local_index(full, corpus: str, root: Path, *, offset_filter) -> int:
    """Local index (among trainable anchors) of a sample with local media."""

    manifest = full.corpus(corpus).media_manifest.with_format("arrow")[:]
    available = {
        recording
        for recording, path, offset in zip(
            manifest["recording_id"].to_pylist(),
            manifest["video_path"].to_pylist(),
            manifest["media_offset_s"].to_pylist(),
            strict=True,
        )
        if (root / path).is_file() and offset_filter(offset)
    }

    if not available:
        pytest.skip(f"No local {corpus} media matching the filter under {root}")

    anchors = full.corpus(corpus).model_ready["train"].data.table
    trainable = anchors.filter(anchors["is_trainable"])["recording_id"]
    value_set = pa.array(sorted(available), type=trainable.type)
    positions = pc.indices_nonzero(pc.is_in(trainable, value_set=value_set))

    if len(positions) == 0:
        pytest.skip(f"No local {corpus} media matching the filter under {root}")

    return positions[0].as_py()


@pytest.mark.parametrize(
    ("modalities", "audio", "video"),
    [
        (None, True, True),
        (("audio",), True, False),
        (("video",), False, True),
    ],
)
def test_real_media_flows_through_one_pipeline(full, modalities, audio, video):
    roots = media_roots()
    selection = {} if modalities is None else {"modalities": modalities}

    dataset = build_dataset(
        full,
        split="train",
        window=WINDOW,
        training=False,
        media_roots=roots,
        **selection,
    )
    # Global start of each corpus, whatever order the builder chose.
    starts, total = {}, 0

    for name, size in dataset.corpus_sizes().items():
        starts[name] = total
        total += size

    egocom_index = starts["egocom"] + first_local_index(
        full, "egocom", roots["egocom"], offset_filter=lambda offset: True
    )
    ego4d_index = starts["ego4d"] + first_local_index(
        full, "ego4d", roots["ego4d"], offset_filter=lambda offset: offset > 1.0
    )

    # Same Dataset, same MediaReader, same collate as the DataLoader uses.
    batch = collate_turn_taking([dataset[egocom_index], dataset[ego4d_index]])

    assert batch["dataset"] == ["egocom", "ego4d"]

    for position, zero_offset in ((0, True), (1, False)):
        context = batch["context_media"][position]
        future = batch["future_media"][position]
        offset = context.start_time_s - context.canonical_start_time_s

        assert (offset == pytest.approx(0.0)) is zero_offset
        assert future.start_time_s == context.end_time_s

        for window in (context, future):
            if video:
                assert window.video is not None and window.video.frames.shape[0] > 0
            else:
                assert window.video is None

            if audio:
                assert window.audio is not None
                assert window.audio.waveform.shape[1] > 0
            else:
                assert window.audio is None
