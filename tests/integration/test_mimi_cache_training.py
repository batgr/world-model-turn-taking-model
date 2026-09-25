"""
Real Mimi cache: published dataset + a local cache from `turn-wm
precompute-mimi` -> aligned features -> LeWM losses. No raw media, no Mimi.

    MIMI_CACHE_ROOT=/path/to/mimi-cache uv run pytest -m integration

The cache's datasets decide what is loaded (EgoCom from the public release,
otherwise the private full release). Skipped when the variable is unset.
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from turn_wm.config import load_config
from turn_wm.data.build import build_dataset
from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.source import EGOCOM, FULL, load_data
from turn_wm.models.encoders import mimi as mimi_module
from turn_wm.training.lewm import (
    LeWMModule,
    lejepa_losses,
    training_window,
    trajectories,
)
from turn_wm.training.train import validate_mimi_cache

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store() -> MimiFeatureStore:
    root = os.environ.get("MIMI_CACHE_ROOT")

    if not root:
        pytest.skip("MIMI_CACHE_ROOT is not set; no local Mimi cache")

    return MimiFeatureStore(Path(root))


def test_real_cache_trains_without_mimi_or_media(store, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("MimiModel.from_pretrained must not be called")

    monkeypatch.setattr(mimi_module.MimiModel, "from_pretrained", refuse)

    datasets = {dataset for dataset, _ in store._records}
    source = EGOCOM if datasets == {"egocom"} else FULL
    loaded = load_data(source)

    cfg = load_config(
        [
            "data.observation_source=mimi_cache",
            f"data.mimi_cache.root={store.root}",
            "trainer.accelerator=cpu",
        ]
    )
    OmegaConf.set_struct(cfg, False)
    validate_mimi_cache(store, loaded, cfg)

    # Only anchors of cached recordings, so a partial cache works too.
    cached = {recording for _, recording in store._records}
    loaded = replace(
        loaded,
        corpora=tuple(
            replace(
                corpus,
                model_ready=corpus.model_ready.filter(
                    lambda rows: [r in cached for r in rows["recording_id"]],
                    batched=True,
                ),
            )
            for corpus in loaded.corpora
            if corpus.name in datasets
        ),
    )

    dataset = build_dataset(
        loaded,
        split="train",
        window=training_window(cfg),
        training=True,
        modalities=("audio",),
        mimi_store=store,
    )
    samples = [dataset[0], dataset[len(dataset) // 2]]

    for sample in samples:
        assert sample["context_features"].shape == (cfg.data.context_steps, 512)
        assert sample["future_features"].shape == (cfg.data.future_steps, 512)
        assert torch.isfinite(sample["context_features"]).all()
        assert "context_media" not in sample

    # Row alignment: the window's first row is anchor - C + 1 on the grid.
    sample = samples[0]
    record = store.record(
        dataset=sample["dataset"], recording_id=sample["recording_id"]
    )
    first = sample["anchor_idx"] - cfg.data.context_steps + 1
    expected = store.get(
        dataset=sample["dataset"],
        recording_id=sample["recording_id"],
        start=first - record.start_index,
        end=first - record.start_index + cfg.data.context_steps,
    )
    torch.testing.assert_close(sample["context_features"], expected, rtol=0, atol=0)

    module = LeWMModule(cfg)
    batch = collate_turn_taking(samples)
    output = lejepa_losses(module.model, module.sigreg, trajectories(batch), cfg)
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
