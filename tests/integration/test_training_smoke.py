"""
Real-data training smoke test: EgoCom audio → Mimi → LeWM losses.

Needs local EgoCom media (EGOCOM_MEDIA_ROOT) and the Mimi weights (downloaded
from the Hub on first use).
"""

import os
from pathlib import Path

import pytest
import torch

from turn_wm.config import load_config
from turn_wm.data.build import build_dataset
from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.source import EGOCOM, load_data
from turn_wm.training.lewm import (
    LeWMModule,
    lejepa_losses,
    training_window,
    trajectories,
)

pytestmark = pytest.mark.integration


def test_lewm_losses_on_real_egocom_audio():
    root = os.environ.get("EGOCOM_MEDIA_ROOT")

    if not root:
        pytest.skip("EGOCOM_MEDIA_ROOT is not set; local raw media unavailable")

    # The raw-audio path: this test is about decoding and encoding real audio.
    cfg = load_config(["data.observation_source=raw_audio"])
    data = load_data(EGOCOM)
    dataset = build_dataset(
        data,
        split="train",
        window=training_window(cfg),
        training=True,
        media_roots={"egocom": Path(root)},
        modalities=tuple(cfg.data.modalities),
    )

    batch = collate_turn_taking([dataset[0], dataset[1]])
    module = LeWMModule(cfg)

    assert set(batch["context_lengths"].tolist()) == {cfg.data.context_steps}

    output = lejepa_losses(module.model, module.sigreg, trajectories(batch), cfg)
    output["loss"].backward()

    assert all(torch.isfinite(value) for value in output.values())
    assert module.model.predictor.pos_embedding.grad is not None
    assert all(p.grad is None for p in module.model.encoder.parameters())
