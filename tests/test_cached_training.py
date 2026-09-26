"""
Training on cached Mimi features: the model's projection boundary, raw vs
cached equivalence, and end-to-end training without Mimi, network or media.
"""

import lightning as L
import pytest
import torch
from datasets import Dataset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from turn_wm.config import load_config
from turn_wm.data.dataset import TurnTakingDataset
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.models.build import build_model
from turn_wm.models.encoders import mimi as mimi_module
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.lewm import (
    LeWMModule,
    Trajectories,
    lejepa_losses,
    training_window,
    trajectories,
)

FIRST_INDEX = 5
GRID_LENGTH = 80


class FakeEncoder(nn.Module):
    """Deterministic stand-in for Mimi: waveform -> (B, T, 512)."""

    def forward(self, waveform, sample_rate, target_length):
        rows = []

        for window in waveform:
            chunks = window.mean(dim=0).chunk(target_length)
            rows.append(torch.stack([chunk.mean() for chunk in chunks]))

        steps = torch.stack(rows)  # (B, T)
        scales = torch.linspace(0.5, 1.5, 512)

        return steps.unsqueeze(-1) * scales


@pytest.fixture
def no_mimi(monkeypatch):
    """Any attempt to load Mimi's weights fails the test."""

    def refuse(*args, **kwargs):
        raise AssertionError("MimiModel.from_pretrained must not be called")

    monkeypatch.setattr(mimi_module.MimiModel, "from_pretrained", refuse)


def cached_config(root=None, **overrides):
    cfg = load_config(
        [
            "data.observation_source=mimi_cache",
            "trainer.accelerator=cpu",
            "trainer.precision=32-true",
        ]
    )
    OmegaConf.set_struct(cfg, False)
    cfg.data.mimi_cache.root = None if root is None else str(root)

    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)

    return cfg


def make_grid() -> Dataset:
    indices = list(range(FIRST_INDEX, FIRST_INDEX + GRID_LENGTH))

    return Dataset.from_dict(
        {
            "dataset": ["egocom"] * GRID_LENGTH,
            "recording_id": ["r1"] * GRID_LENGTH,
            "decision_index": indices,
            "decision_time_s": [index / 10 for index in indices],
            "focal_state_before": ["SILENT"] * GRID_LENGTH,
            "action": ["NO_EVENT", "ONSET", "OFFSET"] * (GRID_LENGTH // 3)
            + ["NO_EVENT"] * (GRID_LENGTH % 3),
            "action_valid": [True] * GRID_LENGTH,
        }
    )


def make_anchors(count: int) -> Dataset:
    anchors = [FIRST_INDEX + 20 + index for index in range(count)]

    return Dataset.from_dict(
        {
            "sample_id": [f"egocom:train#{index}" for index in anchors],
            "dataset": ["egocom"] * count,
            "recording_id": ["r1"] * count,
            "anchor_idx": anchors,
            "anchor_row": [index - FIRST_INDEX for index in anchors],
            "anchor_time": [index / 10 for index in anchors],
            "max_context_steps": [20] * count,
            "future_steps": [10] * count,
            "sample_class": ["event"] * count,
            "is_trainable": [True] * count,
        }
    )


@pytest.fixture
def cache_root(make_mimi_cache):
    return make_mimi_cache({("egocom", "r1"): (FIRST_INDEX, GRID_LENGTH)})


def cached_loader(cfg, root, *, count=4, batch_size=2, training=True):
    dataset = TurnTakingDataset(
        anchors=make_anchors(count),
        action_grid=make_grid(),
        window=training_window(cfg),
        training=training,
        modalities=("audio",),
        mimi_store=MimiFeatureStore(root),
    )

    return build_dataloader(
        dataset, loader=DataLoaderConfig(batch_size=batch_size, shuffle=False)
    )


# ---------------------------------------------------------------------------
# One projection path
# ---------------------------------------------------------------------------


def raw_model(cfg=None):
    cfg = cfg or cached_config()
    return instantiate(cfg.model, encoder=FakeEncoder())


def test_project_features_is_exactly_the_projector():
    model = raw_model()
    features = torch.randn(2, 7, 512)

    expected = model.projector(features.reshape(14, 512)).reshape(2, 7, -1)

    torch.testing.assert_close(
        model.project_features(features), expected, rtol=0, atol=0
    )


def test_encode_is_the_encoder_then_the_same_projection():
    model = raw_model()
    waveforms = [torch.randn(1, 2_500), torch.randn(2, 2_500)]

    raw = model.encode(waveforms, sample_rate=[1_000, 1_000], target_length=25)
    cached = model.project_features(
        model.encoder(waveforms, sample_rate=[1_000, 1_000], target_length=25)
    )

    torch.testing.assert_close(raw, cached, rtol=0, atol=0)


def test_raw_and_cached_trajectories_give_identical_losses():
    # Same features through the raw path (encoder) and the cached path.
    cfg = cached_config()
    model = raw_model(cfg)
    context, future = cfg.data.context_steps, cfg.data.future_steps
    steps = context + future
    waveforms = [torch.randn(1, steps * 100) for _ in range(2)]
    actions = torch.zeros(2, steps, dtype=torch.long)

    raw = Trajectories(
        actions=actions,
        context_steps=context,
        future_steps=future,
        waveforms=waveforms,
        sample_rates=[1_000, 1_000],
    )
    cached = Trajectories(
        actions=actions,
        context_steps=context,
        future_steps=future,
        features=model.encoder(
            waveforms, sample_rate=[1_000, 1_000], target_length=steps
        ),
    )

    torch.manual_seed(0)
    raw_losses = lejepa_losses(model, SIGReg(), raw, cfg)
    torch.manual_seed(0)
    cached_losses = lejepa_losses(model, SIGReg(), cached, cfg)

    for name, value in raw_losses.items():
        torch.testing.assert_close(cached_losses[name], value, rtol=0, atol=0)


def test_float16_cached_features_are_projected():
    model = raw_model()
    features = torch.randn(2, 5, 512)

    torch.testing.assert_close(
        model.project_features(features.half()),
        model.project_features(features.half().float()),
        rtol=0,
        atol=0,
    )


def test_features_must_be_batched_sequences():
    with pytest.raises(ValueError, match=r"shape \(B, T, D\)"):
        raw_model().project_features(torch.randn(5, 512))


def test_model_without_encoder_refuses_raw_observations(no_mimi):
    model = build_model(cached_config())

    assert model.encoder is None

    with pytest.raises(ValueError, match="Raw observation encoding is unavailable"):
        model.encode([torch.randn(1, 100)], sample_rate=[1_000], target_length=1)


def test_trajectories_have_exactly_one_observation_source():
    actions = torch.zeros(1, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="exactly one observation source"):
        Trajectories(actions=actions, context_steps=2, future_steps=1)

    with pytest.raises(ValueError, match="exactly one observation source"):
        Trajectories(
            actions=actions,
            context_steps=2,
            future_steps=1,
            features=torch.zeros(1, 3, 512),
            waveforms=[torch.zeros(1, 30)],
            sample_rates=[1_000],
        )


# ---------------------------------------------------------------------------
# Mimi is never loaded in cached mode; raw mode still builds it
# ---------------------------------------------------------------------------


def test_cached_mode_builds_the_model_without_mimi(no_mimi):
    module = LeWMModule(cached_config())

    assert module.model.encoder is None
    assert not [
        name
        for name, _ in module.named_parameters()
        if name.startswith("model.encoder.")
    ]


def test_raw_mode_still_builds_the_frozen_mimi_encoder(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("stop after the encoder is requested")

    monkeypatch.setattr(mimi_module.MimiModel, "from_pretrained", record)

    with pytest.raises(Exception, match="stop after the encoder is requested"):
        build_model(cached_config(**{"data.observation_source": "raw_audio"}))

    assert calls == [("kyutai/mimi",)]


def test_trajectories_from_a_cached_batch(cache_root):
    cfg = cached_config(cache_root)
    batch = next(iter(cached_loader(cfg, cache_root, training=False)))

    result = trajectories(batch)

    assert result.waveforms is None
    assert result.features is not None
    assert result.features.shape == (2, 25, 512)
    # First sample's anchor is decision_index 25: context 11..25, future 26..35.
    assert result.features[0, :, 0].tolist() == list(range(11, 36))


def test_cached_losses_backpropagate_to_projector_and_predictor(no_mimi, cache_root):
    cfg = cached_config(cache_root)
    module = LeWMModule(cfg)
    batch = next(iter(cached_loader(cfg, cache_root)))

    output = lejepa_losses(module.model, module.sigreg, trajectories(batch), cfg)
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    projector_grads = [p.grad for p in module.model.projector.parameters()]
    predictor_grads = [p.grad for p in module.model.predictor.parameters()]
    assert all(grad is not None for grad in projector_grads)
    assert any(grad is not None and grad.abs().sum() > 0 for grad in predictor_grads)


def test_lightning_trains_on_cached_features_without_mimi(
    no_mimi, cache_root, tmp_path
):
    cfg = cached_config(cache_root)
    module = LeWMModule(cfg)
    before = [p.detach().clone() for p in module.model.projector.parameters()]

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        module,
        train_dataloaders=cached_loader(cfg, cache_root),
        val_dataloaders=cached_loader(cfg, cache_root, training=False),
    )

    after = list(module.model.projector.parameters())

    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["train/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
    # The projector trained on the cached features.
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_cached_losses_run_under_cpu_bf16_autocast(no_mimi, cache_root):
    # bf16-mixed on CPU rejected the float16 cached features in torch.cat.
    cfg = cached_config(cache_root)
    module = LeWMModule(cfg)
    batch = next(iter(cached_loader(cfg, cache_root)))

    assert batch["context_features"].dtype == torch.float16

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = lejepa_losses(module.model, module.sigreg, trajectories(batch), cfg)

    assert torch.isfinite(output["loss"].float())
