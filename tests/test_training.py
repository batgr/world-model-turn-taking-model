import lightning as L
import pytest
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from turn_wm.config import load_config
from turn_wm.data.dataset import MASKED_ACTION_ID, PAD_ACTION_ID
from turn_wm.data.reader import DecodedAudio, MediaWindow
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training import lewm as training_module
from turn_wm.training.lewm import (
    LeWMModule,
    lejepa_losses,
    training_window,
    trajectories,
    validate_config,
)

SAMPLE_RATE = 1_000
SAMPLES_PER_STEP = SAMPLE_RATE // 10


class StepIndexEncoder(nn.Module):
    """Frozen stand-in for Mimi: the feature of grid step t is t everywhere."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls: list[dict] = []

    def forward(self, waveform, sample_rate, target_length):
        self.calls.append(
            {"lengths": [w.shape[-1] for w in waveform], "sample_rate": sample_rate}
        )
        steps = torch.arange(target_length, dtype=torch.float32)

        return steps.view(1, -1, 1).expand(len(waveform), -1, self.dim).clone()


def small_config(**overrides):
    cfg = load_config(
        [
            "data.context_steps=4",
            "data.future_steps=3",
            "prediction.rollout_context_size=2",
        ]
    )
    OmegaConf.set_struct(cfg, False)
    cfg.prediction.rollout_horizons = [1, 3]
    cfg.loss.rollout.horizon_weights = {"1": 1.0, "3": 0.5}

    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)

    return cfg


def make_model(cfg, *, encoder_dim=512, **kwargs):
    return instantiate(cfg.model, encoder=StepIndexEncoder(encoder_dim), **kwargs)


def window(steps: int, *, rate: int = SAMPLE_RATE, with_audio: bool = True):
    audio = DecodedAudio(
        waveform=torch.randn(2, steps * rate // 10),
        sample_rate=rate,
    )

    return MediaWindow(
        start_time_s=0.0,
        end_time_s=steps / 10,
        audio=audio if with_audio else None,
        video=None,
    )


def make_batch(context_steps=4, future_steps=3, *, size=2, context_lengths=None):
    lengths = context_lengths or [context_steps] * size

    return {
        "sample_id": [f"s{i}" for i in range(size)],
        "context_lengths": torch.tensor(lengths),
        "context_action": torch.zeros(size, max(lengths), dtype=torch.long),
        "future_action": torch.ones(size, future_steps, dtype=torch.long),
        "context_media": [window(length) for length in lengths],
        "future_media": [window(future_steps) for _ in lengths],
    }


def test_trajectories_join_context_and_future():
    batch = make_batch(context_steps=4, future_steps=3)
    batch["context_media"][1] = window(4, rate=2_000)
    batch["future_media"][1] = window(3, rate=2_000)

    result = trajectories(batch)

    assert (result.context_steps, result.future_steps) == (4, 3)
    assert result.actions.tolist() == [[0, 0, 0, 0, 1, 1, 1]] * 2
    assert result.sample_rates == [SAMPLE_RATE, 2_000]
    assert [w.shape for w in result.waveforms] == [(2, 700), (2, 1400)]


def test_trajectories_need_one_context_length():
    with pytest.raises(ValueError, match="one context length"):
        trajectories(make_batch(context_lengths=[4, 3]))


def test_trajectories_need_audio():
    batch = make_batch()
    batch["future_media"][1] = window(3, with_audio=False)

    with pytest.raises(ValueError, match="s1 has no audio"):
        trajectories(batch)


def test_trajectories_need_media():
    batch = make_batch()
    del batch["context_media"]

    with pytest.raises(ValueError, match="no media"):
        trajectories(batch)


def predict_spy(model, record):
    """Record every predictor call, then run the real predictor."""

    predict = model.predict

    def spy(emb, act):
        record.append({"emb": emb.detach().clone(), "act": act.detach().clone()})
        return predict(emb, act)

    model.predict = spy


def test_teacher_forcing_is_dense_over_the_whole_context():
    # context_steps = 4: z0..z3 -> z1..z4, whatever rollout_context_size is.
    cfg = small_config()
    # Without projector, latents are the step indices at the predictor width.
    model = make_model(cfg, encoder_dim=cfg.embed_dim, projector=None)
    calls = []
    predict_spy(model, calls)

    lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    teacher_forcing = calls[0]["emb"]

    assert teacher_forcing.shape[1] == 4
    assert teacher_forcing[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_teacher_forcing_targets_are_the_next_four_steps(monkeypatch):
    cfg = small_config()
    # Without projector, latents are the step indices at the predictor width.
    model = make_model(cfg, encoder_dim=cfg.embed_dim, projector=None)
    targets = []
    mse = F.mse_loss

    def spy(prediction, target):
        targets.append(target.detach().clone())
        return mse(prediction, target)

    monkeypatch.setattr(training_module.F, "mse_loss", spy)

    lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    # First MSE is the teacher forcing: z1..z4, the last one being zC.
    assert targets[0][0, :, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_rollout_never_passes_more_than_rollout_context_size():
    cfg = small_config()
    model = make_model(cfg)
    calls = []
    predict_spy(model, calls)

    lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    # Teacher forcing, then one call per rollout step up to horizon 3.
    assert [call["emb"].shape[1] for call in calls] == [4, 2, 2, 2]


def test_rollout_never_reinjects_ground_truth_future_latents():
    # Ground-truth latents are 0..6 (context 0..3, future 4..6); predictions
    # are shifted by 100, so any input in [4, 100) is a leaked GT future.
    cfg = small_config()
    model = make_model(cfg, projector=None)
    calls = []

    def predict(emb, act):
        calls.append(emb.detach().clone())
        return emb + 100

    model.predict = predict

    lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    for rollout_input in calls[1:]:
        values = rollout_input[0, :, 0]
        assert not ((values >= 4) & (values < 100)).any()

    # The window slides over predictions only: ẑ4 = z3 + 100, ẑ5 = ẑ4 + 100.
    assert [call[0, :, 0].tolist() for call in calls[1:]] == [
        [2.0, 3.0],
        [3.0, 103.0],
        [103.0, 203.0],
    ]


def test_rollout_uses_the_real_future_actions():
    cfg = small_config()
    model = make_model(cfg)
    calls = []
    predict_spy(model, calls)
    batch = make_batch()
    batch["context_action"] = torch.tensor([[0, 0, 0, 1]] * 2)
    batch["future_action"] = torch.tensor([[2, 0, 1]] * 2)

    lejepa_losses(model, SIGReg(), trajectories(batch), cfg)

    embed = model.encode_actions
    expected = embed(torch.tensor([[0, 1], [1, 2], [2, 0]]))

    for call, actions in zip(calls[1:], expected, strict=True):
        assert torch.allclose(call["act"][0], actions)


def test_default_config_no_longer_overflows_positions():
    cfg = load_config()
    model = make_model(cfg)
    context, future = cfg.data.context_steps, cfg.data.future_steps

    output = lejepa_losses(
        model, SIGReg(), trajectories(make_batch(context, future)), cfg
    )

    assert torch.isfinite(output["loss"])


def test_targets_are_one_step_ahead():
    # With features equal to the step index and a predictor returning
    # "input + 1", the dense teacher forcing (z0..z3 -> z1..z4) and every
    # rolled-out horizon (ẑ4, ẑ6) match their targets exactly.
    cfg = small_config()
    model = make_model(cfg, projector=None)
    model.predict = lambda emb, act: emb + 1

    output = lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    assert output["tf_loss"].item() == 0.0
    assert output["rollout_1_loss"].item() == 0.0
    assert output["rollout_3_loss"].item() == 0.0


def test_losses_are_finite_and_train_only_the_predictor_side():
    cfg = small_config()
    model = make_model(cfg)

    output = lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)
    output["loss"].backward()

    assert set(output) == {
        "loss",
        "tf_loss",
        "rollout_loss",
        "sigreg_loss",
        "rollout_1_loss",
        "rollout_3_loss",
    }
    assert all(torch.isfinite(value) for value in output.values())
    assert model.predictor.pos_embedding.grad is not None
    assert model.encoder.frozen.grad is None


def test_encoder_receives_whole_trajectories():
    cfg = small_config()
    model = make_model(cfg)

    lejepa_losses(model, SIGReg(), trajectories(make_batch()), cfg)

    assert model.encoder.calls == [
        {"lengths": [700, 700], "sample_rate": [SAMPLE_RATE, SAMPLE_RATE]}
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"data.context_steps": 0}, "context_steps must be >= 1"),
        ({"prediction.rollout_context_size": 0}, "rollout_context_size must lie"),
        ({"prediction.rollout_context_size": 5}, "rollout_context_size must lie"),
        ({"model.predictor.num_frames": 3}, "must cover the teacher-forced"),
        ({"prediction.rollout_horizons": [0, 1]}, "must be positive"),
        ({"prediction.rollout_horizons": []}, "must be positive"),
        ({"prediction.rollout_horizons": [1, 4]}, "must cover every rollout"),
        ({"prediction.rollout_horizons": [2]}, "no weight for \\[2\\]"),
    ],
)
def test_invalid_training_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_config(small_config(**overrides))


def test_default_training_config_is_valid():
    cfg = load_config()

    validate_config(cfg)

    window_config = training_window(cfg)
    assert window_config.min_context_steps == window_config.max_context_steps
    assert window_config.max_context_steps == cfg.data.context_steps
    assert cfg.prediction.rollout_context_size <= cfg.data.context_steps
    assert cfg.model.predictor.num_frames == cfg.data.context_steps


def test_action_vocabulary_matches_the_dataset():
    embedder = load_config().model.action_encoder

    assert embedder.padding_idx == PAD_ACTION_ID
    assert embedder.num_actions == PAD_ACTION_ID + 1
    assert MASKED_ACTION_ID < embedder.padding_idx


def test_sigreg_accepts_bfloat16():
    # The random projection used to be float32 whatever the input dtype.
    latents = torch.randn(4, 8, 16, dtype=torch.bfloat16)

    assert torch.isfinite(SIGReg(num_proj=32)(latents))


def test_optimizer_skips_frozen_parameters():
    cfg = small_config()
    module = LeWMModule(cfg, model=make_model(cfg))

    optimizer = module.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert isinstance(optimizer, torch.optim.AdamW)
    assert id(module.model.encoder.frozen) not in optimized
    assert id(module.model.predictor.pos_embedding) in optimized


def test_lightning_runs_a_training_and_validation_step(tmp_path):
    cfg = small_config()
    module = LeWMModule(cfg, model=make_model(cfg))
    batches = DataLoader([make_batch()], batch_size=None)

    trainer = L.Trainer(
        accelerator="cpu",
        fast_dev_run=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=batches, val_dataloaders=batches)

    # Training logs per step and per epoch (`_step`/`_epoch` suffixes);
    # validation logs per epoch only, under the name ModelCheckpoint monitors.
    assert torch.isfinite(trainer.callback_metrics["train/loss_epoch"])
    assert torch.isfinite(trainer.callback_metrics["val/loss"])


def test_module_leaves_seeding_to_the_caller():
    # The runner seeds; the module must not reseed over it. Identity avoids
    # loading Mimi.
    cfg = small_config(**{"model.encoder._target_": "torch.nn.Identity"})

    torch.manual_seed(0)
    first = LeWMModule(cfg).model.predictor.pos_embedding
    torch.manual_seed(0)
    second = LeWMModule(cfg).model.predictor.pos_embedding
    torch.manual_seed(1)
    other = LeWMModule(cfg).model.predictor.pos_embedding

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
