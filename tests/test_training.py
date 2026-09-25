from types import SimpleNamespace

import lightning as L
import pytest
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from turn_wm.config import load_config
from turn_wm.data.dataset import MASKED_ACTION_ID, PAD_ACTION_ID
from turn_wm.data.reader import DecodedAudio, MediaWindow
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training import lewm as training_module
from turn_wm.training.lewm import (
    LeWMModule,
    lejepa_losses,
    rollout_horizons_for_progress,
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
    cfg.prediction.curriculum.stages = [
        {"until": 0.5, "horizons": [1]},
        {"until": 1.0, "horizons": [1, 3]},
    ]

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


def attach_trainer(module, *, total_steps=100, global_step=0, max_epochs=1):
    """Stand-in for the Trainer attributes the module reads."""

    module._trainer = SimpleNamespace(
        estimated_stepping_batches=total_steps,
        global_step=global_step,
        max_epochs=max_epochs,
    )

    return module


def test_optimizer_skips_frozen_parameters():
    cfg = small_config()
    module = attach_trainer(LeWMModule(cfg, model=make_model(cfg)))

    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == cfg.optimizer.weight_decay
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

    [scheduler_config] = trainer.lr_scheduler_configs

    assert isinstance(trainer.optimizers[0], torch.optim.AdamW)
    assert isinstance(scheduler_config.scheduler, LambdaLR)
    assert scheduler_config.interval == "step"

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


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("progress", "horizons"),
    [
        (0.0, [1]),
        (0.199999, [1]),
        (0.2, [1, 5]),
        (0.499999, [1, 5]),
        (0.5, [1, 5, 10]),
        (0.99, [1, 5, 10]),
        (1.0, [1, 5, 10]),
    ],
)
def test_default_curriculum_boundaries(progress, horizons):
    assert rollout_horizons_for_progress(load_config(), progress) == horizons


def test_disabled_curriculum_uses_every_horizon():
    cfg = load_config(["prediction.curriculum.enabled=false"])

    assert rollout_horizons_for_progress(cfg, 0.0) == [1, 5, 10]


def curriculum(*stages):
    return {
        "prediction.curriculum.stages": [
            {"until": until, "horizons": horizons} for until, horizons in stages
        ]
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (curriculum(), "at least one stage"),
        (curriculum((0.5, [1]), (0.5, [1, 3])), "increasing"),
        (curriculum((0.0, [1]), (1.0, [1, 3])), "increasing"),
        (curriculum((0.5, [1]), (1.5, [1, 3])), "increasing"),
        (curriculum((0.5, [1]), (0.9, [1, 3])), "must end at 1.0"),
        (curriculum((0.5, []), (1.0, [1, 3])), "at least one horizon"),
        (curriculum((0.5, [1]), (1.0, [1, 2])), r"horizons \[2\] are not in"),
        (curriculum((0.5, [-1]), (1.0, [1, 3])), r"horizons \[-1\] are not in"),
        (curriculum((0.5, [1, 3]), (1.0, [3])), r"drops horizons \[1\]"),
        ({"loss.rollout.horizon_weights.3": 0.0}, "must be positive"),
    ],
)
def test_invalid_curriculum_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_config(small_config(**overrides))


def test_curriculum_validation_is_not_tied_to_the_default_horizons():
    cfg = small_config(
        **{
            "prediction.rollout_horizons": [2, 3],
            "loss.rollout.horizon_weights": {"2": 1.0, "3": 1.0},
            **curriculum((0.3, [2]), (1.0, [2, 3])),
        }
    )

    validate_config(cfg)

    assert rollout_horizons_for_progress(cfg, 0.1) == [2]


def test_disabled_curriculum_is_not_validated():
    validate_config(
        small_config(**{"prediction.curriculum.enabled": False, **curriculum()})
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scheduler.type": "onecycle"}, "scheduler.type"),
        ({"scheduler.interval": "epoch"}, "scheduler.interval must be 'step'"),
        ({"scheduler.warmup_ratio": 1.0}, "warmup_ratio"),
        ({"scheduler.warmup_ratio": -0.1}, "warmup_ratio"),
        ({"scheduler.min_lr": 0.0}, "min_lr"),
        ({"scheduler.min_lr": 1.0}, "min_lr"),
    ],
)
def test_invalid_scheduler_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_config(small_config(**overrides))


# ---------------------------------------------------------------------------
# Losses with active horizons
# ---------------------------------------------------------------------------


def default_batch(cfg):
    return trajectories(make_batch(cfg.data.context_steps, cfg.data.future_steps))


@pytest.mark.parametrize(
    ("horizons", "rollout_calls"),
    [([1], 1), ([1, 5], 5), ([1, 5, 10], 10)],
)
def test_rollout_only_runs_to_the_largest_active_horizon(horizons, rollout_calls):
    cfg = load_config()
    model = make_model(cfg)
    calls = []
    predict_spy(model, calls)

    output = lejepa_losses(
        model, SIGReg(), default_batch(cfg), cfg, rollout_horizons=horizons
    )

    # One teacher-forcing call, then one call per rolled-out step.
    assert len(calls) == 1 + rollout_calls
    assert {name for name in output if name.startswith("rollout_")} == {
        "rollout_loss",
        *(f"rollout_{h}_loss" for h in horizons),
    }


def fixed_mse(monkeypatch, values):
    """Make F.mse_loss return `values` in call order: TF, then each horizon."""

    queue = iter(values)

    def mse(prediction, target):
        return torch.tensor(next(queue))

    monkeypatch.setattr(training_module.F, "mse_loss", mse)


def test_rollout_loss_is_the_weighted_mean_of_active_horizons(monkeypatch):
    # L1 = 1, L5 = 3 with weights 1 and 1: mean 2, not the sum 4.
    cfg = load_config()
    fixed_mse(monkeypatch, [0.0, 1.0, 3.0])

    output = lejepa_losses(
        make_model(cfg), SIGReg(), default_batch(cfg), cfg, rollout_horizons=[1, 5]
    )

    assert output["rollout_loss"].item() == pytest.approx(2.0)


def test_rollout_mean_uses_the_horizon_weights(monkeypatch):
    cfg = load_config(["loss.rollout.horizon_weights.5=3.0"])
    fixed_mse(monkeypatch, [0.0, 1.0, 3.0])

    output = lejepa_losses(
        make_model(cfg), SIGReg(), default_batch(cfg), cfg, rollout_horizons=[1, 5]
    )

    # (1 * 1 + 3 * 3) / (1 + 3)
    assert output["rollout_loss"].item() == pytest.approx(2.5)


def test_rollout_magnitude_does_not_grow_with_more_horizons(monkeypatch):
    # Equal per-horizon losses give the same rollout loss in every stage.
    cfg = load_config()
    results = []

    for horizons in ([1], [1, 5], [1, 5, 10]):
        fixed_mse(monkeypatch, [0.0] + [2.0] * len(horizons))
        output = lejepa_losses(
            make_model(cfg),
            SIGReg(),
            default_batch(cfg),
            cfg,
            rollout_horizons=horizons,
        )
        results.append(output["rollout_loss"].item())

    assert results == pytest.approx([2.0, 2.0, 2.0])


def test_default_losses_evaluate_every_configured_horizon():
    cfg = load_config()

    output = lejepa_losses(make_model(cfg), SIGReg(), default_batch(cfg), cfg)

    assert {"rollout_1_loss", "rollout_5_loss", "rollout_10_loss"} <= set(output)


# ---------------------------------------------------------------------------
# Module: progress, curriculum and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("global_step", "total_steps", "progress"),
    [
        (0, 100, 0.0),
        (33, 100, 1 / 3),
        (99, 100, 1.0),
        (150, 100, 1.0),
        (0, 1, 1.0),
    ],
)
def test_progress_counts_optimizer_steps(global_step, total_steps, progress):
    cfg = small_config()
    module = attach_trainer(
        LeWMModule(cfg, model=make_model(cfg)),
        total_steps=total_steps,
        global_step=global_step,
    )

    assert module._training_progress() == pytest.approx(progress)


def test_progress_ignores_epochs():
    cfg = small_config()
    first = attach_trainer(
        LeWMModule(cfg, model=make_model(cfg)), global_step=10, max_epochs=1
    )
    second = attach_trainer(
        LeWMModule(cfg, model=make_model(cfg)), global_step=10, max_epochs=50
    )

    assert first._training_progress() == second._training_progress()


def test_infinite_step_count_is_rejected():
    cfg = small_config()
    module = attach_trainer(
        LeWMModule(cfg, model=make_model(cfg)), total_steps=float("inf")
    )

    with pytest.raises(ValueError, match="finite number of optimizer steps"):
        module.configure_optimizers()


def logging_module(cfg, *, global_step, total_steps=100):
    module = attach_trainer(
        LeWMModule(cfg, model=make_model(cfg)),
        total_steps=total_steps,
        global_step=global_step,
    )
    logged = {}

    def log(name, value, **kwargs):
        logged[name] = value

    def log_dict(values, **kwargs):
        logged.update(values)

    module.log = log
    module.log_dict = log_dict

    return module, logged


@pytest.mark.parametrize(
    ("global_step", "max_horizon", "keys"),
    [
        (0, 1, {"train/rollout_1_loss"}),
        (99, 3, {"train/rollout_1_loss", "train/rollout_3_loss"}),
    ],
)
def test_training_step_follows_the_curriculum(global_step, max_horizon, keys):
    cfg = small_config()
    module, logged = logging_module(cfg, global_step=global_step)

    module.training_step(make_batch(), 0)

    assert logged["train/max_rollout_horizon"] == max_horizon
    assert logged["train/curriculum_progress"] == pytest.approx(global_step / 99)
    assert {name for name in logged if name.startswith("train/rollout_")} == {
        "train/rollout_loss",
        *keys,
    }


def test_validation_always_evaluates_every_horizon():
    # Early in training the curriculum only trains h=1; validation still
    # reports every horizon so its metrics stay comparable across the run.
    cfg = small_config()
    module, logged = logging_module(cfg, global_step=0)

    module.validation_step(make_batch(), 0)

    assert {"val/rollout_1_loss", "val/rollout_3_loss"} <= set(logged)
    assert "train/max_rollout_horizon" not in logged


def test_lightning_steps_the_scheduler_per_optimizer_step(tmp_path):
    # 4 batches with gradient accumulation 2: 2 optimizer steps. The schedule
    # and curriculum follow those, not batches or epochs.
    cfg = small_config()
    module = LeWMModule(cfg, model=make_model(cfg))
    batches = DataLoader([make_batch() for _ in range(4)], batch_size=None)

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=batches)

    scheduler = trainer.lr_scheduler_configs[0].scheduler

    assert trainer.estimated_stepping_batches == 2
    assert trainer.global_step == 2
    assert scheduler.last_epoch == 2


def cfg_lr() -> float:
    return small_config().optimizer.lr


def cfg_min_lr() -> float:
    return small_config().scheduler.min_lr


def test_module_schedule_uses_estimated_stepping_batches_not_epochs():
    # Two modules with the same step count but different max_epochs get the
    # same schedule; a different step count changes it.
    def schedule(total_steps, max_epochs):
        cfg = small_config()
        module = attach_trainer(
            LeWMModule(cfg, model=make_model(cfg)),
            total_steps=total_steps,
            max_epochs=max_epochs,
        )
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        seen = []

        for _ in range(total_steps):
            seen.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        return seen

    assert schedule(40, max_epochs=1) == pytest.approx(schedule(40, max_epochs=50))
    assert schedule(40, max_epochs=1)[-1] == pytest.approx(cfg_min_lr())
    assert schedule(40, max_epochs=1)[2] == pytest.approx(cfg_lr())
    assert schedule(80, max_epochs=1)[2] != pytest.approx(cfg_lr())
