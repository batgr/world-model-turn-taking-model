"""Validation metrics: exact values, epoch aggregation, masks and corpora."""

import math

import pytest
import torch

from turn_wm.training.metrics import (
    EPS,
    ValidationMetrics,
    effective_rank,
    skill_score,
)

HORIZONS = (1, 5, 10)
C = 3  # context steps; trajectories have C + 10 steps


def trajectory(batch=1, dim=2, *, last=(1.0, 2.0), future=(2.0, 4.0)):
    """Latents whose last context step is `last` and every future step `future`."""

    z = torch.zeros(batch, C + 10, dim)
    z[:, C - 1] = torch.tensor(last)
    z[:, C:] = torch.tensor(future)
    return z


def update(metrics, latents, *, rollout=None, tf=None, datasets=None, mask=None):
    batch = latents.shape[0]
    metrics.update(
        latents=latents,
        tf_predictions=latents[:, :C] if tf is None else tf,
        rollout_predictions=rollout
        if rollout is not None
        else {h: latents[:, C - 1] for h in HORIZONS},
        context_steps=C,
        datasets=datasets or ["egocom"] * batch,
        mask=mask,
    )


# -- skill --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "persistence", "skill"), [(2, 4, 0.5), (4, 4, 0.0), (8, 4, -1.0)]
)
def test_skill_score(model, persistence, skill):
    assert skill_score(model, persistence) == skill


def test_skill_with_a_zero_baseline_is_finite_and_unclipped():
    assert skill_score(0.0, 0.0) == 1.0
    assert skill_score(1.0, 0.0) == pytest.approx(1 - 1 / EPS)
    assert math.isfinite(skill_score(1.0, 0.0))


def test_skill_is_the_ratio_of_epoch_errors_not_a_mean_of_batch_skills():
    # Batch 1 (1 sample): model error 1, persistence 4 per element.
    # Batch 2 (9 samples): model error 9, persistence 4 per element.
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    small = trajectory(1, last=(0.0, 0.0), future=(2.0, 2.0))
    large = trajectory(9, last=(0.0, 0.0), future=(2.0, 2.0))
    update(metrics, small, rollout={h: torch.full((1, 2), 1.0) for h in HORIZONS})
    update(metrics, large, rollout={h: torch.full((9, 2), 5.0) for h in HORIZONS})

    result = metrics.compute()

    model_mse = (1 * 1 + 9 * 9) / 10
    assert result["rollout_1_mse"] == pytest.approx(model_mse)
    assert result["persistence_1_mse"] == pytest.approx(4.0)
    assert result["skill_1"] == pytest.approx(1 - model_mse / 4)
    naive = ((1 - 1 / 4) + (1 - 9 / 4)) / 2
    assert result["skill_1"] != pytest.approx(naive)


def test_results_do_not_depend_on_the_batch_size():
    latents = torch.randn(12, C + 10, 4, generator=torch.Generator().manual_seed(0))
    rollout = {h: latents[:, C - 1] + 0.1 * h for h in HORIZONS}

    def run(sizes):
        metrics = ValidationMetrics(HORIZONS, latent_rank_samples=0)
        start = 0
        for size in sizes:
            part = slice(start, start + size)
            update(
                metrics, latents[part], rollout={h: p[part] for h, p in rollout.items()}
            )
            start += size
        return metrics.compute()

    one, many = run([12]), run([1, 3, 8])
    for name in (
        "rollout_5_mse",
        "skill_5",
        "tf_mse",
        "tf_skill",
        "cosine_10",
        "latent_std",
    ):
        assert many[name] == pytest.approx(one[name]), name


# -- baselines ----------------------------------------------------------------


@pytest.mark.parametrize("h", HORIZONS)
def test_rollout_persistence_copies_the_last_context_latent(h):
    # z_last = [1, 2], target = [2, 4]: persistence error ((1)^2 + (2)^2) / 2.
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    update(
        metrics, trajectory(), rollout={k: torch.tensor([[2.0, 4.0]]) for k in HORIZONS}
    )

    result = metrics.compute()

    assert result[f"persistence_{h}_mse"] == pytest.approx(2.5)
    assert result[f"rollout_{h}_mse"] == 0.0
    assert result[f"skill_{h}"] == 1.0


def test_teacher_forcing_persistence_is_each_steps_own_input():
    # z[t] = t on every dim: z[t] -> z[t + 1] errs by exactly 1 per element;
    # the last context latent, zero or a prediction would err differently.
    z = torch.arange(C + 10, dtype=torch.float32).view(1, -1, 1).expand(1, -1, 2)
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    update(metrics, z, tf=z[:, 1 : C + 1])  # perfect teacher-forced predictions

    result = metrics.compute()

    assert result["tf_persistence_mse"] == 1.0
    assert result["tf_mse"] == 0.0
    assert result["tf_skill"] == 1.0


def test_model_and_baseline_share_the_same_masked_positions():
    z = (
        torch.arange(C + 10, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(2, -1, 1)
        .clone()
    )
    z[1] *= 100  # a second sample with huge errors, masked out below
    mask = torch.ones(2, C + 10, dtype=torch.bool)
    mask[1] = False

    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    update(metrics, z, tf=torch.zeros(2, C, 1), mask=mask)
    result = metrics.compute()

    only = ValidationMetrics(HORIZONS, latent_health=False)
    update(only, z[:1], tf=torch.zeros(1, C, 1))
    expected = only.compute()

    for name in ("tf_mse", "tf_persistence_mse", "rollout_5_mse", "persistence_5_mse"):
        assert result[name] == pytest.approx(expected[name]), name
    assert metrics.tf[""].elements == only.tf[""].elements


# -- cosine and deltas ----------------------------------------------------------


def test_cosine_of_equal_and_opposite_predictions():
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    z = trajectory(future=(3.0, 4.0))
    update(
        metrics,
        z,
        rollout={
            1: torch.tensor([[3.0, 4.0]]),
            5: torch.tensor([[-3.0, -4.0]]),
            10: torch.zeros(1, 2),
        },
    )

    result = metrics.compute()

    assert result["cosine_1"] == pytest.approx(1.0)
    assert result["cosine_5"] == pytest.approx(-1.0)
    assert result["cosine_10"] == 0.0  # zero-norm prediction: no NaN


def test_persistence_like_predictions_have_zero_predicted_delta():
    # Prediction == z_last while the target moved: the model "does not move".
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    update(metrics, trajectory(last=(1.0, 2.0), future=(4.0, 6.0)))

    result = metrics.compute()

    for h in HORIZONS:
        assert result[f"prediction_delta_norm_{h}"] == 0.0
        assert result[f"target_delta_norm_{h}"] == pytest.approx(5.0)


# -- latent health ----------------------------------------------------------------


def test_constant_latents_have_zero_std_and_known_norm():
    metrics = ValidationMetrics(HORIZONS)
    z = torch.full((4, C + 10, 2), 3.0)
    update(metrics, z, rollout={h: torch.full((4, 2), 4.0) for h in HORIZONS})

    result = metrics.compute()

    assert result["latent_std"] == pytest.approx(0.0, abs=1e-9)
    assert result["latent_norm"] == pytest.approx(math.sqrt(18))
    assert result["prediction_norm"] == pytest.approx(math.sqrt(32))


def test_effective_rank_orders_concentrated_and_spread_latents():
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(1, 8, generator=generator)
    rank_one = torch.randn(500, 1, generator=generator) * direction
    spread = torch.randn(500, 8, generator=generator)

    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-6)
    assert effective_rank(spread) > 6.0
    assert effective_rank(spread) <= 8.0


def test_rank_sample_is_bounded_and_deterministic():
    def sample():
        metrics = ValidationMetrics(HORIZONS, latent_rank_samples=50)
        generator = torch.Generator().manual_seed(1)
        for _ in range(10):
            update(metrics, torch.randn(4, C + 10, 3, generator=generator))
        return metrics.sample.rows

    first, second = sample(), sample()

    assert first.shape == (50, 3)  # 10 x 4 x 13 = 520 latents seen
    assert torch.equal(first, second)


# -- corpora --------------------------------------------------------------------


def test_global_metrics_pool_every_element_not_the_corpus_metrics():
    # egocom: 1 sample, model error 1, persistence 4; ego4d: 3 samples,
    # model error 16, persistence 4 (per element).
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    z = trajectory(4, last=(0.0, 0.0), future=(2.0, 2.0))
    prediction = torch.tensor([[1.0, 1.0], [6.0, 6.0], [6.0, 6.0], [6.0, 6.0]])
    update(
        metrics,
        z,
        rollout={h: prediction for h in HORIZONS},
        datasets=["egocom", "ego4d", "ego4d", "ego4d"],
    )

    result = metrics.compute()

    assert result["egocom/rollout_5_mse"] == 1.0
    assert result["ego4d/rollout_5_mse"] == 16.0
    assert result["egocom/skill_5"] == pytest.approx(0.75)
    assert result["ego4d/skill_5"] == pytest.approx(-3.0)
    assert result["rollout_5_mse"] == pytest.approx((1 + 3 * 16) / 4)
    assert result["skill_5"] == pytest.approx(1 - 12.25 / 4)
    assert result["skill_5"] != pytest.approx((0.75 - 3.0) / 2)
    assert {"egocom/tf_mse", "ego4d/tf_skill", "tf_persistence_mse"} <= set(result)


def test_corpus_names_come_from_the_batch():
    metrics = ValidationMetrics(HORIZONS, latent_health=False)
    update(metrics, trajectory(2), datasets=["corpus_a", "corpus_b"])

    assert {"corpus_a/skill_10", "corpus_b/skill_10"} <= set(metrics.compute())


def test_toggles_drop_their_metrics():
    metrics = ValidationMetrics(
        HORIZONS,
        persistence_baseline=False,
        cosine_similarity=False,
        latent_health=False,
    )
    update(metrics, trajectory())

    names = set(metrics.compute())

    assert "rollout_1_mse" in names and "tf_mse" in names
    assert not {n for n in names if "skill" in n or "persistence" in n or "cosine" in n}
    assert "latent_std" not in names
