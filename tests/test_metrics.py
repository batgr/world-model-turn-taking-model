"""V2 validation metrics: rollout MSE, baselines, strata and rank."""

import pytest
import torch

from turn_wm.training.metrics import ValidationMetrics, effective_rank

HORIZONS = (1, 5, 10)
C = 3


def trajectory(batch=1, dim=2, *, last=(1.0, 2.0), future=(2.0, 4.0)):
    z = torch.zeros(batch, C + 10, dim)
    z[:, C - 1] = torch.tensor(last)
    z[:, C:] = torch.tensor(future)
    return z


def update(metrics, latents, *, rollout=None, tf=None, datasets=None, mask=None):
    batch = latents.shape[0]
    metrics.update(
        latents=latents,
        tf_predictions=latents[:, :C] if tf is None else tf,
        rollout_predictions=(
            rollout if rollout is not None else {h: latents[:, C - 1] for h in HORIZONS}
        ),
        context_steps=C,
        datasets=datasets or ["egocom"] * batch,
        mask=mask,
    )


def test_results_do_not_depend_on_batch_size():
    latents = torch.randn(12, C + 10, 4, generator=torch.Generator().manual_seed(0))
    rollout = {h: latents[:, C - 1] + 0.1 * h for h in HORIZONS}

    def run(sizes):
        metrics = ValidationMetrics(HORIZONS, latent_rank_samples=0)
        start = 0
        for size in sizes:
            part = slice(start, start + size)
            update(
                metrics,
                latents[part],
                rollout={h: pred[part] for h, pred in rollout.items()},
            )
            start += size
        return metrics.compute()

    one = run([12])
    many = run([1, 3, 8])

    for name in ("tf_mse", "rollout_5_mse", "persistence_5_mse"):
        assert many[name] == pytest.approx(one[name]), name


@pytest.mark.parametrize("h", HORIZONS)
def test_rollout_persistence_copies_last_context_latent(h):
    metrics = ValidationMetrics(HORIZONS, effective_rank_health=False)
    update(
        metrics,
        trajectory(),
        rollout={k: torch.tensor([[2.0, 4.0]]) for k in HORIZONS},
    )

    result = metrics.compute()

    assert result[f"rollout_{h}_mse"] == 0.0
    assert result[f"persistence_{h}_mse"] == pytest.approx(2.5)


def test_teacher_forcing_reports_model_mse_only():
    z = torch.arange(C + 10, dtype=torch.float32).view(1, -1, 1).expand(1, -1, 2)
    metrics = ValidationMetrics(HORIZONS, effective_rank_health=False)
    update(metrics, z, tf=z[:, 1 : C + 1])

    result = metrics.compute()

    assert result["tf_mse"] == 0.0
    assert "tf_persistence_mse" not in result


def test_model_and_baseline_use_same_masked_positions():
    z = (
        torch.arange(C + 10, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(2, -1, 1)
        .clone()
    )
    z[1] *= 100
    mask = torch.ones(2, C + 10, dtype=torch.bool)
    mask[1] = False

    metrics = ValidationMetrics(HORIZONS, effective_rank_health=False)
    update(metrics, z, tf=torch.zeros(2, C, 1), mask=mask)

    only = ValidationMetrics(HORIZONS, effective_rank_health=False)
    update(only, z[:1], tf=torch.zeros(1, C, 1))

    result, expected = metrics.compute(), only.compute()
    for name in ("tf_mse", "rollout_5_mse", "persistence_5_mse"):
        assert result[name] == pytest.approx(expected[name]), name


def transition_update(metrics, z, *, context_action, future_action, predictions):
    metrics.update(
        latents=z,
        tf_predictions=z[:, :C],
        rollout_predictions=predictions,
        context_steps=C,
        datasets=["egocom"] * len(z),
        context_action=context_action,
        context_valid=torch.ones_like(context_action, dtype=torch.bool),
        future_action=future_action,
        future_valid=torch.ones_like(future_action, dtype=torch.bool),
    )


def test_transition_strata_use_actions_required_to_reach_each_horizon():
    metrics = ValidationMetrics(
        HORIZONS, effective_rank_health=False, transition_metrics=True
    )
    z = trajectory(last=(0.0, 0.0), future=(2.0, 2.0))

    # H5 uses a[C-1], a[C], ..., a[C+3]. The event at future_action[4]
    # produces H6, so H5 is stable while H10 contains a transition.
    context_action = torch.zeros(1, C, dtype=torch.long)
    future_action = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
    predictions = {h: torch.tensor([[2.0, 2.0]]) for h in HORIZONS}

    transition_update(
        metrics,
        z,
        context_action=context_action,
        future_action=future_action,
        predictions=predictions,
    )
    result = metrics.compute()

    assert result["stable_1_n"] == 1
    assert result["stable_5_n"] == 1
    assert result["transition_10_n"] == 1
    assert result["stable_5_mse"] == 0.0
    assert result["transition_10_mse"] == 0.0


def test_round_trip_counts_as_transition_even_if_endpoint_state_returns():
    metrics = ValidationMetrics(
        HORIZONS, effective_rank_health=False, transition_metrics=True
    )
    z = trajectory(last=(0.0, 0.0), future=(2.0, 2.0))
    context_action = torch.tensor([[0, 0, 1]])
    future_action = torch.tensor([[2] + [0] * 9])
    predictions = {h: torch.tensor([[2.0, 2.0]]) for h in HORIZONS}

    transition_update(
        metrics,
        z,
        context_action=context_action,
        future_action=future_action,
        predictions=predictions,
    )
    result = metrics.compute()

    assert result["transition_1_n"] == 1
    assert result["transition_5_n"] == 1
    assert result["transition_10_n"] == 1
    assert "stable_5_n" not in result


def test_transition_stratification_requires_action_labels():
    metrics = ValidationMetrics(
        HORIZONS, effective_rank_health=False, transition_metrics=True
    )

    with pytest.raises(ValueError, match="requires context/future actions"):
        update(metrics, trajectory())


def test_transition_strata_report_their_persistence_baseline():
    metrics = ValidationMetrics(
        HORIZONS, effective_rank_health=False, transition_metrics=True
    )
    z = trajectory(last=(0.0, 0.0), future=(2.0, 2.0))
    context_action = torch.tensor([[0, 0, 1]])
    future_action = torch.zeros(1, 10, dtype=torch.long)
    predictions = {h: torch.tensor([[1.0, 1.0]]) for h in HORIZONS}

    transition_update(
        metrics,
        z,
        context_action=context_action,
        future_action=future_action,
        predictions=predictions,
    )
    result = metrics.compute()

    assert result["transition_5_mse"] == 1.0
    assert result["transition_5_persistence_mse"] == 4.0


def test_effective_rank_orders_concentrated_and_spread_latents():
    generator = torch.Generator().manual_seed(0)
    direction = torch.randn(1, 8, generator=generator)
    rank_one = torch.randn(500, 1, generator=generator) * direction
    spread = torch.randn(500, 8, generator=generator)

    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-6)
    assert 6.0 < effective_rank(spread) <= 8.0


def test_effective_rank_sampling_is_bounded_and_deterministic():
    def sample():
        metrics = ValidationMetrics(HORIZONS, latent_rank_samples=50)
        generator = torch.Generator().manual_seed(1)
        for _ in range(10):
            update(metrics, torch.randn(4, C + 10, 3, generator=generator))
        return metrics.sample.rows

    first, second = sample(), sample()

    assert first.shape == (50, 3)
    assert torch.equal(first, second)


def test_global_metrics_pool_elements_and_report_corpora():
    metrics = ValidationMetrics(HORIZONS, effective_rank_health=False)
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
    assert result["rollout_5_mse"] == pytest.approx((1 + 3 * 16) / 4)
    assert result["persistence_5_mse"] == 4.0


def test_optional_diagnostics_can_be_disabled():
    metrics = ValidationMetrics(
        HORIZONS,
        persistence_baseline=False,
        effective_rank_health=False,
    )
    update(metrics, trajectory())

    names = set(metrics.compute())

    assert "rollout_1_mse" in names
    assert "tf_mse" in names
    assert not any("persistence" in name for name in names)
    assert "effective_rank" not in names
    assert not any(
        token in name
        for name in names
        for token in ("skill", "cosine", "latent_norm", "prediction_norm")
    )


def test_constant_latents_have_rank_zero():
    z = torch.full((4, C + 10, 2), 3.0)
    metrics = ValidationMetrics(HORIZONS)
    update(metrics, z)

    assert metrics.compute()["effective_rank"] == 0.0
