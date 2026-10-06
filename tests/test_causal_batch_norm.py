import pytest
import torch

from turn_wm.config import load_config
from turn_wm.models.build import build_model
from turn_wm.models.lewm.mlp import CausalBatchNorm1d
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.objective import lejepa_forward
from turn_wm.training.trajectories import Trajectories


def test_training_prefix_does_not_depend_on_future_positions():
    torch.manual_seed(17)
    norm = CausalBatchNorm1d(3)
    norm.train()
    prefix = torch.randn(4, 3, 3)
    future = torch.randn(4, 2, 3)

    full = norm(torch.cat([prefix, future], dim=1))[:, :3]
    changed = norm(torch.cat([prefix, future + 100], dim=1))[:, :3]

    torch.testing.assert_close(full, changed, rtol=0, atol=0)


def test_eval_uses_fixed_statistics_for_single_sample_and_prefix():
    torch.manual_seed(18)
    norm = CausalBatchNorm1d(3)
    norm.train()
    norm(torch.randn(4, 5, 3))
    norm.eval()
    sample = torch.randn(1, 5, 3)

    torch.testing.assert_close(norm(sample[:, :1]), norm(sample)[:, :1], rtol=0, atol=0)


def test_positional_statistics_match_training_in_eval():
    # Predictor outputs have a mean that depends on the position; eval must
    # normalize each position with that position's statistics.
    torch.manual_seed(20)
    norm = CausalBatchNorm1d(3, momentum=1.0, num_positions=5)
    norm.train()
    offsets = 10 * torch.randn(1, 5, 3)
    x = torch.randn(4096, 5, 3) + offsets

    trained = norm(x)
    norm.eval()

    torch.testing.assert_close(norm(x), trained, rtol=0, atol=1e-3)
    torch.testing.assert_close(norm(x[:1, :2]), norm(x[:1])[:, :2], rtol=0, atol=0)

    with pytest.raises(ValueError, match="statistics for 5 positions"):
        norm(torch.randn(1, 6, 3))


def test_training_rejects_batch_of_one():
    norm = CausalBatchNorm1d(3)

    with pytest.raises(ValueError, match="batch size >= 2"):
        norm(torch.randn(1, 2, 3))


def test_default_model_uses_causal_bn_and_layernorm_variant_is_available():
    v1 = load_config(["model=lewm_ln", "data.observation_source=feature_cache"])
    v2 = load_config(["data.observation_source=feature_cache"])

    assert not getattr(build_model(v1).projector, "expects_sequence", False)
    model = build_model(v2)
    assert isinstance(model.projector.net[1], CausalBatchNorm1d)
    assert isinstance(model.pred_proj.net[1], CausalBatchNorm1d)
    assert model.projector.net[1].num_positions is None
    assert model.pred_proj.net[1].num_positions == v2.model.predictor.num_frames

    model.eval()
    features = torch.randn(1, 4, 512)
    actions = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        z = model.project_features(features)
        output = model.predict(z, model.encode_actions(actions))

    assert z.shape == output.shape == (1, 4, v2.embed_dim)


def test_learned_position_variant_keeps_positional_pred_projector_stats():
    cfg = load_config(
        [
            "model=lewm_learned_pos",
            "data.observation_source=feature_cache",
        ]
    )
    model = build_model(cfg)

    assert model.predictor.position_encoding == "learned"
    assert model.predictor.pos_embedding is not None
    assert isinstance(model.pred_proj.net[1], CausalBatchNorm1d)
    assert model.pred_proj.net[1].num_positions == cfg.model.predictor.num_frames


def test_standard_bn_config_flattens_time_for_both_projectors():
    cfg = load_config(
        [
            "model=lewm_standard_bn",
            "data.observation_source=feature_cache",
        ]
    )
    model = build_model(cfg)

    assert isinstance(model.projector.net[1], torch.nn.BatchNorm1d)
    assert isinstance(model.pred_proj.net[1], torch.nn.BatchNorm1d)
    assert not getattr(model.projector, "expects_sequence", False)
    assert not getattr(model.pred_proj, "expects_sequence", False)

    # Ordinary BatchNorm is applied after JEPA flattens (B, T, D) -> (B*T, D).
    # The resulting shapes must still match the model contract.
    model.eval()
    features = torch.randn(2, 4, 512)
    actions = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        z = model.project_features(features)
        output = model.predict(z, model.encode_actions(actions))

    assert z.shape == output.shape == (2, 4, cfg.embed_dim)


def test_positional_cbn_config_uses_full_v2_trajectory_for_projector():
    cfg = load_config(
        [
            "model=lewm_positional_cbn",
            "train=lewm_v2",
            "data.observation_source=feature_cache",
        ]
    )
    model = build_model(cfg)

    projector_norm = model.projector.net[1]
    pred_norm = model.pred_proj.net[1]

    assert isinstance(projector_norm, CausalBatchNorm1d)
    assert isinstance(pred_norm, CausalBatchNorm1d)
    assert (
        projector_norm.num_positions
        == cfg.data.context_steps + cfg.data.future_steps
        == 40
    )
    assert pred_norm.num_positions == cfg.model.predictor.num_frames == 30

    model.eval()
    features = torch.randn(2, 40, 512)
    actions = torch.zeros(2, 30, dtype=torch.long)

    with torch.no_grad():
        z = model.project_features(features)
        pred = model.predict(z[:, :30], model.encode_actions(actions))

    assert z.shape == (2, 40, cfg.embed_dim)
    assert pred.shape == (2, 30, cfg.embed_dim)


def test_training_projectors_preserve_prefix_and_receive_gradients():
    torch.manual_seed(19)
    cfg = load_config(
        [
            "data.observation_source=feature_cache",
            "data.context_steps=4",
            "data.future_steps=3",
            "prediction.rollout_context_size=4",
            "model.predictor.depth=1",
            "model.predictor.heads=2",
            "model.predictor.mlp_dim=64",
            "model.predictor.dropout=0.0",
            "model.projector.hidden_dim=32",
            "model.pred_proj.hidden_dim=32",
        ]
    )
    model = build_model(cfg).train()
    features = torch.randn(4, 7, 512)
    actions = torch.zeros(4, 7, dtype=torch.long)

    full = model.project_features(features)
    prefix = model.project_features(features[:, :3])
    torch.testing.assert_close(full[:, :3], prefix, rtol=1e-5, atol=1e-6)

    full_pred = model.predict(full[:, :4], model.encode_actions(actions[:, :4]))
    prefix_pred = model.predict(prefix, model.encode_actions(actions[:, :3]))
    torch.testing.assert_close(full_pred[:, :3], prefix_pred, rtol=1e-5, atol=1e-5)

    batch = Trajectories(
        actions=actions,
        context_steps=4,
        future_steps=3,
        features=features,
    )
    loss = lejepa_forward(model, SIGReg(num_proj=16), batch, cfg, rollout_horizons=[1])[
        "loss"
    ]
    loss.backward()

    for projector in (model.projector, model.pred_proj):
        grad = projector.net[1].weight.grad
        assert grad is not None and torch.isfinite(grad).all()
