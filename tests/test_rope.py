import pytest
import torch

from turn_wm.models.lewm.predictor import ARPredictor
from turn_wm.models.lewm.transformer import RotaryEmbedding


def predictor(*, position_encoding="rope", num_frames=4, dim_head=8):
    return ARPredictor(
        num_frames=num_frames,
        depth=1,
        heads=2,
        mlp_dim=32,
        input_dim=16,
        hidden_dim=16,
        output_dim=16,
        dim_head=dim_head,
        dropout=0.0,
        emb_dropout=0.0,
        position_encoding=position_encoding,
    )


def test_rope_is_parameter_free_and_preserves_qk_norms():
    rope = RotaryEmbedding(8)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 3, 7, 8)

    rotated_q, rotated_k = rope(q, k)

    torch.testing.assert_close(
        rotated_q.norm(dim=-1), q.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        rotated_k.norm(dim=-1), k.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
    assert list(rope.parameters()) == []


def test_rope_predictor_has_no_learned_position_table():
    model = predictor()

    assert model.position_encoding == "rope"
    assert model.pos_embedding is None
    assert model.transformer.layers[0].attn.rotary is not None


def test_rope_is_not_limited_by_legacy_num_frames():
    model = predictor(num_frames=4)
    x = torch.randn(2, 7, 16)
    actions = torch.randn(2, 7, 16)

    output = model(x, actions)

    assert output.shape == x.shape


def test_learned_position_variant_keeps_the_old_length_limit():
    model = predictor(position_encoding="learned", num_frames=4)
    x = torch.randn(2, 5, 16)
    actions = torch.randn(2, 5, 16)

    assert model.pos_embedding is not None

    with pytest.raises(ValueError, match="at most 4 steps"):
        model(x, actions)


def test_rope_requires_an_even_head_dimension():
    with pytest.raises(ValueError, match="even head dimension"):
        predictor(dim_head=7)


def test_unknown_position_encoding_is_rejected():
    with pytest.raises(ValueError, match="position_encoding"):
        predictor(position_encoding="sinusoidal")
