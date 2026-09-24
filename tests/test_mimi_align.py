import pytest
import torch

from turn_wm.models.encoders.mimi import causal_align


def frames(count: int) -> torch.Tensor:
    # Feature value = source frame index, so selections are readable.
    return torch.arange(count, dtype=torch.float32).view(1, count, 1)


def selected(aligned: torch.Tensor) -> list[int]:
    return aligned[0, :, 0].long().tolist()


def test_mimi_rate_to_action_grid():
    aligned = causal_align(
        frames(13), source_rate=12.5, target_rate=10.0, target_length=10
    )

    assert aligned.shape == (1, 10, 1)
    # Step k ends at (k + 1) / 10 s; frame i is ready at (i + 1) / 12.5 s.
    assert selected(aligned) == [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]


@pytest.mark.parametrize(("source_rate", "target_rate"), [(12.5, 10.0), (50.0, 10.0)])
def test_no_step_uses_a_frame_from_its_future(source_rate, target_rate):
    aligned = causal_align(
        frames(400), source_rate=source_rate, target_rate=target_rate, target_length=50
    )

    for step, index in enumerate(selected(aligned)):
        assert (index + 1) / source_rate <= (step + 1) / target_rate + 1e-9
        # ...and it is the latest such frame.
        assert (index + 2) / source_rate > (step + 1) / target_rate


def test_keeps_batch_and_feature_dims():
    features = torch.randn(3, 20, 512)

    aligned = causal_align(
        features, source_rate=12.5, target_rate=10.0, target_length=8
    )

    assert aligned.shape == (3, 8, 512)
    assert torch.equal(aligned[:, 0], features[:, 0])


def test_too_few_source_frames_fails():
    with pytest.raises(ValueError, match="not enough source features"):
        causal_align(frames(5), source_rate=12.5, target_rate=10.0, target_length=10)


def test_target_faster_than_source_fails():
    with pytest.raises(ValueError, match="too high"):
        causal_align(frames(5), source_rate=12.5, target_rate=25.0, target_length=2)


@pytest.mark.parametrize(
    ("features", "kwargs"),
    [
        (torch.zeros(4, 1), {}),
        (frames(4), {"source_rate": 0.0}),
        (frames(4), {"target_length": 0}),
    ],
)
def test_invalid_arguments_fail(features, kwargs):
    arguments = {"source_rate": 12.5, "target_rate": 10.0, "target_length": 2}

    with pytest.raises(ValueError):
        causal_align(features, **{**arguments, **kwargs})
