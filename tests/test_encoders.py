"""The encoder contract, with the weight-free log-mel encoder (no download)."""

import pytest
import torch

from turn_wm.models.encoders.base import Encoder
from turn_wm.models.encoders.logmel import LogMelEncoder


def noise(seconds: float, rate: int = 16_000) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(1, round(seconds * rate), generator=generator)


@pytest.mark.parametrize("grid_rate", [10.0, 12.5])
def test_logmel_gives_one_frame_per_grid_step(grid_rate):
    encoder = LogMelEncoder(frame_rate=grid_rate)

    assert isinstance(encoder, Encoder)
    assert (encoder.modality, encoder.frame_rate, encoder.output_dim) == (
        "audio",
        grid_rate,
        80,
    )
    # 2 s of audio, whatever its rate and channels.
    stereo_48k = torch.randn(2, 96_000, generator=torch.Generator().manual_seed(1))
    assert encoder([stereo_48k], [48_000]).shape == (1, round(2 * grid_rate), 80)


def test_a_logmel_frame_only_sees_its_own_step():
    encoder = LogMelEncoder(frame_rate=12.5)
    audio = noise(1.0)
    changed = audio.clone()
    changed[:, 6_400:] = 0.0  # everything after 0.4 s, i.e. from step 5 on

    before = encoder([audio], [16_000])
    after = encoder([changed], [16_000])

    assert torch.equal(before[:, :5], after[:, :5])
    assert not torch.equal(before[:, 5], after[:, 5])


def test_a_frame_rate_without_whole_samples_is_refused():
    with pytest.raises(ValueError, match="no whole number of samples"):
        LogMelEncoder(frame_rate=12.5, sample_rate=16_001)


def test_shorter_inputs_are_end_padded_into_one_batch():
    encoder = LogMelEncoder(frame_rate=10.0)

    features = encoder([noise(1.0), noise(0.5)], [16_000, 16_000])

    assert features.shape == (2, 10, 80)
    single = encoder([noise(0.5)], [16_000])
    assert torch.equal(features[1, :5], single[0])


def test_the_encoder_stays_frozen_in_training_mode():
    encoder = LogMelEncoder(frame_rate=10.0).train()

    assert all(not module.training for module in encoder.children())
