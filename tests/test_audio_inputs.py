"""Audio encoder inputs: down-mixed, resampled, end-padded."""

import pytest
import torch

from turn_wm.models.encoders.audio import stack_waveforms


def test_stack_waveforms_resamples_downmixes_and_end_pads():
    stereo = torch.stack([torch.ones(48_000), -torch.ones(48_000)])
    mono = torch.ones(1, 16_000)

    batch = stack_waveforms([stereo, mono], [48_000, 32_000], sample_rate=24_000)

    assert batch.shape == (2, 1, 24_000)
    # Stereo channels of opposite sign average to silence.
    assert batch[0].abs().max() < 1e-4
    # 16k samples at 32 kHz is 0.5 s: 12k samples at 24 kHz, then zeros.
    assert batch[1, 0, 11_000:11_900].mean() == pytest.approx(1.0, abs=0.05)
    assert torch.all(batch[1, 0, 12_000:] == 0)


def test_stack_waveforms_rejects_bad_input():
    with pytest.raises(ValueError, match="empty batch"):
        stack_waveforms([], [], sample_rate=24_000)

    with pytest.raises(ValueError, match="2 waveforms but 1 sample rates"):
        stack_waveforms([torch.ones(1, 4)] * 2, [16_000], sample_rate=24_000)

    with pytest.raises(ValueError, match="channels, samples"):
        stack_waveforms([torch.ones(4)], [16_000], sample_rate=24_000)
