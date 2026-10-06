"""
Streaming Mimi on a tiny randomly initialized Mimi: the real transformers
architecture (causal convolution and KV caches), without downloading weights.
The pretrained equivalence test is tests/integration/test_mimi_streaming.py.
"""

import math

import pytest
import torch
from transformers import MimiConfig, MimiModel

from turn_wm.models.encoders import mimi as mimi_module
from turn_wm.models.encoders.mimi import FrozenMimiEncoder

# Real Mimi strides (24 kHz, 12.5 Hz) with a tiny width; a 1 s attention
# window (25 transformer steps) so the test audio crosses it.
TINY = MimiConfig(
    hidden_size=32,
    num_filters=4,
    num_residual_layers=1,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=64,
    codebook_dim=32,
    codebook_size=16,
    num_quantizers=2,
    num_semantic_quantizers=1,
    vector_quantization_hidden_dimension=32,
    upsample_groups=32,
    sliding_window=25,
)


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    model = MimiModel(TINY).eval()
    patch = pytest.MonkeyPatch()
    patch.setattr(
        mimi_module.MimiModel, "from_pretrained", lambda *args, **kwargs: model
    )

    try:
        yield FrozenMimiEncoder()
    finally:
        patch.undo()


def audio(encoder, frames: float) -> torch.Tensor:
    """Mono `(1, samples)` at Mimi's own rate."""

    samples = round(frames * encoder.frame_samples)
    generator = torch.Generator().manual_seed(1)

    return 0.1 * torch.randn(1, samples, generator=generator)


def stream(encoder, waveform, **kwargs):
    return encoder.encode_recording(waveform, encoder.sample_rate, **kwargs)


# Observed: at most 1.1e-5 on features up to ~13 (float32 rounding, 9e-7
# relative); the tolerance is about three times that.
ATOL = 3e-5


@pytest.mark.parametrize("chunk_seconds", [0.08, 0.5, 1.0, 2.0])
def test_streaming_matches_one_shot_encoding(encoder, chunk_seconds):
    waveform = audio(encoder, 100)  # 8 s, many chunks and attention windows

    full = encoder([waveform], [encoder.sample_rate])[0]

    streamed = stream(encoder, waveform, chunk_seconds=chunk_seconds)

    torch.testing.assert_close(streamed, full, rtol=0, atol=ATOL)


def test_chunks_are_rounded_down_to_whole_frames(encoder):
    waveform = audio(encoder, 40)

    # 0.5 s is 6.25 frames: streamed as 6-frame chunks, same features.
    rounded = stream(encoder, waveform, chunk_seconds=0.5)
    whole = stream(encoder, waveform, chunk_seconds=0.48)
    tiny = stream(encoder, waveform, chunk_seconds=0.001)

    torch.testing.assert_close(rounded, whole, rtol=0, atol=0)
    torch.testing.assert_close(tiny, whole, rtol=0, atol=ATOL)


@pytest.mark.parametrize("frames", [1, 12.5, 40, 40.3])
def test_one_feature_per_started_frame(encoder, frames):
    waveform = audio(encoder, frames)

    features = stream(encoder, waveform, chunk_seconds=1.0)

    assert features.shape == (math.ceil(frames), encoder.output_dim)
    assert features.device.type == "cpu"


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (torch.zeros(1, 0), "empty recording"),
        (torch.zeros(4_000), r"shape \(channels, samples\)"),
    ],
)
def test_invalid_waveforms_are_rejected(encoder, waveform, message):
    with pytest.raises(ValueError, match=message):
        stream(encoder, waveform)


def test_a_recording_is_down_mixed_and_resampled_once(encoder):
    stereo_48k = torch.randn(2, 2 * 2 * encoder.frame_samples)

    features = encoder.encode_recording(stereo_48k, 48_000)

    assert features.shape == (2, encoder.output_dim)


@pytest.mark.parametrize("chunk_seconds", [0.0, -1.0])
def test_non_positive_chunks_are_rejected(encoder, chunk_seconds):
    with pytest.raises(ValueError, match="chunk_seconds must be positive"):
        stream(encoder, audio(encoder, 10), chunk_seconds=chunk_seconds)


def test_frame_geometry_matches_real_mimi(encoder):
    assert encoder.sample_rate == 24_000
    assert encoder.frame_rate == 12.5
    assert encoder.frame_samples == 1_920
