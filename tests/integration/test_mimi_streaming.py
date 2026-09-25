"""
Streaming the pretrained Mimi must match one-shot Mimi: the proof that the
precomputed cache does not depend on the chunk size.

Uses the real kyutai/mimi weights, only when they are already in the local
Hugging Face cache (the test never downloads them); skipped otherwise.
"""

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from turn_wm.data.mimi_precompute import _encode_recording
from turn_wm.models.encoders.mimi import FrozenMimiEncoder, causal_align

pytestmark = pytest.mark.integration

MODEL = "kyutai/mimi"

# Longer than Mimi's 20 s attention window, so the transformer cache matters.
SECONDS = 36

# Observed: at most 7.1e-7 on features up to ~0.3 (float32 rounding) for
# every chunk size; the tolerance is about three times that.
ATOL = 2e-6


@pytest.fixture(scope="module")
def encoder():
    cached = [
        try_to_load_from_cache(MODEL, filename)
        for filename in ("config.json", "model.safetensors")
    ]

    if not all(isinstance(path, str) for path in cached):
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")

    return FrozenMimiEncoder(model_name=MODEL)


def speech_like(encoder: FrozenMimiEncoder, seconds: float) -> torch.Tensor:
    """Deterministic modulated tone plus seeded noise, `(1, 1, samples)`."""

    generator = torch.Generator().manual_seed(0)
    time = torch.arange(round(seconds * encoder.sample_rate)) / encoder.sample_rate
    tone = torch.sin(2 * torch.pi * 220 * time) * (
        0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * time)
    )
    noise = torch.randn(time.shape, generator=generator)

    return (0.3 * tone + 0.05 * noise).view(1, 1, -1)


@pytest.fixture(scope="module")
def full_native(encoder):
    with torch.no_grad():
        return encoder._encode_mimi(speech_like(encoder, SECONDS))


@pytest.mark.parametrize("chunk_seconds", [2.0, 5.0, 10.0])
def test_streamed_native_features_match_one_shot(encoder, full_native, chunk_seconds):
    streamed = encoder.stream_native_features(
        speech_like(encoder, SECONDS), chunk_seconds=chunk_seconds
    )

    assert streamed.shape == full_native.shape == (1, 450, 512)
    torch.testing.assert_close(streamed, full_native, rtol=0, atol=ATOL)


@pytest.mark.parametrize("chunk_seconds", [2.0, 5.0, 10.0])
def test_streamed_features_match_after_grid_alignment(
    encoder, full_native, chunk_seconds
):
    streamed = encoder.stream_native_features(
        speech_like(encoder, SECONDS), chunk_seconds=chunk_seconds
    )

    def to_grid(features):
        return causal_align(
            features,
            source_rate=encoder.source_rate,
            target_rate=10.0,
            target_length=SECONDS * 10,
        )

    torch.testing.assert_close(
        to_grid(streamed), to_grid(full_native), rtol=0, atol=ATOL
    )


def test_only_the_unused_final_partial_frame_differs(encoder):
    # 36.05 s ends mid-frame: that last frame is padded differently, and
    # causal_align never reads it for a grid of 360 steps.
    audio = speech_like(encoder, SECONDS + 0.05)

    with torch.no_grad():
        full = encoder._encode_mimi(audio)

    streamed = encoder.stream_native_features(audio, chunk_seconds=10.0)

    assert streamed.shape == full.shape == (1, 451, 512)
    torch.testing.assert_close(streamed[:, :-1], full[:, :-1], rtol=0, atol=ATOL)


@pytest.mark.parametrize("steps", [360, 361, 363])
def test_recording_features_have_one_row_per_grid_step(encoder, steps):
    features = _encode_recording(
        encoder=encoder,
        audio=speech_like(encoder, steps / 10),
        steps=steps,
        target_rate=10.0,
        chunk_seconds=20.0,
    )

    assert features.shape == (steps, 512)
    assert torch.isfinite(features).all()
