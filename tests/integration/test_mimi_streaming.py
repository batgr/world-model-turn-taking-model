"""
Streaming the pretrained Mimi must match one-shot Mimi: the proof that the
precomputed cache does not depend on the chunk size.

Uses the real kyutai/mimi weights, only when they are already in the local
Hugging Face cache (the test never downloads them); skipped otherwise.
"""

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from turn_wm.data.feature_precompute import PreparedRecordingAudio, _encode_recording
from turn_wm.models.encoders.mimi import FrozenMimiEncoder

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
    """Deterministic modulated tone plus seeded noise, `(1, samples)` at 24 kHz."""

    generator = torch.Generator().manual_seed(0)
    time = torch.arange(round(seconds * encoder.sample_rate)) / encoder.sample_rate
    tone = torch.sin(2 * torch.pi * 220 * time) * (
        0.5 + 0.5 * torch.sin(2 * torch.pi * 3 * time)
    )
    noise = torch.randn(time.shape, generator=generator)

    return (0.3 * tone + 0.05 * noise).view(1, -1)


@pytest.fixture(scope="module")
def full_native(encoder):
    with torch.no_grad():
        return encoder.encode_native(speech_like(encoder, SECONDS)[None])[0]


@pytest.mark.parametrize("chunk_seconds", [2.0, 5.0, 10.0])
def test_streamed_native_features_match_one_shot(encoder, full_native, chunk_seconds):
    streamed = encoder.encode_recording(
        speech_like(encoder, SECONDS), encoder.sample_rate, chunk_seconds=chunk_seconds
    )

    assert streamed.shape == full_native.shape == (450, 512)
    torch.testing.assert_close(streamed, full_native, rtol=0, atol=ATOL)


def test_only_the_unused_final_partial_frame_differs(encoder):
    # 36.05 s ends mid-frame: that last frame is padded differently, and a
    # 450-step grid never reads it.
    audio = speech_like(encoder, SECONDS + 0.05)

    with torch.no_grad():
        full = encoder.encode_native(audio[None])[0]

    streamed = encoder.encode_recording(audio, encoder.sample_rate, chunk_seconds=10.0)

    assert streamed.shape == full.shape == (451, 512)
    torch.testing.assert_close(streamed[:-1], full[:-1], rtol=0, atol=ATOL)


@pytest.mark.parametrize("steps", [450, 451, 453])
def test_recording_features_have_one_row_per_grid_step(encoder, steps):
    # On a 12.5 Hz grid: `steps` steps of 80 ms are `steps` Mimi frames.
    features = _encode_recording(
        encoder=encoder,
        audio=PreparedRecordingAudio(
            waveform=speech_like(encoder, steps / 12.5),
            sample_rate=encoder.sample_rate,
            audio_gaps=(),
        ),
        steps=steps,
        chunk_seconds=20.0,
    )

    assert features.shape == (steps, 512)
    assert torch.isfinite(features).all()
