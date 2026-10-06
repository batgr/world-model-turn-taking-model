"""Audio inputs of the audio encoders: mono, at the encoder's sample rate."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torchaudio.functional as AF


def stack_waveforms(
    waveforms: Sequence[torch.Tensor],
    sample_rates: Sequence[float],
    *,
    sample_rate: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mono `(channels, samples)` windows at `sample_rate`, end-padded to `(B, 1, S)`."""

    if not waveforms:
        raise ValueError("Cannot stack an empty batch of waveforms")

    if len(waveforms) != len(sample_rates):
        raise ValueError(
            f"Got {len(waveforms)} waveforms but {len(sample_rates)} sample rates"
        )

    mono = []

    for waveform, rate in zip(waveforms, sample_rates, strict=True):
        if waveform.ndim != 2:
            raise ValueError("each waveform must have shape (channels, samples)")

        if rate <= 0:
            raise ValueError("sample rates must be positive")

        channel = waveform.to(device=device, dtype=torch.float32).mean(dim=0)

        if rate != sample_rate:
            channel = AF.resample(channel, orig_freq=int(rate), new_freq=sample_rate)

        mono.append(channel)

    length = max(channel.numel() for channel in mono)
    batch = torch.zeros(len(mono), 1, length, device=mono[0].device)

    for index, channel in enumerate(mono):
        batch[index, 0, : channel.numel()] = channel

    return batch
