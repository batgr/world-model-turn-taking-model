"""
Frozen Mimi audio encoder producing continuous features on the action grid.

Waveforms are down-mixed to mono and resampled to Mimi's sample rate, encoded
to Mimi's continuous pre-quantization latents (12.5 Hz), then causally
aligned to `target_rate` (the 10 Hz action grid by default).

A batch may mix windows of different lengths and sample rates (e.g. EgoCom
and Ego4D in one batch): each is resampled on its own, then zero-padded at
the end. Mimi is causal, so end padding never changes the features of the
audio before it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torchaudio.functional as AF
from torch import nn
from transformers import MimiModel


def causal_align(
    features: torch.Tensor,
    *,
    source_rate: float,
    target_rate: float,
    target_length: int,
) -> torch.Tensor:
    """Resample `(B, T_source, D)` features to `(B, target_length, D)` causally.

    Source frame `i` is available at `(i + 1) / source_rate`. Target step `k`
    covers `[k / target_rate, (k + 1) / target_rate)` and takes the latest
    source frame available by the end of that interval, so no step sees audio
    from after its own interval.
    """

    if features.ndim != 3:
        raise ValueError("features must have shape (B, T_source, D)")

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive")

    if target_length <= 0:
        raise ValueError("target_length must be positive")

    target_end_times = (
        torch.arange(
            1,
            target_length + 1,
            device=features.device,
            dtype=torch.float64,
        )
        / target_rate
    )

    indices = torch.floor(target_end_times * source_rate + 1e-8).long() - 1

    if indices.min() < 0:
        raise ValueError("target_rate is too high for the source frame rate")

    if indices.max() >= features.size(1):
        raise ValueError(
            "not enough source features for requested target_length: "
            f"need source index {indices.max().item()}, "
            f"but only {features.size(1)} source steps are available"
        )

    return features[:, indices, :]


def stack_waveforms(
    waveforms: Sequence[torch.Tensor],
    sample_rates: Sequence[int],
    *,
    target_rate: int,
) -> torch.Tensor:
    """Mono `(channels, samples)` windows at `target_rate`, end-padded to `(B, 1, S)`."""

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

        channel = waveform.to(torch.float32).mean(dim=0)

        if rate != target_rate:
            channel = AF.resample(channel, orig_freq=int(rate), new_freq=target_rate)

        mono.append(channel)

    length = max(channel.numel() for channel in mono)
    batch = torch.zeros(len(mono), 1, length, device=mono[0].device)

    for index, channel in enumerate(mono):
        batch[index, 0, : channel.numel()] = channel

    return batch


class FrozenMimiEncoder(nn.Module):
    """Frozen Mimi encoder: waveform -> `(B, target_length, output_dim)`."""

    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        target_rate: float = 10.0,
    ) -> None:
        super().__init__()

        if target_rate <= 0:
            raise ValueError("target_rate must be positive")

        self.model = MimiModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()

        config = self.model.config

        self.sample_rate = int(config.sampling_rate)
        self.source_rate = float(config.frame_rate)
        self.target_rate = float(target_rate)
        self.output_dim = int(config.hidden_size)

    def train(self, mode: bool = True) -> FrozenMimiEncoder:
        # The pretrained model stays in eval mode even while training.
        super().train(mode)
        self.model.eval()

        return self

    @torch.no_grad()
    def forward(
        self,
        waveform: torch.Tensor | Sequence[torch.Tensor],
        sample_rate: int | Sequence[int],
        target_length: int,
    ) -> torch.Tensor:
        """
        Args:
            waveform:
                `(channels, samples)` for one example, as in `DecodedAudio`,
                `(B, channels, samples)`, or a sequence of `(channels,
                samples)` windows that may differ in length and rate.

            sample_rate:
                Sample rate of `waveform`, or one rate per window.

            target_length:
                Number of grid steps the waveform covers.

        Returns:
            Tensor of shape `(B, target_length, output_dim)`.
        """

        waveform = self._prepare_waveform(
            waveform,
            sample_rate,
        )

        features = self._encode_mimi(waveform)

        return self._align(
            features,
            target_length=target_length,
        )

    def _prepare_waveform(
        self,
        waveform: torch.Tensor | Sequence[torch.Tensor],
        sample_rate: int | Sequence[int],
    ) -> torch.Tensor:
        """Return mono float audio at Mimi's rate, shaped `(B, 1, samples)`."""

        device = next(self.model.parameters()).device

        if not isinstance(waveform, torch.Tensor):
            rates = (
                [sample_rate] * len(waveform)
                if isinstance(sample_rate, int)
                else sample_rate
            )

            return stack_waveforms(
                [window.to(device) for window in waveform],
                rates,
                target_rate=self.sample_rate,
            )

        if not isinstance(sample_rate, int):
            raise TypeError("a batched waveform tensor takes a single sample_rate")

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 3:
            raise ValueError(
                "waveform must have shape (channels, samples) or (B, channels, samples)"
            )

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        waveform = waveform.to(device=device, dtype=torch.float32)
        waveform = waveform.mean(dim=1, keepdim=True)

        if sample_rate != self.sample_rate:
            waveform = AF.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )

        return waveform

    def _encode_mimi(self, waveform: torch.Tensor) -> torch.Tensor:
        """Continuous latents before quantization, `(B, frames, output_dim)`."""

        embeddings = self.model.encoder(waveform)

        encoded = self.model.encoder_transformer(
            embeddings.transpose(1, 2),
            return_dict=True,
        )

        embeddings = self.model.downsample(
            encoded.last_hidden_state.transpose(1, 2),
        )

        return embeddings.transpose(1, 2)

    def _align(
        self,
        features: torch.Tensor,
        *,
        target_length: int,
    ) -> torch.Tensor:
        return causal_align(
            features,
            source_rate=self.source_rate,
            target_rate=self.target_rate,
            target_length=target_length,
        )


if __name__ == "__main__":
    encoder = FrozenMimiEncoder()

    seconds = 1.0
    audio = torch.randn(2, 1, int(48_000 * seconds))

    features = encoder(
        audio,
        sample_rate=48_000,
        target_length=math.floor(seconds * encoder.target_rate),
    )

    print(features.shape)
