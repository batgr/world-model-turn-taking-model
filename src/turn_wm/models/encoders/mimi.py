"""
Frozen Mimi audio encoder producing continuous features on the action grid.

Waveforms are down-mixed to mono and resampled to Mimi's sample rate, encoded
to Mimi's continuous pre-quantization latents (12.5 Hz), then causally
aligned to `target_rate` (the 10 Hz action grid by default).

A batch may mix windows of different lengths and sample rates (e.g. EgoCom
and Ego4D in one batch): each is resampled on its own, then zero-padded at
the end. Mimi is causal, so end padding never changes the features of the
audio before it.

`stream_native_features` encodes a whole recording chunk by chunk while
keeping Mimi's causal convolution and transformer caches, for precomputing
features without holding the full recording's activations at once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio.functional as AF
from torch import nn
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache


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


@dataclass
class MimiStreamState:
    """Caches carried from one streamed chunk to the next."""

    past_key_values: Any | None
    padding_cache: MimiConv1dPaddingCache


class FrozenMimiEncoder(nn.Module):
    """Frozen Mimi encoder: waveform -> `(B, target_length, output_dim)`."""

    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        target_rate: float = 10.0,
        revision: str | None = None,
    ) -> None:
        super().__init__()

        if target_rate <= 0:
            raise ValueError("target_rate must be positive")

        # Pin `revision` to a commit SHA for any released feature cache.
        # "main" is from_pretrained's own default when no revision is pinned.
        self.model = MimiModel.from_pretrained(
            model_name, revision=revision if revision is not None else "main"
        )
        self.model.requires_grad_(False)
        self.model.eval()

        config = self.model.config

        self.sample_rate = int(config.sampling_rate)
        self.source_rate = float(config.frame_rate)
        self.target_rate = float(target_rate)
        self.output_dim = int(config.hidden_size)
        # Commit the weights were loaded from, when the Hub reports it.
        self.resolved_revision: str | None = getattr(config, "_commit_hash", None)
        # Waveform samples per Mimi frame (1920 at 24 kHz / 12.5 Hz).
        self.frame_samples = round(self.sample_rate / self.source_rate)

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

    def _new_stream_state(self) -> MimiStreamState:
        """Empty caches for every causal convolution, as `MimiModel.encode`."""

        paddings = []
        modes = []
        channels = []

        for layer_name in self.model.encoder._mimiconv1d_layer_names:
            layer = self.model.encoder.get_submodule(layer_name)

            paddings.append(layer.padding_total)
            modes.append(layer.pad_mode)
            channels.append(layer.in_channels)

        downsample = self.model.downsample

        if downsample is not None:
            paddings.append(downsample.padding_total)
            modes.append(downsample.pad_mode)
            channels.append(downsample.in_channels)

        padding_cache = MimiConv1dPaddingCache(
            num_layers=len(paddings),
            per_layer_padding=paddings,
            per_layer_padding_mode=modes,
            per_layer_in_channels=channels,
        )

        return MimiStreamState(
            past_key_values=None,
            padding_cache=padding_cache,
        )

    @torch.no_grad()
    def stream_native_features(
        self,
        waveform: torch.Tensor,
        *,
        chunk_seconds: float = 20.0,
    ) -> torch.Tensor:
        """
        Encode a continuous recording without resetting Mimi's state.

        Args:
            waveform:
                Mono waveform `(1, 1, samples)` already resampled to Mimi's
                sample rate.

            chunk_seconds:
                Audio per call, rounded down to whole Mimi frames (at least
                one): with cached padding a convolution never pads the right
                of a chunk, so a chunk ending mid-frame would shift every
                later one. Frame boundaries are the only places the stream
                can be cut without changing the features.

        Returns:
            Continuous Mimi features `(1, T, output_dim)` at Mimi's native
            rate, on the CPU.
        """

        if waveform.ndim != 3:
            raise ValueError("waveform must have shape (1, 1, samples)")

        if waveform.shape[0] != 1 or waveform.shape[1] != 1:
            raise ValueError(
                "streaming Mimi precomputation currently expects mono batch size 1"
            )

        if waveform.shape[-1] == 0:
            raise ValueError("Cannot encode an empty recording")

        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")

        chunk_frames = max(
            1, round(chunk_seconds * self.sample_rate) // self.frame_samples
        )
        chunk_samples = chunk_frames * self.frame_samples

        # End-pad to whole frames, as the one-shot encoder does, so the last
        # chunk also ends on a frame boundary.
        remainder = waveform.shape[-1] % self.frame_samples

        if remainder:
            waveform = nn.functional.pad(waveform, (0, self.frame_samples - remainder))

        state = self._new_stream_state()
        device = next(self.model.parameters()).device
        outputs = []

        for start in range(0, waveform.shape[-1], chunk_samples):
            chunk = waveform[:, :, start : start + chunk_samples].to(
                device=device,
                dtype=torch.float32,
            )

            embeddings = self.model.encoder(
                chunk,
                padding_cache=state.padding_cache,
            )

            encoded = self.model.encoder_transformer(
                embeddings.transpose(1, 2),
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )

            state.past_key_values = encoded.past_key_values

            embeddings = encoded.last_hidden_state.transpose(1, 2)

            if self.model.downsample is not None:
                embeddings = self.model.downsample(
                    embeddings,
                    padding_cache=state.padding_cache,
                )

            outputs.append(embeddings.transpose(1, 2).cpu())

        return torch.cat(outputs, dim=1)

    def _encode_mimi(self, waveform: torch.Tensor) -> torch.Tensor:
        """Continuous latents before quantization, `(B, frames, output_dim)`."""

        embeddings = self.model.encoder(waveform)

        encoded = self.model.encoder_transformer(
            embeddings.transpose(1, 2),
            return_dict=True,
        )

        embeddings = encoded.last_hidden_state.transpose(1, 2)

        if self.model.downsample is not None:
            embeddings = self.model.downsample(embeddings)

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
