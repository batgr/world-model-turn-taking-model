"""
Frozen Mimi: Kyutai's streaming codec, continuous latents before quantization.

Mimi takes 24 kHz mono audio and is causal: its convolutions and transformer
only look back. Its features are 512-d at 12.5 Hz, one frame per 80 ms, so it
runs on a 12.5 Hz decision grid.

`encode_recording` streams a whole recording chunk by chunk while keeping
Mimi's causal convolution and transformer caches, for precomputing features
without holding the full recording's activations at once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import MimiModel
from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache

from turn_wm.models.encoders.audio import stack_waveforms
from turn_wm.models.encoders.base import Encoder


class FrozenMimiEncoder(Encoder):
    """Frozen Mimi encoder: audio -> `(B, frames, 512)` at 12.5 Hz."""

    modality = "audio"

    def __init__(
        self,
        model_name: str = "kyutai/mimi",
        revision: str | None = None,
    ) -> None:
        super().__init__()

        # Pin `revision` to a commit SHA for any released feature cache.
        # "main" is from_pretrained's own default when no revision is pinned.
        self.model = MimiModel.from_pretrained(
            model_name, revision=revision if revision is not None else "main"
        )
        self.model.requires_grad_(False)
        self.model.eval()

        config = self.model.config

        self.name = model_name
        self.revision = revision
        self.sample_rate = int(config.sampling_rate)
        self.frame_rate = float(config.frame_rate)
        self.output_dim = int(config.hidden_size)
        # Commit the weights were loaded from, when the Hub reports it.
        self.resolved_revision = getattr(config, "_commit_hash", None)
        # Waveform samples per Mimi frame (1920 at 24 kHz / 12.5 Hz).
        self.frame_samples = round(self.sample_rate / self.frame_rate)

    @torch.no_grad()
    def forward(
        self, inputs: Sequence[torch.Tensor], rates: Sequence[float]
    ) -> torch.Tensor:
        waveform = stack_waveforms(
            inputs, rates, sample_rate=self.sample_rate, device=self.device()
        )

        return self.encode_native(waveform)

    def _padding_cache(self) -> MimiConv1dPaddingCache:
        """Empty caches for every causal convolution, as `MimiModel.encode`."""

        # Mimi's causal convolutions (typed as plain modules by transformers).
        layers: list[Any] = [
            self.model.encoder.get_submodule(name)
            for name in self.model.encoder._mimiconv1d_layer_names
        ]

        if self.model.downsample is not None:
            layers.append(self.model.downsample)

        return MimiConv1dPaddingCache(
            num_layers=len(layers),
            per_layer_padding=[layer.padding_total for layer in layers],
            per_layer_padding_mode=[layer.pad_mode for layer in layers],
            per_layer_in_channels=[layer.in_channels for layer in layers],
        )

    @torch.no_grad()
    def encode_recording(
        self,
        input: torch.Tensor,
        rate: float,
        *,
        chunk_seconds: float = 20.0,
    ) -> torch.Tensor:
        """
        Encode a continuous recording without resetting Mimi's state.

        Args:
            input:
                Waveform `(channels, samples)` at `rate`; it is down-mixed and
                resampled to 24 kHz at once (resampling chunk by chunk would
                add artifacts at every boundary).

            rate:
                Sample rate of `input`.

            chunk_seconds:
                Audio per call, rounded down to whole Mimi frames (at least
                one): with cached padding a convolution never pads the right
                of a chunk, so a chunk ending mid-frame would shift every
                later one. Frame boundaries are the only places the stream
                can be cut without changing the features.

        Returns:
            Continuous Mimi features `(frames, output_dim)` on the CPU.
        """

        waveform = stack_waveforms([input], [rate], sample_rate=self.sample_rate)

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

        padding_cache = self._padding_cache()
        past_key_values = None
        device = self.device()
        outputs = []

        for start in range(0, waveform.shape[-1], chunk_samples):
            chunk = waveform[:, :, start : start + chunk_samples].to(device)
            latents, past_key_values = self._latents(
                chunk, padding_cache=padding_cache, past_key_values=past_key_values
            )
            outputs.append(latents.cpu())

        return torch.cat(outputs, dim=1)[0]

    def encode_native(self, waveform: torch.Tensor) -> torch.Tensor:
        """Continuous latents before quantization, `(B, frames, output_dim)`."""

        return self._latents(waveform)[0]

    def _latents(
        self,
        waveform: torch.Tensor,
        *,
        padding_cache: MimiConv1dPaddingCache | None = None,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Encoder, encoder transformer and downsampling of `(B, 1, samples)`.

        With caches, a chunk continues the stream they hold (and updates
        them); without, the waveform is encoded on its own.
        """

        embeddings = self.model.encoder(waveform, padding_cache=padding_cache)
        encoded = self.model.encoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=padding_cache is not None,
            return_dict=True,
        )
        embeddings = encoded.last_hidden_state.transpose(1, 2)

        if self.model.downsample is not None:
            embeddings = self.model.downsample(embeddings, padding_cache=padding_cache)

        return embeddings.transpose(1, 2), encoded.past_key_values
