"""
Frozen Mimi audio encoder producing continuous features on the action grid.

Waveforms are down-mixed to mono and resampled to Mimi's sample rate, encoded
to Mimi's continuous pre-quantization latents (12.5 Hz), then causally
aligned to `target_rate` (the 10 Hz action grid by default).
"""

from __future__ import annotations

import math

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
        waveform: torch.Tensor,
        sample_rate: int,
        target_length: int,
    ) -> torch.Tensor:
        """
        Args:
            waveform:
                `(channels, samples)` for one example, as in `DecodedAudio`,
                or `(B, channels, samples)`.

            sample_rate:
                Sample rate of `waveform`.

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
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Return mono float audio at Mimi's rate, shaped `(B, 1, samples)`."""

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim != 3:
            raise ValueError(
                "waveform must have shape (channels, samples) or (B, channels, samples)"
            )

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        device = next(self.model.parameters()).device

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
