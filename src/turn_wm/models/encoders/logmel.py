"""
Log-mel spectrogram: the encoder without learned weights.

A reference point for any learned encoder, and an encoder the whole pipeline
(precompute, cache, training) can run with offline, without downloading
weights. One frame per grid step: `frame_rate` is the decision grid's rate,
and each frame is the mel spectrum of the step's own audio.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torchaudio

from turn_wm.models.encoders.audio import stack_waveforms
from turn_wm.models.encoders.base import Encoder


class LogMelEncoder(Encoder):
    """`n_mels` log-mel energies of each `1 / frame_rate` s of mono audio."""

    modality = "audio"

    def __init__(
        self,
        frame_rate: float,
        n_mels: int = 80,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()

        hop = sample_rate / frame_rate

        if hop != round(hop):
            raise ValueError(
                f"{sample_rate} Hz audio has no whole number of samples per "
                f"{frame_rate:g} Hz frame"
            )

        self.name = f"logmel-{n_mels}"
        self.sample_rate = int(sample_rate)
        self.frame_rate = float(frame_rate)
        self.output_dim = int(n_mels)
        self.hop = round(hop)
        # Non-overlapping windows: frame k covers exactly step k's samples.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.hop,
            hop_length=self.hop,
            n_mels=n_mels,
            center=False,
        )

    def forward(
        self, inputs: Sequence[torch.Tensor], rates: Sequence[float]
    ) -> torch.Tensor:
        waveform = stack_waveforms(
            inputs, rates, sample_rate=self.sample_rate, device=self.device()
        )

        return (self.mel(waveform[:, 0]) + 1e-6).log().transpose(1, 2)
