"""
The contract every frozen encoder of the world model follows.

An encoder reads observations of one `modality` (audio, video, ...) and gives
`frame_rate` features per second, causally: frame `k` depends only on the
input up to the end of frame `k`. It runs at the rate of the decision grid,
so frame `k` is grid step `k`: no feature is resampled or realigned, and the
data's grid rate (`data.grid_rate_hz`) must be the encoder's `frame_rate`.

Adding an encoder means subclassing `Encoder` (set `name`, `modality`,
`frame_rate`, `output_dim`, implement `forward`) and adding a config to
`configs/model/encoder/`. Its features can then be precomputed
(`turn-wm precompute-features`) or computed at every step.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Encoder(nn.Module):
    """A frozen, causal encoder: inputs -> `(B, frames, output_dim)` at `frame_rate`.

    `name`, `revision` and `resolved_revision` identify the weights in a
    feature cache's manifest.
    """

    name: str
    modality: str
    frame_rate: float
    output_dim: int
    revision: str | None = None
    resolved_revision: str | None = None

    def forward(
        self, inputs: Sequence[torch.Tensor], rates: Sequence[float]
    ) -> torch.Tensor:
        """One `(channels, samples)` input per example at its own rate -> features.

        Inputs may differ in length; shorter ones are end-padded, which a
        causal encoder's earlier frames never see.
        """

        raise NotImplementedError

    @torch.no_grad()
    def encode_recording(
        self, input: torch.Tensor, rate: float, *, chunk_seconds: float
    ) -> torch.Tensor:
        """A whole recording `(channels, samples)` -> `(frames, output_dim)` on the CPU.

        One pass by default; an encoder whose memory grows with the input
        (Mimi's transformer) streams it in `chunk_seconds` pieces instead.
        """

        del chunk_seconds

        return self([input], [rate])[0].cpu()

    def train(self, mode: bool = True) -> Encoder:
        # Frozen: the pretrained weights stay in eval mode even while training.
        super().train(mode)

        for module in self.children():
            module.eval()

        return self

    def device(self) -> torch.device:
        tensor = next(self.parameters(), None)

        if tensor is None:
            tensor = next(self.buffers(), None)

        return tensor.device if tensor is not None else torch.device("cpu")
