"""JEPA: a frozen encoder, a trainable projector and an action-conditioned predictor.

Observations reach the latent space through one path: encoder features
`(B, T, D)` go through `project_features` (the trainable projector). Raw
observations are encoded first (`encode_features`) by any `Encoder`;
precomputed features (a feature cache) skip the encoder, which is then `None`
and never built.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class JEPA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode_features(
        self, inputs: Sequence[torch.Tensor], rates: Sequence[float]
    ) -> torch.Tensor:
        """Encoder features `(B, frames, D)` of raw observations, before projection."""

        if self.encoder is None:
            raise ValueError(
                "Raw observation encoding is unavailable: this model was built "
                "without an encoder (precomputed features); use project_features"
            )

        return self.encoder(inputs, rates)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project encoder features `(B, T, D)` to latents `(B, T, embed_dim)`."""

        # Cached features are stored in float16; match the projector's
        # parameters (autocast then applies the training precision).
        parameter = next(self.projector.parameters(), None)

        if parameter is not None and features.dtype != parameter.dtype:
            features = features.to(parameter.dtype)

        return _per_step(self.projector, features)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Action ids `(B, T)` -> action embeddings `(B, T, D_action)`."""

        return self.action_encoder(actions)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Next-step latents `(B, T, D)` from latents and action embeddings."""

        return _per_step(self.pred_proj, self.predictor(emb, act_emb))


def _per_step(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`module` on `(B, T, D)`: on whole sequences if it expects them, else per step."""

    if getattr(module, "expects_sequence", False):
        return module(x)

    return module(x.flatten(0, 1)).unflatten(0, x.shape[:2])
