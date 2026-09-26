"""JEPA Implementation

Observations reach the latent space through one path: encoder features
`(B, T, D)` go through `project_features` (the trainable projector). Raw
observations are encoded first (`encode`); precomputed features, e.g. cached
Mimi features, skip the encoder, which is then `None` and never built.
"""

from __future__ import annotations

import torch
from einops import rearrange
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

    def encode(
        self,
        observation,
        **encoder_kwargs,
    ):
        """Encode raw observations, then project them to the latent space."""

        return self.project_features(
            self.encode_features(observation, **encoder_kwargs)
        )

    def encode_features(
        self,
        observation,
        **encoder_kwargs,
    ) -> torch.Tensor:
        """Encoder features `(B, T, D)` of raw observations, before projection."""

        if self.encoder is None:
            raise ValueError(
                "Raw observation encoding is unavailable: this model was built "
                "without an encoder (precomputed features); use project_features"
            )

        features = self.encoder(
            observation,
            **encoder_kwargs,
        )

        if features.ndim != 3:
            raise ValueError("encoder must return (B, T, D)")

        return features

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Project encoder features `(B, T, D)` to latents `(B, T, embed_dim)`."""

        if features.ndim != 3:
            raise ValueError(
                f"features must have shape (B, T, D), got {tuple(features.shape)}"
            )

        # Cached features are stored in float16; match the projector's
        # parameters (autocast then applies the training precision).
        parameter = next(self.projector.parameters(), None)

        if parameter is not None and features.dtype != parameter.dtype:
            features = features.to(parameter.dtype)

        b = features.size(0)

        emb = rearrange(
            features,
            "b t d -> (b t) d",
        )

        emb = self.projector(emb)

        emb = rearrange(
            emb,
            "(b t) d -> b t d",
            b=b,
        )

        return emb

    def encode_actions(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            actions:
                (B, T)

        Returns:
            action embeddings:
                (B, T, D_action)
        """
        return self.action_encoder(actions)

    def predict(
        self,
        emb: torch.Tensor,
        act_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next-state embeddings.

        Args:
            emb:
                (B, T, D)

            act_emb:
                (B, T, D_action)

        Returns:
            predictions:
                (B, T, D)
        """

        if emb.shape[:2] != act_emb.shape[:2]:
            raise ValueError(
                "embedding and action timelines must match: "
                f"{emb.shape[:2]} != {act_emb.shape[:2]}"
            )

        preds = self.predictor(
            emb,
            act_emb,
        )

        batch_size = preds.size(0)

        preds = rearrange(
            preds,
            "b t d -> (b t) d",
        )

        preds = self.pred_proj(preds)

        preds = rearrange(
            preds,
            "(b t) d -> b t d",
            b=batch_size,
        )

        return preds
