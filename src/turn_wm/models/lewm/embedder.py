import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(
        self,
        num_actions: int,
        smoothed_dim: int,
        emb_dim: int,
        padding_idx: int | None = None,
        mlp_scale: int = 4,
    ):
        super().__init__()

        self.padding_idx = padding_idx

        self.patch_embed = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=smoothed_dim,
            padding_idx=padding_idx,
        )

        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T)

        returns:
            (B, T, emb_dim)
        """
        pad_mask = x.eq(self.padding_idx) if self.padding_idx is not None else None

        x = self.patch_embed(x.long())
        x = self.embed(x)

        if pad_mask is not None:
            x = x.masked_fill(
                pad_mask.unsqueeze(-1),
                0.0,
            )

        return x
