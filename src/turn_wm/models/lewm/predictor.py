import torch
from torch import nn

from turn_wm.models.lewm.transformer import ConditionalBlock, Transformer


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        position_encoding="rope",
        rope_base=10_000.0,
    ):
        super().__init__()

        if position_encoding not in {"rope", "learned"}:
            raise ValueError(
                "position_encoding must be 'rope' or 'learned', "
                f"got {position_encoding!r}"
            )

        self.position_encoding = position_encoding
        self.num_frames = num_frames
        self.pos_embedding = (
            nn.Parameter(torch.randn(1, num_frames, input_dim))
            if position_encoding == "learned"
            else None
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            use_rope=position_encoding == "rope",
            rope_base=rope_base,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)

        if self.pos_embedding is not None:
            if T > self.num_frames:
                raise ValueError(
                    f"learned positional embeddings support at most "
                    f"{self.num_frames} steps, got {T}"
                )
            x = x + self.pos_embedding[:, :T]

        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
