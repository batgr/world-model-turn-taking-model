import torch
from torch import nn


class CausalBatchNorm1d(nn.Module):
    """BatchNorm over examples at each time, without mixing future positions.

    The input is (B, T, D). Training statistics are computed independently
    for each T; the running statistics average those per-time estimates for
    evaluation with one sample or a streaming prefix.
    """

    expects_sequence = True
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"CausalBatchNorm1d expects (B, T, num_features), got {tuple(x.shape)}"
            )

        values = x.float()

        if self.training:
            if x.shape[0] < 2:
                raise ValueError("training CausalBatchNorm1d requires batch size >= 2")

            mean = values.mean(dim=0)
            variance = (values - mean).square().mean(dim=0)

            with torch.no_grad():
                self.num_batches_tracked.add_(1)
                self.running_mean.lerp_(mean.detach().mean(dim=0), self.momentum)
                # Match BatchNorm's unbiased running variance, per time step.
                unbiased = variance.detach() * x.shape[0] / (x.shape[0] - 1)
                self.running_var.lerp_(unbiased.mean(dim=0), self.momentum)

            mean = mean.unsqueeze(0)
            variance = variance.unsqueeze(0)
        else:
            mean = self.running_mean.view(1, 1, -1)
            variance = self.running_var.view(1, 1, -1)

        normalized = (values - mean) * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).to(dtype=x.dtype)


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.expects_sequence = getattr(norm, "expects_sequence", False)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D), or (B, T, D) with CausalBatchNorm1d.
        """
        return self.net(x)
