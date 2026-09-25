"""
Warmup + cosine learning-rate schedule, stepped once per optimizer step.

The factor multiplies the optimizer's base LR:

    step 0                         min_lr / lr
      ↑ linear warmup
    step warmup_steps              1            (the base LR)
      ↓ cosine decay
    step total_steps - 1           min_lr / lr

`total_steps` is the number of optimizer steps of the run (Lightning's
`trainer.estimated_stepping_batches`), so gradient accumulation, batch limits,
`max_steps` or several devices do not change the shape of the schedule.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """LR factor for the optimizer step `step` (0-based) of `total_steps`."""

    if step < warmup_steps:
        return min_lr_ratio + (1.0 - min_lr_ratio) * step / warmup_steps

    # Cosine from the end of the warmup to the last step; steps past the
    # end (e.g. an estimate a few steps short) stay at the minimum.
    decay_steps = max(1, total_steps - 1 - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)

    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def warmup_steps_for(total_steps: int, warmup_ratio: float) -> int:
    """Warmup length in optimizer steps, leaving at least the last step to decay."""

    return min(round(warmup_ratio * total_steps), max(0, total_steps - 1))


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr: float,
) -> LambdaLR:
    """LambdaLR over `total_steps` optimizer steps for every parameter group.

    Its state (the number of steps taken) is saved in checkpoints; resuming
    rebuilds the same lambda from the configuration and continues there.
    """

    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    base_lrs = {group["lr"] for group in optimizer.param_groups}

    if len(base_lrs) != 1:
        raise ValueError("warmup_cosine_scheduler expects one base LR")

    [base_lr] = base_lrs
    warmup_steps = warmup_steps_for(total_steps, warmup_ratio)
    min_lr_ratio = min_lr / base_lr

    def factor(step: int) -> float:
        return warmup_cosine_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return LambdaLR(optimizer, lr_lambda=factor)
