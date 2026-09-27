"""
Causal per-speaker vocal activity from one isolated channel.

The transparent rule of the TurnBench reference baseline
(`baselines/rms_vad/predict.py`): non-overlapping 20 ms windows from the
start of the audio, a window is active when its RMS exceeds 0.01 (linear,
on float audio in [-1, 1]). Window `j` covers [j * 20 ms, (j + 1) * 20 ms)
and is known at its end, so it reads no sample after that. Trailing audio
shorter than one window is dropped.

This module knows nothing of actions: any estimator returning one boolean
per window, known at the window's end, can replace it.
"""

from __future__ import annotations

import torch

from turn_wm.evaluation.turnbench.timing import samples_per

WINDOW_S = 0.02
RMS_THRESHOLD = 0.01


def rms_activity(
    channel: torch.Tensor,
    sample_rate: int,
    *,
    window_s: float = WINDOW_S,
    threshold: float = RMS_THRESHOLD,
) -> torch.Tensor:
    """(W,) bool: is each window's RMS above `threshold`?"""

    if channel.ndim != 1:
        raise ValueError(f"channel must be 1-D, got shape {tuple(channel.shape)}")

    window = samples_per(window_s, sample_rate)
    count = len(channel) // window
    windows = channel[: count * window].to(torch.float64).reshape(count, window)

    return windows.pow(2).mean(dim=1).sqrt() > threshold
