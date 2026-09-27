"""
The temporal contract between TurnBench audio and frozen V1.

V1 runs on a 100 ms control grid. Slot `k` covers

    [t_k, t_k + 0.1 s),    t_k = k / CONTROL_RATE_HZ

and only once that interval has ended are its two inputs known:

    z_k  the scene latent of the slot (Mimi features aligned causally: the
         latest frame available by the end of the slot)
    a_k  the focal vocal action inside the slot (NO_EVENT / ONSET / OFFSET)

V1 then predicts the next latent:

    (z_k, a_k) -> zhat_(k+1)

so anything derived from zhat_(k+1) is committed, at the earliest, at

    t_k + 0.1 s + preprocessing_lookahead

and never at t_k. `preprocessing_lookahead` is whatever audio past the slot
end the preprocessing reads (e.g. a resampling filter); it is 0 only for a
path that reads no sample past the slot end.

Only complete slots exist: trailing audio shorter than one slot is never
assigned to a slot.
"""

from __future__ import annotations

import math

import torch
import torchaudio.functional as AF

CONTROL_RATE_HZ = 10.0
SLOT_S = 1 / CONTROL_RATE_HZ


def slot_start_s(k: int) -> float:
    """t_k: when slot `k` starts."""

    return k / CONTROL_RATE_HZ


def slot_end_s(k: int) -> float:
    """t_k + 0.1 s: when slot `k` ends and (z_k, a_k) become available."""

    return (k + 1) / CONTROL_RATE_HZ


def commit_time_s(k: int, *, preprocessing_lookahead_s: float) -> float:
    """Earliest commit time of anything derived from zhat_(k+1)."""

    if not math.isfinite(preprocessing_lookahead_s) or preprocessing_lookahead_s < 0:
        raise ValueError("preprocessing_lookahead_s must be finite and >= 0")

    return slot_end_s(k) + preprocessing_lookahead_s


def samples_per(duration_s: float, sample_rate: int) -> int:
    """Samples in `duration_s`; refuses a duration that is not whole samples."""

    samples = duration_s * sample_rate
    rounded = round(samples)

    if rounded <= 0 or abs(samples - rounded) > 1e-6:
        raise ValueError(
            f"{duration_s} s is not a whole number of samples at {sample_rate} Hz"
        )

    return rounded


def slot_count(n_samples: int, sample_rate: int) -> int:
    """Complete control slots in `n_samples`."""

    return n_samples // samples_per(SLOT_S, sample_rate)


def resample_lookahead_s(orig_rate: int, new_rate: int) -> float:
    """How far past an output sample's time the resampler reads its input.

    Measured on the resampler V1's Mimi path uses (torchaudio's windowed
    sinc, default parameters): an impulse at input sample `n` first changes
    output sample `m`, so output `m` reads input up to `n / orig_rate`.
    """

    if orig_rate == new_rate:
        return 0.0

    length = 4 * orig_rate // math.gcd(orig_rate, new_rate) + 4_096
    lookahead = 0.0

    # One impulse per input phase covers every output sample's alignment.
    for n in range(
        length // 2, length // 2 + orig_rate // math.gcd(orig_rate, new_rate)
    ):
        impulse = torch.zeros(length, dtype=torch.float64)
        impulse[n] = 1.0
        response = AF.resample(impulse, orig_freq=orig_rate, new_freq=new_rate)
        first = int(torch.nonzero(response.abs() > 1e-12)[0])
        lookahead = max(lookahead, n / orig_rate - first / new_rate)

    return lookahead
