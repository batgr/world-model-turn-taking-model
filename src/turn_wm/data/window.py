"""
Define temporal window semantics around model-ready anchors.

This module contains no storage or framework-specific logic. It converts an
anchor and requested context/future lengths into the temporal indices consumed
by downstream datasets and models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowBounds:
    """Inclusive temporal bounds around one prediction anchor."""

    context_start: int
    context_end: int
    future_start: int
    future_end: int

    @property
    def context_steps(self) -> int:
        return self.context_end - self.context_start + 1

    @property
    def future_steps(self) -> int:
        return self.future_end - self.future_start + 1


def build_window(
    *,
    anchor_idx: int,
    context_steps: int,
    future_steps: int,
) -> WindowBounds:
    """Return temporal bounds for a context/future window.

    The anchor is the final context timestep. Future prediction starts at the
    following timestep.

    Example:
        anchor_idx=100, context_steps=30, future_steps=10

        context: 71..100
        future:  101..110
    """
    if anchor_idx < 0:
        raise ValueError("anchor_idx must be non-negative")

    if context_steps <= 0:
        raise ValueError("context_steps must be positive")

    if future_steps <= 0:
        raise ValueError("future_steps must be positive")

    context_start = anchor_idx - context_steps + 1

    if context_start < 0:
        raise ValueError(
            f"Anchor {anchor_idx} does not provide {context_steps} context steps"
        )

    return WindowBounds(
        context_start=context_start,
        context_end=anchor_idx,
        future_start=anchor_idx + 1,
        future_end=anchor_idx + future_steps,
    )


def validate_against_anchor(
    *,
    context_steps: int,
    future_steps: int,
    max_context_steps: int,
    available_future_steps: int,
) -> None:
    """Validate a requested window against model-ready anchor constraints."""

    if context_steps > max_context_steps:
        raise ValueError(
            f"Requested {context_steps} context steps, "
            f"but anchor supports at most {max_context_steps}"
        )

    if future_steps > available_future_steps:
        raise ValueError(
            f"Requested {future_steps} future steps, "
            f"but anchor supports only {available_future_steps}"
        )
