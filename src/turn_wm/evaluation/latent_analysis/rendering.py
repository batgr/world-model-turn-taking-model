"""Shared figure styling of the latent analyses; matplotlib only when drawing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Reference palette: categorical slots in fixed order, then chart ink.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# A sequential ramp of the first series hue, light to dark.
SEQUENTIAL = ("#d6e6fa", "#2a78d6", "#0b2e59")


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY_INK, labelsize=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def close(figure: Figure) -> None:
    # Figures built from `Figure` are not tracked by pyplot; drop the canvas.
    figure.clear()


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def colors(labels: list[str]) -> dict[str, str]:
    if len(labels) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} categories per figure: {labels}")

    return dict(zip(labels, SERIES, strict=False))


def limits(coordinates: torch.Tensor):
    low = coordinates.min(dim=0).values
    high = coordinates.max(dim=0).values
    pad = (high - low).clamp_min(1e-12) * 0.03
    low, high = (low - pad).tolist(), (high + pad).tolist()

    return (low[0], high[0]), (low[1], high[1]) if len(low) > 1 else (-1.0, 1.0)
