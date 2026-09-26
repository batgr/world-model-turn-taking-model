"""
Progress bars for the long offline commands (extraction, probes, analyses).

One place decides how progress looks: tqdm on stderr, refreshed at most
once a second so notebook logs (e.g. Colab's `!turn-wm ...`) stay small.
`TURN_WM_PROGRESS=0` turns every bar off. Bars never change a result.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from typing import Any

from tqdm.auto import tqdm

ENVIRONMENT_VARIABLE = "TURN_WM_PROGRESS"


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT_VARIABLE, "1") not in {"0", "false", "no"}


def progress(
    iterable: Iterable[Any] | None = None,
    *,
    total: float | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
) -> tqdm:
    """A tqdm bar with the project's defaults."""

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        mininterval=1.0,
        dynamic_ncols=True,
        disable=not enabled(),
    )


def log(message: str) -> None:
    """A line printed above any running bar, on stderr like the bars."""

    if enabled():
        tqdm.write(message, file=sys.stderr)
