"""
A run's messages, kept in its run directory as `train.log`.

Lightning's messages (validation verdicts of `ModelCheckpoint`, resumption,
warnings) are mirrored to `<run_dir>/train.log`, so a run's history survives
the terminal or the notebook it ran in.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

TEXT_LOG = "train.log"


@contextmanager
def text_log(run_dir: Path) -> Iterator[None]:
    """Mirror Lightning's messages into `<run_dir>/train.log` while training."""

    handler = logging.FileHandler(run_dir / TEXT_LOG, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    lightning = logging.getLogger("lightning.pytorch")
    lightning.addHandler(handler)

    try:
        yield
    finally:
        lightning.removeHandler(handler)
        handler.close()
