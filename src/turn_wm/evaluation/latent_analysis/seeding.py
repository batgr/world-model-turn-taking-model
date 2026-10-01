"""Deterministic seeds derived from a base seed and a label (task, fold, ...)."""

from __future__ import annotations

import hashlib
from typing import Any


def derived_seed(seed: int, *parts: Any) -> int:
    """A seed that depends on `seed` and `parts` only, stable across runs."""

    text = ":".join(map(str, (seed, *parts)))

    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=7).digest())
