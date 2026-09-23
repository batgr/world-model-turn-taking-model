"""
Define training-sample selection strategies.

Sampling operates on model-ready sample metadata only. It does not modify the
dataset or its labels. Natural sampling preserves the dataset distribution;
balanced sampling compensates for event/background frequency imbalance.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch.utils.data import Sampler, WeightedRandomSampler

SamplingStrategy = Literal["natural", "balanced"]


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for selecting training examples."""

    strategy: SamplingStrategy = "natural"
    replacement: bool = True
    num_samples: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError("num_samples must be positive")


def build_sampler(
    sample_classes: Sequence[str],
    config: SamplingConfig,
) -> Sampler[int] | None:
    """Build a sampler for the requested training strategy.

    Returns None for natural sampling so the DataLoader can use its standard
    shuffle behavior.
    """

    if config.strategy == "natural":
        return None

    if config.strategy != "balanced":
        raise ValueError(f"Unsupported sampling strategy: {config.strategy!r}")

    if not sample_classes:
        raise ValueError("Cannot build a sampler for an empty dataset")

    counts = Counter(sample_classes)

    if len(counts) < 2:
        raise ValueError("Balanced sampling requires at least two sample classes")

    class_weights = {
        sample_class: 1.0 / count for sample_class, count in counts.items()
    }

    weights = torch.tensor(
        [class_weights[sample_class] for sample_class in sample_classes],
        dtype=torch.double,
    )

    num_samples = (
        config.num_samples if config.num_samples is not None else len(sample_classes)
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=config.replacement,
        generator=generator,
    )
