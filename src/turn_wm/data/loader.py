"""
Construct reproducible PyTorch DataLoaders for turn-taking experiments.

This module connects the temporal Dataset, sampling strategy, and collate
function. It contains no model or training-loop logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Sampler

from turn_wm.data.collate import collate_turn_taking
from turn_wm.data.dataset import TurnTakingDataset
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.sampling import SamplingConfig, build_sampler


@dataclass(frozen=True)
class DataLoaderConfig:
    """Runtime configuration for PyTorch data loading."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    seed: int = 42
    # None: shuffle natural training data, keep evaluation data in order.
    shuffle: bool | None = None
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive when set")

        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")

        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")


class FixedPermutationSampler(Sampler[int]):
    """One seeded permutation, replayed identically at every iteration.

    For evaluation: samples of every corpus are interleaved, and every
    validation sees them in the same order, so a limited number of
    validation batches is the same mixed subset each time.
    """

    def __init__(self, size: int, *, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def build_dataloader(
    dataset: TurnTakingDataset | MultiCorpusDataset,
    *,
    loader: DataLoaderConfig,
    sampling: SamplingConfig | None = None,
) -> DataLoader:
    """Build a DataLoader matching the dataset's train/eval mode."""

    generator = torch.Generator()
    generator.manual_seed(loader.seed)

    sampler = None
    shuffle = False

    if dataset.training:
        sampling = sampling or SamplingConfig()

        # Natural sampling uses standard shuffled iteration; only other
        # strategies need the (potentially millions of) sample classes.
        if sampling.strategy != "natural":
            sampler = build_sampler(
                dataset.sample_classes(),
                sampling,
            )

        shuffle = sampler is None

    elif sampling is not None and sampling.strategy != "natural":
        raise ValueError(
            "Non-natural sampling strategies are only valid during training"
        )

    if loader.shuffle is not None:
        if sampler is not None and loader.shuffle:
            raise ValueError("shuffle cannot be combined with a sampling strategy")

        shuffle = loader.shuffle and sampler is None

    if shuffle and not dataset.training:
        # Evaluation order is shuffled once, not anew at every pass.
        sampler = FixedPermutationSampler(len(dataset), seed=loader.seed)
        shuffle = False

    kwargs = {}

    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers

        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor

    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=collate_turn_taking,
        generator=generator,
        **kwargs,
    )
