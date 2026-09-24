"""
Construct reproducible PyTorch DataLoaders for turn-taking experiments.

This module connects the temporal Dataset, sampling strategy, and collate
function. It contains no model or training-loop logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


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
        persistent_workers=loader.num_workers > 0,
    )
