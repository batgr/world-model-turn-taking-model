"""
Compose corpus-local datasets into one global sample index space.

Each child keeps its own anchors, action grid and media index, so a child's
`anchor_row` is only ever resolved against that child's grid. Only the sample
index space is combined: a global index is routed to (child, local index) with
a binary search over cumulative sizes.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from itertools import accumulate
from typing import Any

from torch.utils.data import Dataset

from turn_wm.data.dataset import TurnTakingDataset


class MultiCorpusDataset(Dataset):
    """Expose several corpus-local datasets through one global index space.

    Global order is the children's order, then each child's own order, so it
    is deterministic given the children.
    """

    def __init__(
        self,
        datasets: Mapping[str, TurnTakingDataset],
    ) -> None:
        if not datasets:
            raise ValueError("MultiCorpusDataset requires at least one dataset")

        empty = [name for name, dataset in datasets.items() if len(dataset) == 0]

        if empty:
            raise ValueError(
                f"MultiCorpusDataset does not accept empty child datasets: {empty}"
            )

        training_modes = {dataset.training for dataset in datasets.values()}

        if len(training_modes) != 1:
            raise ValueError("All child datasets must use the same training mode")

        self.corpora = tuple(datasets)
        self.datasets = tuple(datasets.values())
        self.training = training_modes.pop()

        self._cumulative_sizes = tuple(
            accumulate(len(dataset) for dataset in self.datasets)
        )

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        dataset_index = bisect_right(
            self._cumulative_sizes,
            index,
        )

        previous_size = (
            0 if dataset_index == 0 else self._cumulative_sizes[dataset_index - 1]
        )

        local_index = index - previous_size

        return self.datasets[dataset_index][local_index]

    def corpus_sizes(self) -> dict[str, int]:
        """Number of samples each corpus contributes, in global order."""

        return {
            name: len(dataset)
            for name, dataset in zip(self.corpora, self.datasets, strict=True)
        }

    def sample_classes(self) -> list[str]:
        """Return classes in global dataset order."""

        classes: list[str] = []

        for dataset in self.datasets:
            classes.extend(dataset.sample_classes())

        return classes
