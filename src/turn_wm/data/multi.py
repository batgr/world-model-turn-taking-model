"""
Compose corpus-local datasets into one global sample index space.

Each child keeps its own anchors, action grid and media index, so a child's
`anchor_row` is only ever resolved against that child's grid. Only the sample
index space is combined, by PyTorch's `ConcatDataset`: global order is the
children's order, then each child's own order.
"""

from __future__ import annotations

from collections.abc import Mapping

from torch.utils.data import ConcatDataset

from turn_wm.data.dataset import TurnTakingDataset


class MultiCorpusDataset(ConcatDataset):
    """`ConcatDataset` of corpus-local datasets that remembers their corpus names."""

    datasets: list[TurnTakingDataset]

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

        super().__init__(list(datasets.values()))
        self.corpora = tuple(datasets)
        self.training = training_modes.pop()

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
