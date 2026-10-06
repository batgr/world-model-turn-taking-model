"""
The training run's data as a LightningDataModule.

The train and validation splits are built with the run's fixed window and
observation source (`build_run_dataset`); the train loader shuffles, the
validation loader keeps one fixed interleaved order across corpora. Building
the datasets draws no random numbers, so where Lightning calls `setup`
changes nothing in the run.
"""

from __future__ import annotations

from dataclasses import replace

import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from turn_wm.data.build import build_dataset
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.source import LoadedData
from turn_wm.training.config import training_window
from turn_wm.training.observations import RunObservations


def build_run_dataset(
    cfg: DictConfig,
    loaded: LoadedData,
    observations: RunObservations,
    *,
    split: str,
    training: bool,
) -> MultiCorpusDataset:
    """The dataset of `split` exactly as the run `cfg` sees it."""

    return build_dataset(
        loaded,
        split=split,
        window=training_window(cfg),
        training=training,
        media_roots=observations.media_roots,
        modalities=observations.modalities,
        feature_store=observations.feature_store,
    )


class TurnTakingDataModule(L.LightningDataModule):
    """Train and validation loaders of one run."""

    def __init__(
        self, cfg: DictConfig, loaded: LoadedData, observations: RunObservations
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loaded = loaded
        self.observations = observations
        self.train_dataset: MultiCorpusDataset | None = None
        self.val_dataset: MultiCorpusDataset | None = None

    def setup(self, stage: str) -> None:
        if self.train_dataset is None:
            self.train_dataset = build_run_dataset(
                self.cfg, self.loaded, self.observations, split="train", training=True
            )
            self.val_dataset = build_run_dataset(
                self.cfg,
                self.loaded,
                self.observations,
                split="validation",
                training=False,
            )

    def _loader_config(self) -> DataLoaderConfig:
        loader = self.cfg.loader
        return DataLoaderConfig(
            batch_size=loader.batch_size,
            drop_last=loader.drop_last,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=loader.persistent_workers,
            prefetch_factor=loader.prefetch_factor,
            seed=self.cfg.seed,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "setup() builds the datasets"
        return build_dataloader(self.train_dataset, loader=self._loader_config())

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "setup() builds the datasets"
        # Corpora interleaved in one fixed order: every validation (and any
        # limit_val_batches subset) covers the same mix of EgoCom and Ego4D.
        config = replace(self._loader_config(), shuffle=True, drop_last=False)
        return build_dataloader(self.val_dataset, loader=config)
