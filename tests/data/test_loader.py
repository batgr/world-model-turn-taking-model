import pytest
import torch
from torch.utils.data import (
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from turn_wm.data.loader import (
    DataLoaderConfig,
    build_dataloader,
)
from turn_wm.data.sampling import SamplingConfig


class FakeTurnTakingDataset(Dataset):
    def __init__(
        self,
        *,
        training: bool,
        size: int = 8,
    ):
        self.training = training
        self.size = size

    def __len__(self):
        return self.size

    def sample_classes(self):
        return ["event" if i % 4 == 0 else "background" for i in range(self.size)]

    def __getitem__(self, index):
        length = 2 + index % 3

        return {
            "context_state": torch.zeros(
                length,
                dtype=torch.long,
            ),
            "context_action": torch.zeros(
                length,
                dtype=torch.long,
            ),
            "context_valid": torch.ones(
                length,
                dtype=torch.bool,
            ),
            "future_state": torch.zeros(
                3,
                dtype=torch.long,
            ),
            "future_action": torch.zeros(
                3,
                dtype=torch.long,
            ),
            "future_valid": torch.ones(
                3,
                dtype=torch.bool,
            ),
            "context_length": length,
            "sample_id": f"sample-{index}",
            "dataset": "synthetic",
            "recording_id": "r1",
            "anchor_idx": index,
            "anchor_time": index / 10,
            "sample_class": self.sample_classes()[index],
        }


def test_training_natural_sampling_uses_shuffle():
    dataset = FakeTurnTakingDataset(
        training=True,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=2,
        ),
        sampling=SamplingConfig(
            strategy="natural",
        ),
    )

    assert isinstance(
        loader.sampler,
        RandomSampler,
    )


def test_training_balanced_sampling_uses_weighted_sampler():
    dataset = FakeTurnTakingDataset(
        training=True,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=2,
        ),
        sampling=SamplingConfig(
            strategy="balanced",
        ),
    )

    assert isinstance(
        loader.sampler,
        WeightedRandomSampler,
    )


def test_evaluation_uses_sequential_sampling():
    dataset = FakeTurnTakingDataset(
        training=False,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=2,
        ),
    )

    assert isinstance(
        loader.sampler,
        SequentialSampler,
    )


def test_balanced_sampling_is_rejected_for_evaluation():
    dataset = FakeTurnTakingDataset(
        training=False,
    )

    with pytest.raises(
        ValueError,
        match="only valid during training",
    ):
        build_dataloader(
            dataset,
            loader=DataLoaderConfig(),
            sampling=SamplingConfig(
                strategy="balanced",
            ),
        )


def test_loader_produces_padded_batch():
    dataset = FakeTurnTakingDataset(
        training=False,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=3,
        ),
    )

    batch = next(iter(loader))

    assert batch["context_state"].shape == (3, 4)
    assert batch["context_action"].shape == (3, 4)
    assert batch["context_mask"].shape == (3, 4)

    assert batch["future_state"].shape == (3, 3)
    assert batch["future_action"].shape == (3, 3)


def test_batch_size_is_respected():
    dataset = FakeTurnTakingDataset(
        training=False,
        size=8,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=4,
        ),
    )

    batch = next(iter(loader))

    assert batch["context_state"].shape[0] == 4


def test_drop_last_is_respected():
    dataset = FakeTurnTakingDataset(
        training=False,
        size=5,
    )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=2,
            drop_last=True,
        ),
    )

    assert len(loader) == 2


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_invalid_batch_size_raises(batch_size):
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=batch_size,
        )


def test_negative_worker_count_raises():
    with pytest.raises(ValueError):
        DataLoaderConfig(
            num_workers=-1,
        )
