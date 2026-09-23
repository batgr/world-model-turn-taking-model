import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from turn_wm.data.sampling import (
    SamplingConfig,
    build_sampler,
)


def test_natural_sampling_returns_no_sampler():
    sampler = build_sampler(
        ["background", "event"],
        SamplingConfig(strategy="natural"),
    )

    assert sampler is None


def test_balanced_sampling_uses_inverse_frequency_weights():
    classes = [
        "background",
        "background",
        "background",
        "event",
    ]

    sampler = build_sampler(
        classes,
        SamplingConfig(strategy="balanced"),
    )

    assert isinstance(
        sampler,
        WeightedRandomSampler,
    )

    background_weight = sampler.weights[0]
    event_weight = sampler.weights[-1]

    assert torch.isclose(
        background_weight,
        torch.tensor(
            1 / 3,
            dtype=torch.double,
        ),
    )

    assert torch.isclose(
        event_weight,
        torch.tensor(
            1.0,
            dtype=torch.double,
        ),
    )


def test_default_sample_count_matches_dataset_size():
    classes = [
        "background",
        "background",
        "event",
    ]

    sampler = build_sampler(
        classes,
        SamplingConfig(strategy="balanced"),
    )

    assert sampler.num_samples == len(classes)


def test_explicit_sample_count_is_respected():
    sampler = build_sampler(
        ["background", "event"],
        SamplingConfig(
            strategy="balanced",
            num_samples=100,
        ),
    )

    assert sampler.num_samples == 100


def test_balanced_sampling_is_reproducible():
    classes = [
        "background",
        "background",
        "background",
        "event",
    ]

    config = SamplingConfig(
        strategy="balanced",
        num_samples=20,
        seed=123,
    )

    first = list(build_sampler(classes, config))

    second = list(build_sampler(classes, config))

    assert first == second


def test_balanced_sampling_requires_two_classes():
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        build_sampler(
            ["background"] * 10,
            SamplingConfig(strategy="balanced"),
        )


def test_empty_balanced_dataset_raises():
    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        build_sampler(
            [],
            SamplingConfig(strategy="balanced"),
        )


def test_invalid_num_samples_raises():
    with pytest.raises(ValueError):
        SamplingConfig(
            strategy="balanced",
            num_samples=0,
        )
