import pytest
import torch
from corpora import make_corpus

from turn_wm.data.build import build_dataset
from turn_wm.data.dataset import STATE_TO_ID, TurnTakingDataset, WindowConfig
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.sampling import SamplingConfig
from turn_wm.data.source import LoadedData

WINDOW = WindowConfig(min_context_steps=10, max_context_steps=10, future_steps=10)


def child(name: str, *, state: str, count: int, training=False, **kwargs):
    corpus = make_corpus(name, state=state, splits={"train": count}, **kwargs)

    return TurnTakingDataset(
        anchors=corpus.model_ready["train"],
        action_grid=corpus.action_grid,
        window=WINDOW,
        training=training,
    )


@pytest.fixture
def combined() -> MultiCorpusDataset:
    return MultiCorpusDataset(
        {
            "a": child("a", state="SILENT", count=3),
            "b": child("b", state="SPEAKING", count=2),
        }
    )


@pytest.mark.parametrize(
    ("index", "sample_id"),
    [
        (0, "a:train#19"),  # first sample of first child
        (2, "a:train#21"),  # last sample of first child
        (3, "b:train#19"),  # boundary: first sample of next child
        (4, "b:train#20"),  # last sample of last child
        (-1, "b:train#20"),
        (-5, "a:train#19"),
    ],
)
def test_global_index_routes_to_child(combined, index, sample_id):
    assert combined[index]["sample_id"] == sample_id


@pytest.mark.parametrize("index", [5, 100, -6])
def test_out_of_range_index_raises(combined, index):
    with pytest.raises(IndexError):
        combined[index]


def test_length_and_provenance(combined):
    assert len(combined) == 5
    assert combined.corpora == ("a", "b")
    assert combined.corpus_sizes() == {"a": 3, "b": 2}


def test_empty_corpus_collection_is_rejected():
    with pytest.raises(ValueError, match="at least one dataset"):
        MultiCorpusDataset({})


def test_empty_child_is_rejected():
    empty = child("b", state="SPEAKING", count=1, is_trainable=False)

    with pytest.raises(ValueError, match="empty child datasets: \\['b'\\]"):
        MultiCorpusDataset({"a": child("a", state="SILENT", count=1), "b": empty})


def test_children_must_share_training_mode():
    with pytest.raises(ValueError, match="same training mode"):
        MultiCorpusDataset(
            {
                "a": child("a", state="SILENT", count=1, training=False),
                "b": child("b", state="SPEAKING", count=1, training=True),
            }
        )


def test_same_local_anchor_row_reads_each_corpus_own_grid(combined):
    """Both corpora use recording r1 and anchor_row 19; only grids differ."""

    from_a = combined[0]
    from_b = combined[3]

    assert (from_a["recording_id"], from_b["recording_id"]) == ("r1", "r1")
    assert (from_a["anchor_idx"], from_b["anchor_idx"]) == (19, 19)

    # Corpus A's grid is SILENT everywhere, corpus B's is SPEAKING.
    assert set(from_a["context_state"].tolist()) == {STATE_TO_ID["SILENT"]}
    assert set(from_a["future_state"].tolist()) == {STATE_TO_ID["SILENT"]}
    assert set(from_b["context_state"].tolist()) == {STATE_TO_ID["SPEAKING"]}
    assert set(from_b["future_state"].tolist()) == {STATE_TO_ID["SPEAKING"]}


def test_samples_share_one_contract_across_corpora(combined):
    from_a = combined[0]
    from_b = combined[3]

    assert from_a.keys() == from_b.keys()

    for key in from_a:
        assert type(from_a[key]) is type(from_b[key]), key

        if isinstance(from_a[key], torch.Tensor):
            assert from_a[key].dtype == from_b[key].dtype, key
            assert from_a[key].shape == from_b[key].shape, key

    varying = {key for key in from_a if _differs(from_a[key], from_b[key])}

    assert varying == {"dataset", "sample_id", "context_state", "future_state"}


def _differs(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return not torch.equal(left, right)

    return left != right


def test_ordinary_dataloader_collates_mixed_batch(combined):
    loader = build_dataloader(combined, loader=DataLoaderConfig(batch_size=5))

    batch = next(iter(loader))

    assert batch["dataset"] == ["a", "a", "a", "b", "b"]
    assert batch["sample_id"][3] == "b:train#19"
    assert batch["context_state"].shape == (5, 10)
    assert batch["future_state"].shape == (5, 10)


def test_shuffled_batch_draws_from_both_corpora(combined):
    loader = build_dataloader(
        combined,
        loader=DataLoaderConfig(batch_size=5, shuffle=True, seed=0),
    )

    batch = next(iter(loader))

    # Every sample appears once; order comes from the seeded shuffle.
    assert sorted(batch["sample_id"]) == sorted(
        combined[i]["sample_id"] for i in range(5)
    )
    assert sorted(batch["dataset"]) == ["a", "a", "a", "b", "b"]


def test_natural_sampling_does_not_request_sample_classes(monkeypatch):
    dataset = MultiCorpusDataset(
        {
            "a": child("a", state="SILENT", count=3, training=True),
            "b": child("b", state="SPEAKING", count=2, training=True),
        }
    )

    def fail():
        raise AssertionError("natural sampling must not read sample classes")

    monkeypatch.setattr(dataset, "sample_classes", fail)

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=5),
        sampling=SamplingConfig(strategy="natural"),
    )

    assert len(next(iter(loader))["dataset"]) == 5


def test_balanced_sampling_uses_classes_from_all_corpora():
    # Each corpus holds one class; balancing is only possible across both.
    dataset = MultiCorpusDataset(
        {
            "a": child("a", state="SILENT", count=3, training=True),
            "b": child(
                "b",
                state="SPEAKING",
                count=1,
                training=True,
                sample_class="background",
            ),
        }
    )

    assert dataset.sample_classes() == ["event"] * 3 + ["background"]

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=4),
        sampling=SamplingConfig(strategy="balanced", num_samples=400, seed=0),
    )

    weights = loader.sampler.weights

    # Classes are weighted by class frequency, independent of corpus.
    assert weights.tolist() == pytest.approx([1 / 3] * 3 + [1.0])

    drawn = list(loader.sampler)
    background_share = sum(index == 3 for index in drawn) / len(drawn)

    assert background_share == pytest.approx(0.5, abs=0.1)


def test_build_dataset_matches_manual_composition():
    loaded = LoadedData(
        corpora=(
            make_corpus("a", state="SILENT", splits={"train": 3}),
            make_corpus("b", state="SPEAKING", splits={"train": 2}),
        )
    )

    dataset = build_dataset(loaded, split="train", window=WINDOW, training=False)

    assert isinstance(dataset, MultiCorpusDataset)
    assert dataset.corpora == ("a", "b")
    assert [dataset[i]["dataset"] for i in range(len(dataset))] == [
        "a",
        "a",
        "a",
        "b",
        "b",
    ]


def test_training_batches_interleave_the_corpora():
    # Training draws from the combined dataset, not corpus after corpus.
    corpus_a = child("a", state="SILENT", count=10, training=True)
    corpus_b = child("b", state="SPEAKING", count=10, training=True)
    loader = build_dataloader(
        MultiCorpusDataset({"a": corpus_a, "b": corpus_b}),
        loader=DataLoaderConfig(batch_size=10, seed=0),
    )

    first = next(iter(loader))

    assert set(first["dataset"]) == {"a", "b"}


def test_validation_batches_interleave_the_corpora_in_a_fixed_order():
    corpus_a = child("a", state="SILENT", count=10)
    corpus_b = child("b", state="SPEAKING", count=10)
    loader = build_dataloader(
        MultiCorpusDataset({"a": corpus_a, "b": corpus_b}),
        loader=DataLoaderConfig(batch_size=10, shuffle=True, seed=0),
    )

    first, again = next(iter(loader)), next(iter(loader))

    assert set(first["dataset"]) == {"a", "b"}
    assert first["sample_id"] == again["sample_id"]
