from pathlib import Path

import pytest
from corpora import make_corpus, make_manifest

from turn_wm.data import dataset as dataset_module
from turn_wm.data.build import build_dataset
from turn_wm.data.dataset import WindowConfig
from turn_wm.data.reader import MediaWindow
from turn_wm.data.source import LoadedData

WINDOW = WindowConfig(min_context_steps=10, max_context_steps=10, future_steps=10)


def loaded_ab(**corpus_kwargs) -> LoadedData:
    return LoadedData(
        corpora=(
            make_corpus(
                "a",
                state="SILENT",
                splits={"train": 3, "validation": 2, "test": 1},
                **corpus_kwargs.get("a", {}),
            ),
            make_corpus(
                "b",
                state="SPEAKING",
                splits={"train": 2, "validation": 1},
                **corpus_kwargs.get("b", {}),
            ),
        )
    )


@pytest.mark.parametrize(
    ("split", "corpora", "sizes"),
    [
        ("train", ("a", "b"), {"a": 3, "b": 2}),
        ("validation", ("a", "b"), {"a": 2, "b": 1}),
        ("test", ("a",), {"a": 1}),
    ],
)
def test_split_uses_only_corpora_that_publish_it(split, corpora, sizes):
    dataset = build_dataset(loaded_ab(), split=split, window=WINDOW, training=False)

    assert dataset.corpora == corpora
    assert dataset.corpus_sizes() == sizes
    assert {dataset[i]["dataset"] for i in range(len(dataset))} == set(corpora)


def test_single_corpus_uses_same_api():
    loaded = LoadedData(
        corpora=(make_corpus("a", state="SILENT", splits={"train": 3}),)
    )

    dataset = build_dataset(loaded, split="train", window=WINDOW, training=True)

    assert dataset.corpora == ("a",)
    assert len(dataset) == 3
    assert dataset.training is True


def test_unpublished_split_fails():
    with pytest.raises(ValueError, match="'dev' is not published by any"):
        build_dataset(loaded_ab(), split="dev", window=WINDOW, training=False)


def test_corpus_without_usable_anchors_is_left_out():
    loaded = loaded_ab(b={"is_trainable": False})

    dataset = build_dataset(loaded, split="train", window=WINDOW, training=False)

    assert dataset.corpora == ("a",)


def test_no_usable_anchors_anywhere_fails():
    loaded = loaded_ab(a={"is_trainable": False}, b={"is_trainable": False})

    with pytest.raises(ValueError, match="no usable anchors"):
        build_dataset(loaded, split="train", window=WINDOW, training=False)


class EchoReader:
    def read_window(self, media, *, start_time_s, end_time_s):
        return MediaWindow(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            audio=None,
            video=None,
        )


@pytest.fixture
def media_roots(tmp_path, monkeypatch) -> dict[str, Path]:
    monkeypatch.setattr(dataset_module, "MediaReader", EchoReader)

    roots = {}

    for name in ("a", "b"):
        root = tmp_path / name
        (root / "videos").mkdir(parents=True)
        (root / "videos/r1.mp4").touch()
        roots[name] = root

    return roots


def loaded_with_media() -> LoadedData:
    return loaded_ab(
        a={"media_manifest": make_manifest(dataset="a", media_offset_s=0.0)},
        b={"media_manifest": make_manifest(dataset="b", media_offset_s=300.0)},
    )


def test_media_roots_route_to_each_corpus(media_roots):
    dataset = build_dataset(
        loaded_with_media(),
        split="train",
        window=WINDOW,
        training=False,
        media_roots=media_roots,
    )

    from_a = dataset[0]
    from_b = dataset[3]

    # Same canonical window (anchor_row 19) in both corpora...
    for sample in (from_a, from_b):
        context = sample["context_media"]
        assert (context.canonical_start_time_s, context.canonical_end_time_s) == (
            pytest.approx(1.0),
            pytest.approx(2.0),
        )

    # ...read from each corpus's own media timeline.
    assert from_a["context_media"].start_time_s == pytest.approx(1.0)
    assert from_b["context_media"].start_time_s == pytest.approx(301.0)
    assert from_b["future_media"].end_time_s == pytest.approx(303.0)


def test_media_requires_a_root_for_every_included_corpus(media_roots):
    with pytest.raises(ValueError, match="No media root configured.*'b'"):
        build_dataset(
            loaded_with_media(),
            split="train",
            window=WINDOW,
            training=False,
            media_roots={"a": media_roots["a"]},
        )


def test_media_requires_a_manifest(media_roots):
    with pytest.raises(ValueError, match="'a' does not publish a media manifest"):
        build_dataset(
            loaded_ab(),
            split="train",
            window=WINDOW,
            training=False,
            media_roots=media_roots,
        )


def test_without_media_roots_samples_have_no_media():
    dataset = build_dataset(
        loaded_with_media(), split="train", window=WINDOW, training=False
    )

    assert "context_media" not in dataset[0]
    assert "context_media" not in dataset[3]
