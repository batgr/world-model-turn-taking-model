from pathlib import Path

import pytest
from corpora import make_corpus, make_manifest
from datasets import Dataset
from synthetic_media import make_audio, make_video, make_video_with_audio

from turn_wm.data import dataset as dataset_module
from turn_wm.data.build import build_dataset
from turn_wm.data.dataset import WindowConfig
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
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
    def read_window(self, media, *, start_time_s, end_time_s, modalities):
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


def test_modalities_default_to_audio_and_video_in_every_corpus(media_roots):
    dataset = build_dataset(
        loaded_with_media(),
        split="train",
        window=WINDOW,
        training=False,
        media_roots=media_roots,
    )

    assert [child.modalities for child in dataset.datasets] == [
        ("audio", "video"),
        ("audio", "video"),
    ]


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (("audio",), ("audio",)),
        (("video",), ("video",)),
        (("video", "audio"), ("audio", "video")),
    ],
)
def test_modalities_reach_every_corpus(media_roots, modalities, expected):
    dataset = build_dataset(
        loaded_with_media(),
        split="train",
        window=WINDOW,
        training=False,
        media_roots=media_roots,
        modalities=modalities,
    )

    assert [child.modalities for child in dataset.datasets] == [expected, expected]


@pytest.mark.parametrize("modalities", [(), ("text",), ("audio", "depth")])
def test_build_rejects_invalid_modalities(modalities):
    with pytest.raises(ValueError, match="modalit"):
        build_dataset(
            loaded_ab(),
            split="train",
            window=WINDOW,
            training=False,
            modalities=modalities,
        )


def test_modalities_without_media_roots_keep_state_action_samples():
    dataset = build_dataset(
        loaded_ab(), split="train", window=WINDOW, training=False, modalities=("audio",)
    )

    assert "context_media" not in dataset[0]


@pytest.fixture
def real_media_corpora(tmp_path) -> tuple[LoadedData, dict[str, Path]]:
    """EgoCom-like (separate audio file) and Ego4D-like (embedded, offset)."""

    egocom, ego4d = tmp_path / "egocom", tmp_path / "ego4d"
    (egocom / "videos").mkdir(parents=True)
    (ego4d / "videos").mkdir(parents=True)

    make_video(egocom / "videos/r1.mp4", frames=50)
    make_audio(egocom / "r1.wav", duration_s=5.0)
    make_video_with_audio(ego4d / "videos/r1.mp4", duration_s=5.0)

    def manifest(dataset: str, **row) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "dataset": dataset,
                    "recording_id": "r1",
                    "video_path": "videos/r1.mp4",
                    **row,
                }
            ]
        )

    loaded = LoadedData(
        corpora=(
            make_corpus(
                "egocom",
                state="SILENT",
                splits={"train": 3},
                media_manifest=manifest(
                    "egocom",
                    audio_path="r1.wav",
                    media_offset_s=0.0,
                    video_has_audio=False,
                ),
            ),
            make_corpus(
                "ego4d",
                state="SPEAKING",
                splits={"train": 2},
                media_manifest=manifest(
                    "ego4d",
                    audio_path=None,
                    media_offset_s=0.5,
                    video_has_audio=True,
                ),
            ),
        )
    )

    return loaded, {"egocom": egocom, "ego4d": ego4d}


@pytest.mark.parametrize(
    ("modalities", "audio", "video"),
    [
        (None, True, True),
        (("audio",), True, False),
        (("video",), False, True),
    ],
)
def test_multi_corpus_batch_decodes_only_selected_modalities(
    real_media_corpora, decode_spies, modalities, audio, video
):
    loaded, roots = real_media_corpora
    selection = {} if modalities is None else {"modalities": modalities}

    dataset = build_dataset(
        loaded,
        split="train",
        window=WINDOW,
        training=False,
        media_roots=roots,
        **selection,
    )
    loader = build_dataloader(dataset, loader=DataLoaderConfig(batch_size=5))

    [batch] = list(loader)

    assert batch["dataset"] == ["egocom"] * 3 + ["ego4d"] * 2

    for key in ("context_media", "future_media"):
        for window in batch[key]:
            assert (window.audio is not None) is audio
            assert (window.video is not None) is video

    # Each sample reads two windows; nothing unselected is ever decoded.
    reads = 2 * len(batch["dataset"])
    assert len(decode_spies["audio"]) == (reads if audio else 0)
    assert len(decode_spies["video"]) == (reads if video else 0)

    if audio:
        # Ego4D-like audio still comes from its video container.
        assert roots["ego4d"] / "videos/r1.mp4" in decode_spies["audio"]
