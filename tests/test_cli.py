import pytest
import torch
from datasets import Dataset, DatasetDict

from turn_wm import cli
from turn_wm.data import dataset as dataset_module
from turn_wm.data.reader import DecodedAudio, DecodedVideo, MediaWindow
from turn_wm.data.source import LoadedCorpus, LoadedData


def make_grid(length: int = 40) -> Dataset:
    return Dataset.from_dict(
        {
            "recording_id": ["r1"] * length,
            "decision_index": list(range(length)),
            "decision_time_s": [i / 10 for i in range(length)],
            "focal_state_before": ["SILENT"] * 20 + ["SPEAKING"] * (length - 20),
            "action": ["NO_EVENT"] * 20 + ["ONSET"] + ["NO_EVENT"] * (length - 21),
            "action_valid": [True] * length,
        }
    )


def make_anchors(
    *, count: int = 3, is_trainable: bool = True, dataset: str = "synthetic"
) -> Dataset:
    anchor_idx = [19 + i for i in range(count)]

    return Dataset.from_dict(
        {
            "sample_id": [f"r1#{i}" for i in anchor_idx],
            "dataset": [dataset] * count,
            "recording_id": ["r1"] * count,
            "anchor_idx": anchor_idx,
            "anchor_row": anchor_idx,
            "anchor_time": [i / 10 for i in anchor_idx],
            "max_context_steps": [15] * count,
            "future_steps": [10] * count,
            "sample_class": ["event"] * count,
            "is_trainable": [is_trainable] * count,
        }
    )


def make_data(
    *, media_manifest: Dataset | None = None, **splits: Dataset
) -> LoadedData:
    return LoadedData(corpora=(make_corpus(media_manifest=media_manifest, **splits),))


def make_corpus(
    name: str = "synthetic",
    *,
    media_manifest: Dataset | None = None,
    **splits: Dataset,
) -> LoadedCorpus:
    return LoadedCorpus(
        name=name,
        model_ready=DatasetDict(splits or {"train": make_anchors(dataset=name)}),
        action_grid=make_grid(),
        metadata={
            "splits": {"recordings": {"train": 1}},
            "counts": {"recordings": 1},
            "grid": {"frequency_hz": 10.0},
        },
        media_manifest=media_manifest,
    )


@pytest.fixture
def fake_load(monkeypatch):
    """Replace the Hub boundary; returns the list of sources requested."""

    calls = []
    state = {"data": make_data()}

    def load(source):
        calls.append(source)
        return state["data"]

    monkeypatch.setattr(cli, "load_data", load)

    return calls, state


def test_inspect_data_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["inspect-data", "--help"])

    assert exit_info.value.code == 0
    assert "--batch-size" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect-data", "--dataset", "unknown"],
        ["inspect-data", "--split", "dev"],
        ["inspect-data", "--batch-size", "0"],
        ["inspect-data", "--batch-size", "four"],
        ["inspect-data", "--context-min", "20", "--context-max", "10"],
    ],
)
def test_invalid_arguments_fail_before_loading(argv, fake_load, capsys):
    calls, _ = fake_load

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert calls == []


def test_invalid_dataset_message(capsys):
    with pytest.raises(SystemExit):
        cli.main(["inspect-data", "--dataset", "unknown"])

    assert "invalid choice: 'unknown'" in capsys.readouterr().err


def test_batch_size_message(capsys):
    with pytest.raises(SystemExit):
        cli.main(["inspect-data", "--batch-size", "-1"])

    assert "must be positive" in capsys.readouterr().err


def test_inspect_data_formats_batch(fake_load, capsys):
    calls, _ = fake_load

    assert cli.main(["inspect-data", "--batch-size", "2"]) == 0

    out = capsys.readouterr().out

    assert calls == [cli.DATASETS["egocom"]]
    assert "source: batgre/conversational-dynamics-egocom" in out
    assert "revision: default branch (unpinned)" in out
    assert "samples: 3" in out
    assert "batch size: 2" in out
    assert "context_state:  (2, 15)" in out
    assert "future_action:  (2, 10)" in out
    assert "grid: 10 Hz" in out
    assert "sample_id: r1#19" in out
    assert "3 PAD" in out
    assert "3 MASKED\n4 PAD" in out


def test_missing_split_fails_clearly(fake_load):
    _, state = fake_load
    state["data"] = make_data(train=make_anchors())

    with pytest.raises(SystemExit, match="'test' is not published"):
        cli.main(["inspect-data", "--split", "test"])


def test_no_usable_anchors_fails_clearly(fake_load):
    _, state = fake_load
    state["data"] = make_data(train=make_anchors(is_trainable=False))

    with pytest.raises(SystemExit, match="no usable"):
        cli.main(["inspect-data"])


def test_network_failure_is_reported(monkeypatch):
    def load(source):
        raise ConnectionError("offline")

    monkeypatch.setattr(cli, "load_data", load)

    with pytest.raises(SystemExit, match="could not reach Hugging Face"):
        cli.main(["inspect-data"])


def test_schema_mismatch_is_reported(monkeypatch):
    def load(source):
        raise ValueError("action_grid is missing required columns: ['action']")

    monkeypatch.setattr(cli, "load_data", load)

    with pytest.raises(SystemExit, match="does not match the modelling data"):
        cli.main(["inspect-data"])


def make_manifest(*rows: dict) -> Dataset:
    defaults = {
        "dataset": "synthetic",
        "recording_id": "r1",
        "video_path": "videos/r1.mp4",
        "audio_path": None,
        "media_offset_s": 0.0,
        "video_has_audio": True,
    }

    return Dataset.from_list([{**defaults, **row} for row in rows or [{}]])


class FakeMediaReader:
    """Decodes nothing; returns small tensors shaped like real media."""

    def read_window(self, media, *, start_time_s, end_time_s, modalities):
        frames = round((end_time_s - start_time_s) * 30)
        audio = DecodedAudio(
            waveform=torch.zeros(2, round((end_time_s - start_time_s) * 16_000)),
            sample_rate=16_000,
        )
        video = DecodedVideo(
            frames=torch.zeros(frames, 3, 24, 32, dtype=torch.uint8),
            timestamps_s=start_time_s + torch.arange(frames) / 30,
        )

        return MediaWindow(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            audio=audio if "audio" in modalities else None,
            video=video if "video" in modalities else None,
        )


@pytest.fixture
def media_root(tmp_path, monkeypatch):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/r1.mp4").touch()
    monkeypatch.setattr(dataset_module, "MediaReader", FakeMediaReader)

    return tmp_path


def test_inspect_data_without_media_has_no_media_section(fake_load, capsys):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    assert cli.main(["inspect-data", "--batch-size", "2"]) == 0

    assert "Media\n-----" not in capsys.readouterr().out


def test_inspect_data_with_media_manifest(fake_load, media_root, capsys):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    assert cli.main(["inspect-data", "--media-root", str(media_root)]) == 0

    out = capsys.readouterr().out

    assert "Media\n-----" in out
    assert "dataset: synthetic" in out
    assert f"video file: {media_root / 'videos/r1.mp4'}" in out
    assert "source: embedded video" in out
    assert "sample rate: 16000 Hz" in out
    # First anchor (19) with 15 context steps: rows 5..19, future 20..29.
    assert "canonical time: 0.500 s → 2.000 s" in out
    assert "physical media time: 0.500 s → 2.000 s" in out
    assert "shape (T, C, H, W): (45, 3, 24, 32)" in out


def test_inspect_data_reports_default_modalities(fake_load, media_root, capsys):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    assert cli.main(["inspect-data", "--media-root", str(media_root)]) == 0

    assert "modalities: audio, video" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("value", "reported", "absent"),
    [
        ("audio", "audio", "video"),
        ("video", "video", "audio"),
        ("video,audio", "audio, video", None),
    ],
)
def test_inspect_data_modalities(
    fake_load, media_root, capsys, value, reported, absent
):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    argv = ["inspect-data", "--media-root", str(media_root), "--modalities", value]

    assert cli.main(argv) == 0

    out = capsys.readouterr().out

    assert f"modalities: {reported}" in out

    for modality in ("audio", "video"):
        present = "no" if modality == absent else "yes"
        assert f"  {modality}:\n    present: {present}" in out


@pytest.mark.parametrize("value", ["", "text", "audio,depth", "audio,audio"])
def test_inspect_data_rejects_invalid_modalities(media_root, capsys, value):
    argv = ["inspect-data", "--media-root", str(media_root), "--modalities", value]

    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    assert error.value.code == 2
    assert "argument --modalities" in capsys.readouterr().err


def test_modalities_require_a_media_root(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["inspect-data", "--modalities", "audio"])

    assert error.value.code == 2
    assert "--modalities requires --media-root" in capsys.readouterr().err


def test_nonzero_offset_formatting(fake_load, media_root, capsys):
    _, state = fake_load
    state["data"] = make_data(
        media_manifest=make_manifest({"media_offset_s": 300.0}),
        train=make_anchors(),
    )

    assert cli.main(["inspect-data", "--media-root", str(media_root)]) == 0

    out = capsys.readouterr().out

    assert "media offset: 300 s" in out
    assert "canonical time: 2.000 s → 3.000 s" in out
    assert "physical media time: 302.000 s → 303.000 s" in out


def test_named_media_root(fake_load, media_root, capsys):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    argv = ["inspect-data", "--media-root", f"synthetic={media_root}"]

    assert cli.main(argv) == 0
    assert "dataset: synthetic" in capsys.readouterr().out


def test_media_root_must_be_a_directory(fake_load, tmp_path, capsys):
    calls, _ = fake_load

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["inspect-data", "--media-root", str(tmp_path / "missing")])

    assert exit_info.value.code == 2
    assert "media root is not a directory" in capsys.readouterr().err
    assert calls == []


def test_missing_root_for_manifest_dataset(fake_load, media_root):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    with pytest.raises(SystemExit, match="No media root configured.*synthetic"):
        cli.main(["inspect-data", "--media-root", f"other={media_root}"])


def test_unnamed_root_needs_single_dataset_manifest(fake_load, media_root):
    _, state = fake_load
    state["data"] = make_data(
        media_manifest=make_manifest({}, {"dataset": "other"}),
        train=make_anchors(),
    )

    with pytest.raises(SystemExit, match="DATASET=PATH"):
        cli.main(["inspect-data", "--media-root", str(media_root)])


def test_media_root_without_published_manifest(fake_load, media_root):
    with pytest.raises(SystemExit, match="does not publish a media manifest"):
        cli.main(["inspect-data", "--media-root", str(media_root)])


def test_missing_local_media_file(fake_load, media_root):
    _, state = fake_load
    state["data"] = make_data(
        media_manifest=make_manifest({"video_path": "videos/absent.mp4"}),
        train=make_anchors(),
    )

    with pytest.raises(SystemExit, match="raw media for the first batch"):
        cli.main(["inspect-data", "--media-root", str(media_root)])


def make_two_corpora() -> LoadedData:
    return LoadedData(
        corpora=(
            make_corpus(
                "a",
                train=make_anchors(dataset="a", count=3),
                test=make_anchors(dataset="a", count=1),
            ),
            make_corpus("b", train=make_anchors(dataset="b", count=2)),
        )
    )


def test_multi_corpus_summary(fake_load, capsys):
    calls, state = fake_load
    state["data"] = make_two_corpora()

    assert cli.main(["inspect-data", "--dataset", "full", "--batch-size", "5"]) == 0

    out = capsys.readouterr().out

    assert calls == [cli.DATASETS["full"]]
    assert "corpora:\n  - a\n  - b\n" in out
    assert "usable samples: 5" in out
    assert "Corpus: a" in out and "Corpus: b" in out
    assert "datasets:\n  a: 3\n  b: 2\n" in out
    assert "context_state:  (5, 15)" in out


def test_multi_corpus_split_lists_absent_corpora(fake_load, capsys):
    _, state = fake_load
    state["data"] = make_two_corpora()

    assert cli.main(["inspect-data", "--dataset", "full", "--split", "test"]) == 0

    out = capsys.readouterr().out

    assert "corpora:\n  - a\n" in out
    assert "not in this split: b" in out
    assert "Corpus: b" not in out


def test_shuffled_inspection_is_seeded(fake_load, capsys):
    _, state = fake_load
    state["data"] = make_two_corpora()

    argv = ["inspect-data", "--dataset", "full", "--batch-size", "5", "--shuffle"]

    cli.main(argv)
    first = capsys.readouterr().out
    cli.main(argv)
    second = capsys.readouterr().out

    assert first == second
    assert "datasets:\n  a: 3\n  b: 2\n" in first


def test_multi_corpus_media_needs_named_roots(fake_load, media_root):
    _, state = fake_load
    state["data"] = LoadedData(
        corpora=(
            make_corpus(
                "a",
                train=make_anchors(dataset="a"),
                media_manifest=make_manifest({"dataset": "a"}),
            ),
            make_corpus(
                "b",
                train=make_anchors(dataset="b"),
                media_manifest=make_manifest({"dataset": "b"}),
            ),
        )
    )

    with pytest.raises(SystemExit, match="DATASET=PATH"):
        cli.main(["inspect-data", "--media-root", str(media_root)])

    assert (
        cli.main(
            [
                "inspect-data",
                "--dataset",
                "full",
                "--media-root",
                f"a={media_root}",
                "--media-root",
                f"b={media_root}",
            ]
        )
        == 0
    )
