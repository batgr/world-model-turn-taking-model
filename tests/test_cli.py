import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, DatasetDict

from turn_wm import cli
from turn_wm.data import cli as data_cli
from turn_wm.data import dataset as dataset_module
from turn_wm.data.reader import DecodedAudio, DecodedVideo, MediaWindow
from turn_wm.data.source import LoadedCorpus, LoadedData
from turn_wm.models.encoders.logmel import LogMelEncoder
from turn_wm.training import cli as training_cli
from turn_wm.training import train as train_module


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
    name: str = "synthetic", *, media_manifest: Dataset | None = None, **splits: Dataset
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

    monkeypatch.setattr(data_cli, "load_data", load)

    return calls, state


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


def test_train_loads_config_and_runs(monkeypatch):
    cfg = object()
    received = {}

    def fake_load_config(overrides):
        received["overrides"] = list(overrides)
        return cfg

    def fake_run(received_cfg):
        received["cfg"] = received_cfg

    monkeypatch.setattr(training_cli, "load_config", fake_load_config)
    monkeypatch.setattr(training_cli, "run_training", fake_run)

    assert cli.main(["train", "data.dataset=egocom", "trainer.max_epochs=3"]) == 0

    assert received["overrides"] == ["data.dataset=egocom", "trainer.max_epochs=3"]
    assert received["cfg"] is cfg


def test_train_accepts_no_overrides(monkeypatch):
    cfg = object()
    received = {}

    def fake_load_config(overrides):
        received["overrides"] = list(overrides)
        return cfg

    monkeypatch.setattr(training_cli, "load_config", fake_load_config)
    monkeypatch.setattr(training_cli, "run_training", lambda _: None)

    assert cli.main(["train"]) == 0
    assert received["overrides"] == []


def test_train_composes_the_real_config(monkeypatch):
    received = {}
    monkeypatch.setattr(
        training_cli, "run_training", lambda cfg: received.setdefault("cfg", cfg)
    )

    assert cli.main(["train", "data.dataset=egocom", "loader.batch_size=16"]) == 0

    assert received["cfg"].data.dataset == "egocom"
    assert received["cfg"].loader.batch_size == 16


@pytest.mark.parametrize("override", ["model.nope=1", "train=missing", "==="])
def test_train_invalid_override_is_a_usage_error(monkeypatch, capsys, override):
    monkeypatch.setattr(
        training_cli, "run_training", lambda _: pytest.fail("must not run")
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["train", override])

    assert error.value.code == 2
    assert "invalid configuration override" in capsys.readouterr().err


def test_train_rejected_config_is_a_clear_cli_error(monkeypatch):
    def reject(cfg):
        raise ValueError("prediction.rollout_context_size must lie in [1, 15]")

    monkeypatch.setattr(training_cli, "run_training", reject)

    with pytest.raises(SystemExit) as error:
        cli.main(["train", "prediction.rollout_context_size=100"])

    assert error.value.code == (
        "turn-wm: error: prediction.rollout_context_size must lie in [1, 15]"
    )


def test_train_reports_a_real_validation_error(monkeypatch):
    # No mock of run: validation fails before any data is loaded.
    monkeypatch.setattr(
        train_module, "load_data", lambda *_: pytest.fail("must not load")
    )

    with pytest.raises(SystemExit) as error:
        cli.main(["train", "prediction.rollout_context_size=100"])

    assert str(error.value.code).startswith(
        "turn-wm: error: prediction.rollout_context_size must lie in"
    )


@pytest.fixture
def fake_precompute(monkeypatch, tmp_path):
    """Replace the feature cache generation; records its arguments."""

    received = {}

    def precompute(loaded, **kwargs):
        received["loaded"] = loaded
        received.update(kwargs)
        span = SimpleNamespace(dataset="synthetic", recording_id="r1")
        kwargs["progress"](1, 1, span)

        output = kwargs["output_root"]
        output.mkdir(parents=True)
        manifest = output / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "recordings": [{"recording_id": "r1"}],
                    "features": {"rate_hz": 10.0, "dim": 512},
                }
            )
        )
        return manifest

    monkeypatch.setattr(data_cli, "precompute_features", precompute)

    return received


def test_precompute_features_propagates_every_option(
    fake_load, fake_precompute, tmp_path, capsys
):
    calls, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())
    output = tmp_path / "cache"

    argv = [
        "precompute-features",
        "--dataset",
        "egocom",
        "--media-root",
        str(tmp_path),
        "--output",
        str(output),
        "--device",
        "mps",
        "--chunk-seconds",
        "10",
        "--encoder",
        "logmel",
        "feature_dim=40",
    ]

    assert cli.main(argv) == 0

    assert calls == [data_cli.DATASETS["egocom"]]
    assert fake_precompute["loaded"] is state["data"]
    assert fake_precompute["media_roots"] == {"synthetic": tmp_path}
    assert fake_precompute["output_root"] == output
    assert fake_precompute["device"] == "mps"
    assert fake_precompute["chunk_seconds"] == 10.0
    # The encoder config and its overrides, at the data's grid rate.
    encoder = fake_precompute["encoder"]
    assert isinstance(encoder, LogMelEncoder)
    assert (encoder.output_dim, encoder.frame_rate) == (40, 10.0)

    captured = capsys.readouterr()
    out = captured.out
    # Progress and stage logs on stderr; stdout keeps only the results.
    assert "synthetic / r1" not in out
    assert "precompute-features: dataset egocom" in captured.err
    assert "precompute-features: device mps" in captured.err
    assert f"precompute-features: output {output}" in captured.err
    assert "precompute-features: 1 recordings" in captured.err
    assert "1/1" in captured.err and "recording" in captured.err
    assert "precompute-features: done in" in captured.err
    assert f"Feature cache written to {output}" in out
    assert f"manifest: {output / 'manifest.json'}" in out
    assert "recordings: 1" in out
    assert "feature rate: 10 Hz" in out
    assert "feature dim: 512" in out


def test_precompute_features_defaults(
    fake_load, fake_precompute, tmp_path, monkeypatch
):
    built = []

    def build_encoder(cfg):
        # Mimi itself would download its weights.
        built.append(cfg)
        return LogMelEncoder(frame_rate=cfg.data.grid_rate_hz)

    monkeypatch.setattr("turn_wm.models.build.build_encoder", build_encoder)
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    argv = [
        "precompute-features",
        "--media-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "cache"),
    ]

    assert cli.main(argv) == 0

    assert fake_precompute["device"] == "cpu"
    assert fake_precompute["chunk_seconds"] == 20.0
    [cfg] = built
    assert cfg.model.encoder._target_.endswith("FrozenMimiEncoder")
    assert cfg.model.encoder.get("revision") is None


def test_precompute_features_accepts_one_root_per_corpus(
    fake_load, fake_precompute, tmp_path
):
    _, state = fake_load
    state["data"] = LoadedData(
        corpora=(
            make_corpus("egocom", media_manifest=make_manifest({"dataset": "egocom"})),
            make_corpus("ego4d", media_manifest=make_manifest({"dataset": "ego4d"})),
        )
    )
    egocom, ego4d = tmp_path / "egocom", tmp_path / "ego4d"
    egocom.mkdir()
    ego4d.mkdir()

    argv = [
        "precompute-features",
        "--encoder",
        "logmel",
        "--dataset",
        "full",
        "--media-root",
        f"egocom={egocom}",
        "--media-root",
        f"ego4d={ego4d}",
        "--output",
        str(tmp_path / "cache"),
    ]

    assert cli.main(argv) == 0
    assert fake_precompute["media_roots"] == {"egocom": egocom, "ego4d": ego4d}


def test_precompute_features_requires_a_media_root(fake_precompute, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["precompute-features", "--output", str(tmp_path / "cache")])

    assert error.value.code == 2
    assert "--media-root" in capsys.readouterr().err
    assert fake_precompute == {}


def test_precompute_features_requires_an_output(fake_precompute, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["precompute-features", "--media-root", str(tmp_path)])

    assert error.value.code == 2
    assert "--output" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "nan", "abc"])
def test_precompute_features_rejects_invalid_chunk_seconds(
    fake_precompute, tmp_path, capsys, value
):
    argv = [
        "precompute-features",
        "--encoder",
        "logmel",
        "--media-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "cache"),
        "--chunk-seconds",
        value,
    ]

    with pytest.raises(SystemExit) as error:
        cli.main(argv)

    assert error.value.code == 2
    assert "argument --chunk-seconds" in capsys.readouterr().err
    assert fake_precompute == {}


def test_precompute_features_needs_named_roots_for_several_corpora(
    fake_load, fake_precompute, tmp_path
):
    _, state = fake_load
    state["data"] = LoadedData(
        corpora=(
            make_corpus("egocom", media_manifest=make_manifest({"dataset": "egocom"})),
            make_corpus("ego4d", media_manifest=make_manifest({"dataset": "ego4d"})),
        )
    )

    argv = [
        "precompute-features",
        "--encoder",
        "logmel",
        "--dataset",
        "full",
        "--media-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "cache"),
    ]

    with pytest.raises(SystemExit, match="pass --media-root DATASET=PATH"):
        cli.main(argv)

    assert fake_precompute == {}


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No media root configured for dataset 'ego4d'"),
        FileNotFoundError("Video file does not exist: /media/r1.mp4"),
        ValueError("Non-contiguous action grid for 'r1'"),
        ValueError("Output /cache already exists and is not an empty directory"),
        OSError("kyutai/mimi is not a valid git identifier (branch name, tag)"),
    ],
)
def test_precompute_features_errors_are_clear_cli_errors(
    fake_load, monkeypatch, tmp_path, error
):
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())

    def fail(loaded, **kwargs):
        raise error

    monkeypatch.setattr(data_cli, "precompute_features", fail)

    argv = [
        "precompute-features",
        "--encoder",
        "logmel",
        "--media-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "cache"),
    ]

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == f"turn-wm: error: {error}"


def test_precompute_features_real_output_check_is_a_cli_error(
    fake_load, monkeypatch, tmp_path
):
    # Not mocked: the real precompute refuses a non-empty output before
    # loading Mimi.
    _, state = fake_load
    state["data"] = make_data(media_manifest=make_manifest(), train=make_anchors())
    output = tmp_path / "cache"
    output.mkdir()
    (output / "manifest.json").write_text("{}")

    argv = [
        "precompute-features",
        "--encoder",
        "logmel",
        "--media-root",
        str(tmp_path),
        "--output",
        str(output),
    ]

    with pytest.raises(SystemExit, match="not an empty directory"):
        cli.main(argv)
