import pytest
from datasets import Dataset, DatasetDict

from turn_wm import cli
from turn_wm.data.source import LoadedData


def make_grid(length: int = 40) -> Dataset:
    return Dataset.from_dict(
        {
            "recording_id": ["r1"] * length,
            "decision_index": list(range(length)),
            "focal_state_before": ["SILENT"] * 20 + ["SPEAKING"] * (length - 20),
            "action": ["NO_EVENT"] * 20 + ["ONSET"] + ["NO_EVENT"] * (length - 21),
            "action_valid": [True] * length,
        }
    )


def make_anchors(*, count: int = 3, is_trainable: bool = True) -> Dataset:
    anchor_idx = [19 + i for i in range(count)]

    return Dataset.from_dict(
        {
            "sample_id": [f"r1#{i}" for i in anchor_idx],
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


def make_data(**splits: Dataset) -> LoadedData:
    return LoadedData(
        model_ready=DatasetDict(splits or {"train": make_anchors()}),
        action_grid=make_grid(),
        metadata={
            "splits": {"recordings": {"train": 1}},
            "counts": {"recordings": 1},
            "grid": {"frequency_hz": 10.0},
        },
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
