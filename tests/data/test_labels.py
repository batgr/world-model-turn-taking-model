"""The optional label consumer: selection, projection and the disabled fast path."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from turn_wm.data import labels as labels_module
from turn_wm.data.labels import (
    LabelsConfig,
    LabelSelectionError,
    LabelUnavailableError,
    check_grid_alignment,
    load_labels,
    local_store,
)


def entry(
    name,
    *,
    modalities=("audio",),
    table="grid",
    columns=None,
    shape="[]",
    available=True,
    row_filter=None,
    extractor="speech",
):
    return {
        "name": name,
        "family": name.split(".")[0],
        "modalities": list(modalities),
        "availability": "available" if available else "unsupported",
        "unsupported_reason": None if available else "needs human annotation",
        "extractor": extractor if available else None,
        "table": table if available else None,
        "columns": list(columns or [name.split(".")[1]]) if available else [],
        "shape": shape,
        "row_filter": list(row_filter) if row_filter else None,
    }


REGISTRY = {
    "registry_version": 1,
    "families": {
        "instantaneous": "",
        "timing": "",
        "turns": "",
        "social_states": "",
        "social_native": "",
    },
    "labels": [
        entry("instantaneous.ego_speaking"),
        entry("instantaneous.speaker_activity", shape="[participant]"),
        entry(
            "timing.time_to_next_ego_onset",
            columns=("time_to_next_ego_onset", "time_to_next_ego_onset_valid"),
        ),
        entry(
            "turns.floor_transfers",
            table="events",
            columns=("fto_s",),
            row_filter=("event_type", "floor_change"),
        ),
        entry(
            "social_native.talking_to_wearer_subframes",
            modalities=("audio", "video"),
            extractor="social",
        ),
        entry("social_states.dominance", available=False),
    ],
}


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "labels"
    (root / "speech").mkdir(parents=True)
    (root / "registry.json").write_text(json.dumps(REGISTRY))
    grid = pa.table(
        {
            "recording_id": ["a", "a", "b"],
            "decision_index": [0, 1, 0],
            "decision_time_s": [0.0, 0.1, 0.0],
            "participant_ids": [["w", "x"]] * 3,
            "ego_speaking": [True, False, None],
            "speaker_activity": [[True, False]] * 3,
            "time_to_next_ego_onset": [0.5, None, 1.0],
            "time_to_next_ego_onset_valid": [True, False, True],
        }
    )
    events = pa.table(
        {
            "recording_id": ["a", "a"],
            "event_type": ["onset", "floor_change"],
            "time_s": [0.05, 0.3],
            "fto_s": [None, 0.2],
        }
    )
    pq.write_table(grid, root / "speech" / "grid.parquet")
    pq.write_table(events, root / "speech" / "events.parquet")
    (root / "speech" / "manifest.json").write_text(
        json.dumps(
            {
                "materialized_labels": [
                    "instantaneous.ego_speaking",
                    "instantaneous.speaker_activity",
                    "timing.time_to_next_ego_onset",
                    "turns.floor_transfers",
                ],
                "unavailable_labels": {},
                "tables": {
                    "grid": {"file": "grid.parquet"},
                    "events": {"file": "events.parquet"},
                },
            }
        )
    )
    return root


def test_disabled_labels_fetch_nothing():
    def fetch(_):
        raise AssertionError("fetched a label file")

    assert not load_labels(fetch, LabelsConfig())
    assert not load_labels(None, LabelsConfig.from_mapping({"enabled": False}))


def test_exact_labels_read_only_their_columns(store, monkeypatch):
    calls = []
    original = pq.read_table

    def spy(path, *args, **kwargs):
        calls.append(tuple(kwargs["columns"]))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(labels_module.pq, "read_table", spy)
    loaded = load_labels(
        local_store(store),
        LabelsConfig(True, ("timing.time_to_next_ego_onset", "turns.floor_transfers")),
    )
    assert calls == [
        ("recording_id", "event_type", "time_s", "fto_s"),
        (
            "recording_id",
            "decision_index",
            "decision_time_s",
            "time_to_next_ego_onset",
            "time_to_next_ego_onset_valid",
        ),
    ]
    assert loaded.tables["events"].column("event_type").to_pylist() == ["floor_change"]


def test_family_and_all_report_what_is_skipped(store):
    loaded = load_labels(local_store(store), LabelsConfig(True, ("all",)))
    assert "social_states.dominance" in loaded.skipped
    assert (
        "social_native.talking_to_wearer_subframes" in loaded.skipped
    )  # not published
    assert "instantaneous.speaker_activity" in loaded.labels
    assert "participant_ids" in loaded.tables["grid"].column_names
    family = load_labels(local_store(store), LabelsConfig(True, ("timing.*",)))
    assert family.labels == ("timing.time_to_next_ego_onset",)


def test_modalities_filter(store):
    loaded = load_labels(local_store(store), LabelsConfig(True, ("all",), ("video",)))
    assert loaded.labels == ()
    both = load_labels(
        local_store(store), LabelsConfig(True, ("all",), ("audio", "video"))
    )
    assert "social_native.talking_to_wearer_subframes" in both.skipped


def test_errors_are_explicit(store):
    fetch = local_store(store)
    with pytest.raises(LabelSelectionError, match="unknown label"):
        load_labels(fetch, LabelsConfig(True, ("timing.nope",)))
    with pytest.raises(LabelUnavailableError, match="unsupported"):
        load_labels(fetch, LabelsConfig(True, ("social_states.dominance",)))
    with pytest.raises(LabelUnavailableError, match="not published"):
        load_labels(
            fetch, LabelsConfig(True, ("social_native.talking_to_wearer_subframes",))
        )
    with pytest.raises(LabelUnavailableError, match="publishes no label"):
        load_labels(None, LabelsConfig(True, ("all",)))


def test_recording_filter(store):
    loaded = load_labels(
        local_store(store),
        LabelsConfig(True, ("instantaneous.ego_speaking",)),
        recording_ids=["b"],
    )
    assert loaded.tables["grid"].column("recording_id").to_pylist() == ["b"]


def test_grid_alignment_with_the_action_grid(store):
    loaded = load_labels(
        local_store(store), LabelsConfig(True, ("instantaneous.ego_speaking",))
    )
    grid = loaded.tables["grid"]
    check_grid_alignment(grid, grid.select(["recording_id", "decision_index"]))
    shifted = pa.table({"recording_id": ["a", "b", "a"], "decision_index": [0, 0, 1]})
    with pytest.raises(ValueError, match="not aligned"):
        check_grid_alignment(grid, shifted)


def test_alignment_ignores_the_arrow_string_flavour(store):
    loaded = load_labels(
        local_store(store), LabelsConfig(True, ("instantaneous.ego_speaking",))
    )
    grid = loaded.tables["grid"]
    action = pa.table(
        {
            "recording_id": pa.array(["a", "a", "b"], pa.large_string()),
            "decision_index": [0, 1, 0],
        }
    )
    check_grid_alignment(grid, action)
