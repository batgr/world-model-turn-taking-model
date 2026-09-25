import json

import pytest
import torch

from turn_wm.data.mimi_cache import (
    MimiAudioGap,
    MimiCacheExclusion,
    MimiFeatureRecord,
    MimiFeatureStore,
    write_features,
    write_manifest,
)


def test_mimi_features_round_trip(tmp_path):
    features = torch.randn(100, 512)

    path = write_features(
        tmp_path,
        dataset="synthetic",
        recording_id="r1",
        features=features,
    )

    write_manifest(
        tmp_path,
        recordings=[
            MimiFeatureRecord(
                dataset="synthetic",
                recording_id="r1",
                path=str(path.relative_to(tmp_path)),
                steps=100,
                start_time_s=0.0,
                start_index=0,
            )
        ],
        model_name="kyutai/mimi",
        model_revision="test",
        source_dataset_revision="dataset-test",
        feature_rate_hz=10.0,
        feature_dim=512,
    )

    store = MimiFeatureStore(tmp_path)

    result = store.get(
        dataset="synthetic",
        recording_id="r1",
        start=20,
        end=30,
    )

    assert result.shape == (10, 512)

    assert torch.allclose(
        result.float(),
        features[20:30].half().float(),
    )


def test_mimi_store_rejects_out_of_bounds_slice(tmp_path):
    features = torch.randn(10, 512)

    path = write_features(
        tmp_path,
        dataset="synthetic",
        recording_id="r1",
        features=features,
    )

    write_manifest(
        tmp_path,
        recordings=[
            MimiFeatureRecord(
                dataset="synthetic",
                recording_id="r1",
                path=str(path.relative_to(tmp_path)),
                steps=10,
                start_time_s=0.0,
                start_index=0,
            )
        ],
        model_name="kyutai/mimi",
        model_revision=None,
        source_dataset_revision=None,
        feature_rate_hz=10.0,
        feature_dim=512,
    )

    store = MimiFeatureStore(tmp_path)

    with pytest.raises(IndexError):
        store.get(
            dataset="synthetic",
            recording_id="r1",
            start=0,
            end=11,
        )


def test_manifest_is_deterministic(tmp_path):
    records = [
        MimiFeatureRecord(
            dataset="b",
            recording_id="r2",
            path="b/r2.safetensors",
            steps=10,
            start_time_s=0.0,
            start_index=0,
        ),
        MimiFeatureRecord(
            dataset="a",
            recording_id="r1",
            path="a/r1.safetensors",
            steps=20,
            start_time_s=0.0,
            start_index=0,
        ),
    ]

    write_manifest(
        tmp_path,
        recordings=records,
        model_name="kyutai/mimi",
        model_revision="abc",
        source_dataset_revision="def",
        feature_rate_hz=10.0,
        feature_dim=512,
    )

    first = (tmp_path / "manifest.json").read_bytes()

    write_manifest(
        tmp_path,
        recordings=list(reversed(records)),
        model_name="kyutai/mimi",
        model_revision="abc",
        source_dataset_revision="def",
        feature_rate_hz=10.0,
        feature_dim=512,
    )

    second = (tmp_path / "manifest.json").read_bytes()

    assert first == second


def test_manifest_round_trips_sorted_gaps_and_exclusions_without_a_timestamp(
    tmp_path,
):
    gaps = (
        MimiAudioGap(start_time_s=2.0, end_time_s=2.25),
        MimiAudioGap(start_time_s=1.0, end_time_s=1.101),
    )
    exclusions = (
        MimiCacheExclusion(
            dataset="ego4d",
            recording_id="z",
            reason="audio_annotation_clock_drift",
            max_drift_s=0.525,
        ),
        MimiCacheExclusion(
            dataset="ego4d",
            recording_id="a",
            reason="audio_annotation_clock_drift",
            max_drift_s=0.3,
        ),
    )

    def write(gap_order, exclusion_order):
        write_manifest(
            tmp_path,
            recordings=[
                MimiFeatureRecord(
                    dataset="ego4d",
                    recording_id="kept",
                    path="ego4d/kept.safetensors",
                    steps=10,
                    start_time_s=0.0,
                    start_index=0,
                    audio_gaps=gap_order,
                )
            ],
            model_name="kyutai/mimi",
            model_revision="abc",
            source_dataset_revision="def",
            feature_rate_hz=10.0,
            feature_dim=512,
            excluded_recordings=exclusion_order,
        )

    write(gaps, exclusions)
    first = (tmp_path / "manifest.json").read_bytes()
    payload = json.loads(first)

    write(tuple(reversed(gaps)), tuple(reversed(exclusions)))

    assert (tmp_path / "manifest.json").read_bytes() == first
    assert "created_at" not in payload
    assert "generated_at" not in payload
    assert payload["recordings"][0]["audio_gaps"] == [
        {
            "duration_s": pytest.approx(0.101),
            "end_time_s": 1.101,
            "start_time_s": 1.0,
        },
        {
            "duration_s": 0.25,
            "end_time_s": 2.25,
            "start_time_s": 2.0,
        },
    ]
    assert [row["recording_id"] for row in payload["excluded_recordings"]] == [
        "a",
        "z",
    ]

    store = MimiFeatureStore(tmp_path)
    assert store.recording_keys == frozenset({("ego4d", "kept")})
    assert store.record(dataset="ego4d", recording_id="kept").audio_gaps == tuple(
        reversed(gaps)
    )
    assert store.exclusions == tuple(reversed(exclusions))


def test_store_reads_back_the_record_fields(tmp_path):
    path = write_features(
        tmp_path,
        dataset="synthetic",
        recording_id="r1",
        features=torch.zeros(5, 512),
    )
    record = MimiFeatureRecord(
        dataset="synthetic",
        recording_id="r1",
        path=str(path.relative_to(tmp_path)),
        steps=5,
        start_time_s=1.5,
        start_index=15,
    )

    write_manifest(
        tmp_path,
        recordings=[record],
        model_name="kyutai/mimi",
        model_revision="test",
        source_dataset_revision=None,
        feature_rate_hz=10.0,
        feature_dim=512,
    )

    # start_index used to be read back as a one-element tuple.
    assert MimiFeatureStore(tmp_path)._records[("synthetic", "r1")] == record


def test_get_by_index_maps_decision_indices_to_rows(make_mimi_cache):
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (100, 50)}))

    features = store.get_by_index(
        dataset="egocom", recording_id="r1", start_index=105, end_index=108
    )

    # Local rows [5:8] of a recording starting at decision_index 100.
    assert features.shape == (3, 512)
    assert features[:, 0].tolist() == [105.0, 106.0, 107.0]
    assert features.dtype == torch.float16


def test_get_by_index_covers_the_whole_recording(make_mimi_cache):
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (100, 50)}))

    features = store.get_by_index(
        dataset="egocom", recording_id="r1", start_index=100, end_index=150
    )

    assert features.shape == (50, 512)
    assert features[[0, -1], 0].tolist() == [100.0, 149.0]


@pytest.mark.parametrize(("start", "end"), [(99, 102), (148, 151), (120, 110)])
def test_get_by_index_rejects_rows_outside_the_recording(make_mimi_cache, start, end):
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (100, 50)}))

    with pytest.raises(IndexError, match=r"covers \[100, 150\)"):
        store.get_by_index(
            dataset="egocom", recording_id="r1", start_index=start, end_index=end
        )


def test_unknown_recording_is_a_clear_error(make_mimi_cache):
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (0, 5)}))

    with pytest.raises(KeyError, match="'ego4d', 'r1'"):
        store.get_by_index(
            dataset="ego4d", recording_id="r1", start_index=0, end_index=1
        )


def test_only_the_requested_rows_are_read(make_mimi_cache, monkeypatch):
    from turn_wm.data import mimi_cache as module

    requested = []
    real_open = module.safe_open

    class Recording:
        def __init__(self, *args, **kwargs):
            self.file = real_open(*args, **kwargs)

        def __enter__(self):
            self.opened = self.file.__enter__()
            return self

        def __exit__(self, *exc):
            return self.file.__exit__(*exc)

        def get_slice(self, name):
            sliced = self.opened.get_slice(name)

            class Slice:
                def __getitem__(self, key):
                    requested.append(key)
                    return sliced[key]

            return Slice()

    monkeypatch.setattr(module, "safe_open", Recording)
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (100, 5_000)}))

    store.get_by_index(
        dataset="egocom", recording_id="r1", start_index=200, end_index=225
    )

    assert requested == [slice(100, 125)]


def test_store_exposes_the_manifest_identity(make_mimi_cache):
    store = MimiFeatureStore(make_mimi_cache({("egocom", "r1"): (0, 5)}, dim=64))

    assert store.feature_dim == 64
    assert store.feature_rate_hz == 10.0
    assert store.dtype == "float16"
    assert store.schema_version == 2
    assert store.model_name == "kyutai/mimi"
    assert store.model_revision == "requested-sha"
    assert store.model_resolved_revision == "resolved-sha"
    assert store.source_dataset_revision == "rev-123"


def test_store_keeps_read_compatibility_with_schema_v1(make_mimi_cache):
    root = make_mimi_cache({("egocom", "r1"): (0, 5)})
    path = root / "manifest.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    payload.pop("excluded_recordings")

    for record in payload["recordings"]:
        record.pop("audio_gaps")

    path.write_text(json.dumps(payload))
    store = MimiFeatureStore(root)

    assert store.schema_version == 1
    assert store.recording_keys == frozenset({("egocom", "r1")})
    assert store.record(dataset="egocom", recording_id="r1").audio_gaps == ()
    assert store.exclusions == ()
