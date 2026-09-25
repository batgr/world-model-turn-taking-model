import pytest
import torch

from turn_wm.data.mimi_cache import (
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
