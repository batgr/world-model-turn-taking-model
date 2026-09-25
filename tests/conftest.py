from pathlib import Path

import pytest
import torch

from turn_wm.data.mimi_cache import MimiFeatureRecord, write_features, write_manifest


@pytest.fixture
def make_mimi_cache(tmp_path):
    """Write a synthetic Mimi cache; feature[k, 0] is the row's decision_index.

    `recordings` maps (dataset, recording_id) to (start_index, steps).
    """

    def make(
        recordings: dict[tuple[str, str], tuple[int, int]],
        *,
        dim: int = 512,
        rate_hz: float = 10.0,
        source_dataset_revision: str | None = "rev-123",
        root: Path | None = None,
    ) -> Path:
        root = root or tmp_path / "mimi-cache"
        records = []

        for (dataset, recording_id), (start_index, steps) in recordings.items():
            features = torch.zeros(steps, dim)
            features[:, 0] = torch.arange(start_index, start_index + steps)
            path = write_features(
                root, dataset=dataset, recording_id=recording_id, features=features
            )
            records.append(
                MimiFeatureRecord(
                    dataset=dataset,
                    recording_id=recording_id,
                    path=str(path.relative_to(root)),
                    steps=steps,
                    start_time_s=start_index / rate_hz,
                    start_index=start_index,
                )
            )

        write_manifest(
            root,
            recordings=records,
            model_name="kyutai/mimi",
            model_revision="requested-sha",
            model_resolved_revision="resolved-sha",
            source_dataset_revision=source_dataset_revision,
            feature_rate_hz=rate_hz,
            feature_dim=dim,
        )

        return root

    return make
