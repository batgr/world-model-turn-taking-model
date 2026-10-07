import json
from types import SimpleNamespace

import httpx
import pytest
from datasets import Dataset, DatasetDict

from turn_wm.data import source as source_module
from turn_wm.data.source import CorpusConfig, HuggingFaceSource, load_data


def make_model_ready(split: str) -> Dataset:
    return Dataset.from_dict(
        {
            "sample_id": [f"{split}-0"],
            "dataset": ["synthetic"],
            "recording_id": ["r1"],
            "conversation_id": ["c1"],
            "view_id": ["v1"],
            "wearer_id": ["p1"],
            "split": [split],
            "split_source": ["upstream"],
            "segment_id": [0],
            "anchor_idx": [10],
            "anchor_time": [1.0],
            "anchor_row": [10],
            "max_context_steps": [10],
            "future_steps": [10],
            "context_valid_ratio": [1.0],
            "future_valid_ratio": [1.0],
            "future_event_count": [1],
            "sample_class": ["event"],
            "is_trainable": [True],
            "window_schema_version": ["1"],
            "action_schema_version": ["1"],
        }
    )


def make_action_grid() -> Dataset:
    return Dataset.from_dict(
        {
            "dataset": ["synthetic"],
            "recording_id": ["r1"],
            "sync_group_id": ["c1"],
            "view_id": ["v1"],
            "wearer_id": ["p1"],
            "decision_index": [0],
            "decision_time_s": [0.0],
            "focal_state_before": ["SILENT"],
            "action": ["NO_EVENT"],
            "action_valid": [True],
            "mask_reason": [None],
        }
    )


def make_media_manifest(**overrides) -> Dataset:
    row = {
        "dataset": "synthetic",
        "recording_id": "r1",
        "video_path": "videos/r1.mp4",
        "audio_path": None,
        "media_offset_s": 0.0,
        "video_has_audio": True,
    }
    row.update(overrides)

    return Dataset.from_list([row])


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Replace every Hub call made by source.py; record what was requested."""

    calls = {"load_dataset": [], "download": [], "resolve": []}
    state = {"sha": "abc123", "offline": False}

    artifacts = {
        "model_ready": lambda: DatasetDict({"train": make_model_ready("train")}),
        "action_grid": make_action_grid,
        "media_manifest": make_media_manifest,
    }
    calls["returned"] = {}

    def load_dataset(repo_id, config, *, revision, split=None):
        # Configs are named "<corpus>/<artifact>" or just "<artifact>".
        calls["load_dataset"].append((config, revision))
        artifact = artifacts[config.rsplit("/", 1)[-1]]()
        calls["returned"][config] = artifact
        return artifact

    def hf_hub_download(*, repo_id, filename, repo_type, revision):
        calls["download"].append((filename, revision))
        path = tmp_path / "metadata.json"
        path.write_text(json.dumps({"dataset": "synthetic"}), encoding="utf-8")
        return str(path)

    class FakeApi:
        def dataset_info(self, repo_id, *, revision):
            calls["resolve"].append(revision)

            if state["offline"]:
                raise httpx.ConnectError("offline")

            return SimpleNamespace(sha=state["sha"])

    monkeypatch.setattr(source_module, "load_dataset", load_dataset)
    monkeypatch.setattr(source_module, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(source_module, "HfApi", FakeApi)

    return calls, state


def hf_source(
    *, media_manifest_config: str | None = None, revision: str | None = None
) -> HuggingFaceSource:
    return HuggingFaceSource(
        repo_id="owner/synthetic",
        corpora=(
            CorpusConfig(
                name="synthetic",
                model_ready_config="model_ready",
                action_grid_config="action_grid",
                media_manifest_config=media_manifest_config,
            ),
        ),
        revision=revision,
    )


def corpus_config(name: str) -> CorpusConfig:
    return CorpusConfig(
        name=name,
        model_ready_config=f"{name}/model_ready",
        action_grid_config=f"{name}/action_grid",
        media_manifest_config=f"{name}/media_manifest",
        metadata_file=f"{name}/metadata.json",
    )


def test_huggingface_source_loads_media_manifest(fake_hub):
    calls, _ = fake_hub

    loaded = load_data(hf_source(media_manifest_config="media_manifest"))

    assert loaded.corpora[0].media_manifest is not None
    assert loaded.corpora[0].media_manifest[0]["recording_id"] == "r1"
    assert ("media_manifest", "abc123") in calls["load_dataset"]


def test_huggingface_source_without_media_manifest(fake_hub):
    calls, _ = fake_hub

    loaded = load_data(hf_source())

    assert loaded.corpora[0].media_manifest is None
    assert [config for config, _ in calls["load_dataset"]] == [
        "model_ready",
        "action_grid",
    ]


def test_all_artifacts_come_from_one_resolved_revision(fake_hub):
    calls, _ = fake_hub

    loaded = load_data(
        hf_source(media_manifest_config="media_manifest", revision="main")
    )

    assert calls["resolve"] == ["main"]
    assert loaded.revision == "abc123"

    revisions = {revision for _, revision in calls["load_dataset"]}
    revisions |= {revision for _, revision in calls["download"]}

    assert revisions == {"abc123"}


def test_offline_load_falls_back_to_configured_revision(fake_hub):
    calls, state = fake_hub
    state["offline"] = True

    loaded = load_data(hf_source(media_manifest_config="media_manifest"))

    assert loaded.revision is None
    assert {revision for _, revision in calls["load_dataset"]} == {None}


def test_malformed_published_manifest_raises(fake_hub, monkeypatch):
    original = source_module.load_dataset

    def load_dataset(repo_id, config, **kwargs):
        if config == "media_manifest":
            return make_media_manifest().remove_columns("video_has_audio")
        return original(repo_id, config, **kwargs)

    monkeypatch.setattr(source_module, "load_dataset", load_dataset)

    with pytest.raises(ValueError, match="media_manifest is missing"):
        load_data(hf_source(media_manifest_config="media_manifest"))


def test_multi_corpus_source_keeps_each_corpus_bundle(fake_hub):
    calls, _ = fake_hub

    loaded = load_data(
        HuggingFaceSource(
            repo_id="owner/synthetic",
            corpora=(corpus_config("a"), corpus_config("b")),
            revision="main",
        )
    )

    assert loaded.names == ("a", "b")
    assert loaded.revision == "abc123"

    for name in ("a", "b"):
        corpus = loaded.corpus(name)
        returned = calls["returned"]

        assert corpus.action_grid is returned[f"{name}/action_grid"]
        assert corpus.model_ready is returned[f"{name}/model_ready"]
        assert corpus.media_manifest is returned[f"{name}/media_manifest"]

    # One resolution; every artifact of every corpus from that commit.
    assert calls["resolve"] == ["main"]
    assert {revision for _, revision in calls["load_dataset"]} == {"abc123"}
    assert calls["download"] == [
        ("a/metadata.json", "abc123"),
        ("b/metadata.json", "abc123"),
    ]


def test_source_requires_unique_non_empty_corpora():
    with pytest.raises(ValueError, match="at least one corpus"):
        HuggingFaceSource(repo_id="owner/x", corpora=())

    with pytest.raises(ValueError, match="Duplicate corpus names"):
        HuggingFaceSource(
            repo_id="owner/x", corpora=(corpus_config("a"), corpus_config("a"))
        )


def test_unknown_corpus_lookup(fake_hub):
    loaded = load_data(hf_source())

    with pytest.raises(KeyError, match="No corpus 'other'"):
        loaded.corpus("other")


def test_the_12_5_hz_releases_are_registered_next_to_the_10_hz_ones():
    from turn_wm.data.source import DATASETS

    for name in ("egocom", "ego4d", "full"):
        base, faster = DATASETS[name], DATASETS[f"{name}_12.5hz"]
        assert faster.repo_id == f"{base.repo_id}-12.5hz"
        assert faster.corpora == base.corpora


def test_a_split_column_disagreeing_with_its_partition_is_refused():
    with pytest.raises(ValueError, match="split"):
        source_module._validate_model_ready(
            DatasetDict({"train": make_model_ready("validation")}), corpus="c"
        )


def test_a_release_without_a_train_split_is_refused():
    with pytest.raises(ValueError, match="train split"):
        source_module._validate_model_ready(
            DatasetDict({"validation": make_model_ready("validation")}), corpus="c"
        )
