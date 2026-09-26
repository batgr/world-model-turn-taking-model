"""
Latent analysis: anchor representations of a trained run, and the runner
rebuilding a run's exact data and model from its run directory.
"""

import json

import lightning as L
import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn

from turn_wm.config import load_config
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.mimi_cache import open_mimi_cache
from turn_wm.data.source import LoadedCorpus, LoadedData
from turn_wm.evaluation.latent_analysis import run as run_module
from turn_wm.evaluation.latent_analysis.extract import (
    FEATURES,
    LATENT,
    RepresentationSnapshot,
    anchor_representations,
    extract_snapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.run import extract_run
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.training.lewm import LeWMModule, Trajectories
from turn_wm.training.train import (
    _config_hash,
    _write_config,
    _write_metadata,
    build_run_dataset,
    prepare_observations,
)

# ---------------------------------------------------------------------------
# Extraction from batches
# ---------------------------------------------------------------------------


def _model() -> JEPA:
    """Projects (a, b, c) to (a, b)."""

    projector = nn.Linear(3, 2, bias=False)

    with torch.no_grad():
        projector.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    return JEPA(
        encoder=None,
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
        projector=projector,
    )


def _batch(offset: int = 0) -> dict:
    """Two samples, 3 context steps and 1 future step of 3-d features."""

    steps = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) + offset

    return {
        "context_features": steps[:, :3].half(),
        "future_features": steps[:, 3:].half(),
        "context_lengths": torch.tensor([3, 3]),
        "context_action": torch.tensor([[0, 0, 1], [0, 3, 2]]),
        "future_action": torch.tensor([[0], [0]]),
        "sample_id": [f"s{offset}-0", f"s{offset}-1"],
        "dataset": ["egocom", "ego4d"],
        "recording_id": ["r0", "r1"],
        "anchor_idx": torch.tensor([12, 34]),
        "anchor_time": torch.tensor([1.2, 3.4]),
        "sample_class": ["event", "background"],
    }


def test_representations_are_taken_at_the_anchor_step():
    batch = _batch()

    snapshot = extract_snapshot(_model(), [batch])

    # The anchor is the last context step.
    anchor = batch["context_features"][:, 2].float()
    assert torch.equal(snapshot.representations[FEATURES], anchor)
    assert torch.equal(snapshot.representations[LATENT], anchor[:, :2])
    assert snapshot.metadata["anchor_idx"] == [12, 34]
    assert snapshot.metadata["sample_id"] == ["s0-0", "s0-1"]
    # The action taken at the anchor step itself.
    assert snapshot.metadata["action_id"] == [1, 2]
    assert snapshot.metadata["action"] == ["ONSET", "OFFSET"]


def test_raw_observations_go_through_the_encoder():
    class Encoder(nn.Module):
        def forward(self, waveform, sample_rate, target_length):
            # Step k of sample b is (b, k, 0).
            return torch.stack(
                [
                    torch.stack(
                        [torch.tensor([b, k, 0.0]) for k in range(target_length)]
                    )
                    for b in range(len(waveform))
                ]
            )

    model = _model()
    model.encoder = Encoder()
    batch = Trajectories(
        actions=torch.zeros(2, 5, dtype=torch.long),
        context_steps=3,
        future_steps=2,
        waveforms=[torch.zeros(1, 50), torch.zeros(1, 50)],
        sample_rates=[10, 10],
    )

    representations = anchor_representations(model, batch)

    expected = torch.tensor([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    assert torch.equal(representations[FEATURES], expected)
    assert torch.equal(representations[LATENT], expected[:, :2])


def test_max_samples_keeps_the_first_samples_in_order():
    snapshot = extract_snapshot(_model(), [_batch(0), _batch(100)], max_samples=3)

    assert snapshot.samples == 3
    assert snapshot.metadata["sample_id"] == ["s0-0", "s0-1", "s100-0"]


def test_without_a_limit_every_sample_is_kept():
    snapshot = extract_snapshot(_model(), [_batch(0), _batch(100)])

    assert snapshot.samples == 4


@pytest.mark.parametrize("max_samples", [0, -1])
def test_invalid_sample_limit_is_refused(max_samples):
    with pytest.raises(ValueError, match="max_samples must be positive"):
        extract_snapshot(_model(), [_batch()], max_samples=max_samples)


def test_empty_input_is_refused():
    with pytest.raises(ValueError, match="No samples were extracted"):
        extract_snapshot(_model(), [])


def test_snapshot_rows_must_align():
    with pytest.raises(ValueError, match="same number of rows"):
        RepresentationSnapshot(
            representations={LATENT: torch.zeros(2, 2)},
            metadata={"sample_id": ["only-one"]},
        )


def test_written_artifact(tmp_path):
    snapshot = extract_snapshot(_model(), [_batch()])
    output = tmp_path / "artifact"

    write_snapshot(snapshot, output, provenance={"sampling": {"seed": 3}})

    tensors = load_file(output / "representations.safetensors")
    metadata = pq.read_table(output / "metadata.parquet").to_pydict()
    manifest = json.loads((output / "manifest.json").read_text())

    assert torch.equal(tensors[FEATURES], snapshot.representations[FEATURES])
    assert torch.equal(tensors[LATENT], snapshot.representations[LATENT])
    assert metadata["action"] == ["ONSET", "OFFSET"]
    assert manifest["schema_version"] == 1
    assert manifest["samples"] == 2
    assert manifest["representations"] == {
        FEATURES: {"shape": [3], "dtype": "float32"},
        LATENT: {"shape": [2], "dtype": "float32"},
    }
    assert manifest["provenance"] == {"sampling": {"seed": 3}}


def test_further_representations_extend_the_artifact(tmp_path):
    # E.g. predicted latents per horizon, or whole trajectories.
    snapshot = RepresentationSnapshot(
        representations={
            LATENT: torch.zeros(2, 4),
            "trajectory_latent": torch.zeros(2, 25, 4),
        },
        metadata={"sample_id": ["a", "b"]},
    )

    write_snapshot(snapshot, tmp_path / "artifact")

    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["representations"]["trajectory_latent"]["shape"] == [25, 4]


def test_a_non_empty_output_directory_is_refused(tmp_path):
    (tmp_path / "existing.txt").write_text("keep")

    with pytest.raises(ValueError, match="Output directory is not empty"):
        write_snapshot(extract_snapshot(_model(), [_batch()]), tmp_path)


# ---------------------------------------------------------------------------
# The runner: a synthetic run directory, trained for one step
# ---------------------------------------------------------------------------

GRID_STEPS = 60
REVISION = "rev-123"


def _anchors(split: str, rows: range) -> Dataset:
    rows = list(rows)

    return Dataset.from_dict(
        {
            "sample_id": [f"egocom:{split}#{row}" for row in rows],
            "dataset": ["egocom"] * len(rows),
            "recording_id": ["r1"] * len(rows),
            "split": [split] * len(rows),
            "anchor_idx": rows,
            "anchor_row": rows,
            "anchor_time": [row / 10 for row in rows],
            "max_context_steps": [row + 1 for row in rows],
            "future_steps": [GRID_STEPS - 1 - row for row in rows],
            "sample_class": ["event"] * len(rows),
            "is_trainable": [True] * len(rows),
        }
    )


def _loaded(revision: str | None = REVISION) -> LoadedData:
    grid = Dataset.from_dict(
        {
            "dataset": ["egocom"] * GRID_STEPS,
            "recording_id": ["r1"] * GRID_STEPS,
            "decision_index": list(range(GRID_STEPS)),
            "decision_time_s": [i / 10 for i in range(GRID_STEPS)],
            "focal_state_before": ["SILENT"] * GRID_STEPS,
            "action": (["NO_EVENT", "ONSET", "OFFSET"] * GRID_STEPS)[:GRID_STEPS],
            "action_valid": [True] * GRID_STEPS,
        }
    )

    return LoadedData(
        corpora=(
            LoadedCorpus(
                name="egocom",
                model_ready=DatasetDict(
                    {
                        "train": _anchors("train", range(20, 30)),
                        "validation": _anchors("validation", range(20, 45)),
                        "test": _anchors("test", range(20, 30)),
                    }
                ),
                action_grid=grid,
                metadata={},
            ),
        ),
        revision=revision,
    )


BATCHNORM = (
    "+model.projector.norm_fn={_target_:hydra.utils.get_class,"
    "path:torch.nn.BatchNorm1d}"
)


def _config(cache_root, *overrides):
    return load_config(
        [
            "data.dataset=egocom",
            "data.observation_source=mimi_cache",
            f"data.mimi_cache.root={cache_root}",
            "trainer.accelerator=cpu",
            "trainer.precision=32-true",
            *overrides,
        ]
    )


@pytest.fixture
def cache_root(make_mimi_cache):
    # feature[k, 0] is the step's decision_index.
    return make_mimi_cache({("egocom", "r1"): (0, GRID_STEPS)})


@pytest.fixture
def loads(monkeypatch):
    """Serve the synthetic data; record the sources the runner asked for."""

    sources = []

    def load_data(source):
        sources.append(source)
        return _loaded(source.revision)

    monkeypatch.setattr(run_module, "load_data", load_data)

    return sources


def _make_run(tmp_path, cfg) -> tuple:
    """A run directory as training writes it, with a one-step checkpoint."""

    run_dir = tmp_path / "outputs" / "lewm" / "run-1"
    run_dir.mkdir(parents=True)
    loaded = _loaded()
    observations = prepare_observations(cfg, loaded)

    _write_config(cfg, run_dir)
    _write_metadata(
        run_dir=run_dir,
        run_id="run-1",
        config_hash=_config_hash(cfg),
        cfg=cfg,
        git={"commit": "abc", "dirty": False},
        dataset_revision=REVISION,
        mimi_store=observations.mimi_store,
    )

    train = build_run_dataset(cfg, loaded, observations, split="train", training=True)
    module = LeWMModule(cfg)
    trainer = L.Trainer(
        accelerator="cpu",
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(
        module,
        train_dataloaders=build_dataloader(
            train, loader=DataLoaderConfig(batch_size=4)
        ),
    )
    trainer.save_checkpoint(run_dir / "checkpoints" / "last.ckpt")

    return run_dir, module.model.eval()


def _read(output):
    return (
        load_file(output / "representations.safetensors"),
        pq.read_table(output / "metadata.parquet").to_pydict(),
        json.loads((output / "manifest.json").read_text()),
    )


@pytest.mark.parametrize(
    "overrides",
    [(), ("data.context_steps=12",), (BATCHNORM,)],
    ids=["v1", "longer-context", "batchnorm-projector"],
)
def test_run_extracts_the_trained_model_on_validation(
    tmp_path, cache_root, loads, overrides
):
    cfg = _config(cache_root, *overrides)
    run_dir, model = _make_run(tmp_path, cfg)

    output = extract_run(run_dir, max_samples=7, batch_size=3)

    tensors, metadata, manifest = _read(output)

    assert output == run_dir / "latent_analysis" / "last-validation"
    assert [source.revision for source in loads] == [REVISION]
    assert all(":validation#" in sample for sample in metadata["sample_id"])
    assert tensors[FEATURES].shape == (7, 512)
    # The cached features of each sample's anchor step.
    assert tensors[FEATURES][:, 0].tolist() == metadata["anchor_idx"]
    # The checkpoint's projector, in eval mode.
    with torch.no_grad():
        expected = model.project_features(tensors[FEATURES].unsqueeze(1))[:, 0]
    torch.testing.assert_close(tensors[LATENT], expected)

    provenance = manifest["provenance"]
    assert manifest["samples"] == provenance["sampling"]["samples"] == 7
    assert provenance["run"]["run_id"] == "run-1"
    assert provenance["run"]["config_hash"] == _config_hash(cfg)
    assert provenance["data"]["dataset_revision"] == REVISION
    assert provenance["data"]["split"] == "validation"
    assert provenance["data"]["observation_source"] == "mimi_cache"
    assert provenance["data"]["context_steps"] == cfg.data.context_steps
    assert provenance["data"]["anchor_step"] == cfg.data.context_steps - 1
    assert provenance["data"]["feature_caches"]["mimi"]["caches"]
    assert provenance["sampling"]["seed"] == cfg.seed
    assert provenance["checkpoint"]["filename"] == "last.ckpt"
    assert provenance["checkpoint"]["global_step"] == 1
    assert len(provenance["checkpoint"]["sha256"]) == 64


def test_samples_are_a_seeded_prefix_independent_of_batching(
    tmp_path, cache_root, loads
):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))

    few = extract_run(run_dir, output_dir=tmp_path / "few", max_samples=5)
    every = extract_run(run_dir, output_dir=tmp_path / "all", batch_size=4)
    again = extract_run(run_dir, output_dir=tmp_path / "again", batch_size=7)
    other = extract_run(run_dir, output_dir=tmp_path / "other", seed=1)

    few_tensors, few_meta, _ = _read(few)
    every_tensors, every_meta, _ = _read(every)
    again_tensors, again_meta, _ = _read(again)
    _, other_meta, _ = _read(other)

    # A limit keeps the first samples of the same order.
    assert few_meta["sample_id"] == every_meta["sample_id"][:5]
    assert torch.equal(few_tensors[LATENT], every_tensors[LATENT][:5])
    # The batch size changes nothing.
    assert again_meta == every_meta
    assert torch.equal(again_tensors[LATENT], every_tensors[LATENT])
    # The whole split, shuffled: every validation anchor once.
    assert sorted(every_meta["anchor_idx"]) == list(range(20, 45))
    assert every_meta["anchor_idx"] != sorted(every_meta["anchor_idx"])
    # Another seed, another order of the same samples.
    assert other_meta["sample_id"] != every_meta["sample_id"]
    assert sorted(other_meta["sample_id"]) == sorted(every_meta["sample_id"])


def test_test_split_only_when_asked(tmp_path, cache_root, loads):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))

    output = extract_run(run_dir, split="test")

    _, metadata, manifest = _read(output)
    assert output.name == "last-test"
    assert all(":test#" in sample for sample in metadata["sample_id"])
    assert manifest["provenance"]["data"]["split"] == "test"


def test_an_edited_config_is_refused(tmp_path, cache_root, loads):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))
    edited = OmegaConf.load(run_dir / "config.yaml")
    edited.data.context_steps = 5
    OmegaConf.save(edited, run_dir / "config.yaml")

    with pytest.raises(ValueError, match="edited after the run"):
        extract_run(run_dir)


def test_a_run_without_a_dataset_revision_is_refused(tmp_path, cache_root, loads):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset_revision"] = None
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="recorded no dataset revision"):
        extract_run(run_dir)


def test_another_dataset_revision_is_refused(tmp_path, cache_root, monkeypatch):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))
    monkeypatch.setattr(run_module, "load_data", lambda source: _loaded("moved"))

    with pytest.raises(ValueError, match="instead of the run's rev-123"):
        extract_run(run_dir)


def test_another_cache_is_refused(tmp_path, cache_root, loads, make_mimi_cache):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))
    other = make_mimi_cache(
        {("egocom", "r1"): (0, GRID_STEPS)},
        source_dataset_revision=REVISION,
        root=tmp_path / "other-cache",
    )
    # Same grid, but features from another Mimi build.
    manifest_path = other / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["resolved_revision"] = "another-sha"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not the cache the run trained on"):
        extract_run(run_dir, mimi_cache_root=other)


def test_a_moved_cache_is_accepted(tmp_path, cache_root, loads):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))
    moved = tmp_path / "moved-cache"
    cache_root.rename(moved)

    output = extract_run(run_dir, mimi_cache_root=moved, max_samples=2)

    _, _, manifest = _read(output)
    caches = manifest["provenance"]["data"]["feature_caches"]["mimi"]
    assert caches["root"] == str(open_mimi_cache(moved).root)


def test_a_missing_checkpoint_fails_before_loading_data(tmp_path, cache_root, loads):
    run_dir, _ = _make_run(tmp_path, _config(cache_root))

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        extract_run(run_dir, checkpoint="best.ckpt")

    assert loads == []


def test_not_a_run_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a training run directory"):
        extract_run(tmp_path)
