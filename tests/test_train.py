"""run() wiring, with data loading, the model and Lightning all mocked."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf

from turn_wm.config import load_config
from turn_wm.data.mimi_cache import MimiFeatureCaches
from turn_wm.data.source import DATASETS
from turn_wm.training import train as train_module
from turn_wm.training.train import (
    _build_callbacks,
    _build_logger,
    _config_hash,
    _write_config,
    _write_metadata,
)

GRID_STEPS = 50


def fake_corpus(name: str) -> SimpleNamespace:
    """A corpus whose only recording, r1, spans decision_index 0..49."""

    return SimpleNamespace(
        name=name,
        action_grid=Dataset.from_dict(
            {
                "dataset": [name] * GRID_STEPS,
                "recording_id": ["r1"] * GRID_STEPS,
                "decision_index": list(range(GRID_STEPS)),
                "decision_time_s": [i / 10 for i in range(GRID_STEPS)],
            }
        ),
    )


class Recorder:
    """Stand-ins for everything run() calls, logging calls in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.loaded_sources = []
        self.dataset_calls: list[dict] = []
        self.loader_calls: list[dict] = []
        self.trainers: list[SimpleNamespace] = []
        self.modules: list[SimpleNamespace] = []

    def load_data(self, source):
        self.events.append("load_data")
        self.loaded_sources.append(source)
        return SimpleNamespace(
            names=("egocom", "ego4d"),
            revision="rev-123",
            corpora=(fake_corpus("egocom"), fake_corpus("ego4d")),
        )

    def build_dataset(self, loaded, **kwargs):
        self.events.append(f"build_dataset:{kwargs['split']}")
        self.dataset_calls.append(kwargs)
        return f"{kwargs['split']}-dataset"

    def build_dataloader(self, dataset, *, loader):
        self.loader_calls.append({"dataset": dataset, "loader": loader})
        return f"{dataset}-loader"

    def seed_everything(self, seed, workers=False):
        self.events.append(f"seed:{seed}:workers={workers}")

    def module(self, cfg):
        self.events.append("module")
        module = SimpleNamespace(cfg=cfg)
        self.modules.append(module)
        return module

    def trainer(self, **kwargs):
        recorder = self

        class FakeTrainer:
            def __init__(self) -> None:
                self.kwargs = kwargs
                self.fit_calls = []

            def fit(self, module, *, train_dataloaders, val_dataloaders, ckpt_path):
                recorder.events.append("fit")
                self.fit_calls.append(
                    (module, train_dataloaders, val_dataloaders, ckpt_path)
                )

        trainer = FakeTrainer()
        self.trainers.append(trainer)
        return trainer


@pytest.fixture(autouse=True)
def isolated_outputs(tmp_path, monkeypatch):
    """Run from tmp_path so experiment.output_root never lands in the repo."""

    monkeypatch.chdir(tmp_path)


def run_dirs(tmp_path) -> list[Path]:
    root = tmp_path / "outputs" / "lewm"

    return sorted(root.iterdir()) if root.exists() else []


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    recorder = Recorder()

    monkeypatch.setattr(train_module, "load_data", recorder.load_data)
    monkeypatch.setattr(train_module, "build_dataset", recorder.build_dataset)
    monkeypatch.setattr(train_module, "build_dataloader", recorder.build_dataloader)
    monkeypatch.setattr(train_module, "LeWMModule", recorder.module)
    monkeypatch.setattr(
        train_module,
        "L",
        SimpleNamespace(
            seed_everything=recorder.seed_everything, Trainer=recorder.trainer
        ),
    )

    return recorder


@pytest.fixture
def media_roots(tmp_path):
    roots = {name: tmp_path / name for name in ("egocom", "ego4d")}

    for root in roots.values():
        root.mkdir()

    return roots


RAW_AUDIO = "data.observation_source=raw_audio"


def run(media_roots, *overrides):
    # The raw-audio path unless a test selects the cache.
    cfg = load_config([RAW_AUDIO, *overrides])
    train_module.run(cfg, media_roots=media_roots)
    return cfg


def calls_by_split(recorder):
    return {call["split"]: call for call in recorder.dataset_calls}


def test_run_builds_train_and_validation_splits(recorder, media_roots):
    run(media_roots)

    calls = calls_by_split(recorder)

    assert list(calls) == ["train", "validation"]
    assert calls["train"]["training"] is True
    assert calls["validation"]["training"] is False
    assert calls["train"]["media_roots"] == media_roots


def test_run_loads_the_configured_dataset(recorder, media_roots):
    run(media_roots, "data.dataset=egocom")

    assert recorder.loaded_sources == [DATASETS["egocom"]]


def test_run_uses_fixed_training_window(recorder, media_roots):
    run(media_roots)

    calls = calls_by_split(recorder)
    train_call, val_call = calls["train"], calls["validation"]

    assert train_call["window"].min_context_steps == 15
    assert train_call["window"].max_context_steps == 15
    assert train_call["window"].future_steps == 10
    assert val_call["window"] == train_call["window"]


def test_run_only_requests_configured_modalities(recorder, media_roots):
    run(media_roots)

    for call in recorder.dataset_calls:
        assert call["modalities"] == ("audio",)


def test_run_propagates_loader_config(recorder, media_roots):
    cfg = run(media_roots)

    train_loader, val_loader = (call["loader"] for call in recorder.loader_calls)

    assert [call["dataset"] for call in recorder.loader_calls] == [
        "train-dataset",
        "validation-dataset",
    ]

    for loader in (train_loader, val_loader):
        assert loader.batch_size == cfg.loader.batch_size
        assert loader.num_workers == cfg.loader.num_workers
        assert loader.pin_memory is cfg.loader.pin_memory
        assert loader.persistent_workers is cfg.loader.persistent_workers
        assert loader.prefetch_factor == cfg.loader.prefetch_factor
        assert loader.seed == cfg.seed

    # Training order comes from the dataset mode; validation stays in order.
    assert train_loader.shuffle is None
    assert val_loader.shuffle is False
    assert val_loader.drop_last is False


def test_run_seeds_before_model_construction(recorder, media_roots):
    run(media_roots, "seed=7")

    seed = recorder.events.index("seed:7:workers=True")

    assert seed < recorder.events.index("load_data")
    assert seed < recorder.events.index("module")


def test_run_calls_trainer_fit(recorder, media_roots):
    cfg = run(media_roots, "trainer.max_epochs=3")

    [trainer] = recorder.trainers
    [module] = recorder.modules

    callbacks = trainer.kwargs.pop("callbacks")
    default_root_dir = trainer.kwargs.pop("default_root_dir")
    logger = trainer.kwargs.pop("logger")

    assert trainer.kwargs == OmegaConf.to_container(cfg.trainer, resolve=True)
    assert Path(default_root_dir).parent == Path("outputs") / "lewm"
    assert logger is False
    assert trainer.kwargs["max_epochs"] == 3
    assert [type(callback) for callback in callbacks] == [ModelCheckpoint]
    assert trainer.fit_calls == [
        (module, "train-dataset-loader", "validation-dataset-loader", None)
    ]
    assert module.cfg is cfg
    assert recorder.events[-1] == "fit"


def test_run_rejects_unknown_dataset(recorder, media_roots, tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset 'nope'"):
        run(media_roots, "data.dataset=nope")

    assert recorder.events == []
    assert run_dirs(tmp_path) == []


def test_run_rejects_an_invalid_config_before_loading(recorder, media_roots, tmp_path):
    with pytest.raises(ValueError, match="rollout_context_size"):
        run(media_roots, "prediction.rollout_context_size=100")

    assert recorder.events == []
    assert run_dirs(tmp_path) == []


def test_missing_media_root_is_rejected(recorder, monkeypatch, tmp_path):
    monkeypatch.delenv("EGOCOM_MEDIA_ROOT", raising=False)
    monkeypatch.delenv("EGO4D_MEDIA_ROOT", raising=False)

    with pytest.raises(ValueError, match="EGOCOM_MEDIA_ROOT is not set"):
        train_module.run(load_config([RAW_AUDIO]))

    assert "module" not in recorder.events
    assert run_dirs(tmp_path) == []


def test_explicit_media_roots_must_cover_every_corpus(recorder, media_roots):
    with pytest.raises(ValueError, match="No media root supplied for corpus 'ego4d'"):
        run({"egocom": media_roots["egocom"]})


def test_media_root_must_be_a_directory(recorder, media_roots, tmp_path):
    roots = {**media_roots, "ego4d": tmp_path / "missing"}

    with pytest.raises(ValueError, match="'ego4d' is not a directory"):
        run(roots)


def test_media_roots_come_from_the_environment(recorder, media_roots, monkeypatch):
    monkeypatch.setenv("EGOCOM_MEDIA_ROOT", str(media_roots["egocom"]))
    monkeypatch.setenv("EGO4D_MEDIA_ROOT", str(media_roots["ego4d"]))

    train_module.run(load_config([RAW_AUDIO]))

    assert calls_by_split(recorder)["train"]["media_roots"] == media_roots


def test_checkpoint_callback_uses_config():
    cfg = load_config()

    callbacks = _build_callbacks(cfg, run_dir=Path("run"))

    assert len(callbacks) == 1

    checkpoint = callbacks[0]

    assert isinstance(checkpoint, ModelCheckpoint)
    assert checkpoint.monitor == "val/loss"
    assert checkpoint.mode == "min"
    assert checkpoint.save_top_k == 3
    assert checkpoint.save_last is True
    assert checkpoint.every_n_epochs == 1


def test_checkpoint_can_be_disabled():
    cfg = load_config(["checkpoint.enabled=false"])

    callbacks = _build_callbacks(cfg, run_dir=Path("run"))

    assert callbacks == []


def test_disabled_checkpoint_passes_no_callbacks(recorder, media_roots):
    run(media_roots, "checkpoint.enabled=false")

    assert recorder.trainers[0].kwargs["callbacks"] == []


def test_run_resumes_from_the_configured_checkpoint(recorder, media_roots):
    run(media_roots, "checkpoint.resume_from=/tmp/model.ckpt")

    [trainer] = recorder.trainers
    [(_, _, _, ckpt_path)] = trainer.fit_calls

    assert ckpt_path == "/tmp/model.ckpt"


def test_run_starts_fresh_without_resume(recorder, media_roots):
    run(media_roots)

    [(_, _, _, ckpt_path)] = recorder.trainers[0].fit_calls

    assert ckpt_path is None


def test_resume_path_expands_the_home_directory(recorder, media_roots):
    # Hydra needs "~" quoted: checkpoint.resume_from='"~/runs/last.ckpt"'.
    run(media_roots, 'checkpoint.resume_from="~/runs/last.ckpt"')

    [(_, _, _, ckpt_path)] = recorder.trainers[0].fit_calls

    assert ckpt_path == str(Path("~/runs/last.ckpt").expanduser())


def test_config_hash_is_stable():
    first = load_config()
    second = load_config()

    assert _config_hash(first) == _config_hash(second)


def test_config_hash_changes_with_experiment():
    first = load_config()
    second = load_config(["optimizer.lr=3e-4"])

    assert _config_hash(first) != _config_hash(second)


def test_wandb_disabled_returns_false(tmp_path):
    cfg = load_config()

    assert (
        _build_logger(
            cfg,
            run_id="test",
            run_dir=tmp_path,
        )
        is False
    )


def test_wandb_enabled_without_the_package_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module.importlib.util, "find_spec", lambda name: None)
    cfg = load_config(["logging.wandb.enabled=true"])

    with pytest.raises(RuntimeError, match="uv sync --extra wandb"):
        _build_logger(cfg, run_id="test", run_dir=tmp_path)


def test_checkpoint_is_inside_run_directory(tmp_path):
    cfg = load_config()

    callbacks = _build_callbacks(
        cfg,
        run_dir=tmp_path,
    )

    checkpoint = callbacks[0]

    assert checkpoint.dirpath == str(tmp_path / "checkpoints")


def test_write_config_saves_the_resolved_config(tmp_path):
    cfg = load_config(["data.context_steps=20"])

    _write_config(cfg, tmp_path)

    saved = OmegaConf.load(tmp_path / "config.yaml")

    assert (tmp_path / "config.yaml").is_file()
    # Interpolations are resolved: the predictor size is written as a value.
    assert saved.model.predictor.num_frames == 20
    assert _config_hash(saved) == _config_hash(cfg)


def test_write_metadata_records_the_run(tmp_path):
    cfg = load_config()
    git = {"commit": "abc", "dirty": False}

    _write_metadata(
        run_dir=tmp_path,
        run_id="run-1",
        config_hash="hash",
        cfg=cfg,
        git=git,
        dataset_revision="rev-123",
    )

    assert (tmp_path / "metadata.json").is_file()
    assert json.loads((tmp_path / "metadata.json").read_text()) == {
        "run_id": "run-1",
        "seed": cfg.seed,
        "config_hash": "hash",
        "git": git,
        "dataset": "full",
        "dataset_revision": "rev-123",
        "observation_source": "mimi_cache",
    }


def test_run_writes_config_and_metadata_into_its_run_directory(
    recorder, media_roots, tmp_path
):
    cfg = run(media_roots)

    [run_dir] = run_dirs(tmp_path)
    config_hash = _config_hash(cfg)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert run_dir.name.endswith(config_hash[:8])
    assert (run_dir / "config.yaml").is_file()
    assert metadata["run_id"] == run_dir.name
    assert metadata["config_hash"] == config_hash
    assert metadata["dataset_revision"] == "rev-123"
    assert Path(recorder.trainers[0].kwargs["default_root_dir"]) == Path(
        "outputs", "lewm", run_dir.name
    )


# ---------------------------------------------------------------------------
# Cached Mimi features
# ---------------------------------------------------------------------------


def run_cached(cache_root, *overrides, media_roots=None):
    cfg = load_config(
        [
            "data.observation_source=mimi_cache",
            f"data.mimi_cache.root={cache_root}",
            *overrides,
        ]
    )
    train_module.run(cfg, media_roots=media_roots)
    return cfg


BOTH = {("egocom", "r1"): (0, GRID_STEPS), ("ego4d", "r1"): (0, GRID_STEPS)}


@pytest.fixture
def cache_root(make_mimi_cache):
    return make_mimi_cache(BOTH)


@pytest.fixture
def release_root(make_mimi_cache, tmp_path):
    """A release: one cache per corpus, as uploaded to the Hub."""

    root = tmp_path / "release"
    make_mimi_cache({("egocom", "r1"): (0, GRID_STEPS)}, root=root / "egocom")
    make_mimi_cache({("ego4d", "r1"): (0, GRID_STEPS)}, root=root / "ego4d")
    (root / "release_manifest.json").write_text(
        json.dumps({"corpora": {"egocom": {}, "ego4d": {}}})
    )
    return root


def test_cached_run_needs_no_media_roots(recorder, cache_root, monkeypatch):
    monkeypatch.delenv("EGOCOM_MEDIA_ROOT", raising=False)
    monkeypatch.delenv("EGO4D_MEDIA_ROOT", raising=False)

    run_cached(cache_root)

    for call in recorder.dataset_calls:
        assert call["media_roots"] is None
        assert call["modalities"] == ("audio",)
        assert isinstance(call["mimi_store"], MimiFeatureCaches)
        assert call["mimi_store"].root == cache_root

    assert recorder.events[-1] == "fit"


def test_raw_run_passes_no_store(recorder, media_roots):
    run(media_roots)

    assert all(call["mimi_store"] is None for call in recorder.dataset_calls)


def test_cached_run_still_needs_media_for_other_modalities(
    recorder, cache_root, monkeypatch
):
    monkeypatch.delenv("EGOCOM_MEDIA_ROOT", raising=False)

    with pytest.raises(ValueError, match="EGOCOM_MEDIA_ROOT is not set"):
        run_cached(cache_root, "data.modalities=[audio,video]")


def test_cached_run_requires_a_cache_root(recorder, tmp_path):
    cfg = load_config(["data.observation_source=mimi_cache"])

    with pytest.raises(ValueError, match="data.mimi_cache.root is required"):
        train_module.run(cfg)

    assert recorder.events == []
    assert run_dirs(tmp_path) == []


def test_unknown_observation_source_is_rejected(recorder):
    cfg = load_config(["data.observation_source=video_cache"])

    with pytest.raises(ValueError, match="observation_source must be one of"):
        train_module.run(cfg)

    assert recorder.events == []


def test_missing_cache_is_a_clear_error(recorder, tmp_path):
    with pytest.raises(FileNotFoundError, match="expected manifest.json"):
        run_cached(tmp_path / "nowhere")


def test_a_release_root_serves_every_corpus(recorder, release_root):
    # data.mimi_cache.root may be the release root downloaded from the Hub.
    run_cached(release_root)

    [store] = {
        id(call["mimi_store"]): call["mimi_store"] for call in recorder.dataset_calls
    }.values()
    assert store.recording_keys == {("egocom", "r1"), ("ego4d", "r1")}
    assert len(store.stores) == 2
    assert recorder.events[-1] == "fit"


def test_same_grid_from_another_revision_is_accepted(recorder, make_mimi_cache):
    # E.g. the EgoCom cache built from the public repository, used by `full`.
    root = make_mimi_cache(BOTH, source_dataset_revision="public-rev")

    run_cached(root)

    assert recorder.events[-1] == "fit"


def test_cache_without_a_recorded_revision_is_accepted(recorder, make_mimi_cache):
    run_cached(make_mimi_cache(BOTH, source_dataset_revision=None))

    assert recorder.events[-1] == "fit"


@pytest.mark.parametrize(
    ("recordings", "message"),
    [
        (
            {("egocom", "r1"): (0, 40), ("ego4d", "r1"): (0, GRID_STEPS)},
            "1 recordings differ from the grid",
        ),
        (
            {("egocom", "r1"): (5, GRID_STEPS), ("ego4d", "r1"): (0, GRID_STEPS)},
            "1 recordings differ from the grid",
        ),
        (
            {("egocom", "r2"): (0, GRID_STEPS), ("ego4d", "r1"): (0, GRID_STEPS)},
            "1 are missing",
        ),
    ],
)
def test_cache_of_another_grid_is_refused(
    recorder, make_mimi_cache, tmp_path, recordings, message
):
    root = make_mimi_cache(recordings, source_dataset_revision="old")

    with pytest.raises(ValueError, match=message):
        run_cached(root)

    assert "module" not in recorder.events
    assert run_dirs(tmp_path) == []


def test_cache_missing_a_loaded_corpus_is_refused(recorder, make_mimi_cache):
    root = make_mimi_cache({("egocom", "r1"): (0, GRID_STEPS)})

    with pytest.raises(ValueError, match="no features for corpus 'ego4d'"):
        run_cached(root)


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        ({"rate_hz": 12.5}, "12.5 Hz features; the action grid is 10 Hz"),
        ({"dim": 256}, "256-d features; model.projector.input_dim is 512"),
    ],
)
def test_incompatible_cache_is_refused(recorder, make_mimi_cache, cache, message):
    root = make_mimi_cache({("egocom", "r1"): (0, 5)}, **cache)

    with pytest.raises(ValueError, match=message):
        run_cached(root)


def test_cached_run_records_the_cache_in_metadata(recorder, cache_root, tmp_path):
    run_cached(cache_root)

    [run_dir] = run_dirs(tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert metadata["observation_source"] == "mimi_cache"
    assert metadata["mimi_cache"] == {
        "root": str(cache_root),
        "caches": {
            "ego4d,egocom": {
                "schema_version": 2,
                "model_name": "kyutai/mimi",
                "model_revision": "requested-sha",
                "model_resolved_revision": "resolved-sha",
                "source_dataset_revision": "rev-123",
                "feature_rate_hz": 10.0,
                "feature_dim": 512,
            }
        },
    }


def test_raw_run_records_its_observation_source(recorder, media_roots, tmp_path):
    run(media_roots)

    [run_dir] = run_dirs(tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())

    assert metadata["observation_source"] == "raw_audio"
    assert "mimi_cache" not in metadata


def test_learning_rate_is_monitored_when_a_logger_is_active(
    recorder, media_roots, monkeypatch
):
    monkeypatch.setattr(train_module, "_build_logger", lambda cfg, **kwargs: "logger")

    run(media_roots)

    callbacks = recorder.trainers[0].kwargs["callbacks"]
    monitors = [c for c in callbacks if isinstance(c, LearningRateMonitor)]

    assert len(monitors) == 1
    assert monitors[0].logging_interval == "step"
    assert sum(isinstance(c, ModelCheckpoint) for c in callbacks) == 1


def test_no_learning_rate_monitor_without_a_logger(recorder, media_roots):
    run(media_roots)

    callbacks = recorder.trainers[0].kwargs["callbacks"]

    assert recorder.trainers[0].kwargs["logger"] is False
    assert not any(isinstance(c, LearningRateMonitor) for c in callbacks)


def test_fit_never_uses_the_test_split(recorder, media_roots):
    run(media_roots)

    assert [call["split"] for call in recorder.dataset_calls] == [
        "train",
        "validation",
    ]
    [trainer] = recorder.trainers
    [(_, train_loader, val_loader, _)] = trainer.fit_calls
    assert (train_loader, val_loader) == (
        "train-dataset-loader",
        "validation-dataset-loader",
    )
    assert "test" not in recorder.events
    assert not hasattr(trainer, "test_calls")


def test_the_module_defines_no_test_step():
    import lightning as L

    from turn_wm.training.lewm import LeWMModule

    assert LeWMModule.test_step is L.LightningModule.test_step
