"""run() wiring, with data loading, the model and Lightning all mocked."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from turn_wm.config import load_config
from turn_wm.data.source import DATASETS
from turn_wm.training import train as train_module
from turn_wm.training.train import (
    _build_callbacks,
    _build_logger,
    _config_hash,
    _write_config,
    _write_metadata,
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
        return SimpleNamespace(names=("egocom", "ego4d"), revision="rev-123")

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


def run(media_roots, *overrides):
    cfg = load_config(list(overrides))
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
        train_module.run(load_config())

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

    train_module.run(load_config())

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
