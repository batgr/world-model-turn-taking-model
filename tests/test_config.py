import re

import pytest
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from torch import nn

from turn_wm.config import CONFIG_DIR, load_config
from turn_wm.models.build import build_model
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg

# Keeps build tests offline: the real encoder downloads pretrained weights.
OFFLINE_ENCODER = "model.encoder._target_=torch.nn.Identity"


def targets(node, path="") -> list[tuple[str, str]]:
    if isinstance(node, dict):
        found = [(path, node["_target_"])] if "_target_" in node else []
        for key, value in node.items():
            found += targets(value, f"{path}.{key}".lstrip("."))
        return found

    return []


def test_default_config_selects_every_group():
    cfg = load_config()

    assert isinstance(cfg, DictConfig)
    assert set(cfg) == {
        "embed_dim",
        "model",
        "seed",
        "data",
        "prediction",
        "trainer",
        "loader",
        "optimizer",
        "scheduler",
        "loss",
        "checkpoint",
        "experiment",
        "logging",
    }
    assert cfg.model.encoder.model_name == "kyutai/mimi"
    assert cfg.model.encoder.target_rate == 10.0


def test_shared_sizes_are_interpolated_into_the_model():
    cfg = load_config(["embed_dim=256", "data.context_steps=20"])

    # The predictor's positions cover the teacher-forced context.
    assert cfg.model.predictor.num_frames == 20
    assert cfg.model.action_encoder.emb_dim == 256
    assert cfg.model.projector.output_dim == 256
    assert cfg.model.pred_proj.input_dim == cfg.model.pred_proj.output_dim == 256


def test_every_target_resolves_to_a_class():
    cfg = OmegaConf.to_container(load_config(), resolve=True)

    found = targets(cfg)

    assert len(found) == 6

    for path, target in found:
        assert isinstance(get_class(target), type), path


def test_overrides_apply():
    cfg = load_config(["model.predictor.depth=2", "model.encoder.target_rate=5.0"])

    assert cfg.model.predictor.depth == 2
    assert cfg.model.encoder.target_rate == 5.0


def test_unknown_override_fails():
    with pytest.raises(Exception, match="not_a_key"):
        load_config(["model.not_a_key=1"])


def test_config_dir_is_the_repository_configs():
    assert (CONFIG_DIR / "config.yaml").is_file()


def test_build_model_from_config():
    model = build_model(load_config([OFFLINE_ENCODER]))

    assert isinstance(model, JEPA)
    assert isinstance(model.encoder, nn.Identity)
    assert model.predictor is not None
    assert model.action_encoder.embed[-1].out_features == 192


def test_training_recipe_sits_at_the_root():
    cfg = load_config(["trainer.max_epochs=10", "optimizer.lr=1e-4"])

    assert cfg.trainer.max_epochs == 10
    assert cfg.optimizer.lr == 1e-4
    assert cfg.loss.sigreg.weight == 0.09


def test_sigreg_kwargs_match_the_regularizer():
    cfg = load_config()

    regularizer = SIGReg(**cfg.loss.sigreg.kwargs)

    assert regularizer.num_proj == cfg.loss.sigreg.kwargs.num_proj
    assert regularizer.t.numel() == cfg.loss.sigreg.kwargs.knots


def test_training_docs_match_the_recipe():
    # Every YAML excerpt in docs/training.md must agree with the composed
    # configuration, so the documentation cannot drift from the recipe.
    docs = (CONFIG_DIR.parent / "docs" / "training.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)```", docs, flags=re.DOTALL)
    cfg = OmegaConf.to_container(load_config(), resolve=True)

    assert blocks

    def assert_subset(expected, actual, path="config"):
        if isinstance(expected, dict):
            assert isinstance(actual, dict), path
            for key, value in expected.items():
                assert key in actual, f"{path}.{key}"
                assert_subset(value, actual[key], f"{path}.{key}")
        else:
            assert expected == actual, path

    for block in blocks:
        assert_subset(OmegaConf.to_container(OmegaConf.create(block)), cfg)
