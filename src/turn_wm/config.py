"""
Compose the experiment configuration with Hydra.

The YAML tree lives in the repository's `configs/` directory, in the spirit
of le-wm: `config.yaml` holds the shared `embed_dim` and selects a `model` and
a `train` recipe. The model config nests its sub-modules, each naming its
class with `_target_` and interpolating `embed_dim` and the recipe's
`data.context_steps`; its frozen encoder is the `model/encoder` group, which
also sets the root `feature_dim`. `turn_wm.models.build.build_model`
instantiates it. The training recipe (`seed`, `trainer`, `loader`,
`optimizer`, `loss`, ...) is merged at the root of the composed config.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(
    overrides: Sequence[str] = (),
    *,
    config_name: str = "config",
    config_dir: Path = CONFIG_DIR,
) -> DictConfig:
    """Compose `config_name` from `config_dir` with Hydra command-line overrides."""

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


# Every run before `data.grid_rate_hz` existed trained on the 10 Hz grid.
LEGACY_GRID_RATE_HZ = 10.0


def upgrade_run_config(cfg: DictConfig) -> DictConfig:
    """A saved run's config in the current layout; the saved file is unchanged.

    Runs saved before encoders were pluggable name their cache `mimi_cache`
    (`data.mimi_cache`, `observation_source: mimi_cache`), have no
    `data.grid_rate_hz` (10 Hz) and no root `feature_dim` (the projector's
    input size). The architecture and every value are kept; only names move.
    """

    cfg = copy.deepcopy(cfg)

    with open_dict(cfg):
        data = cfg.get("data")

        if data is not None:
            if "mimi_cache" in data:
                data.feature_cache = data.pop("mimi_cache")

            if data.get("observation_source") == "mimi_cache":
                data.observation_source = "feature_cache"

            if "grid_rate_hz" not in data:
                data.grid_rate_hz = LEGACY_GRID_RATE_HZ

        if "feature_dim" not in cfg and "model" in cfg:
            cfg.feature_dim = cfg.model.projector.input_dim

    return cfg
