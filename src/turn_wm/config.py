"""
Compose the experiment configuration with Hydra.

The YAML tree lives in the repository's `configs/` directory, in the spirit
of le-wm: `config.yaml` holds shared sizes (`embed_dim`, `history_size`) and
selects a `model` and a `train` recipe. The model config nests its
sub-modules, encoder included, each naming its class with `_target_` and
interpolating the shared sizes; `turn_wm.models.build.build_model`
instantiates it. The training recipe (`seed`, `trainer`, `loader`,
`optimizer`, `loss`, ...) is merged at the root of the composed config.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

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
