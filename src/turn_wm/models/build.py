from hydra.utils import instantiate
from omegaconf import DictConfig

from turn_wm.models.lewm.jepa import JEPA

# Observation sources a training recipe can select (`data.observation_source`).
RAW_AUDIO = "raw_audio"
MIMI_CACHE = "mimi_cache"
OBSERVATION_SOURCES = (RAW_AUDIO, MIMI_CACHE)


def observation_source(cfg: DictConfig) -> str:
    """The configured source; configs without a data section are raw audio."""

    data = cfg.get("data")
    source = RAW_AUDIO if data is None else data.get("observation_source", RAW_AUDIO)

    if source not in OBSERVATION_SOURCES:
        raise ValueError(
            f"data.observation_source must be one of {list(OBSERVATION_SOURCES)}, "
            f"got {source!r}"
        )

    return source


def build_model(cfg: DictConfig) -> JEPA:
    """Instantiate `cfg.model`; its encoder only when raw audio needs it.

    With precomputed Mimi features the encoder is never built, so Mimi's
    weights are not even loaded; the model projects the cached features.
    """

    if observation_source(cfg) == MIMI_CACHE:
        return instantiate(cfg.model, encoder=None)

    # The encoder is nested in the model config and built recursively.
    return instantiate(cfg.model)
