import math

from hydra.utils import instantiate
from omegaconf import DictConfig

from turn_wm.models.encoders.base import Encoder
from turn_wm.models.lewm.jepa import JEPA

# Observation sources a training recipe can select (`data.observation_source`).
RAW_AUDIO = "raw_audio"
FEATURE_CACHE = "feature_cache"
OBSERVATION_SOURCES = (RAW_AUDIO, FEATURE_CACHE)


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


def build_encoder(cfg: DictConfig) -> Encoder:
    """The frozen encoder of `cfg.model.encoder`, checked against the config.

    It must give `feature_dim` features, one per step of the decision grid
    (`data.grid_rate_hz`): frame `k` is grid step `k`.
    """

    encoder = instantiate(cfg.model.encoder)

    if not isinstance(encoder, Encoder):
        raise TypeError(
            f"model.encoder must build an Encoder, got {type(encoder).__name__}"
        )

    if not math.isclose(encoder.frame_rate, float(cfg.data.grid_rate_hz)):
        raise ValueError(
            f"{encoder.name} gives {encoder.frame_rate:g} Hz frames; the decision "
            f"grid is {cfg.data.grid_rate_hz:g} Hz (data.grid_rate_hz): an encoder "
            "runs on a grid at its own frame rate"
        )

    if encoder.output_dim != int(cfg.feature_dim):
        raise ValueError(
            f"{encoder.name} gives {encoder.output_dim}-d features; feature_dim "
            f"is {cfg.feature_dim}"
        )

    return encoder


def build_model(cfg: DictConfig) -> JEPA:
    """Instantiate `cfg.model`; its encoder only when raw audio needs it.

    With precomputed features the encoder is never built, so its weights are
    not even loaded; the model projects the cached features.
    """

    if observation_source(cfg) == FEATURE_CACHE:
        return instantiate(cfg.model, encoder=None)

    return instantiate(cfg.model, encoder=build_encoder(cfg))
