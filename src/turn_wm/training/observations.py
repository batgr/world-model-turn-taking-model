"""
Where a run's samples get their observations: an encoder's feature cache or media.

`prepare_observations` opens and checks the cache against the loaded data
(same grid, every anchor covered) and resolves the local media roots of
whatever is still decoded. Training and post-hoc analysis both use it.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
from omegaconf import DictConfig

from turn_wm.data.feature_cache import (
    FeatureCaches,
    FeatureStore,
    open_feature_cache,
    store_datasets,
)
from turn_wm.data.media import MediaModality
from turn_wm.data.source import LoadedCorpus, LoadedData
from turn_wm.models.build import FEATURE_CACHE, observation_source


@dataclass(frozen=True)
class RunObservations:
    """Where the samples of a run get their observations from.

    What `build_dataset` needs beyond a split and a window: the precomputed
    features standing in for audio, and the local media of whatever is still
    decoded. Training and post-hoc analysis both build their datasets from it
    (`build_run_dataset`), so a new observation source is wired here once.
    """

    modalities: tuple[MediaModality, ...]
    feature_store: FeatureCaches | None = None
    media_roots: dict[str, Path] | None = None


def prepare_observations(
    cfg: DictConfig,
    loaded: LoadedData,
    *,
    media_roots: Mapping[str, Path] | None = None,
) -> RunObservations:
    """Open and check what `cfg` observes `loaded` through.

    With `observation_source: feature_cache`, the cache under
    `data.feature_cache.root` is opened and checked against the loaded data.
    Media roots, when anything is still decoded from media, are
    `media_roots` or else the `<DATASET>_MEDIA_ROOT` environment variables.
    """

    require_grid_rate(cfg, loaded)
    modalities = cast(tuple[MediaModality, ...], tuple(cfg.data.modalities))
    feature_store = None

    if observation_source(cfg) == FEATURE_CACHE:
        require_feature_cache_root(cfg)
        # One corpus cache, or a release root holding one cache per corpus.
        feature_store = open_feature_cache(
            Path(cfg.data.feature_cache.root).expanduser()
        )
        validate_feature_cache(feature_store, loaded, cfg)

    # Cached features replace audio only; other modalities still need media.
    needs_media = feature_store is None or any(m != "audio" for m in modalities)

    roots = None

    if needs_media:
        roots = (
            _resolve_media_roots(loaded)
            if media_roots is None
            else _validate_media_roots(loaded, media_roots)
        )

    return RunObservations(
        modalities=modalities,
        feature_store=feature_store,
        media_roots=roots,
    )


def feature_cache_identity(caches: FeatureCaches) -> dict[str, object]:
    """What identifies a feature cache, per corpus set, not its whole manifest."""

    return {
        ",".join(sorted(store_datasets(store))): {
            "schema_version": store.schema_version,
            "model_name": store.model_name,
            "model_revision": store.model_revision,
            "model_resolved_revision": store.model_resolved_revision,
            "source_dataset_revision": store.source_dataset_revision,
            "feature_rate_hz": store.feature_rate_hz,
            "feature_dim": store.feature_dim,
        }
        for store in caches.stores
    }


def require_grid_rate(cfg: DictConfig, loaded: LoadedData) -> None:
    """`data.grid_rate_hz` must be the loaded data's decision grid rate.

    Every step count of the configuration (context, horizons, curriculum) is
    in steps of that grid, so a run never silently changes its time scale.
    """

    configured = float(cfg.data.grid_rate_hz)
    actual = loaded.grid_rate_hz

    if not math.isclose(configured, actual):
        raise ValueError(
            f"data.grid_rate_hz is {configured:g} Hz but the loaded data's decision "
            f"grid is {actual:g} Hz; set data.grid_rate_hz={actual:g} and express "
            "every *_steps setting in steps of that grid"
        )


def require_feature_cache_root(cfg: DictConfig) -> None:
    if observation_source(cfg) == FEATURE_CACHE and cfg.data.feature_cache.root is None:
        raise ValueError(
            "data.feature_cache.root is required with data.observation_source="
            "feature_cache (create the cache with `turn-wm precompute-features`)"
        )


def validate_feature_cache(
    caches: FeatureCaches,
    loaded: LoadedData,
    cfg: DictConfig,
) -> None:
    """Refuse caches that do not match the grid, the model or the loaded data.

    Every loaded corpus must be covered by a cache. A cache computed from the
    loaded dataset revision is accepted as is; otherwise (e.g. the EgoCom
    cache built from the public repository, used by the private `full`
    release) each of its recordings must have exactly the loaded grid's span
    (start_index, steps, start_time_s), and every loaded recording must be
    cached or explicitly excluded: the features are then those of the grid.
    """

    input_dim = int(cfg.model.projector.input_dim)
    grid_rate = loaded.grid_rate_hz

    for store in caches.stores:
        if not math.isclose(store.feature_rate_hz, grid_rate):
            raise ValueError(
                f"Feature cache {store.root} has {store.feature_rate_hz:g} Hz "
                f"features; the action grid is {grid_rate:g} Hz"
            )

        if store.feature_dim != input_dim:
            raise ValueError(
                f"Feature cache {store.root} has {store.feature_dim}-d features; "
                f"model.projector.input_dim is {input_dim}"
            )

    for corpus in loaded.corpora:
        stores = [
            store for store in caches.stores if corpus.name in store_datasets(store)
        ]

        if not stores:
            raise ValueError(
                f"Feature cache {caches.root} has no features for corpus {corpus.name!r}"
            )

        for store in stores:
            if (
                store.source_dataset_revision is not None
                and store.source_dataset_revision == loaded.revision
            ):
                continue

            _require_grid_spans(store, corpus, loaded_revision=loaded.revision)


def _require_grid_spans(
    store: FeatureStore,
    corpus: LoadedCorpus,
    *,
    loaded_revision: str | None,
) -> None:
    """The cache's recordings of `corpus` must be exactly the loaded grid's."""

    table = cast(
        pa.Table,
        corpus.action_grid.select_columns(
            ["dataset", "recording_id", "decision_index", "decision_time_s"]
        ).with_format("arrow")[:],
    )
    spans = {
        (row["dataset"], row["recording_id"]): (
            int(row["decision_index_min"]),
            int(row["decision_index_count"]),
            float(row["decision_time_s_min"]),
        )
        for row in table.group_by(["dataset", "recording_id"])
        .aggregate(
            [
                ("decision_index", "min"),
                ("decision_index", "count"),
                ("decision_time_s", "min"),
            ]
        )
        .to_pylist()
    }
    cached = {
        key: store.record(dataset=key[0], recording_id=key[1])
        for key in store.recording_keys
        if key[0] == corpus.name
    }
    excluded = {
        (e.dataset, e.recording_id)
        for e in store.exclusions
        if e.dataset == corpus.name
    }
    mismatched = sorted(
        key
        for key, record in cached.items()
        if key not in spans
        or spans[key][:2] != (record.start_index, record.steps)
        or not math.isclose(spans[key][2], record.start_time_s, abs_tol=1e-6)
    )
    uncovered = sorted(set(spans) - set(cached) - excluded)

    if mismatched or uncovered:
        raise ValueError(
            f"Feature cache {store.root} (dataset revision "
            f"{store.source_dataset_revision}) does not match corpus {corpus.name!r} "
            f"as loaded (revision {loaded_revision}): {len(mismatched)} recordings "
            f"differ from the grid (e.g. {mismatched[:3]}), {len(uncovered)} are "
            f"missing (e.g. {uncovered[:3]}); recompute the cache"
        )


def _resolve_media_roots(
    loaded: LoadedData,
) -> dict[str, Path]:
    """Resolve corpus media roots from environment variables."""

    roots = {}

    for name in loaded.names:
        variable = _media_root_variable(name)
        value = os.environ.get(variable)

        if value is None:
            raise ValueError(
                f"{variable} is not set; training requires local raw "
                f"media for corpus {name!r}"
            )

        path = Path(value).expanduser()

        if not path.is_dir():
            raise ValueError(f"{variable} does not point to a directory: {path}")

        roots[name] = path

    return roots


def _validate_media_roots(
    loaded: LoadedData,
    media_roots: Mapping[str, Path],
) -> dict[str, Path]:
    """Validate explicitly supplied media roots."""

    roots = {}

    for name in loaded.names:
        if name not in media_roots:
            raise ValueError(f"No media root supplied for corpus {name!r}")

        path = Path(media_roots[name]).expanduser()

        if not path.is_dir():
            raise ValueError(f"Media root for {name!r} is not a directory: {path}")

        roots[name] = path

    return roots


def _media_root_variable(dataset: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", dataset).upper()

    return f"{normalized}_MEDIA_ROOT"
