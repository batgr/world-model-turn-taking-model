"""
Build the dataset for one split from loaded canonical data.

This is the construction boundary between loaded corpora and modelling code.
Whether one or several corpora were loaded, the result is one Dataset with one
global index space; callers never branch on corpus count.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path

from turn_wm.data.dataset import TurnTakingDataset, WindowConfig
from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaIndex,
    MediaModality,
    validate_modalities,
)
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.source import LoadedCorpus, LoadedData

logger = logging.getLogger(__name__)


def build_dataset(
    loaded: LoadedData,
    *,
    split: str,
    window: WindowConfig,
    training: bool,
    media_roots: Mapping[str, Path] | None = None,
    trainable_only: bool = True,
    modalities: Iterable[MediaModality] = MEDIA_MODALITIES,
    mimi_store: MimiFeatureStore | None = None,
) -> MultiCorpusDataset:
    """Build one dataset over every loaded corpus that publishes `split`.

    Corpora without the split are left out rather than substituted. With
    `media_roots` (manifest `dataset` value → local corpus root), samples also
    carry decoded raw media, restricted to `modalities` (audio and video by
    default) in every corpus alike. With `mimi_store`, audio comes from its
    precomputed Mimi features instead of the media; the caller opens the
    store and decides whether media roots are still needed.
    """

    selected = validate_modalities(modalities)

    children: dict[str, TurnTakingDataset] = {}
    without_split = []
    without_usable = []

    for corpus in loaded.corpora:
        if split not in corpus.model_ready:
            without_split.append(corpus.name)
            continue

        dataset = TurnTakingDataset(
            anchors=corpus.model_ready[split],
            action_grid=corpus.action_grid,
            window=window,
            training=training,
            trainable_only=trainable_only,
            media_index=(
                None if media_roots is None else _media_index(corpus, media_roots)
            ),
            modalities=selected,
            mimi_store=mimi_store,
        )

        if mimi_store is not None:
            canonical_recording_keys = frozenset(
                zip(
                    corpus.action_grid["dataset"],
                    corpus.action_grid["recording_id"],
                    strict=True,
                )
            )
            cached_recording_keys = canonical_recording_keys & mimi_store.recording_keys
            logger.info(
                "Mimi cache coverage for %s: canonical recordings: %d; "
                "cached recordings: %d; excluded recordings: %d; "
                "anchors filtered from %s: %d",
                corpus.name,
                len(canonical_recording_keys),
                len(cached_recording_keys),
                len(canonical_recording_keys - cached_recording_keys),
                split,
                dataset.cache_filtered_anchor_count,
            )

        if len(dataset) == 0:
            without_usable.append(corpus.name)
            continue

        children[corpus.name] = dataset

    if without_split:
        logger.info("Split %r is not published by: %s", split, without_split)

    if without_usable:
        logger.info("Split %r has no usable anchors in: %s", split, without_usable)

    if not children:
        if len(without_split) == len(loaded.corpora):
            raise ValueError(
                f"Split {split!r} is not published by any loaded corpus "
                f"({list(loaded.names)})"
            )

        raise ValueError(f"Split {split!r} has no usable anchors in any corpus")

    return MultiCorpusDataset(children)


def _media_index(
    corpus: LoadedCorpus,
    media_roots: Mapping[str, Path],
) -> MediaIndex:
    if corpus.media_manifest is None:
        raise ValueError(f"Corpus {corpus.name!r} does not publish a media manifest")

    return MediaIndex.from_manifest(corpus.media_manifest, media_roots)
