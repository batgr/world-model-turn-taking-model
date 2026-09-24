"""
Resolve canonical media records to local raw-media files.

The data repository publishes a `media_manifest` keyed by
`(dataset, recording_id)` with paths relative to each corpus root and a
`media_offset_s` relating the canonical grid clock to the media file clock:

    media_time_s = decision_time_s + media_offset_s

This module joins those records with locally configured corpus roots. It
contains no corpus-specific logic. It also defines the media modalities a
dataset can decode.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from datasets import Dataset

type MediaKey = tuple[str, str]

MediaModality = Literal["audio", "video"]

# Every supported modality, in canonical order; also the default selection.
MEDIA_MODALITIES: tuple[MediaModality, ...] = get_args(MediaModality)


def validate_modalities(
    modalities: Iterable[MediaModality],
) -> tuple[MediaModality, ...]:
    """Check a modality selection and return it in canonical order.

    The selection must be non-empty, contain only supported modalities and
    name each at most once; order does not matter.
    """

    if isinstance(modalities, str):
        raise TypeError(
            f"modalities must be a collection such as ({modalities!r},), not a string"
        )

    values = tuple(modalities)

    if not values:
        raise ValueError("At least one media modality must be selected")

    unknown = [value for value in values if value not in MEDIA_MODALITIES]

    if unknown:
        raise ValueError(
            f"Unsupported media modalities {unknown}; "
            f"supported: {list(MEDIA_MODALITIES)}"
        )

    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate media modalities: {list(values)}")

    return tuple(modality for modality in MEDIA_MODALITIES if modality in values)


@dataclass(frozen=True)
class MediaPaths:
    """Physical media associated with one canonical recording."""

    dataset: str
    recording_id: str
    video_path: Path | None = None
    audio_path: Path | None = None
    media_offset_s: float = 0.0
    # None means the publisher could not probe the container.
    video_has_audio: bool | None = None

    def __post_init__(self) -> None:
        if self.video_path is None and self.audio_path is None:
            raise ValueError(f"Recording {self.key!r} has no audio or video media")

        if not math.isfinite(self.media_offset_s):
            raise ValueError(f"Recording {self.key!r} has a non-finite media offset")

    @property
    def key(self) -> MediaKey:
        return (self.dataset, self.recording_id)

    @property
    def audio_source(self) -> Path | None:
        """File to decode audio from, or None when audio is unavailable."""

        if self.audio_path is not None:
            return self.audio_path

        if self.video_path is not None and self.video_has_audio is not False:
            return self.video_path

        return None

    def to_media_time(self, decision_time_s: float) -> float:
        """Convert canonical grid time to the media file's own timeline."""

        return decision_time_s + self.media_offset_s


class MediaIndex:
    """Resolve `(dataset, recording_id)` to local media files."""

    def __init__(
        self,
        records: Iterable[MediaPaths],
        *,
        validate_paths: bool = True,
    ) -> None:
        self._records: dict[MediaKey, MediaPaths] = {}

        for record in records:
            if record.key in self._records:
                raise ValueError(f"Duplicate media record for {record.key!r}")

            self._records[record.key] = record

        if not self._records:
            raise ValueError("Media index cannot be empty")

        self.validate_paths = validate_paths

    @classmethod
    def from_manifest(
        cls,
        manifest: Dataset,
        roots: Mapping[str, Path],
        *,
        validate_paths: bool = True,
    ) -> MediaIndex:
        """Join a canonical media manifest with local corpus roots.

        `roots` maps each manifest `dataset` value to its corpus root.
        File existence is checked on lookup, so a partially available local
        corpus can still serve the recordings it contains.
        """

        records = []

        for row in manifest:
            dataset = row["dataset"]

            if dataset not in roots:
                raise ValueError(
                    f"No media root configured for dataset {dataset!r}; "
                    f"configured: {sorted(roots)}"
                )

            root = Path(roots[dataset])

            records.append(
                MediaPaths(
                    dataset=dataset,
                    recording_id=row["recording_id"],
                    video_path=_resolve(root, row["video_path"]),
                    audio_path=_resolve(root, row["audio_path"]),
                    media_offset_s=float(row["media_offset_s"]),
                    video_has_audio=row["video_has_audio"],
                )
            )

        return cls(records, validate_paths=validate_paths)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: MediaKey) -> bool:
        return key in self._records

    def get(self, *, dataset: str, recording_id: str) -> MediaPaths:
        try:
            record = self._records[(dataset, recording_id)]
        except KeyError as exc:
            raise KeyError(
                f"No media found for recording {(dataset, recording_id)!r}"
            ) from exc

        if self.validate_paths:
            _require_file(record.video_path, kind="Video")
            _require_file(record.audio_path, kind="Audio")

        return record


def _resolve(root: Path, relative: str | None) -> Path | None:
    if relative is None:
        return None

    path = PurePosixPath(relative)

    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(
            f"Canonical media path must be relative to the corpus root: {relative!r}"
        )

    return root.joinpath(*path.parts)


def _require_file(path: Path | None, *, kind: str) -> None:
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"{kind} file does not exist: {path}")
