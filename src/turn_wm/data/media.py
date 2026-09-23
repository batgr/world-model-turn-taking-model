from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class MediaPaths:
    """Physical media associated with one recording."""

    recording_id: str
    video_path: Path | None = None
    audio_path: Path | None = None

    def __post_init__(self) -> None:
        if self.video_path is None and self.audio_path is None:
            raise ValueError(
                f"Recording {self.recording_id!r} has no audio or video media"
            )


class MediaIndex:
    """Resolve canonical recording IDs to local media files."""

    def __init__(
        self,
        records: Mapping[str, MediaPaths],
        *,
        validate_paths: bool = True,
    ) -> None:
        if not records:
            raise ValueError("Media index cannot be empty")

        self._records = dict(records)

        if validate_paths:
            self._validate_paths()

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, recording_id: str) -> bool:
        return recording_id in self._records

    def get(self, recording_id: str) -> MediaPaths:
        try:
            return self._records[recording_id]
        except KeyError as exc:
            raise KeyError(f"No media found for recording {recording_id!r}") from exc

    def _validate_paths(self) -> None:
        for record in self._records.values():
            if record.video_path is not None and not record.video_path.is_file():
                raise FileNotFoundError(
                    f"Video file does not exist: {record.video_path}"
                )

            if record.audio_path is not None and not record.audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file does not exist: {record.audio_path}"
                )
