"""
Build PyTorch samples from model-ready anchors and the temporal action grid.

This module connects the dataset artifacts to modelling code. It resolves an
anchor into context and future trajectories, encodes categorical vocal states
and actions, and returns tensors. Batching and sampling are handled elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaIndex,
    MediaModality,
    MediaPaths,
    validate_modalities,
)
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.reader import MediaReader, MediaWindow
from turn_wm.data.window import build_window, validate_against_anchor

STATE_TO_ID = {
    "SILENT": 0,
    "SPEAKING": 1,
    "UNKNOWN": 2,
}

PAD_STATE_ID = 3


ACTION_TO_ID = {
    "NO_EVENT": 0,
    "ONSET": 1,
    "OFFSET": 2,
}

MASKED_ACTION_ID = len(ACTION_TO_ID)
PAD_ACTION_ID = MASKED_ACTION_ID + 1


@dataclass(frozen=True)
class WindowConfig:
    """Temporal geometry requested by the modelling experiment."""

    min_context_steps: int = 10
    max_context_steps: int = 50
    future_steps: int = 10

    def __post_init__(self) -> None:
        if self.min_context_steps <= 0:
            raise ValueError("min_context_steps must be positive")

        if self.max_context_steps < self.min_context_steps:
            raise ValueError("max_context_steps must be >= min_context_steps")

        if self.future_steps <= 0:
            raise ValueError("future_steps must be positive")


class TurnTakingDataset(Dataset):
    """Expose temporal turn-taking samples from model-ready anchors.

    With a `media_index`, samples also carry `context_media`/`future_media`
    windows in which only the selected `modalities` are decoded.

    With a `mimi_store`, audio comes from precomputed Mimi features instead:
    samples carry `context_features`/`future_features`, one row per grid
    step of the window, and audio is never decoded. Other modalities still
    come from the media.
    """

    def __init__(
        self,
        *,
        anchors: HFDataset,
        action_grid: HFDataset,
        window: WindowConfig,
        training: bool,
        trainable_only: bool = True,
        media_index: MediaIndex | None = None,
        media_reader: MediaReader | None = None,
        modalities: Iterable[MediaModality] = MEDIA_MODALITIES,
        mimi_store: MimiFeatureStore | None = None,
    ) -> None:
        self.modalities = validate_modalities(modalities)

        if mimi_store is not None and "audio" not in self.modalities:
            raise ValueError("mimi_store provides audio; modalities must include audio")

        self.canonical_anchor_count = len(anchors)
        self.cache_filtered_anchor_count = 0

        if mimi_store is not None:
            available = mimi_store.recording_keys
            anchors = anchors.filter(
                lambda datasets, recording_ids: [
                    (dataset, recording_id) in available
                    for dataset, recording_id in zip(
                        datasets,
                        recording_ids,
                        strict=True,
                    )
                ],
                input_columns=["dataset", "recording_id"],
                batched=True,
                desc="Filtering anchors to recordings available in the Mimi cache",
            )
            self.cache_filtered_anchor_count = self.canonical_anchor_count - len(
                anchors
            )

        # Modalities decoded from media; cached audio is not.
        self.media_modalities: tuple[MediaModality, ...] = tuple(
            modality
            for modality in self.modalities
            if not (modality == "audio" and mimi_store is not None)
        )

        if trainable_only:
            anchors = anchors.filter(
                lambda batch: batch["is_trainable"],
                batched=True,
                desc="Filtering trainable anchors",
            )

        # Anchors too close to a recording's start (or end) for this window
        # would fail in __getitem__; keep only those that support it. With
        # the published minimum context this removes nothing unless the
        # window asks for more (e.g. a fixed training context).
        anchors = anchors.filter(
            lambda context, future: [
                c >= window.min_context_steps and f >= window.future_steps
                for c, f in zip(context, future, strict=True)
            ],
            input_columns=["max_context_steps", "future_steps"],
            batched=True,
            desc="Filtering anchors supporting the window",
        )

        if media_reader is not None and media_index is None:
            raise ValueError("media_reader requires media_index")

        self.anchors = anchors
        self.action_grid = action_grid
        self.window = window
        self.training = training

        self.mimi_store = mimi_store
        self.media_index = media_index if self.media_modalities else None

        self.media_reader = (
            media_reader
            if media_reader is not None
            else MediaReader()
            if self.media_index is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = self.anchors[index]

        context_steps = self._context_steps(anchor)

        validate_against_anchor(
            context_steps=context_steps,
            future_steps=self.window.future_steps,
            max_context_steps=int(anchor["max_context_steps"]),
            available_future_steps=int(anchor["future_steps"]),
        )

        anchor_idx = int(anchor["anchor_idx"])
        anchor_row = int(anchor["anchor_row"])

        # Validates the logical window bounds; row offsets are derived below.
        build_window(
            anchor_idx=anchor_idx,
            context_steps=context_steps,
            future_steps=self.window.future_steps,
        )

        # `anchor_row` is the physical position in action_grid.
        # WindowBounds expresses the same geometry in logical timestep space,
        # so offsets relative to the anchor map directly to table rows.
        context_start_row = anchor_row - context_steps + 1
        future_end_row = anchor_row + self.window.future_steps

        rows = self.action_grid[context_start_row : future_end_row + 1]

        self._validate_slice(
            rows=rows,
            recording_id=str(anchor["recording_id"]),
            expected_steps=context_steps + self.window.future_steps,
        )

        context = slice(0, context_steps)
        future = slice(
            context_steps,
            context_steps + self.window.future_steps,
        )

        states = rows["focal_state_before"]
        actions = rows["action"]
        valid = rows["action_valid"]

        sample = {
            "context_state": torch.tensor(
                self._encode_states(states[context]),
                dtype=torch.long,
            ),
            "context_action": torch.tensor(
                self._encode_actions(actions[context]),
                dtype=torch.long,
            ),
            "context_valid": torch.tensor(
                valid[context],
                dtype=torch.bool,
            ),
            "future_state": torch.tensor(
                self._encode_states(states[future]),
                dtype=torch.long,
            ),
            "future_action": torch.tensor(
                self._encode_actions(actions[future]),
                dtype=torch.long,
            ),
            "future_valid": torch.tensor(
                valid[future],
                dtype=torch.bool,
            ),
            "context_length": context_steps,
            "sample_id": anchor["sample_id"],
            "dataset": anchor["dataset"],
            "recording_id": anchor["recording_id"],
            "anchor_idx": anchor_idx,
            "anchor_time": anchor["anchor_time"],
            "sample_class": anchor["sample_class"],
        }

        if self.mimi_store is not None:
            self._attach_features(
                sample=sample,
                rows=rows,
                context_steps=context_steps,
                anchor_idx=anchor_idx,
                dataset=anchor["dataset"],
                recording_id=anchor["recording_id"],
            )

        if self.media_index is not None:
            self._attach_media(
                sample=sample,
                rows=rows,
                context_steps=context_steps,
                dataset=anchor["dataset"],
                recording_id=anchor["recording_id"],
            )

        return sample

    def sample_classes(self) -> list[str]:
        """Return the sampling class associated with each exposed anchor."""
        return list(self.anchors["sample_class"])

    def _attach_features(
        self,
        *,
        sample: dict[str, Any],
        rows: dict[str, list[Any]],
        context_steps: int,
        anchor_idx: int,
        dataset: str,
        recording_id: str,
    ) -> None:
        """Cached Mimi rows for exactly the grid rows of the window."""

        assert self.mimi_store is not None

        indices = [int(value) for value in rows["decision_index"]]
        first, last = indices[0], indices[-1]

        if last - first + 1 != len(indices) or indices[context_steps - 1] != anchor_idx:
            raise ValueError(
                f"Grid rows of {recording_id!r} are not the contiguous window "
                f"around decision_index {anchor_idx}"
            )

        features = self.mimi_store.get_by_index(
            dataset=dataset,
            recording_id=recording_id,
            start_index=first,
            end_index=last + 1,
        )

        sample["context_features"] = features[:context_steps]
        sample["future_features"] = features[context_steps:]

    def _attach_media(
        self,
        *,
        sample: dict[str, Any],
        rows: dict[str, list[Any]],
        context_steps: int,
        dataset: str,
        recording_id: str,
    ) -> None:
        if self.media_index is None or self.media_reader is None:
            return

        decision_times = [float(value) for value in rows["decision_time_s"]]

        if len(decision_times) <= context_steps:
            raise ValueError("Cannot derive media boundaries from window")

        context_start_s = decision_times[0]

        # First future grid point.
        future_start_s = decision_times[context_steps]

        # Grid spacing comes directly from the canonical
        # action grid instead of being hard-coded to 10 Hz.
        grid_step_s = decision_times[context_steps] - decision_times[context_steps - 1]

        if grid_step_s <= 0:
            raise ValueError("Action-grid timestamps must be strictly increasing")

        future_end_s = decision_times[-1] + grid_step_s

        media = self.media_index.get(
            dataset=dataset,
            recording_id=recording_id,
        )

        sample["context_media"] = self._read_media(
            media,
            start_time_s=context_start_s,
            end_time_s=future_start_s,
        )

        sample["future_media"] = self._read_media(
            media,
            start_time_s=future_start_s,
            end_time_s=future_end_s,
        )

    def _read_media(
        self,
        media: MediaPaths,
        *,
        start_time_s: float,
        end_time_s: float,
    ) -> MediaWindow:
        """Decode a canonical-time interval from the media file's timeline."""

        assert self.media_reader is not None

        window = self.media_reader.read_window(
            media,
            start_time_s=media.to_media_time(start_time_s),
            end_time_s=media.to_media_time(end_time_s),
            modalities=self.media_modalities,
        )

        return replace(
            window,
            canonical_start_time_s=start_time_s,
            canonical_end_time_s=end_time_s,
        )

    def _context_steps(self, anchor: dict[str, Any]) -> int:
        available = min(
            int(anchor["max_context_steps"]),
            self.window.max_context_steps,
        )

        if available < self.window.min_context_steps:
            raise ValueError(
                f"Anchor {anchor['sample_id']} supports only "
                f"{available} context steps, but the experiment requires "
                f"at least {self.window.min_context_steps}"
            )

        if not self.training:
            return available

        return int(
            torch.randint(
                low=self.window.min_context_steps,
                high=available + 1,
                size=(1,),
            ).item()
        )

    @staticmethod
    def _encode_states(values: list[str]) -> list[int]:
        encoded = []

        for value in values:
            try:
                encoded.append(STATE_TO_ID[value])
            except KeyError as error:
                raise ValueError(f"Unknown vocal state: {value!r}") from error

        return encoded

    @staticmethod
    def _encode_actions(values: list[str | None]) -> list[int]:
        return [
            MASKED_ACTION_ID if value is None else ACTION_TO_ID[value]
            for value in values
        ]

    @staticmethod
    def _validate_slice(
        *,
        rows: dict[str, list[Any]],
        recording_id: str,
        expected_steps: int,
    ) -> None:
        row_count = len(rows["recording_id"])

        if row_count != expected_steps:
            raise ValueError(f"Expected {expected_steps} grid rows, got {row_count}")

        recordings = set(rows["recording_id"])

        if recordings != {recording_id}:
            raise ValueError(
                f"Extracted window crosses a recording boundary: {sorted(recordings)}"
            )
