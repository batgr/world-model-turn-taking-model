"""
Batch variable-length turn-taking samples for PyTorch training.

This module pads context sequences to the longest sample in the batch while
preserving true lengths and validity masks. Future sequences are fixed-length
in the current dataset contract and are stacked directly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from turn_wm.data.dataset import PAD_ACTION_ID, PAD_STATE_ID


def collate_turn_taking(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate variable-length temporal samples into a padded batch."""

    if not samples:
        raise ValueError("Cannot collate an empty batch")

    context_state = pad_sequence(
        [sample["context_state"] for sample in samples],
        batch_first=True,
        padding_value=PAD_STATE_ID,
    )

    context_action = pad_sequence(
        [sample["context_action"] for sample in samples],
        batch_first=True,
        padding_value=PAD_ACTION_ID,
    )

    context_valid = pad_sequence(
        [sample["context_valid"] for sample in samples],
        batch_first=True,
        padding_value=False,
    )

    context_lengths = torch.tensor(
        [sample["context_length"] for sample in samples],
        dtype=torch.long,
    )

    max_context_length = context_state.shape[1]

    positions = torch.arange(max_context_length).unsqueeze(0)

    context_mask = positions < context_lengths.unsqueeze(1)

    future_state = torch.stack([sample["future_state"] for sample in samples])

    future_action = torch.stack([sample["future_action"] for sample in samples])

    future_valid = torch.stack([sample["future_valid"] for sample in samples])

    batch = {
        "context_state": context_state,
        "context_action": context_action,
        "context_valid": context_valid,
        "context_mask": context_mask,
        "context_lengths": context_lengths,
        "future_state": future_state,
        "future_action": future_action,
        "future_valid": future_valid,
        "sample_id": [sample["sample_id"] for sample in samples],
        "dataset": [sample["dataset"] for sample in samples],
        "recording_id": [sample["recording_id"] for sample in samples],
        "anchor_idx": torch.tensor(
            [sample["anchor_idx"] for sample in samples],
            dtype=torch.long,
        ),
        "anchor_time": torch.tensor(
            [sample["anchor_time"] for sample in samples],
            dtype=torch.float32,
        ),
        "sample_class": [sample["sample_class"] for sample in samples],
    }

    has_features = "context_features" in samples[0]

    if any(("context_features" in sample) != has_features for sample in samples):
        raise ValueError("Cannot collate mixed cached-feature and plain samples")

    if has_features:
        # [B, C, D] and [B, F, D]; variable contexts are zero-padded in time
        # like context_state (context_mask marks the real rows).
        batch["context_features"] = pad_sequence(
            [sample["context_features"] for sample in samples],
            batch_first=True,
        )
        batch["future_features"] = torch.stack(
            [sample["future_features"] for sample in samples]
        )

    has_media = "context_media" in samples[0]

    if any(("context_media" in sample) != has_media for sample in samples):
        raise ValueError("Cannot collate mixed media and non-media samples")

    if has_media:
        batch["context_media"] = [sample["context_media"] for sample in samples]

        batch["future_media"] = [sample["future_media"] for sample in samples]

    return batch
