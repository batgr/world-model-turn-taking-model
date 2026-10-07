"""
Context + future trajectories: the model inputs of one batch.

`trajectories` assembles them from a collated data batch; `encode_trajectories`
is the only place that knows where observations come from (cached encoder
features or raw audio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from turn_wm.models.lewm.jepa import JEPA


@dataclass(frozen=True)
class Trajectories:
    """Model inputs for a batch of context + future trajectories.

    The observations are either precomputed encoder `features` (B, T, D) or
    raw `waveforms` with their `sample_rates`, never both.
    """

    actions: torch.Tensor  # (B, T)
    context_steps: int
    future_steps: int
    features: torch.Tensor | None = None  # (B, T, D), context then future
    waveforms: list[torch.Tensor] | None = None  # each (channels, samples)
    sample_rates: list[int] | None = None

    def __post_init__(self) -> None:
        has_features = self.features is not None
        has_audio = self.waveforms is not None and self.sample_rates is not None

        if has_features == has_audio or (
            not has_audio and (self.waveforms or self.sample_rates)
        ):
            raise ValueError(
                "Trajectories need exactly one observation source: features, or "
                "waveforms with sample_rates"
            )

    @property
    def total_steps(self) -> int:
        return self.context_steps + self.future_steps


def trajectories(batch: dict[str, Any]) -> Trajectories:
    """Assemble context + future trajectories from a collated data batch.

    Cached features (`context_features`/`future_features`) are used when the
    batch has them; otherwise the audio of the media windows.
    """

    has_features = "context_features" in batch

    if not has_features and "context_media" not in batch:
        raise ValueError(
            "Batch has no observations; build the dataset with a feature_store or "
            "media_roots"
        )

    lengths = batch["context_lengths"]

    if not bool((lengths == lengths[0]).all()):
        raise ValueError(
            "Trajectories need one context length per batch; build the training "
            "dataset with training_window(cfg)"
        )

    context_steps = int(lengths[0])
    future_steps = int(batch["future_action"].shape[1])

    actions = torch.cat(
        [batch["context_action"][:, :context_steps], batch["future_action"]], dim=1
    )

    if has_features:
        return Trajectories(
            actions=actions,
            context_steps=context_steps,
            future_steps=future_steps,
            # Cached features are float16; float32 before any op, as CPU
            # autocast (bf16-mixed) rejects float16 inputs.
            features=torch.cat(
                [
                    batch["context_features"][:, :context_steps].float(),
                    batch["future_features"].float(),
                ],
                dim=1,
            ),
        )

    waveforms = []
    sample_rates = []

    for index, (context, future) in enumerate(
        zip(batch["context_media"], batch["future_media"], strict=True)
    ):
        if context.audio is None or future.audio is None:
            raise ValueError(
                f"Sample {batch['sample_id'][index]} has no audio; "
                "LeWM trajectories need audio in every window"
            )

        if context.audio.sample_rate != future.audio.sample_rate:
            raise ValueError("Context and future audio must share a sample rate")

        # The windows are contiguous: context ends where the future starts.
        waveforms.append(
            torch.cat([context.audio.waveform, future.audio.waveform], dim=1)
        )
        sample_rates.append(context.audio.sample_rate)

    return Trajectories(
        actions=actions,
        context_steps=context_steps,
        future_steps=future_steps,
        waveforms=waveforms,
        sample_rates=sample_rates,
    )


def encode_trajectories(
    model: JEPA, batch: Trajectories
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encoder features and projected latents of every step, each (B, T, ·).

    The only place that knows where observations come from: cached encoder
    features are used as they are, raw audio goes through the encoder first.
    Everything downstream sees the same features and latents.
    """

    if batch.features is not None:
        features = batch.features
    else:
        # One encoder frame per grid step: frame k is step k. Frames past the
        # last step come from end padding of shorter windows.
        assert batch.waveforms is not None and batch.sample_rates is not None
        features = model.encode_features(batch.waveforms, batch.sample_rates)

        if features.size(1) < batch.total_steps:
            raise ValueError(
                f"The encoder gave {features.size(1)} frames for "
                f"{batch.total_steps} grid steps of audio"
            )

        features = features[:, : batch.total_steps]

    return features, model.project_features(features)
