"""
Extract anchor-level representations of a trained world model.

The anchor step `t` of a sample is the last step of its context: the dataset
ends the context window on the anchor's own grid row. At that step the
extractor keeps, for every sample:

- `features`: the projector's input, i.e. the encoder features (cached or
  computed from raw observations);
- `latent`: the projected latent z_t;

and the anchor's metadata, including the action taken at `t`.

Observations reach the model through `trajectories()` and
`encode_trajectories()`, the same path as training, so this module does not
know whether a batch carries cached features or raw media. Further
representations (predicted latents, whole trajectories, other feature
streams) are added as new named tensors without changing the existing ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from turn_wm.data.dataset import ACTION_TO_ID, MASKED_ACTION_ID
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.training.lewm import Trajectories, encode_trajectories, trajectories

SCHEMA_VERSION = 1

FEATURES = "features"
LATENT = "latent"

_ID_TO_ACTION = {
    **{value: name for name, value in ACTION_TO_ID.items()},
    MASKED_ACTION_ID: "MASKED",
}

# Metadata copied from the collated batch, one value per sample.
_BATCH_COLUMNS = (
    "sample_id",
    "dataset",
    "recording_id",
    "anchor_idx",
    "anchor_time",
    "sample_class",
)


@dataclass(frozen=True)
class RepresentationSnapshot:
    """Named per-sample representations and their metadata, row-aligned."""

    representations: dict[str, torch.Tensor]  # name -> (N, ...)
    metadata: dict[str, list[Any]]  # column -> N values

    def __post_init__(self) -> None:
        if not self.representations:
            raise ValueError("A snapshot needs at least one representation")

        rows = {len(tensor) for tensor in self.representations.values()}
        rows |= {len(values) for values in self.metadata.values()}

        if len(rows) != 1:
            raise ValueError(
                "Every representation and metadata column must have the same "
                f"number of rows, got {sorted(rows)}"
            )

    @property
    def samples(self) -> int:
        return len(next(iter(self.representations.values())))


def anchor_representations(
    model: JEPA,
    batch: Trajectories,
) -> dict[str, torch.Tensor]:
    """Representations of each trajectory at its anchor, each (B, ·)."""

    features, latents = encode_trajectories(model, batch)
    anchor = batch.context_steps - 1

    return {
        FEATURES: features[:, anchor],
        LATENT: latents[:, anchor],
    }


def extract_snapshot(
    model: JEPA,
    batches: Iterable[dict[str, Any]],
    *,
    max_samples: int | None = None,
    device: torch.device | str = "cpu",
    representations: Callable[
        [JEPA, Trajectories], dict[str, torch.Tensor]
    ] = anchor_representations,
) -> RepresentationSnapshot:
    """Representations of the first `max_samples` samples of `batches`.

    `batches` are collated data batches, in the order to keep; `None` keeps
    every sample. `representations` maps the model and one batch of
    trajectories to named (B, ...) tensors (by default the anchor
    representations). The model runs in eval mode and float32.
    """

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    device = torch.device(device)
    model = model.to(device).eval()

    chunks: dict[str, list[torch.Tensor]] = {}
    metadata: dict[str, list[Any]] = {
        column: [] for column in (*_BATCH_COLUMNS, "action_id", "action")
    }
    collected = 0

    with torch.inference_mode():
        for batch in batches:
            if max_samples is not None and collected >= max_samples:
                break

            trajectory = _to_device(trajectories(batch), device)
            take = len(batch["sample_id"])

            if max_samples is not None:
                take = min(take, max_samples - collected)

            for name, tensor in representations(model, trajectory).items():
                # Floating tensors in float32; integer ones (ids) as they are.
                if tensor.is_floating_point():
                    tensor = tensor.float()

                chunks.setdefault(name, []).append(tensor[:take].cpu())

            for column in _BATCH_COLUMNS:
                values = batch[column][:take]
                metadata[column].extend(
                    values.tolist() if isinstance(values, torch.Tensor) else values
                )

            # The action taken at the anchor step itself.
            action_ids = batch["context_action"][:take, trajectory.context_steps - 1]
            metadata["action_id"].extend(action_ids.tolist())
            metadata["action"].extend(_action_names(action_ids.tolist()))

            collected += take

    if not chunks:
        raise ValueError("No samples were extracted")

    representations = {
        name: torch.cat(parts).contiguous() for name, parts in chunks.items()
    }

    for name, tensor in representations.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Extracted {name!r} contains non-finite values")

    return RepresentationSnapshot(representations=representations, metadata=metadata)


def write_snapshot(
    snapshot: RepresentationSnapshot,
    output_dir: Path,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write representations, metadata and manifest into a new directory.

    - `representations.safetensors`: one tensor per representation name;
    - `metadata.parquet`: one row per sample, aligned with the tensors;
    - `manifest.json`: schema, shapes and the provenance given.
    """

    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    save_file(
        snapshot.representations,
        str(output_dir / "representations.safetensors"),
    )

    pq.write_table(
        pa.table(snapshot.metadata),
        output_dir / "metadata.parquet",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "samples": snapshot.samples,
        "representations": {
            name: {
                "shape": list(tensor.shape[1:]),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for name, tensor in snapshot.representations.items()
        },
        "metadata_columns": list(snapshot.metadata),
        "provenance": dict(provenance or {}),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return output_dir


def _to_device(batch: Trajectories, device: torch.device) -> Trajectories:
    # Raw waveforms are moved by the encoder itself, as in training.
    if batch.features is None:
        return batch

    return replace(batch, features=batch.features.to(device))


def _action_names(action_ids: list[int]) -> list[str]:
    try:
        return [_ID_TO_ACTION[action_id] for action_id in action_ids]
    except KeyError as error:
        raise ValueError(f"Unknown action id at an anchor: {error.args[0]}") from None
