"""
A representation snapshot (`extract.write_snapshot`), as the analyses read it.

`read_snapshot` loads the representations, the per-row metadata and the
manifest of a snapshot directory; nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

# Metadata column whose values define the per-group analyses.
GROUP_COLUMN = "dataset"


@dataclass(frozen=True)
class Snapshot:
    """The contents of a snapshot directory."""

    path: Path
    representations: dict[str, torch.Tensor]
    metadata: dict[str, list[Any]]  # column -> one value per row
    manifest: dict[str, Any]

    @property
    def groups(self) -> list[str] | None:
        """The per-group label of each row (its corpus), if recorded."""

        if GROUP_COLUMN not in self.metadata:
            return None

        return [str(value) for value in self.metadata[GROUP_COLUMN]]

    @property
    def seed(self) -> int:
        """The seed the snapshot's samples were drawn with (0 if unrecorded)."""

        sampling = (self.manifest.get("provenance") or {}).get("sampling") or {}
        seed = sampling.get("seed")

        return 0 if seed is None else int(seed)


def read_snapshot(path: Path) -> Snapshot:
    """Read representations, metadata and manifest; nothing is written."""

    path = Path(path).expanduser()
    files = {
        name: path / name
        for name in (
            "representations.safetensors",
            "metadata.parquet",
            "manifest.json",
        )
    }

    for file in files.values():
        if not file.is_file():
            raise FileNotFoundError(f"Not a representation snapshot, missing {file}")

    return Snapshot(
        path=path,
        representations=load_file(files["representations.safetensors"]),
        metadata=pq.read_table(files["metadata.parquet"]).to_pydict(),
        manifest=json.loads(files["manifest.json"].read_text(encoding="utf-8")),
    )


def describe_snapshot(snapshot: Snapshot) -> str:
    """One log line: the snapshot, its rows, split and checkpoint."""

    provenance = snapshot.manifest.get("provenance") or {}
    checkpoint = provenance.get("checkpoint") or {}
    split = (provenance.get("data") or {}).get("split")
    rows = len(next(iter(snapshot.metadata.values()), []))

    return (
        f"{snapshot.path} ({rows:,} anchors, {split} split, checkpoint "
        f"{checkpoint.get('filename')} step {checkpoint.get('global_step')})"
    )
