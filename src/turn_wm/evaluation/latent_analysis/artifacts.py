"""
Files an analysis writes: provenance of its input snapshot, and output checks.

Every analysis records the snapshot it read (`snapshot_source`) and refuses
to write into a non-empty directory (`require_empty`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from turn_wm.evaluation.latent_analysis.snapshot import Snapshot


def snapshot_source(snapshot: Snapshot) -> dict[str, Any]:
    """The snapshot an analysis read, with the hash of its representations."""

    representations_file = snapshot.path / "representations.safetensors"

    return {
        "snapshot": str(snapshot.path),
        "representations_sha256": sha256(representations_file),
        "samples": snapshot.manifest.get("samples"),
        "snapshot_provenance": snapshot.manifest.get("provenance"),
    }


def require_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
