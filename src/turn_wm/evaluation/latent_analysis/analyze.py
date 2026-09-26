"""
Analyze an extracted representation snapshot (`extract.write_snapshot`).

The snapshot directory is the only input: no checkpoint, dataset or feature
cache is loaded. Each analysis runs in three stages, so its results can be
logged elsewhere (e.g. W&B) without recomputing them:

    computation (spectrum.py)  ->  structured results (summary, long table)
                               ->  local rendering (JSON, Parquet, figures)

Results go to `<snapshot>/analysis/<analysis>/` unless another output root
is given. Figures need matplotlib (`uv sync --extra analysis`); nothing else
does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from turn_wm.evaluation.latent_analysis.spectrum import (
    CUMULATIVE_AT,
    VARIANCE_THRESHOLDS,
    Spectrum,
    analyze_spectra,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SCHEMA_VERSION = 1

SPECTRUM = "spectrum"
ANALYSES = (SPECTRUM,)

# Metadata column whose values define the per-group analyses.
GROUP_COLUMN = "dataset"

# Reference palette: categorical slots in fixed order, then chart ink.
_SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


@dataclass(frozen=True)
class Snapshot:
    """The contents of a snapshot directory."""

    path: Path
    representations: dict[str, torch.Tensor]
    groups: list[str] | None
    manifest: dict[str, Any]


def read_snapshot(path: Path) -> Snapshot:
    """Read representations, the group column (if any) and the manifest."""

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

    groups = None

    if GROUP_COLUMN in pq.read_schema(files["metadata.parquet"]).names:
        table = pq.read_table(files["metadata.parquet"], columns=[GROUP_COLUMN])
        groups = [str(value) for value in table[GROUP_COLUMN].to_pylist()]

    return Snapshot(
        path=path,
        representations=load_file(files["representations.safetensors"]),
        groups=groups,
        manifest=json.loads(files["manifest.json"].read_text(encoding="utf-8")),
    )


def analyze_snapshot(
    path: Path,
    *,
    analyses: Sequence[str] = ANALYSES,
    output_root: Path | None = None,
) -> dict[str, Path]:
    """Run `analyses` on a snapshot; return each analysis's output directory."""

    unknown = [name for name in analyses if name not in ANALYSES]

    if unknown:
        raise ValueError(f"Unknown analyses {unknown}; expected some of {ANALYSES}")

    # Before any computation: figures are part of every analysis's output.
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "Figures need matplotlib, an optional dependency. Run "
            "`uv sync --extra analysis`."
        )

    snapshot = read_snapshot(path)
    output_root = (
        snapshot.path / "analysis" if output_root is None else Path(output_root)
    )
    outputs = {}

    for name in dict.fromkeys(analyses):
        output_dir = output_root / name
        _require_empty(output_dir)

        if name == SPECTRUM:
            write_spectrum(snapshot, output_dir)

        outputs[name] = output_dir

    return outputs


# ---------------------------------------------------------------------------
# Spectrum: structured results
# ---------------------------------------------------------------------------


def spectrum_summary(spectra: Sequence[Spectrum]) -> dict[str, Any]:
    """Aggregated metrics, keyed representation -> group."""

    summary: dict[str, dict[str, Any]] = {}

    for spectrum in spectra:
        summary.setdefault(spectrum.representation, {})[spectrum.group] = (
            spectrum.summary()
        )

    return summary


def spectrum_table(spectra: Sequence[Spectrum]) -> pa.Table:
    """One row per (representation, group, component)."""

    return pa.Table.from_pylist(
        [row for spectrum in spectra for row in spectrum.rows()],
        schema=pa.schema(
            [
                ("representation", pa.string()),
                ("group", pa.string()),
                ("component", pa.int64()),
                ("eigenvalue", pa.float64()),
                ("singular_value", pa.float64()),
                ("explained_variance_ratio", pa.float64()),
                ("cumulative_explained_variance", pa.float64()),
            ]
        ),
    )


def write_spectrum(snapshot: Snapshot, output_dir: Path) -> None:
    """Compute the spectra of every representation and write the results."""

    spectra = analyze_spectra(snapshot.representations, groups=snapshot.groups)

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": SPECTRUM,
        "source": _source(snapshot),
        "settings": {
            "centering": "per group (global mean for 'all')",
            "standardization": None,
            "dtype": "float64",
            "group_column": GROUP_COLUMN if snapshot.groups is not None else None,
            "cumulative_at": list(CUMULATIVE_AT),
            "variance_thresholds": list(VARIANCE_THRESHOLDS),
        },
        "representations": spectrum_summary(spectra),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pq.write_table(spectrum_table(spectra), output_dir / "spectrum.parquet")

    for name, figure in spectrum_figures(spectra).items():
        figure.savefig(output_dir / f"{name}.png", dpi=150, facecolor=_SURFACE)
        _close(figure)


# ---------------------------------------------------------------------------
# Spectrum: local rendering
# ---------------------------------------------------------------------------


def spectrum_figures(spectra: Sequence[Spectrum]) -> dict[str, Figure]:
    """The cumulative explained variance and the eigenvalue spectrum.

    One panel per group, one line per representation (colored by its fixed
    slot), components on a log axis so spaces of different D compare.
    """

    return {
        "cumulative_variance": _spectrum_figure(
            spectra,
            values=lambda s: s.cumulative_explained_variance,
            ylabel="Cumulative explained variance",
            title="Cumulative explained variance",
            log_y=False,
        ),
        "eigenvalue_spectrum": _spectrum_figure(
            spectra,
            values=lambda s: s.explained_variance_ratio,
            ylabel="Eigenvalue / total variance",
            title="Covariance eigenvalue spectrum (normalized)",
            log_y=True,
        ),
    }


def _spectrum_figure(
    spectra: Sequence[Spectrum],
    *,
    values,
    ylabel: str,
    title: str,
    log_y: bool,
) -> Figure:
    from matplotlib.figure import Figure

    representations = list(dict.fromkeys(s.representation for s in spectra))
    groups = list(dict.fromkeys(s.group for s in spectra))

    if len(representations) > len(_SERIES):
        raise ValueError(
            f"At most {len(_SERIES)} representations per figure, "
            f"got {len(representations)}"
        )

    colors = dict(zip(representations, _SERIES, strict=False))
    # Every representation has the same rows, so a group has one N.
    samples = {s.group: s.samples for s in spectra}

    figure = Figure(figsize=(4.2 * len(groups), 3.6), facecolor=_SURFACE)
    axes = figure.subplots(1, len(groups), sharey=True, squeeze=False)[0]

    for ax, group in zip(axes, groups, strict=True):
        for spectrum in spectra:
            if spectrum.group != group:
                continue

            y = values(spectrum)
            x = torch.arange(1, spectrum.dim + 1)

            if log_y:
                keep = y > 0
                x, y = x[keep], y[keep]

            ax.plot(
                x.numpy(),
                y.numpy(),
                color=colors[spectrum.representation],
                linewidth=1.5,
                label=f"{spectrum.representation} (D={spectrum.dim})",
            )

        ax.set_xscale("log")

        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.02)

        ax.set_title(
            f"{group} (N={samples[group]:,})",
            color=_SECONDARY_INK,
            fontsize=10,
        )
        ax.set_xlabel("Component", color=_SECONDARY_INK)
        _style(ax)

    axes[0].set_ylabel(ylabel, color=_SECONDARY_INK)
    axes[0].legend(frameon=False, labelcolor=_INK, fontsize=9)
    figure.suptitle(title, color=_INK, fontsize=12)
    figure.tight_layout()

    return figure


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, which="major", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelcolor=_SECONDARY_INK, labelsize=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)


def _close(figure: Figure) -> None:
    # Figures built from `Figure` are not tracked by pyplot; drop the canvas.
    figure.clear()


# ---------------------------------------------------------------------------
# Provenance and output
# ---------------------------------------------------------------------------


def _source(snapshot: Snapshot) -> dict[str, Any]:
    representations_file = snapshot.path / "representations.safetensors"

    return {
        "snapshot": str(snapshot.path),
        "representations_sha256": _sha256(representations_file),
        "samples": snapshot.manifest.get("samples"),
        "snapshot_provenance": snapshot.manifest.get("provenance"),
    }


def _require_empty(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
