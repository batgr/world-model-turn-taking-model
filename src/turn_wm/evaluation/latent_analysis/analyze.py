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

import importlib.util
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from turn_wm.evaluation.latent_analysis.artifacts import require_empty, snapshot_source
from turn_wm.evaluation.latent_analysis.label_report import write_labels
from turn_wm.evaluation.latent_analysis.label_source import (
    CorpusLabelSource,
    hub_label_sources,
)
from turn_wm.evaluation.latent_analysis.label_structure import DEFAULT_BALANCED_CAP
from turn_wm.evaluation.latent_analysis.pca import (
    DEFAULT_MAX_PLOT_SAMPLES,
    DEFAULT_SILHOUETTE_SAMPLES,
)
from turn_wm.evaluation.latent_analysis.pca_report import PCA, write_pca
from turn_wm.evaluation.latent_analysis.snapshot import (
    describe_snapshot,
    read_snapshot,
)
from turn_wm.evaluation.latent_analysis.spectrum_report import SPECTRUM, write_spectrum
from turn_wm.progress import log

LABELS = "labels"
ANALYSES = (SPECTRUM, PCA, LABELS)
# Run without --analysis: those reading the snapshot alone. The label
# analysis also reads the Hub, so it runs only when asked for.
DEFAULT_ANALYSES = (SPECTRUM, PCA)


def analyze_snapshot(
    path: Path,
    *,
    analyses: Sequence[str] = DEFAULT_ANALYSES,
    output_root: Path | None = None,
    silhouette_samples: int = DEFAULT_SILHOUETTE_SAMPLES,
    max_plot_samples: int = DEFAULT_MAX_PLOT_SAMPLES,
    balanced_cap: int = DEFAULT_BALANCED_CAP,
    labels_revision: str | None = None,
    label_sources: Mapping[str, CorpusLabelSource] | None = None,
) -> dict[str, Path]:
    """Run `analyses` on a snapshot; return each analysis's output directory.

    The snapshot itself is only read. The label analysis reads the label
    sidecars of the snapshot's dataset revision from the Hub (or
    `labels_revision`, explicitly), unless `label_sources` are given.
    """

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

    if "rollout" in (snapshot.manifest.get("provenance") or {}):
        raise ValueError(
            f"{snapshot.path} is a rollout snapshot; analyze it with "
            "`turn-wm analyze-rollouts`"
        )

    output_root = (
        snapshot.path / "analysis" if output_root is None else Path(output_root)
    )
    outputs = {name: output_root / name for name in dict.fromkeys(analyses)}

    for output_dir in outputs.values():
        require_empty(output_dir)

    start = time.perf_counter()
    log(f"analyze-latents: {describe_snapshot(snapshot)}")

    for name, output_dir in outputs.items():
        log(f"analyze-latents: {name} -> {output_dir}")

        if name == SPECTRUM:
            write_spectrum(snapshot, output_dir)
        elif name == PCA:
            write_pca(
                snapshot,
                output_dir,
                silhouette_samples=silhouette_samples,
                max_plot_samples=max_plot_samples,
            )
        elif name == LABELS:
            write_labels(
                snapshot.representations,
                snapshot.metadata,
                output_dir,
                source=snapshot_source(snapshot),
                sources=(
                    label_sources
                    if label_sources is not None
                    else hub_label_sources(
                        snapshot.manifest.get("provenance") or {},
                        labels_revision=labels_revision,
                    )
                ),
                seed=snapshot.seed,
                silhouette_samples=silhouette_samples,
                balanced_cap=balanced_cap,
                max_plot_samples=max_plot_samples,
            )

    log(f"analyze-latents: done in {time.perf_counter() - start:.0f}s")

    return outputs
