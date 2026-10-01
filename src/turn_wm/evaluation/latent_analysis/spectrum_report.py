"""
Files, figures and report of the spectrum analysis (`spectrum.analyze_spectra`).

`write_spectrum` writes `summary.json`, the long spectrum table, the
figures and `report.md`: how concentrated each representation's variance is.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from turn_wm.evaluation.latent_analysis.artifacts import snapshot_source
from turn_wm.evaluation.latent_analysis.rendering import (
    INK,
    SECONDARY_INK,
    SERIES,
    SURFACE,
    close,
    style,
)
from turn_wm.evaluation.latent_analysis.snapshot import (
    GROUP_COLUMN,
    Snapshot,
)
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
        "source": snapshot_source(snapshot),
        "settings": {
            "centering": "per group (global mean for 'all')",
            "standardization": None,
            "dtype": "float64",
            "group_column": GROUP_COLUMN if snapshot.groups is not None else None,
            "cumulative_at": list(CUMULATIVE_AT),
            "variance_thresholds": list(VARIANCE_THRESHOLDS),
        },
        "representations": spectrum_summary(spectra),
        "figures": [f"{name}.png" for name in SPECTRUM_FIGURES],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(spectrum_report(summary), encoding="utf-8")
    pq.write_table(spectrum_table(spectra), output_dir / "spectrum.parquet")

    for name, figure in spectrum_figures(spectra).items():
        figure.savefig(output_dir / f"{name}.png", dpi=150, facecolor=SURFACE)
        close(figure)


SPECTRUM_FIGURES = ("cumulative_variance", "eigenvalue_spectrum")


def spectrum_report(summary: Mapping[str, Any]) -> str:
    """Question, results, interpretation, limitations, implication."""

    representations = summary["representations"]
    rows = [
        (
            "| representation | group | N | D | PC1 variance (isotropic 1/D) | "
            "components for 90% / 95% / 99% | effective rank | participation ratio |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]

    for name, groups in representations.items():
        for group, s in groups.items():
            rows.append(
                f"| {name} | {group} | {s['samples']:,} | {s['dim']} | "
                f"{s['cumulative_explained_variance']['1']:.3f} "
                f"({1 / s['dim']:.3f}) | {s['dimensions_for_90_percent']} / "
                f"{s['dimensions_for_95_percent']} / "
                f"{s['dimensions_for_99_percent']} | "
                f"{s['effective_rank_singular']:.1f} "
                f"({s['effective_rank_singular_fraction']:.0%} of D) | "
                f"{s['participation_ratio']:.1f} |"
            )

    return "\n".join(
        [
            "# Spectrum",
            "",
            "## Question",
            "",
            (
                "Is the representation collapsed, excessively anisotropic, or "
                "effectively much lower-dimensional than its nominal dimension D?"
            ),
            "",
            "## Results",
            "",
            (
                "Covariance spectrum of the centred, unstandardized rows (per "
                "corpus for the corpus groups). Effective rank: exp(entropy) of "
                "the normalized singular values, the training metric."
            ),
            "",
            *rows,
            "",
            "## Interpretation",
            "",
            *(
                f"- **{name}**: {_spectrum_reading(groups['all'])}"
                for name, groups in representations.items()
                if "all" in groups
            ),
            "",
            "## Limitations",
            "",
            (
                "- Second-order, linear geometry only: it says how variance "
                "spreads, not what the directions encode."
            ),
            (
                "- No standardization: directions with a large scale dominate. "
                "Numbers depend on the snapshot's sample and split."
            ),
            (
                "- The feature and latent D differ; compare their ranks as "
                "fractions of D."
            ),
            "",
            "## Implication",
            "",
            (
                "Read the PCA figures and the probes with the effective "
                "dimensionality, not D, in mind. A low effective rank alone does "
                "not show that information is lost: whether the representation "
                "keeps turn-taking information is tested by the probes."
            ),
            "",
        ]
    )


def _spectrum_reading(s: Mapping[str, Any]) -> str:
    """What one representation's 'all' spectrum says, from its numbers only."""

    dim, k99 = s["dim"], s["dimensions_for_99_percent"]

    if s["total_variance"] == 0:
        return "collapsed: every row is the same point (no variance)."

    if k99 == 1:
        return "collapsed onto one direction: one component holds 99% of the variance."

    pc1 = s["cumulative_explained_variance"]["1"]

    return (
        f"no trivial collapse (to a point or one direction): 99% of the "
        f"variance needs {k99} of {dim} directions "
        f"({k99 / dim:.0%}); effective rank {s['effective_rank_singular']:.1f} "
        f"({s['effective_rank_singular_fraction']:.0%} of D). Anisotropy: the "
        f"first component holds {pc1:.1%} of the variance, {pc1 * dim:.1f} times "
        "the share of an isotropic space."
    )


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

    if len(representations) > len(SERIES):
        raise ValueError(
            f"At most {len(SERIES)} representations per figure, "
            f"got {len(representations)}"
        )

    colors = dict(zip(representations, SERIES, strict=False))
    # Every representation has the same rows, so a group has one N.
    samples = {s.group: s.samples for s in spectra}

    figure = Figure(figsize=(4.2 * len(groups), 3.6), facecolor=SURFACE)
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
            color=SECONDARY_INK,
            fontsize=10,
        )
        ax.set_xlabel("Component", color=SECONDARY_INK)
        style(ax)

    axes[0].set_ylabel(ylabel, color=SECONDARY_INK)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=9)
    figure.suptitle(title, color=INK, fontsize=12)
    figure.tight_layout()

    return figure
