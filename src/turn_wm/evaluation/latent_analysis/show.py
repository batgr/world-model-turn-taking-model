"""
Display written PCA results: a diagnostics table, then the figures.

Only reads `summary.json` and the figures an analysis already wrote, so
showing never changes a result. Standard library only, plus IPython when it
is there: from a notebook kernel (even without the project's training
dependencies),

    import sys; sys.path.insert(0, "<repo>/src")
    from turn_wm.evaluation.latent_analysis.show import show_pca
    show_pca("<snapshot>/analysis/pca")

renders inline. Elsewhere, including `!turn-wm ... --show` (a subprocess,
which cannot draw in the notebook), the table is printed as text with the
figure paths.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

_SECTIONS = (
    ("dataset", "Dataset structure"),
    ("action", "Action structure"),
    ("action_by_dataset", "Action structure within each dataset"),
)


def pca_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per representation: the main PCA, dataset and action numbers."""

    rows = []

    for name, result in summary["representations"].items():
        dataset = result.get("dataset") or {}
        row = {
            "representation": name,
            "D": result["dim"],
            "PC1": result["pca"]["pc1_explained_variance"],
            "PC2": result["pca"]["pc2_explained_variance"],
            "dataset distance / within RMS": dataset.get(
                "centroid_distance_over_pooled_within_rms"
            ),
            "dataset between/within": dataset.get("between_to_within_variance_ratio"),
            "dataset silhouette": dataset.get("silhouette"),
        }

        for condition, structure in (result.get("action") or {}).items():
            row[f"action silhouette | {condition}"] = structure["silhouette"]

        rows.append(row)

    return rows


def show_pca(output_dir: Path | str, *, out: Callable[[str], None] = print) -> None:
    """Show a written PCA analysis, inline in a notebook when possible."""

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = pca_diagnostics(summary)
    sections = [
        (title, [output_dir / path for path in summary["figures"].get(key, [])])
        for key, title in _SECTIONS
    ]

    display = _notebook_display()

    if display is None:
        out(_text_table(rows))

        for title, paths in sections:
            if paths:
                out(f"\n{title}:")
                out("\n".join(f"  {path}" for path in paths))

        out(
            "\nFigures are shown inline when show_pca() runs in the notebook "
            "kernel itself (see turn_wm.evaluation.latent_analysis.show)."
        )
        return

    show, html_block, image = display
    show(html_block(_html_table(rows)))

    for title, paths in sections:
        if paths:
            show(html_block(f"<h4>{html.escape(title)}</h4>"))

            for path in paths:
                show(image(filename=str(path)))


def _notebook_display():
    """IPython's display, HTML and Image when running in a kernel, else None."""

    try:
        from IPython import get_ipython
        from IPython.display import HTML, Image, display
    except ImportError:
        return None

    if get_ipython() is None:
        return None

    return display, HTML, Image


def _format(value: Any) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def _text_table(rows: list[dict[str, Any]]) -> str:
    """One line per metric, one column per representation: stays narrow."""

    metrics = list(dict.fromkeys(key for row in rows for key in row))
    cells = [[_format(row.get(metric)) for metric in metrics] for row in rows]
    label_width = max(len(metric) for metric in metrics)
    widths = [max(len(value) for value in column) for column in cells]

    return "\n".join(
        f"{metric:<{label_width}}  "
        + "  ".join(
            column[i].rjust(width) for column, width in zip(cells, widths, strict=True)
        )
        for i, metric in enumerate(metrics)
    )


def _html_table(rows: list[dict[str, Any]]) -> str:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    header = "".join(f"<th>{html.escape(row['representation'])}</th>" for row in rows)
    body = "".join(
        f"<tr><th style='text-align:left'>{html.escape(column)}</th>"
        + "".join(
            f"<td style='text-align:right'>{html.escape(_format(row.get(column)))}</td>"
            for row in rows
        )
        + "</tr>"
        for column in columns
        if column != "representation"
    )

    return (
        f"<table><thead><tr><th></th>{header}</tr></thead><tbody>{body}</tbody></table>"
    )
