"""
Display written PCA, label and rollout-dynamics results: tables, then the
figures.

Only reads `summary.json` and the figures an analysis already wrote, so
showing never changes a result. Standard library only, plus IPython when it
is there: from a notebook kernel (even without the project's training
dependencies),

    import sys; sys.path.insert(0, "<repo>/src")
    from turn_wm.evaluation.latent_analysis.show import show_pca
    show_pca("<snapshot>/analysis/pca")
    show_labels("<snapshot>/analysis/labels")
    show_rollouts("<rollout-snapshot>/analysis/rollout_dynamics")

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


_LABEL_SECTIONS = (
    ("conversational_state", "Conversational state"),
    ("temporal_state", "Temporal state"),
    ("future", "Future structure"),
    ("nuisance", "Nuisance controls"),
)


def label_coverage(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "variable": row["variable"],
            "corpus": row["dataset"],
            "snapshot rows": row["n_snapshot"],
            "joined": row["n_joined"],
            "valid": row["n_valid"],
            "coverage": row["coverage_fraction"],
        }
        for row in summary["coverage"]
    ]


def label_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Per variable: strength (between-group variance fraction) and structure."""

    rows = []

    for name, variable in summary["variables"].items():
        row: dict[str, Any] = {"variable": name}

        for representation, by_condition in variable["representations"].items():
            for condition, metrics in by_condition.items():
                row[f"{representation} | {condition}"] = metrics.get(
                    "between_bin_variance_fraction",
                    metrics.get("between_variance_fraction"),
                )

        change = ((variable.get("features_to_latent") or {}).get("all") or {}).get(
            "change"
        )
        row["features→latent"] = change
        domain = (variable.get("domain_structure") or {}).get("latent") or {}
        row["latent domain"] = domain.get("class")
        rows.append(row)

    return rows


def show_labels(output_dir: Path | str, *, out: Callable[[str], None] = print) -> None:
    """Show a written label analysis, inline in a notebook when possible."""

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    tables = [
        ("Coverage", label_coverage(summary)),
        ("Between-group variance fraction", label_diagnostics(summary)),
    ]
    sections = [
        (title, [output_dir / path for path in summary["figures"].get(key, [])])
        for key, title in _LABEL_SECTIONS
    ]
    display = _notebook_display()

    if display is None:
        for title, rows in tables:
            out(f"{title}:")
            out(_rows_text(rows))
            out("")

        for title, paths in sections:
            if paths:
                out(f"{title}:")
                out("\n".join(f"  {path}" for path in paths))

        out(
            "\nFigures are shown inline when show_labels() runs in the notebook "
            "kernel itself (see turn_wm.evaluation.latent_analysis.show)."
        )
        return

    show, html_block, image = display

    for title, rows in tables:
        show(html_block(f"<h4>{html.escape(title)}</h4>" + _rows_html(rows)))

    for title, paths in sections:
        if paths:
            show(html_block(f"<h4>{html.escape(title)}</h4>"))

            for path in paths:
                show(image(filename=str(path)))


def rollout_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per (horizon, condition): the three measures and their counts."""

    rows = []

    for h, seconds in zip(
        summary["horizons_steps"], summary["horizons_s"], strict=True
    ):
        for condition, values in summary["metrics"][str(h)].items():
            rows.append(
                {
                    "horizon": f"{seconds:g} s",
                    "condition": condition,
                    "n": values["n"],
                    "recordings": values["n_recordings"],
                    "skill": _with_interval(values, "skill"),
                    "alignment": _with_interval(values, "displacement_alignment"),
                    "alignment rows valid": values["direction_defined_fraction"],
                    "movement ratio": _with_interval(values, "movement_ratio"),
                    "ONSET/OFFSET in future tokens": values["future_event_fraction"],
                }
            )

    return rows


def show_rollouts(
    output_dir: Path | str, *, out: Callable[[str], None] = print
) -> None:
    """Show a written rollout-dynamics analysis, inline in a notebook if possible."""

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    conditioned = any(
        summary["rollout"]["conditioned_on_ground_truth_future_actions"].values()
    )
    note = (
        "Rollout conditioned on ground-truth future action/event tokens: "
        f"{'yes' if conditioned else 'no'}. Intervals: 95% cluster bootstrap "
        "over recordings within each corpus."
    )
    rows = rollout_diagnostics(summary)
    paths = [output_dir / path for path in summary["figures"]]
    display = _notebook_display()

    if display is None:
        out(note)
        out(_rows_text(rows))
        out("\nFigures:")
        out("\n".join(f"  {path}" for path in paths))
        out(f"Report: {output_dir / 'report.md'}")
        out(
            "\nFigures are shown inline when show_rollouts() runs in the notebook "
            "kernel itself (see turn_wm.evaluation.latent_analysis.show)."
        )
        return

    show, html_block, image = display
    show(html_block(f"<p>{html.escape(note)}</p>" + _rows_html(rows)))

    for path in paths:
        show(image(filename=str(path)))


def _with_interval(values: dict[str, Any], name: str) -> str:
    value, interval = values[name], values[f"{name}_ci"]

    if value is None:
        return "n/a"

    if interval is None:
        return f"{value:.3f}"

    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"


def _rows_text(rows: list[dict[str, Any]]) -> str:
    """One line per row, one column per key."""

    if not rows:
        return "(none)"

    columns = list(dict.fromkeys(key for row in rows for key in row))
    cells = [[_format(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(line[i]) for line in cells))
        for i, column in enumerate(columns)
    ]

    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(line, widths, strict=True))
        for line in [columns, *cells]
    )


def _rows_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>(none)</p>"

    columns = list(dict.fromkeys(key for row in rows for key in row))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(row.get(c)))}</td>" for c in columns)
        + "</tr>"
        for row in rows
    )

    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


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
