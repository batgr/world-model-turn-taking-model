"""Display persisted analysis reports and figures without recomputing them."""

from __future__ import annotations

import html
import importlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


def show_spectrum(
    output_dir: Path | str, *, out: Callable[[str], None] = print
) -> None:
    _show(output_dir, out=out)


def show_pca(output_dir: Path | str, *, out: Callable[[str], None] = print) -> None:
    _show(output_dir, out=out)


def show_labels(output_dir: Path | str, *, out: Callable[[str], None] = print) -> None:
    _show(output_dir, out=out)


def show_rollouts(
    output_dir: Path | str, *, out: Callable[[str], None] = print
) -> None:
    _show(output_dir, out=out)


def show_probes(output_dir: Path | str, *, out: Callable[[str], None] = print) -> None:
    _show(output_dir, out=out)


def show_concepts(
    output_dir: Path | str, *, out: Callable[[str], None] = print
) -> None:
    _show(output_dir, out=out)


def show_action_ablation(
    output_dir: Path | str, *, out: Callable[[str], None] = print
) -> None:
    _show(output_dir, out=out)


def _show(output_dir: Path | str, *, out: Callable[[str], None]) -> None:
    """Read one analysis's report and figure manifest; never recompute it."""

    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    report_path = output_dir / "report.md"
    report = (
        report_path.read_text(encoding="utf-8")
        if report_path.is_file()
        else f"Missing report: {report_path}"
    )
    figures = [output_dir / name for name in _figure_names(summary.get("figures", []))]
    notebook = _notebook_display()

    if notebook is None:
        out(report.rstrip())
        out("\nFigures:")
        out(
            "\n".join(
                f"  {path}{'' if path.is_file() else ' (missing)'}" for path in figures
            )
            or "  (none recorded)"
        )
        out(f"Report: {report_path}{'' if report_path.is_file() else ' (missing)'}")
        return

    display, html_block, image = notebook
    display(html_block(f"<pre>{html.escape(report)}</pre>"))

    for path in figures:
        if path.is_file():
            display(image(filename=str(path)))
        else:
            display(html_block(f"<p>Missing figure: {html.escape(str(path))}</p>"))


def _figure_names(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _figure_names(child)
    elif isinstance(value, list):
        for child in value:
            yield from _figure_names(child)


def _notebook_display():
    try:
        ipython = importlib.import_module("IPython")
        display_module = importlib.import_module("IPython.display")
    except ImportError:
        return None

    return (
        None
        if ipython.get_ipython() is None
        else (display_module.display, display_module.HTML, display_module.Image)
    )
