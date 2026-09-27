"""
The TurnBench DEV report: Mimi vs current z_t vs predicted [zpred_A ; zpred_B].

Official metrics only (recall, fp_rate, latency p10/p50/p90 per task), at
each condition's own DEV operating point (highest recall at fp_rate <= 0.1).
Observation, interpretation and limitations are kept apart; the
interpretation only compares the numbers above it and claims nothing they
do not show.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from turn_wm.evaluation.turnbench.extract import CONDITIONS, CURRENT, MIMI, PREDICTED

NAMES = {
    MIMI: "Mimi features",
    CURRENT: "V1 current z_t",
    PREDICTED: "V1 predicted [zpred_A ; zpred_B]",
}
TASK_NAMES = {"eot": "EOT", "int": "INT"}

QUESTIONS = """\
## Questions

1. Does V1's current latent z_t keep enough information to match or improve on Mimi?
2. Does the one-step predicted state improve on z_t?
3. Does it reduce EOT and/or INT latency at a comparable FP budget?
"""

LIMITATIONS = """\
## Limitations

- V1 was trained on ego / focal action semantics; TurnBench has two symmetric,
  isolated speaker channels and no ego.
- zpred_A / zpred_B are action-conditioned predictions of one scene latent, not
  learned per-speaker world states.
- TRAIN supervision is single-annotator (annotator A); the DEV gold is the
  official 2-of-3 consensus.
- Domain shift: V1 learned on EgoCom / Ego4D egocentric audio; the TurnBench
  scene is a digital sum of studio booth channels.
- Operating points are selected and reported on the same DEV set, as
  TurnBench's baselines are; DEV numbers are therefore optimistic.
"""


def turnbench_report(
    results: Mapping[str, Mapping[str, Any]], context: Mapping[str, Any]
) -> str:
    """Markdown report from each condition's `scores.json` content."""

    rows = []

    for task, task_name in TASK_NAMES.items():
        for name in CONDITIONS:
            score = results[name][task]
            latency = score["latency_ms"]
            rows.append(
                f"| {task_name} | {NAMES[name]} | {_fmt(score['threshold'], 4)} | "
                f"{_fmt(score['recall'])} | {_fmt(score['fp_rate'])} | "
                f"{_fmt(latency['p10'], 0)} / {_fmt(latency['p50'], 0)} / "
                f"{_fmt(latency['p90'], 0)} |"
            )

    setup = (
        "## Setup\n\n"
        f"- V1 checkpoint sha256 `{context['checkpoint_sha256']}`; TurnBench "
        f"scorer `{context['scorer_revision']}`; DEV `{context['dev_revision']}`; "
        f"TRAIN `{context['train_revision']}`.\n"
        "- Matched heads (same architecture, loss, schedule, eligible frames) "
        "trained on TRAIN only, with a whole-conversation train/validation "
        f"split (seed {context['seed']}).\n"
        "- Each condition's EOT and INT thresholds are chosen on DEV, "
        "independently: highest recall at fp_rate <= 0.1 (TurnBench's baseline "
        "rule); the same DEV then reports the scores.\n"
    )
    observation = "\n".join(
        [
            "## Observation",
            "",
            (
                "| task | representation | threshold | recall | fp_rate | "
                "latency ms p10 / p50 / p90 |"
            ),
            "|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )
    interpretation = "\n".join(
        [
            "## Interpretation",
            "",
            *_comparisons(results),
            "",
            (
                "These are single DEV operating points without confidence "
                "intervals; a small difference is not evidence of a real one."
            ),
            "",
        ]
    )

    title = "# TurnBench DEV: V1 representations\n"

    return (
        f"{title}\n{QUESTIONS}\n{setup}\n{observation}\n{interpretation}\n{LIMITATIONS}"
    )


def _comparisons(results: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = []
    pairs = (((CURRENT, MIMI), "1"), ((PREDICTED, CURRENT), "2-3"))

    for task, task_name in TASK_NAMES.items():
        for (a, b), question in pairs:
            left, right = results[a][task], results[b][task]
            lines.append(
                f"- {task_name}, {NAMES[a]} vs {NAMES[b]} (Q{question}): recall "
                f"{_delta(left['recall'], right['recall'])}, fp_rate "
                f"{_delta(left['fp_rate'], right['fp_rate'])}, latency p50 "
                f"{_delta(left['latency_ms']['p50'], right['latency_ms']['p50'], 0)} ms."
            )

    return lines


def _delta(a: float | None, b: float | None, digits: int = 3) -> str:
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return "not comparable (no operating point or no detection)"

    return f"{a:.{digits}f} vs {b:.{digits}f} ({a - b:+.{digits}f})"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"

    return f"{value:.{digits}f}"
