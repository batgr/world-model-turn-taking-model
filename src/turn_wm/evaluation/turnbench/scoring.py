"""
Official TurnBench DEV scoring of a head's continuous scores.

Per condition and task (EOT, INT), independently:

    per-frame probabilities (official `ProbsFile`, 10 Hz canonical grid)
        -> official threshold sweep (`turnbench.sweep.sweep`)
        -> official operating point: highest recall at fp_rate <= 0.1
        -> official commit rule (`commit_events`, rising edge, 2 s refractory)
        -> predictions-dev.json, scored by `turnbench.score.score_submission`

Frame i of the grid is our slot i; its scores exist at the slot end plus the
Mimi resampler's lookahead, so every committed time gets that lookahead added
(clamped to the audio's end) before the final scoring: the reported latencies
are those of the causal availability time. The sweep itself uses the grid's
frame ends, which differ by that lookahead only (0.25 ms at 48 kHz). The
+0.1 s prediction horizon of zpred is not added: it predicts the future and
does not wait for it. Frames without a prediction (the first W - 1) score 0
for every condition, so no condition can fire there.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from turnbench.durations import load_durations
from turnbench.score import score_submission
from turnbench.submission import (
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)
from turnbench.sweep import (
    ConversationProbs,
    ProbsFile,
    SpeakerProbs,
    commit_events,
    operating_point,
    sweep,
    validate_probs,
)

from turn_wm.evaluation.turnbench.extract import condition
from turn_wm.evaluation.turnbench.heads import CausalHead
from turn_wm.evaluation.turnbench.labels import OUTPUTS
from turn_wm.evaluation.turnbench.timing import CONTROL_RATE_HZ

FP_BUDGET = 0.1
TASKS = {
    "eot": ("eot_speaker_1", "eot_speaker_2"),
    "int": ("int_speaker_1", "int_speaker_2"),
}


@torch.no_grad()
def dev_scores(
    head: CausalHead,
    extraction_dir: Path,
    conversation_ids: Sequence[str],
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    """(K, 4) probabilities per conversation; 0 where no prediction exists."""

    scores = {}

    for conversation_id in conversation_ids:
        tensors = load_file(
            str(Path(extraction_dir) / f"{conversation_id}.safetensors")
        )
        inputs = condition(tensors, name).nan_to_num(0.0)[None]
        probabilities = head(inputs)[0].sigmoid()
        scores[conversation_id] = probabilities * tensors["prediction_valid"][:, None]

    return scores


def probs_file(task: str, scores: Mapping[str, torch.Tensor]) -> ProbsFile:
    """The official per-frame probabilities file of one task."""

    columns = [OUTPUTS.index(output) for output in TASKS[task]]

    return ProbsFile(
        schema_version=1,
        task=task,  # type: ignore[arg-type]
        frame_rate_hz=CONTROL_RATE_HZ,
        probs=[
            ConversationProbs(
                conversation_id=conversation_id,
                speaker_1=SpeakerProbs(prob=probabilities[:, columns[0]].tolist()),
                speaker_2=SpeakerProbs(prob=probabilities[:, columns[1]].tolist()),
            )
            for conversation_id, probabilities in scores.items()
        ],
    )


def score_condition(
    scores: Mapping[str, torch.Tensor],
    *,
    lookahead_s: Mapping[str, float],
    dataset,
    durations: Mapping[str, float] | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Sweep, pick the operating points, write and score predictions-dev.json."""

    durations = dict(load_durations("dev") if durations is None else durations)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {task: probs_file(task, scores) for task in TASKS}
    chosen: dict[str, Any] = {}

    for task, probs in files.items():
        validate_probs(probs, durations)
        (output_dir / f"probs-{task}.json").write_text(probs.model_dump_json() + "\n")
        rows = sweep(probs, dataset)
        point = operating_point(rows, fp_budget=FP_BUDGET)
        chosen[task] = None if point is None else asdict(point)
        (output_dir / f"sweep-{task}.json").write_text(
            json.dumps([asdict(row) for row in rows]) + "\n"
        )

    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            ConversationPrediction(
                conversation_id=conversation_id,
                **{
                    f"speaker_{speaker}": SpeakerEvents(
                        **{
                            key: _committed(
                                files[task].by_conversation()[conversation_id],
                                speaker,
                                chosen[task],
                                lookahead_s[conversation_id],
                                durations[conversation_id],
                            )
                            for task, key in (("eot", "eot"), ("int", "interruption"))
                        }
                    )
                    for speaker in (1, 2)
                },
            )
            for conversation_id in scores
        ],
    )
    (output_dir / "predictions-dev.json").write_text(
        submission.model_dump_json(indent=2) + "\n"
    )
    final = score_submission(submission, dataset)
    result = {"fp_budget": FP_BUDGET, "operating_points": chosen}

    for task, score in (("eot", final.task_eot), ("int", final.task_int)):
        latency = score.latency()
        result[task] = {
            "threshold": None if chosen[task] is None else chosen[task]["theta"],
            "recall": score.recall,
            "fp_rate": score.fp_rate,
            "latency_ms": {"p10": latency.p10, "p50": latency.p50, "p90": latency.p90},
            "tp": score.tp,
            "fn": score.fn,
            "fp": score.fp,
            "tn": score.tn,
        }

    (output_dir / "scores.json").write_text(json.dumps(result, indent=2) + "\n")

    return result


def _committed(
    probs: ConversationProbs,
    speaker: int,
    point: Mapping[str, Any] | None,
    lookahead_s: float,
    duration_s: float,
) -> list[float]:
    """Official commit rule at the chosen threshold, at the availability time.

    No operating point under the budget: nothing is committed for that task.
    """

    if point is None:
        return []

    prob = (probs.speaker_1 if speaker == 1 else probs.speaker_2).prob
    times = [
        min(time_s + lookahead_s, duration_s)
        for time_s in commit_events(prob, CONTROL_RATE_HZ, point["theta"])
    ]

    return sorted(set(times))
