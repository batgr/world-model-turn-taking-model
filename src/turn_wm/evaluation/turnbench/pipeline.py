"""
The V1 TurnBench TRAIN -> DEV experiment, one stage after another.

    smoke DEV (1 conversation)   -> stop on any contract failure
    smoke TRAIN (1 conversation) -> stop on any contract failure
    TRAIN extraction, DEV extraction
    TRAIN labels (annotator A, official floor semantics)
    three matched heads: mimi / current / predicted  (W&B runs v1-<condition>)
    DEV scoring per condition (official sweep, operating point, scorer)
    report.md

This module only chains the stages; each stage is the one in its own module.
A stage whose manifest exists is reused only if it matches what this run
would produce (its identity) and every file hash still validates; otherwise
it is refused as stale, never silently reused. An interrupted extraction
resumes (`extract_turnbench`); the TEST split is never read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from turnbench.durations import load_durations

from turn_wm.evaluation.turnbench.data import (
    DEV_REVISION,
    TRAIN_REVISION,
    load_dev,
    load_train,
    scorer_revision,
    train_conversation_ids,
)
from turn_wm.evaluation.turnbench.extract import (
    CONDITIONS,
    extract_turnbench,
    extraction_identity,
    validate_extraction,
)
from turn_wm.evaluation.turnbench.heads import (
    CausalHead,
    TrainingConfig,
    load_sequences,
    split_conversations,
    train_head,
)
from turn_wm.evaluation.turnbench.labels import (
    OUTPUTS,
    POSITIVE_FRAMES,
    download_annotations,
    frame_targets,
    parse_srt,
    single_annotator_events,
)
from turn_wm.evaluation.turnbench.report import turnbench_report
from turn_wm.evaluation.turnbench.scoring import FP_BUDGET, dev_scores, score_condition
from turn_wm.evaluation.turnbench.timing import CONTROL_RATE_HZ
from turn_wm.progress import log, progress
from turn_wm.training.train import _git_metadata

WANDB_PROJECT = "turn-wm-turnbench"
SOURCE_SAMPLE_RATE = 48_000  # both TurnBench splits, as published
LABEL_SCHEMA_VERSION = 1


class SmokeTestFailed(RuntimeError):
    """A one-conversation extraction broke the contract: nothing else runs."""


@dataclass(frozen=True)
class PipelineConfig:
    run_dir: Path
    work_dir: Path
    checkpoint: str
    device: str = "cuda"
    scratch_dir: Path | None = None  # temporary TRAIN audio (one conversation)
    wandb_mode: str = "online"
    training: TrainingConfig = field(default_factory=TrainingConfig)


def run_pipeline(config: PipelineConfig) -> Path:
    """Run (or reuse) every stage; return the report path."""

    work = Path(config.work_dir)
    identities = {
        split: extraction_identity(
            config.run_dir, checkpoint=config.checkpoint, split=split
        )
        for split in ("dev", "train")
    }
    scorer = scorer_revision()
    ids = {
        "dev": sorted(load_durations("dev"), key=int),
        "train": train_conversation_ids(),
    }
    sources: dict[str, Callable[[Any], Any]] = {
        "dev": lambda done: load_dev(skip=done),
        "train": lambda done: load_train(skip=done, scratch_dir=config.scratch_dir),
    }

    for split in ("dev", "train"):
        smoke = work / f"smoke-{split}"
        manifest = _extraction_stage(
            config, smoke, split, identities[split], sources[split], limit=1
        )
        check_smoke(smoke, manifest)
        log(f"turnbench-pipeline: smoke {split} passed")

    manifests = {
        split: _extraction_stage(
            config,
            work / split,
            split,
            identities[split],
            sources[split],
            expected=ids[split],
        )
        for split in ("train", "dev")
    }
    labels = _labels_stage(work / "labels", work / "train", ids["train"])
    context = {
        "checkpoint_sha256": identities["dev"]["checkpoint_sha256"],
        "scorer_revision": scorer,
        "dev_revision": DEV_REVISION,
        "train_revision": TRAIN_REVISION,
        "seed": config.training.seed,
        "train_extraction_sha256": _sha256(work / "train" / "manifest.json"),
        "dev_extraction_sha256": _sha256(work / "dev" / "manifest.json"),
        "labels_sha256": _sha256(labels / "manifest.json"),
    }
    results = {}

    for name in CONDITIONS:
        heads = _heads_stage(
            config, work / "heads" / name, name, ids["train"], labels, context
        )
        results[name] = _scoring_stage(
            config,
            work / "scoring" / name,
            name,
            heads,
            manifests["dev"],
            ids["dev"],
            context,
        )

    report = work / "report.md"
    report.write_text(turnbench_report(results, context), encoding="utf-8")
    log(f"turnbench-pipeline: report {report}")

    return report


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _extraction_stage(
    config: PipelineConfig,
    output: Path,
    split: str,
    identity: dict[str, Any],
    source: Callable[[Any], Any],
    *,
    limit: int | None = None,
    expected: list[str] | None = None,
) -> dict[str, Any]:
    if (output / "manifest.json").exists():
        manifest = validate_extraction(output, identity=identity, expected_ids=expected)

        if manifest.get("conversation_limit") != limit:
            raise ValueError(f"Stale extraction in {output}: made with another limit")

        log(f"turnbench-pipeline: reusing {output} (identity and hashes validated)")

        return manifest

    extract_turnbench(
        source,
        run_dir=config.run_dir,
        output_dir=output,
        checkpoint=config.checkpoint,
        device=config.device,
        split=split,
        total=None if expected is None else len(expected),
        limit=limit,
    )

    return validate_extraction(output, identity=identity, expected_ids=expected)


def check_smoke(output: Path, manifest: Mapping[str, Any]) -> None:
    """The scientific / temporal contract on a one-conversation extraction."""

    try:
        [(conversation_id, entry)] = manifest["files"].items()
        tensors = load_file(str(output / entry["file"]))
        slots = len(tensors["latent"])
        window = manifest["identity"]["window"]
        lookahead = entry["resample_lookahead_s"]
        embed = tensors["latent"].shape[1]
        expected_shapes = {
            "mimi": (slots, 512),
            "latent": (slots, embed),
            "zpred_speaker_1": (slots, embed),
            "zpred_speaker_2": (slots, embed),
        }
        checks = {
            "source sample rate": entry["sample_rate"] == SOURCE_SAMPLE_RATE,
            "slots = floor(duration * 10 Hz)": slots
            == int(entry["duration_s"] * CONTROL_RATE_HZ + 1e-9),
            "shapes": all(
                tuple(tensors[k].shape) == v for k, v in expected_shapes.items()
            ),
            "first W - 1 slots prediction-invalid": tensors["prediction_valid"].tolist()
            == [k >= window - 1 for k in range(slots)],
            "predictions finite where valid": bool(
                torch.isfinite(tensors["zpred_speaker_1"][window - 1 :]).all()
                and torch.isfinite(tensors["zpred_speaker_2"][window - 1 :]).all()
            ),
            "timestamps increasing": bool((tensors["slot_start_s"].diff() > 0).all()),
            "available_s = slot end + lookahead": torch.allclose(
                tensors["available_s"],
                torch.arange(1, slots + 1, dtype=torch.float64) / CONTROL_RATE_HZ
                + lookahead,
            ),
            "no annotation read": "annotation"
            not in json.dumps(manifest["inputs"]["turnbench_dataset"]),
        }
    except (KeyError, ValueError, RuntimeError) as error:
        raise SmokeTestFailed(
            f"{output}: unreadable smoke extraction: {error}"
        ) from error

    failed = [name for name, ok in checks.items() if not ok]

    if failed:
        raise SmokeTestFailed(f"{output} ({conversation_id}): failed {failed}")


def _labels_stage(output: Path, train_dir: Path, conversation_ids: list[str]) -> Path:
    identity = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "train_revision": TRAIN_REVISION,
        "train_extraction_sha256": _sha256(train_dir / "manifest.json"),
        "annotator": "a",
        "positive_frames": POSITIVE_FRAMES,
        "outputs": list(OUTPUTS),
        "scorer_revision": scorer_revision(),
    }

    if _reuse(output, identity):
        return output

    files, counts = {}, {}
    output.mkdir(parents=True, exist_ok=True)

    for conversation_id in progress(
        conversation_ids, desc="labels", unit="conversation"
    ):
        srt = download_annotations(conversation_id, output / "srt")
        events = single_annotator_events(
            {
                speaker: parse_srt(path.read_text(encoding="utf-8"))
                for speaker, path in srt.items()
            }
        )
        available = load_file(str(train_dir / f"{conversation_id}.safetensors"))[
            "available_s"
        ]
        targets, supervised = frame_targets(events, available)
        path = output / f"{conversation_id}.safetensors"
        save_file({"targets": targets, "supervised": supervised}, str(path))
        files[path.name] = _sha256(path)
        counts[conversation_id] = {
            "eot_positive": len(events.eot_positive_events),
            "eot_negative_spans": len(events.eot_negative_spans),
            "int_positive": len(events.int_positive_events),
            "int_negative_spans": len(events.int_negative_spans),
            "int_excluded": len(events.int_excluded),
        }

    _write_stage(output, identity, files, {"event_counts": counts})

    return output


def _heads_stage(
    config: PipelineConfig,
    output: Path,
    name: str,
    train_ids: list[str],
    labels: Path,
    context: Mapping[str, Any],
) -> Path:
    identity = {
        "condition": name,
        "train_extraction_sha256": context["train_extraction_sha256"],
        "labels_sha256": context["labels_sha256"],
        "training": asdict(config.training),
    }

    if _reuse(output, identity):
        return output

    train, validation = split_conversations(
        train_ids,
        seed=config.training.seed,
        validation_fraction=config.training.validation_fraction,
    )

    def label(conversation_id: str, available_s: torch.Tensor):
        tensors = load_file(str(labels / f"{conversation_id}.safetensors"))
        return tensors["targets"], tensors["supervised"].bool()

    train_data = load_sequences(output.parent.parent / "train", train, label, name=name)
    validation_data = load_sequences(
        output.parent.parent / "train", validation, label, name=name
    )
    input_dim = train_data[0].inputs.shape[1]
    run = _wandb(
        config,
        name,
        {
            "representation_condition": name,
            "input_dim": input_dim,
            "v1_checkpoint_sha256": context["checkpoint_sha256"],
            "train_extraction_manifest_sha256": context["train_extraction_sha256"],
            "dev_extraction_manifest_sha256": context["dev_extraction_sha256"],
            "turnbench_train_revision": TRAIN_REVISION,
            "turnbench_dev_revision": DEV_REVISION,
            "turnbench_scorer_revision": context["scorer_revision"],
            "split_seed": config.training.seed,
            "train_conversations": train,
            "validation_conversations": validation,
            "head": f"LayerNorm -> causal Conv1d(k={config.training.kernel}, "
            f"{config.training.hidden}) -> GELU -> Conv1d(1, {len(OUTPUTS)})",
            "optimizer": "AdamW",
            **asdict(config.training),
            "decision_frame_rate_hz": CONTROL_RATE_HZ,
            "git": _git_metadata(),
        },
    )
    log(
        f"turnbench-pipeline: head {name}: {len(train)} train / {len(validation)} validation conversations"
    )
    head, record = train_head(
        train_data,
        validation_data,
        config=config.training,
        device=config.device,
        log_metrics=lambda metrics, step: run.log(metrics, step=step),
    )
    run.summary.update(
        {f"class_counts/{k}": v for k, v in record["class_counts"].items()}
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "head.safetensors"
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(path))
    _write_stage(
        output,
        identity,
        {path.name: _sha256(path)},
        {
            **record,
            "train_conversations": train,
            "validation_conversations": validation,
            "wandb_run_id": run.id,
        },
    )
    run.finish()

    return output


def _scoring_stage(
    config: PipelineConfig,
    output: Path,
    name: str,
    heads: Path,
    dev_manifest: Mapping[str, Any],
    dev_ids: list[str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "condition": name,
        "head_sha256": _sha256(heads / "manifest.json"),
        "dev_extraction_sha256": context["dev_extraction_sha256"],
        "scorer_revision": context["scorer_revision"],
        "dev_revision": DEV_REVISION,
        "fp_budget": FP_BUDGET,
    }

    if _reuse(output, identity):
        return json.loads((output / "scores.json").read_text(encoding="utf-8"))

    from turnbench.data import DEV_DATASET, resolve_dataset

    record = json.loads((heads / "manifest.json").read_text(encoding="utf-8"))
    head = CausalHead(
        record["input_dim"],
        hidden=config.training.hidden,
        kernel=config.training.kernel,
    )
    head.load_state_dict(load_file(str(heads / "head.safetensors")))
    scores = dev_scores(head.eval(), output.parent.parent / "dev", dev_ids, name=name)
    lookahead = {
        cid: entry["resample_lookahead_s"]
        for cid, entry in dev_manifest["files"].items()
    }
    result = score_condition(
        scores,
        lookahead_s=lookahead,
        dataset=resolve_dataset(DEV_DATASET, DEV_REVISION, skip_audio=True),
        output_dir=output,
    )
    files = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.suffix == ".json" and path.name != "manifest.json"
    }
    _write_stage(output, identity, files, {})
    run = _wandb(config, name, None, run_id=record["wandb_run_id"])

    for task in ("eot", "int"):
        score = result[task]
        run.summary.update(
            {
                f"dev/{task}/threshold": score["threshold"],
                f"dev/{task}/recall": score["recall"],
                f"dev/{task}/fp_rate": score["fp_rate"],
                **{
                    f"dev/{task}/latency_{p}_ms": v
                    for p, v in score["latency_ms"].items()
                },
            }
        )

    run.finish()

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reuse(output: Path, identity: Mapping[str, Any]) -> bool:
    """True for a valid finished stage; False for none; raises for a stale one."""

    path = output / "manifest.json"

    if not path.exists():
        return False

    manifest = json.loads(path.read_text(encoding="utf-8"))

    if manifest["identity"] != identity:
        raise ValueError(
            f"Stale stage in {output}: made for {manifest['identity']}, expected "
            f"{identity}; remove it to recompute"
        )

    for name, digest in manifest["files"].items():
        if not (output / name).is_file() or _sha256(output / name) != digest:
            raise ValueError(f"{output / name}: missing or changed since the stage ran")

    log(f"turnbench-pipeline: reusing {output} (identity and hashes validated)")

    return True


def _write_stage(
    output: Path,
    identity: Mapping[str, Any],
    files: Mapping[str, str],
    extra: Mapping[str, Any],
) -> None:
    content = {"identity": identity, "files": files, **extra}
    tmp = output / "manifest.tmp"
    tmp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output / "manifest.json")


def _wandb(
    config: PipelineConfig,
    name: str,
    run_config: Mapping[str, Any] | None,
    *,
    run_id: str | None = None,
):
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        name=f"v1-{name}",
        config=None if run_config is None else dict(run_config),
        dir=str(config.work_dir),
        mode=config.wandb_mode,  # type: ignore[arg-type]
        id=run_id,
        resume="allow" if run_id is not None else None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
