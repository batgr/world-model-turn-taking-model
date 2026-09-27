"""
The three frozen V1 representations of TurnBench conversations.

Per complete 100 ms slot k of a conversation:

- `mimi`:    Mimi features of the scene (speaker_1 + speaker_2), streamed
             and aligned causally to the 10 Hz grid exactly as V1's cache
             was (`turn_wm.data.mimi_precompute`);
- `latent`:  z_k, the frozen V1 projector applied to `mimi`;
- `zpred_<speaker>`: one V1 predictor step from the same latent history,
             conditioned on that speaker's own observed actions:

                 zpred_s(k) = predict(z[k-W+1 .. k], a_s[k-W+1 .. k])[-1]

             with W = the run's `prediction.rollout_context_size`, the
             window V1's rollout predicts from. Slots with fewer than W
             latents of history have no prediction (`prediction_valid`).

A downstream head sees exactly one condition (`condition`): the Mimi
features, the current latent, or concat(zpred_speaker_1, zpred_speaker_2)
and nothing else (no z_k, no action).

Timing: z_k and a_k exist at the end of slot k; z_k also waits for the
resampler's lookahead (`resample_lookahead_s`). Every row is stamped with
`available_s = slot_end_s + lookahead`, the earliest time anything derived
from it can be committed. zpred predicts slot k + 1 but does not wait for it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from turn_wm.data.dataset import ACTION_TO_ID, MASKED_ACTION_ID
from turn_wm.evaluation.latent_analysis.run import load_checkpoint, load_run
from turn_wm.evaluation.turnbench.actions import SlotActions, conversation_actions
from turn_wm.evaluation.turnbench.data import (
    SOURCES,
    SPEAKERS,
    Conversation,
    input_provenance,
)
from turn_wm.evaluation.turnbench.timing import (
    CONTROL_RATE_HZ,
    resample_lookahead_s,
    slot_count,
    slot_end_s,
    slot_start_s,
)
from turn_wm.models.encoders.mimi import (
    FrozenMimiEncoder,
    causal_align,
    stack_waveforms,
)
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.progress import log, progress
from turn_wm.training.train import _git_metadata

SCHEMA_VERSION = 1
MIMI, CURRENT, PREDICTED = "mimi", "current", "predicted"
CONDITIONS = (MIMI, CURRENT, PREDICTED)
CHUNK_SECONDS = 20.0  # V1's precompute-mimi default; features do not depend on it
PREDICT_BATCH = 1_024


def scene_features(
    conversation: Conversation,
    encoder: FrozenMimiEncoder,
    *,
    chunk_seconds: float = CHUNK_SECONDS,
) -> torch.Tensor:
    """(K, D) Mimi features of the scene, one row per complete slot."""

    slots = slot_count(len(conversation.speaker_1), conversation.sample_rate)

    if slots == 0:
        raise ValueError(f"{conversation.conversation_id}: shorter than one slot")

    # As V1's cache: resampled once, then streamed and aligned causally.
    waveform = stack_waveforms(
        [conversation.scene()[None]],
        [conversation.sample_rate],
        target_rate=encoder.sample_rate,
    )
    waveform = waveform[..., : round(slots / CONTROL_RATE_HZ * encoder.sample_rate)]
    native = encoder.stream_native_features(waveform, chunk_seconds=chunk_seconds)

    return causal_align(
        native,
        source_rate=encoder.source_rate,
        target_rate=CONTROL_RATE_HZ,
        target_length=slots,
    )[0]


def action_ids(actions: SlotActions) -> torch.Tensor:
    """(K,) V1 action ids; a masked action is V1's MASKED id, as in training."""

    return torch.tensor(
        [MASKED_ACTION_ID if a is None else ACTION_TO_ID[a] for a in actions.action]
    )


@torch.inference_mode()
def one_step_predictions(
    model: JEPA, latents: torch.Tensor, actions: torch.Tensor, *, window: int
) -> torch.Tensor:
    """(K, D) one predictor step per slot k >= window - 1; NaN before.

    Row k predicts z_(k+1) from latents and actions k - window + 1 .. k,
    exactly V1's first rollout step.
    """

    if len(latents) != len(actions):
        raise ValueError("latents and actions must cover the same slots")

    predictions = torch.full_like(latents, float("nan"))
    ends = range(window - 1, len(latents))
    action_embeddings = model.encode_actions(actions[None].to(latents.device))[0]

    for start in range(0, len(ends), PREDICT_BATCH):
        batch = ends[start : start + PREDICT_BATCH]
        rows = torch.stack(
            [torch.arange(k - window + 1, k + 1, device=latents.device) for k in batch]
        )
        predictions[batch.start : batch.stop] = model.predict(
            latents[rows], action_embeddings[rows]
        )[:, -1]

    return predictions


@torch.inference_mode()
def conversation_representations(
    conversation: Conversation,
    *,
    model: JEPA,
    encoder: FrozenMimiEncoder,
    window: int,
    lookahead_s: float,
) -> dict[str, torch.Tensor]:
    """Every tensor persisted for one conversation, one row per slot."""

    features = scene_features(conversation, encoder)
    slots = len(features)
    parameter = next(model.parameters())
    latents = model.project_features(features[None].to(parameter))[0].float()
    actions = conversation_actions(conversation)
    tensors: dict[str, torch.Tensor] = {
        "slot_start_s": torch.tensor([slot_start_s(k) for k in range(slots)]),
        "available_s": torch.tensor(
            [slot_end_s(k) + lookahead_s for k in range(slots)], dtype=torch.float64
        ),
        MIMI: features.float(),
        "latent": latents,
        "prediction_valid": torch.arange(slots) >= window - 1,
    }

    for speaker in SPEAKERS:
        ids = action_ids(actions[speaker])

        if len(ids) != slots:
            raise RuntimeError(
                f"{conversation.conversation_id}: {len(ids)} action slots for "
                f"{slots} feature slots"
            )

        tensors[f"zpred_{speaker}"] = one_step_predictions(
            model, latents, ids.to(latents.device), window=window
        ).float()
        tensors[f"action_valid_{speaker}"] = torch.tensor(actions[speaker].action_valid)

    return {name: tensor.cpu().contiguous() for name, tensor in tensors.items()}


def condition(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """The (K, ·) input a downstream head sees for one condition, and only it."""

    if name == MIMI:
        return tensors[MIMI]
    if name == CURRENT:
        return tensors["latent"]
    if name == PREDICTED:
        return torch.cat([tensors[f"zpred_{s}"] for s in SPEAKERS], dim=-1)

    raise ValueError(f"Unknown condition {name!r}; expected one of {CONDITIONS}")


def extraction_identity(
    run_dir: Path, *, checkpoint: str, split: str
) -> dict[str, Any]:
    """What an extraction must match to be reused: data, model and Mimi.

    Computed from files only (no model is loaded), so a stage can be checked
    before any expensive work.
    """

    input_provenance(split=split, source_sample_rate=1)  # an unknown split fails
    record = load_run(run_dir)
    path = Path(checkpoint).expanduser()

    if not path.is_file():
        path = record.run_dir / "checkpoints" / str(checkpoint)

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "dataset": {k: SOURCES[split][k] for k in ("repo", "revision")},
        "run_id": record.metadata["run_id"],
        "config_hash": record.metadata.get("config_hash"),
        "checkpoint_sha256": _sha256(path),
        "mimi": mimi_identity(record.metadata),
        "window": int(record.cfg.prediction.rollout_context_size),
    }


def validate_extraction(
    output_dir: Path,
    *,
    identity: dict[str, Any],
    expected_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """The manifest of a complete, matching extraction; raises otherwise.

    A different identity (data, checkpoint, Mimi, window), a file whose hash
    changed or a missing conversation make the artifact stale: it is refused,
    never reused.
    """

    path = Path(output_dir) / "manifest.json"

    if not path.is_file():
        raise ValueError(f"No complete extraction in {output_dir} (no manifest.json)")

    manifest = json.loads(path.read_text(encoding="utf-8"))

    if manifest.get("identity") != identity:
        raise ValueError(
            f"Stale extraction in {output_dir}: made for {manifest.get('identity')}, "
            f"expected {identity}"
        )

    for entry in manifest["files"].values():
        file = Path(output_dir) / entry["file"]

        if not file.is_file() or _sha256(file) != entry["sha256"]:
            raise ValueError(f"{file}: missing or changed since the extraction")

    if expected_ids is not None and set(manifest["files"]) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(manifest["files"]))
        raise ValueError(
            f"Extraction in {output_dir} does not cover the split: missing "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return manifest


def extract_turnbench(
    source: Callable[[Collection[str]], Iterable[Conversation]],
    *,
    run_dir: Path,
    output_dir: Path,
    checkpoint: str = "last.ckpt",
    device: str = "cpu",
    split: str = "dev",
    total: int | None = None,
    limit: int | None = None,
) -> Path:
    """Write one safetensors file per conversation and `manifest.json`.

    `source(done)` yields the conversations not in `done`. An interrupted
    extraction resumes from `partial.json` when its identity matches (each
    done file's hash re-checked); anything else in `output_dir` is refused.
    `limit` stops after that many conversations (smoke runs). `total`
    (conversations expected) only sizes the progress bar.
    """

    start = time.perf_counter()
    output_dir = Path(output_dir)
    identity = extraction_identity(run_dir, checkpoint=checkpoint, split=split)
    partial_path = output_dir / "partial.json"
    files: dict[str, dict[str, Any]] = {}

    if (output_dir / "manifest.json").exists():
        raise ValueError(
            f"{output_dir} already holds a complete extraction; validate it or "
            "choose another directory"
        )

    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))

        if partial["identity"] != identity:
            raise ValueError(
                f"Stale partial extraction in {output_dir}: made for "
                f"{partial['identity']}, expected {identity}"
            )

        for entry in partial["files"].values():
            if _sha256(output_dir / entry["file"]) != entry["sha256"]:
                raise ValueError(f"{entry['file']} changed since it was extracted")

        files = partial["files"]
        log(f"turnbench: resuming, {len(files)} conversations already extracted")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    record = load_run(run_dir)
    loaded = load_checkpoint(record, checkpoint)
    model = loaded.model.to(device).eval()
    mimi = identity["mimi"]
    encoder = FrozenMimiEncoder(
        model_name=mimi["model_name"], revision=mimi["model_resolved_revision"]
    ).to(device)
    encoder.eval()
    window = identity["window"]
    dataset = identity["dataset"]
    log(f"turnbench: split {split}, {dataset['repo']} @ {dataset['revision']}")
    log(
        f"turnbench: checkpoint {loaded.path} (step {loaded.global_step}, "
        f"sha256 {loaded.sha256[:12]}), window {window}"
    )
    log(
        f"turnbench: Mimi {mimi['model_name']} @ {encoder.resolved_revision} "
        f"(run's cache: {mimi['model_resolved_revision']})"
    )
    log(f"turnbench: device {_describe_device(device)}")

    if limit is not None:
        total = limit if total is None else min(total, limit)

    remaining = None if total is None else max(0, total - len(files))
    log(
        f"turnbench: {'unknown number of' if remaining is None else remaining} "
        f"conversations to extract -> {output_dir}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    conversations = source(set(files))

    if limit is not None:
        conversations = itertools.islice(conversations, max(0, limit - len(files)))

    for conversation in progress(
        conversations, total=remaining, desc="turnbench", unit="conversation"
    ):
        if conversation.conversation_id in files:
            raise ValueError(f"Duplicate conversation {conversation.conversation_id}")

        lookahead = resample_lookahead_s(conversation.sample_rate, encoder.sample_rate)
        tensors = conversation_representations(
            conversation,
            model=model,
            encoder=encoder,
            window=window,
            lookahead_s=lookahead,
        )
        path = output_dir / f"{conversation.conversation_id}.safetensors"
        save_file(tensors, str(path))
        files[conversation.conversation_id] = {
            "file": path.name,
            "sha256": _sha256(path),
            "slots": len(tensors["latent"]),
            "duration_s": conversation.duration_s,
            "sample_rate": conversation.sample_rate,
            "resample_lookahead_s": lookahead,
        }
        _write_json(partial_path, {"identity": identity, "files": files})

    if not files:
        raise ValueError("No conversation was extracted")

    rates = {entry["sample_rate"] for entry in files.values()}

    if len(rates) != 1:
        raise ValueError(f"Conversations differ in sample rate: {sorted(rates)}")

    rate = rates.pop()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "TurnBench",
        "split": split,
        "identity": identity,
        "inputs": input_provenance(split=split, source_sample_rate=rate),
        "actions": {
            "source": "each speaker's own channel, causal RMS activity; no annotation",
            "vocabulary": ["NO_EVENT", "ONSET", "OFFSET"],
            "state_before": "activity of the window ending at t_k",
            "action": (
                "transition inside slot k: none -> NO_EVENT, one "
                "SILENT->SPEAKING -> ONSET, one SPEAKING->SILENT -> OFFSET, "
                "several -> masked (compound_transition)"
            ),
            "slot_0": "state UNKNOWN, action masked (recording_start)",
            "masked_to_v1": "V1's MASKED action id, as in its training",
        },
        "model": {
            "run_id": record.metadata["run_id"],
            "run_dir": str(record.run_dir),
            "config_hash": record.metadata.get("config_hash"),
            "checkpoint": loaded.path.name,
            "checkpoint_sha256": loaded.sha256,
            "epoch": loaded.epoch,
            "global_step": loaded.global_step,
            "train_dataset_revision": record.metadata.get("dataset_revision"),
        },
        "mimi": {
            **mimi,
            "loaded_revision": encoder.resolved_revision,
            "sample_rate": encoder.sample_rate,
            "frame_rate_hz": encoder.source_rate,
            "chunk_seconds": CHUNK_SECONDS,
            "pipeline": (
                "scene resampled once (torchaudio sinc, defaults), streamed "
                "through Mimi with persistent caches, causal_align to 10 Hz: "
                "V1's precompute-mimi path"
            ),
        },
        "timing": {
            "slot_s": 1 / CONTROL_RATE_HZ,
            "row_k": "slot [t_k, t_k + 0.1 s): z_k, a_k known at its end",
            "resample_lookahead_s": resample_lookahead_s(rate, encoder.sample_rate),
            "available_s": "slot_end_s + resample_lookahead_s",
            "prediction": (
                "zpred(k) = predict(z[k-W+1..k], a_s[k-W+1..k])[-1] predicts "
                "z_(k+1) at available_s; the +0.1 s horizon is not latency"
            ),
            "mimi_streaming": "causal (persistent caches), as V1's cache",
        },
        "conditions": {
            MIMI: "mimi: (K, 512) Mimi features of the scene",
            CURRENT: "latent: (K, embed_dim) z_k, frozen V1 projector",
            PREDICTED: (
                "concat(zpred_speaker_1, zpred_speaker_2): (K, 2 * embed_dim), "
                f"one predictor step, window W = {window}; no z_k, no action"
            ),
        },
        "rows": {
            "prediction_valid": f"k >= W - 1 = {window - 1}",
            "action_valid_<speaker>": "that speaker's a_k is not masked",
        },
        "conversation_limit": limit,
        "files": files,
        "extraction": {
            "git": _git_metadata(),
            "device": device,
            "precision": "float32",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "elapsed_s": round(time.perf_counter() - start, 1),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    partial_path.unlink(missing_ok=True)
    log(
        f"turnbench: done, {len(files)} conversations in "
        f"{time.perf_counter() - start:.0f}s; manifest {output_dir / 'manifest.json'}"
    )

    return output_dir


def mimi_identity(run_metadata: dict[str, Any]) -> dict[str, Any]:
    """The Mimi model of the cache V1 trained on; refuses to guess one."""

    caches = (run_metadata.get("mimi_cache") or {}).get("caches") or {}
    identities = {
        (c["model_name"], c["model_resolved_revision"]) for c in caches.values()
    }

    if len(identities) != 1 or None in next(iter(identities), (None,)):
        raise ValueError(
            "The run does not record exactly one resolved Mimi model revision; "
            f"found {sorted(identities, key=str)}"
        )

    name, revision = identities.pop()

    return {"model_name": name, "model_resolved_revision": revision}


def _describe_device(device: str) -> str:
    resolved = torch.device(device)

    if resolved.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(resolved)})"
    if resolved.type == "mps":
        return f"{device} (Apple GPU)"

    return device


def _write_json(path: Path, content: dict[str, Any]) -> None:
    """Write atomically: a crash never leaves a truncated manifest."""

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()
