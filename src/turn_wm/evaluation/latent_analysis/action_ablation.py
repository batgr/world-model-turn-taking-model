"""
Extract the action/event ablation of a run's validation rollout.

Two questions:

1. Does the predictor use its future ego-action conditioning?
2. Does changing only the ego action change the predicted transition?

Every prediction below comes from `rollout_representations`, i.e. the
validation rollout (`lejepa_forward`); only the trajectory's action tensor
changes. With C context steps and the anchor t = C - 1:

- OBSERVED: the real WAIT/START/HOLD/STOP actions;
- STATE_PRESERVING: after the unchanged anchor action, every future action
  preserves the resulting ego vocal state (WAIT if silent, HOLD if speaking);
- SHUFFLED: the future steps (>= C) are replaced by another sample's complete
  future sequence, drawn deterministically (seed 3072) within the same
  corpus and focal state at the anchor.

Steps < C (the context and the anchor's own action), the latents, targets,
anchors and horizons never change. The rollout at h = 1 reads no future
action, so the three conditions must agree exactly there; extraction
refuses a batch where they do not.

One-step action effect: the anchor's own action (step C - 1, the one
conditioning the prediction of z_(t+1)) is forced to each semantic ego action,
everything else unchanged; the h = 1 prediction is z_hat_next(action).
Downstream analysis reports only the state-valid pair: WAIT vs START from
SILENT, and HOLD vs STOP from SPEAKING.

Only the validation split is read.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from turn_wm.data.dataset import ACTION_TO_ID, MASKED_ACTION_ID, STATE_TO_ID
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    extract_snapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
    rollout_provenance,
    rollout_representations,
)
from turn_wm.evaluation.latent_analysis.run import DEFAULT_CHECKPOINT, open_run
from turn_wm.models.lewm.jepa import JEPA
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.progress import log, progress
from turn_wm.training.trajectories import Trajectories

ABLATION_SPLIT = "validation"
DEFAULT_ABLATION_SAMPLES = 10_000
SHUFFLE_SEED = 3072

OBSERVED = "observed"
STATE_PRESERVING = "state_preserving"
SHUFFLED = "shuffled"
CONDITIONS = (OBSERVED, STATE_PRESERVING, SHUFFLED)
FORCED_ACTIONS = ("WAIT", "START", "HOLD", "STOP")

PRED = {condition: f"pred_future_latent_{condition}" for condition in CONDITIONS}
SHUFFLED_ACTION_IDS = "rollout_action_ids_shuffled"
COUNTERFACTUAL_NEXT = "counterfactual_next_latent"  # (N, 4, D), FORCED_ACTIONS
FOCAL_STATE = "focal_state_id"  # at the anchor (focal_state_before of step t)
SHUFFLE_DONOR = "shuffle_donor_row"
ABLATION_TENSORS = (
    ANCHOR_LATENT,
    TRUE_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    *PRED.values(),
    SHUFFLED_ACTION_IDS,
    COUNTERFACTUAL_NEXT,
    FOCAL_STATE,
    SHUFFLE_DONOR,
)


# ---------------------------------------------------------------------------
# Action conditions
# ---------------------------------------------------------------------------


def state_preserving_actions(
    actions: torch.Tensor, context_steps: int
) -> torch.Tensor:
    """Replace future actions by the valid action that preserves ego state.

    The anchor action (step C - 1) is unchanged. Its result determines the
    state entering the future: WAIT/STOP lead to SILENT, START/HOLD to
    SPEAKING. Every future step then uses WAIT or HOLD respectively.
    Masked anchors remain masked.
    """

    ablated = actions.clone()
    anchor = actions[:, context_steps - 1]
    preserving = torch.full_like(anchor, MASKED_ACTION_ID)

    silent = (anchor == ACTION_TO_ID["WAIT"]) | (anchor == ACTION_TO_ID["STOP"])
    speaking = (anchor == ACTION_TO_ID["START"]) | (anchor == ACTION_TO_ID["HOLD"])

    preserving[silent] = ACTION_TO_ID["WAIT"]
    preserving[speaking] = ACTION_TO_ID["HOLD"]
    ablated[:, context_steps:] = preserving[:, None]

    return ablated


def shuffled_actions(
    actions: torch.Tensor, context_steps: int, donor_future: torch.Tensor
) -> torch.Tensor:
    """The future steps (>= C) replaced by whole donor sequences (B, F)."""

    if donor_future.shape != actions[:, context_steps:].shape:
        raise ValueError(
            "A donor future must have the recipient's future length: "
            f"{tuple(donor_future.shape)} vs {tuple(actions[:, context_steps:].shape)}"
        )

    ablated = actions.clone()
    ablated[:, context_steps:] = donor_future

    return ablated


def forced_anchor_action(
    actions: torch.Tensor, context_steps: int, action: str
) -> torch.Tensor:
    """The anchor's own action (step C - 1) forced; nothing else changes."""

    forced = actions.clone()
    forced[:, context_steps - 1] = ACTION_TO_ID[action]

    return forced


def shuffle_donors(
    sample_ids: Sequence[str],
    groups: Sequence[Any],
    *,
    seed: int = SHUFFLE_SEED,
) -> list[int]:
    """Each row's donor row, within its group, never itself if avoidable.

    Members of a group are ordered by a seeded key of their sample id (so
    the assignment does not depend on row order), and each one receives the
    whole future sequence of the next member in that order (a cycle). A
    group of one keeps its own sequence.
    """

    members: dict[Any, list[int]] = defaultdict(list)

    for row, group in enumerate(groups):
        members[group].append(row)

    donors = list(range(len(sample_ids)))

    for rows in members.values():
        rows = sorted(rows, key=lambda r: (_key(seed, sample_ids[r]), sample_ids[r]))

        for position, row in enumerate(rows):
            donors[row] = rows[(position + 1) % len(rows)]

    return donors


def _key(seed: int, sample_id: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{sample_id}".encode(), digest_size=8).digest()

    return int.from_bytes(digest)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def ablation_representations(
    model: JEPA,
    batch: Trajectories,
    *,
    sigreg: SIGReg,
    cfg: DictConfig,
    focal_state: torch.Tensor,
    donor_future: torch.Tensor,
    donor_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The three rollout conditions and the counterfactual next latents.

    `focal_state` (B,), `donor_future` (B, F) and `donor_rows` (B,) belong
    to the rows of `batch`, in order.
    """

    c = batch.context_steps

    def rollout(actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return rollout_representations(
            model, replace(batch, actions=actions), sigreg=sigreg, cfg=cfg
        )

    observed = rollout(batch.actions)
    horizons = sorted(int(h) for h in cfg.prediction.rollout_horizons)

    if 1 not in horizons:
        raise ValueError("The ablation needs the 1-step rollout horizon")

    first = horizons.index(1)
    shuffled = shuffled_actions(batch.actions, c, donor_future)
    conditions = {
        OBSERVED: observed,
        STATE_PRESERVING: rollout(state_preserving_actions(batch.actions, c)),
        SHUFFLED: rollout(shuffled),
    }

    for name, result in conditions.items():
        # Same anchors and targets, and at h = 1 (no future action read) the
        # same prediction: anything else is a broken ablation.
        for tensor in (ANCHOR_LATENT, TRUE_FUTURE_LATENT):
            if not torch.equal(result[tensor], observed[tensor]):
                raise RuntimeError(
                    f"{name}: {tensor} differs from the observed rollout"
                )

        if not torch.equal(
            result[PRED_FUTURE_LATENT][:, first], observed[PRED_FUTURE_LATENT][:, first]
        ):
            raise RuntimeError(
                f"Integrity check failed: the {name} rollout differs from the "
                "observed one at h = 1, where no future action is read"
            )

    counterfactual = torch.stack(
        [
            rollout(forced_anchor_action(batch.actions, c, action))[PRED_FUTURE_LATENT][
                :, first
            ]
            for action in FORCED_ACTIONS
        ],
        dim=1,
    )

    return {
        ANCHOR_LATENT: observed[ANCHOR_LATENT],
        TRUE_FUTURE_LATENT: observed[TRUE_FUTURE_LATENT],
        ROLLOUT_ACTION_IDS: observed[ROLLOUT_ACTION_IDS],
        **{
            PRED[name]: result[PRED_FUTURE_LATENT]
            for name, result in conditions.items()
        },
        SHUFFLED_ACTION_IDS: shuffled[:, : observed[ROLLOUT_ACTION_IDS].shape[1]],
        COUNTERFACTUAL_NEXT: counterfactual,
        FOCAL_STATE: focal_state,
        SHUFFLE_DONOR: donor_rows,
    }


def collect_futures(batches, *, max_samples: int | None, total: int | None):
    """First pass: ids, corpora, anchor states/actions and future actions."""

    sample_ids: list[str] = []
    datasets: list[str] = []
    states: list[torch.Tensor] = []
    anchor_actions: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    if max_samples is not None:
        total = max_samples if total is None else min(total, max_samples)

    with progress(total=total, desc="collect futures", unit="anchor") as bar:
        for batch in batches:
            if max_samples is not None and len(sample_ids) >= max_samples:
                break

            take = len(batch["sample_id"])

            if max_samples is not None:
                take = min(take, max_samples - len(sample_ids))

            context = int(batch["context_lengths"][0])
            sample_ids += list(batch["sample_id"][:take])
            datasets += [str(d) for d in batch["dataset"][:take]]
            states.append(batch["context_state"][:take, context - 1])
            anchor_actions.append(batch["context_action"][:take, context - 1])
            futures.append(batch["future_action"][:take])
            bar.update(take)

    return (
        sample_ids,
        datasets,
        torch.cat(states),
        torch.cat(anchor_actions),
        torch.cat(futures),
    )


def action_embedding_geometry(model: JEPA) -> dict[str, Any]:
    """Norms, pairwise cosines and distances of the four semantic action embeddings."""

    with torch.inference_mode():
        ids = torch.tensor([[ACTION_TO_ID[a] for a in FORCED_ACTIONS]])
        parameter = next(model.action_encoder.parameters())
        embeddings = model.action_encoder(ids.to(parameter.device))[0].double().cpu()

    pairs = [
        (a, b) for i, a in enumerate(FORCED_ACTIONS) for b in FORCED_ACTIONS[i + 1 :]
    ]
    index = {a: FORCED_ACTIONS.index(a) for a in FORCED_ACTIONS}

    return {
        "norm": {a: float(embeddings[index[a]].norm()) for a in FORCED_ACTIONS},
        "cosine": {
            f"{a} vs {b}": float(
                F.cosine_similarity(embeddings[index[a]], embeddings[index[b]], dim=0)
            )
            for a, b in pairs
        },
        "euclidean": {
            f"{a} vs {b}": float((embeddings[index[a]] - embeddings[index[b]]).norm())
            for a, b in pairs
        },
    }


def extract_action_ablation(
    model: JEPA,
    cfg: DictConfig,
    batches,
    *,
    max_samples: int | None = DEFAULT_ABLATION_SAMPLES,
    device: torch.device | str = "cpu",
    total: int | None = None,
    seed: int = SHUFFLE_SEED,
) -> tuple[RepresentationSnapshot, dict[str, Any]]:
    """The ablation tensors of the first `max_samples` samples of `batches`.

    `batches` is iterated twice and must yield the same order both times
    (e.g. the fixed-permutation validation loader).
    """

    sample_ids, datasets, _states, anchor_actions, futures = collect_futures(
        batches, max_samples=max_samples, total=total
    )
    groups = [
        (d, int(a), int(f.numel()))
        for d, a, f in zip(datasets, anchor_actions, futures, strict=True)
    ]
    donors = shuffle_donors(sample_ids, groups, seed=seed)
    log(
        f"action ablation: {len(sample_ids):,} anchors, "
        f"{len(set(groups))} shuffle groups (seed {seed})"
    )
    row_of = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    def representations(m, trajectory, batch):
        c = trajectory.context_steps
        # Rows past `max_samples` in the last batch are computed, then dropped
        # by extract_snapshot: they keep their own sequence (donor -1).
        rows = [row_of.get(s) for s in batch["sample_id"]]
        donor_rows = torch.tensor([-1 if r is None else donors[r] for r in rows])
        donor_future = batch["future_action"].clone()

        for i, donor in enumerate(donor_rows.tolist()):
            if donor >= 0:
                donor_future[i] = futures[donor]

        return ablation_representations(
            m,
            trajectory,
            sigreg=sigreg,
            cfg=cfg,
            focal_state=batch["context_state"][:, c - 1],
            donor_future=donor_future,
            donor_rows=donor_rows,
        )

    log("action ablation: rollout under observed, state-preserving and shuffled actions")
    snapshot = extract_snapshot(
        model,
        batches,
        max_samples=len(sample_ids),
        device=device,
        representations=representations,
        total=len(sample_ids),
    )

    if snapshot.metadata["sample_id"] != sample_ids:
        raise RuntimeError("The two passes over the batches saw different samples")

    sizes: dict[Any, int] = defaultdict(int)

    for group in groups:
        sizes[group] += 1

    provenance = {
        "conditions": list(CONDITIONS),
        "future_steps_replaced": "trajectory steps >= context_steps (all future)",
        "unchanged": (
            "context actions (steps < C, including the anchor's), latents, "
            "targets, anchors, horizons"
        ),
        "shuffle": {
            "seed": seed,
            "grouping": "(dataset, anchor ego action, future length)",
            "unit": "whole future action sequence",
            "assignment": "cycle over group members in seeded sample-key order",
            "groups": len(sizes),
            "rows_in_singleton_groups": sum(n for n in sizes.values() if n == 1),
        },
        "counterfactual": {
            "forced_actions": list(FORCED_ACTIONS),
            "reported_valid_pairs": {
                "SILENT": ["WAIT", "START"],
                "SPEAKING": ["HOLD", "STOP"],
            },
            "forced_step": "the anchor's own action, trajectory step C - 1",
            "prediction": "1-step rollout prediction of z_(t+1)",
        },
        "focal_state_ids": dict(STATE_TO_ID),
        "integrity_check": (
            "per batch: anchors, targets and h = 1 predictions identical across "
            "conditions (exact equality)"
        ),
        "action_embeddings": action_embedding_geometry(model),
    }

    return snapshot, provenance


def extract_action_ablation_run(
    run_dir: Path,
    *,
    output_dir: Path | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    max_samples: int | None = DEFAULT_ABLATION_SAMPLES,
    seed: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    device: str = "cpu",
    mimi_cache_root: Path | None = None,
    media_roots: Mapping[str, Path] | None = None,
) -> Path:
    """Write the validation action-ablation snapshot of one run; return it.

    The anchors are the seeded fixed permutation of the validation split
    that `extract-rollouts` uses: with the same seed and `max_samples`,
    they are the same anchors.
    """

    start = time.perf_counter()
    opened = open_run(
        run_dir,
        checkpoint=checkpoint,
        split=ABLATION_SPLIT,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        mimi_cache_root=mimi_cache_root,
        media_roots=dict(media_roots) if media_roots is not None else None,
    )
    output_dir = output_dir or opened.default_output_dir("-action-ablation")
    log(
        f"action ablation: device {device}, max samples {max_samples}, "
        f"output {output_dir}"
    )
    snapshot, ablation = extract_action_ablation(
        opened.checkpoint.model.to(device),
        opened.cfg,
        opened.loader,
        max_samples=max_samples,
        device=device,
        total=len(opened.dataset),
    )

    written = write_snapshot(
        snapshot,
        output_dir,
        provenance={
            **opened.provenance(
                max_samples=max_samples, samples=snapshot.samples, device=device
            ),
            "rollout": rollout_provenance(opened.cfg),
            "action_ablation": ablation,
        },
    )
    log(f"action ablation: written in {time.perf_counter() - start:.0f}s")

    return written
