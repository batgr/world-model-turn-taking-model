"""
Matched EOT / INT heads on frozen TurnBench representations.

Question: given one frozen representation per 100 ms frame (Mimi features,
V1's current latent, or V1's one-step predictions for both speakers), how
well can a small causal readout detect each speaker's end of turn and
interruption?

The three conditions share everything but their input: the same eligible
frames (prediction_valid and the label mask, never the representation), the
same head (per-frame LayerNorm without affine, a causal temporal convolution
of HEAD_KERNEL frames to HEAD_HIDDEN channels, GELU, a linear map to the
four outputs), the same loss (binary cross-entropy weighted per output by
its negative/positive ratio on the training frames) and the same optimizer,
schedule and early stopping. The head sees `condition(...)` only: for
`predicted`, concat(zpred_speaker_1, zpred_speaker_2), never z_t, action ids
or activity.

TRAIN conversations are split by whole conversation (seeded); the internal
validation part serves early stopping and detecting broken training only.
Operating points are chosen on the official DEV set, not here.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from turn_wm.evaluation.turnbench.extract import condition
from turn_wm.evaluation.turnbench.labels import OUTPUTS
from turn_wm.progress import log, progress

HEAD_HIDDEN = 64
HEAD_KERNEL = 5  # 0.5 s of causal context at 10 Hz


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    validation_fraction: float = 0.1
    batch_size: int = 32
    crop_frames: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 20
    patience: int = 3
    hidden: int = HEAD_HIDDEN
    kernel: int = HEAD_KERNEL


@dataclass(frozen=True)
class Sequence_:
    """One conversation's head inputs and supervision."""

    conversation_id: str
    inputs: torch.Tensor  # (K, D) float16
    targets: torch.Tensor  # (K, 4)
    mask: torch.Tensor  # (K, 4) bool: eligible and supervised


class CausalHead(nn.Module):
    """Per-frame scores for OUTPUTS from the last `kernel` frames only."""

    def __init__(
        self, input_dim: int, *, hidden: int = HEAD_HIDDEN, kernel: int = HEAD_KERNEL
    ):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.conv = nn.Conv1d(input_dim, hidden, kernel)
        self.out = nn.Conv1d(hidden, len(OUTPUTS), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, D) -> (B, K, 4) logits; frame k reads frames k - kernel + 1 .. k."""

        x = self.norm(x).transpose(1, 2)
        x = F.pad(x, (self.kernel - 1, 0))

        return self.out(F.gelu(self.conv(x))).transpose(1, 2)


def split_conversations(
    conversation_ids: Sequence[str], *, seed: int, validation_fraction: float
) -> tuple[list[str], list[str]]:
    """(train, validation) whole-conversation split, deterministic in `seed`."""

    ids = sorted(set(conversation_ids))
    order = random.Random(seed).sample(ids, len(ids))
    count = max(1, math.ceil(validation_fraction * len(ids)))

    return sorted(order[count:]), sorted(order[:count])


def eligible_mask(
    tensors: dict[str, torch.Tensor], supervised: torch.Tensor
) -> torch.Tensor:
    """(K, 4): frames every condition is trained and compared on.

    Built from `prediction_valid` and the label mask only, never from a
    representation, so it is the same for the three conditions.
    """

    return supervised & tensors["prediction_valid"].bool()[:, None]


def load_sequences(
    extraction_dir: Path,
    conversation_ids: Sequence[str],
    labels: Callable[[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    name: str,
) -> list[Sequence_]:
    """Condition `name`'s inputs, targets and mask for each conversation.

    `labels(conversation_id, available_s)` gives (targets, supervised).
    """

    sequences = []

    for conversation_id in progress(
        conversation_ids, desc=f"load {name}", unit="conversation"
    ):
        tensors = load_file(
            str(Path(extraction_dir) / f"{conversation_id}.safetensors")
        )
        targets, supervised = labels(conversation_id, tensors["available_s"])
        mask = eligible_mask(tensors, supervised)
        # Rows without a prediction are not eligible; fill them to keep shapes.
        inputs = condition(tensors, name).nan_to_num(0.0).half()
        sequences.append(Sequence_(conversation_id, inputs, targets, mask))

    return sequences


def class_counts(sequences: Sequence[Sequence_]) -> dict[str, dict[str, int]]:
    """Positive / negative eligible frames per output."""

    counts = {}

    for column, output in enumerate(OUTPUTS):
        positive = sum(
            int((s.targets[:, column][s.mask[:, column]] > 0.5).sum())
            for s in sequences
        )
        total = sum(int(s.mask[:, column].sum()) for s in sequences)
        counts[output] = {"positive": positive, "negative": total - positive}

    return counts


def masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean over eligible frames, per-output (4,) mean) weighted BCE."""

    losses = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    mask = mask.float()
    per_output = (losses * mask).sum(dim=tuple(range(mask.ndim - 1))) / mask.sum(
        dim=tuple(range(mask.ndim - 1))
    ).clamp_min(1)

    return (losses * mask).sum() / mask.sum().clamp_min(1), per_output


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Area under the precision-recall curve (step interpolation)."""

    if targets.sum() == 0:
        return float("nan")

    order = scores.argsort(descending=True)
    hits = targets[order]
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)

    return float((precision * hits).sum() / hits.sum())


@torch.no_grad()
def evaluate(
    head: CausalHead,
    sequences: Sequence[Sequence_],
    pos_weight: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Validation loss (all, EOT, INT) and per-output diagnostics.

    The head runs on `device`; the loss and metrics are computed on the CPU,
    where the targets and masks live, so `pos_weight` is brought there too.
    """

    head.eval()
    logits, targets, masks = [], [], []

    for sequence in sequences:
        logits.append(head(sequence.inputs.float()[None].to(device))[0].cpu())
        targets.append(sequence.targets)
        masks.append(sequence.mask)

    logits_, targets_, masks_ = torch.cat(logits), torch.cat(targets), torch.cat(masks)
    loss, per_output = masked_loss(logits_, targets_, masks_, pos_weight.cpu())
    metrics = {
        "val/loss": float(loss),
        "val/eot_loss": float(per_output[:2].mean()),
        "val/int_loss": float(per_output[2:].mean()),
    }
    probabilities = logits_.sigmoid()

    for column, output in enumerate(OUTPUTS):
        eligible = masks_[:, column]
        target = targets_[eligible, column]
        score = probabilities[eligible, column]
        metrics[f"val/{output}/average_precision"] = average_precision(score, target)
        metrics[f"val/{output}/mean_score_positive"] = (
            float(score[target > 0.5].mean()) if (target > 0.5).any() else float("nan")
        )
        metrics[f"val/{output}/mean_score_negative"] = (
            float(score[target <= 0.5].mean())
            if (target <= 0.5).any()
            else float("nan")
        )

    return metrics


def train_head(
    train: Sequence[Sequence_],
    validation: Sequence[Sequence_],
    *,
    config: TrainingConfig,
    device: str = "cpu",
    log_metrics: Callable[[dict[str, Any], int], None] | None = None,
) -> tuple[CausalHead, dict[str, Any]]:
    """Train one head; return it at its best validation loss, with its history."""

    torch.manual_seed(config.seed)
    generator = random.Random(config.seed)
    input_dim = train[0].inputs.shape[1]
    head = CausalHead(input_dim, hidden=config.hidden, kernel=config.kernel).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    counts = class_counts(train)
    # Same imbalance policy for every condition: weight positives by neg/pos.
    pos_weight = torch.tensor(
        [counts[o]["negative"] / max(1, counts[o]["positive"]) for o in OUTPUTS]
    )
    frames = sum(len(s.inputs) for s in train)
    steps = max(1, math.ceil(frames / (config.batch_size * config.crop_frames)))
    weights = [len(s.inputs) for s in train]
    best: tuple[float, dict[str, torch.Tensor], int] | None = None
    history = []
    epochs_without_gain = 0

    for epoch in range(1, config.max_epochs + 1):
        head.train()
        total = 0.0

        for _ in progress(
            range(steps), desc=f"epoch {epoch}", unit="step", leave=False
        ):
            batch = [
                _crop(s, config.crop_frames, generator)
                for s in generator.choices(train, weights, k=config.batch_size)
            ]
            inputs = torch.stack([b[0] for b in batch]).float().to(device)
            targets = torch.stack([b[1] for b in batch]).to(device)
            mask = torch.stack([b[2] for b in batch]).to(device)
            loss, _ = masked_loss(head(inputs), targets, mask, pos_weight.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach().item()

        metrics = {"epoch": epoch, "train/loss": total / steps}
        metrics |= evaluate(head, validation, pos_weight, device)
        history.append(metrics)
        log(
            f"heads: epoch {epoch}: train {metrics['train/loss']:.4f}, "
            f"val {metrics['val/loss']:.4f} (eot {metrics['val/eot_loss']:.4f}, "
            f"int {metrics['val/int_loss']:.4f})"
        )

        if log_metrics is not None:
            log_metrics(metrics, epoch)

        if best is None or metrics["val/loss"] < best[0]:
            best = (
                metrics["val/loss"],
                {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                epoch,
            )
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

            if epochs_without_gain >= config.patience:
                log(f"heads: early stop after epoch {epoch} (best {best[2]})")
                break

    assert best is not None
    head.load_state_dict(best[1])

    return head.eval(), {
        "config": asdict(config),
        "input_dim": input_dim,
        "class_counts": counts,
        "pos_weight": pos_weight.tolist(),
        "steps_per_epoch": steps,
        "best_epoch": best[2],
        "best_val_loss": best[0],
        "history": history,
    }


def _crop(sequence: Sequence_, frames: int, generator: random.Random):
    """A random `frames`-long crop, zero-padded (and masked) when shorter."""

    length = len(sequence.inputs)
    start = generator.randrange(max(1, length - frames + 1))
    stop = min(length, start + frames)
    pad = frames - (stop - start)

    def cut(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor[start:stop], (0, 0, 0, pad))

    return cut(sequence.inputs), cut(sequence.targets), cut(sequence.mask)
