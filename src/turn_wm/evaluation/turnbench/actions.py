"""
Per-speaker V1 actions on the 100 ms control grid, from causal activity.

The semantics follow the published V1 action grid:

- `state_before[k]`: the speaker's state when slot `k` starts, i.e. the
  activity of the last window ending at t_k;
- `action[k]`: the transition inside slot `k`, from `state_before[k]` through
  the slot's windows to `state_before[k + 1]`:
  no change -> NO_EVENT, one SILENT -> SPEAKING flip -> ONSET, one
  SPEAKING -> SILENT flip -> OFFSET, more than one flip -> masked
  (`compound_transition`), never NO_EVENT;
- slot 0: no window ends at t_0, so its state is UNKNOWN and its action is
  masked (`recording_start`), as the V1 grid masks every recording start.

`action[k]` reads activity up to the end of slot `k` only; it is available at
`slot_end_s[k]`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch

from turn_wm.evaluation.turnbench.activity import WINDOW_S, rms_activity
from turn_wm.evaluation.turnbench.data import SPEAKERS, Conversation
from turn_wm.evaluation.turnbench.timing import SLOT_S, slot_end_s, slot_start_s

SILENT, SPEAKING, UNKNOWN = "SILENT", "SPEAKING", "UNKNOWN"
NO_EVENT, ONSET, OFFSET = "NO_EVENT", "ONSET", "OFFSET"
RECORDING_START = "recording_start"
COMPOUND_TRANSITION = "compound_transition"


@dataclass(frozen=True)
class SlotActions:
    """One speaker's state and action per complete control slot."""

    state_before: list[str]
    action: list[str | None]
    action_valid: list[bool]
    mask_reason: list[str | None]
    slot_start_s: list[float]
    slot_end_s: list[float]  # when the action becomes available


def slot_actions(activity: torch.Tensor, *, window_s: float = WINDOW_S) -> SlotActions:
    """Actions of the complete slots covered by `activity` (one bool per window)."""

    per_slot = round(SLOT_S / window_s)

    if per_slot < 1 or abs(per_slot * window_s - SLOT_S) > 1e-9:
        raise ValueError(f"{window_s} s windows do not tile a {SLOT_S} s slot")

    active = activity.bool().tolist()
    slots = len(active) // per_slot
    result = SlotActions([], [], [], [], [], [])

    for k in range(slots):
        if k == 0:
            state, action, reason = UNKNOWN, None, RECORDING_START
        else:
            path = active[k * per_slot - 1 : (k + 1) * per_slot]
            flips = sum(a != b for a, b in pairwise(path))
            state = SPEAKING if path[0] else SILENT

            if flips == 0:
                action, reason = NO_EVENT, None
            elif flips == 1:
                action, reason = (OFFSET if path[0] else ONSET), None
            else:
                action, reason = None, COMPOUND_TRANSITION

        result.state_before.append(state)
        result.action.append(action)
        result.action_valid.append(action is not None)
        result.mask_reason.append(reason)
        result.slot_start_s.append(slot_start_s(k))
        result.slot_end_s.append(slot_end_s(k))

    return result


def conversation_actions(
    conversation: Conversation,
    *,
    activity: Callable[[torch.Tensor, int], torch.Tensor] = rms_activity,
    window_s: float = WINDOW_S,
) -> dict[str, SlotActions]:
    """Each speaker's actions, from that speaker's own channel only."""

    return {
        speaker: slot_actions(
            activity(conversation.channel(speaker), conversation.sample_rate),
            window_s=window_s,
        )
        for speaker in SPEAKERS
    }
