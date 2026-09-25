import math
from itertools import pairwise

import pytest
import torch

from turn_wm.training.scheduler import (
    warmup_cosine_factor,
    warmup_cosine_scheduler,
    warmup_steps_for,
)

BASE_LR = 1e-4
MIN_LR = 1e-6


def make_scheduler(total_steps, *, warmup_ratio=0.05):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=BASE_LR, weight_decay=1e-3)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr=MIN_LR,
    )

    return optimizer, scheduler


def lrs(total_steps, **kwargs) -> list[float]:
    """LR used by each optimizer step 0 .. total_steps - 1."""

    optimizer, scheduler = make_scheduler(total_steps, **kwargs)
    seen = []

    for _ in range(total_steps):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return seen


def test_starts_at_min_lr():
    assert lrs(1_000)[0] == pytest.approx(MIN_LR)


def test_warmup_reaches_the_base_lr_at_five_percent():
    schedule = lrs(1_000)

    assert warmup_steps_for(1_000, 0.05) == 50
    assert schedule[50] == pytest.approx(BASE_LR)
    # Linear warmup: halfway between min and base LR at step 25.
    assert schedule[25] == pytest.approx((MIN_LR + BASE_LR) / 2)
    assert all(a < b for a, b in pairwise(schedule[:51]))


def test_cosine_decays_after_the_warmup():
    schedule = lrs(1_000)

    assert all(a > b for a, b in pairwise(schedule[50:]))
    # Cosine midpoint: halfway through the decay the LR is halfway down.
    middle = 50 + (1_000 - 1 - 50) // 2
    assert schedule[middle] == pytest.approx((MIN_LR + BASE_LR) / 2, rel=1e-2)


def test_ends_at_min_lr_on_the_last_optimizer_step():
    assert lrs(1_000)[-1] == pytest.approx(MIN_LR)


def test_steps_past_the_end_stay_at_min_lr():
    factor = warmup_cosine_factor(
        1_200, total_steps=1_000, warmup_steps=50, min_lr_ratio=MIN_LR / BASE_LR
    )

    assert factor * BASE_LR == pytest.approx(MIN_LR)


@pytest.mark.parametrize("total_steps", [1, 2, 3, 5])
def test_tiny_runs_stay_within_bounds(total_steps):
    schedule = lrs(total_steps)

    assert all(MIN_LR - 1e-12 <= lr <= BASE_LR + 1e-12 for lr in schedule)
    assert all(math.isfinite(lr) for lr in schedule)


def test_a_single_step_run_uses_the_base_lr():
    # No room for a warmup: the only step trains at the configured LR.
    assert lrs(1) == pytest.approx([BASE_LR])


def test_no_warmup_starts_at_the_base_lr():
    schedule = lrs(100, warmup_ratio=0.0)

    assert schedule[0] == pytest.approx(BASE_LR)
    assert schedule[-1] == pytest.approx(MIN_LR)


def test_schedule_follows_total_optimizer_steps():
    # Same step, different run lengths: the run length sets the schedule.
    assert lrs(200)[100] != pytest.approx(lrs(1_000)[100])


def test_state_dict_resumes_at_the_same_lr():
    optimizer, scheduler = make_scheduler(1_000)

    for _ in range(300):
        optimizer.step()
        scheduler.step()

    state = scheduler.state_dict()
    resumed_optimizer, resumed = make_scheduler(1_000)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed.load_state_dict(state)

    assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())

    for _ in range(10):
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()

    assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())
    assert resumed.last_epoch == scheduler.last_epoch == 310


def test_several_base_lrs_are_rejected():
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3},
    ]

    with pytest.raises(ValueError, match="one base LR"):
        warmup_cosine_scheduler(
            torch.optim.AdamW(groups), total_steps=10, warmup_ratio=0.05, min_lr=1e-6
        )
