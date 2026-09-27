# ADR 0003 — Separate observed events from causal agent actions in V2

- Status: **Accepted**
- Date: 2026-09-27

## Problem in V1

V1 uses `NO_EVENT / ONSET / OFFSET / MASKED` tokens from the action grid.
During multi-step validation rollouts, future ground-truth event tokens are
provided to the predictor.

Consequently, V1 answers:

> Given this observed future event sequence, can the model evolve the latent
> state through the associated interaction transition?

It does not answer:

> What will happen if the agent chooses this action?

This distinction is the main reason the V1 action-ablation results cannot yet
be called planning evidence.

## Decision

V2 has two semantically separate channels:

```text
environment / human events e_t
        ↓
     observation

agent command a_t
        ↓
 controllable intervention
        ↓
world-model transition
```

Human speaking activity, other participants and naturally occurring
conversation events remain part of the observed state. They are not treated as
actions chosen by the planner.

## Primitive control semantics

The accepted minimal causal control representation is a vocal command on the
10 Hz decision grid:

```text
SPEAK
SILENT
```

Observed transitions such as ONSET and OFFSET are derived from state/command
changes and remain useful labels for diagnostics and evaluation; they are not
the primary causal action representation.

A richer primitive action vocabulary may be introduced later only through an
explicit decision, after the binary control contract works.

## Causal alignment

The temporal convention is:

```text
(z_t, a_t) → z_hat_(t+1)
```

The action selected at decision time `t` must influence the very next
predicted latent. V2 must not have an off-by-one rollout in which the first
candidate action is added only after predicting the first future state.

Tests must explicitly verify this alignment.

## Fast and slow action representations

Fast model:

```text
a_t = primitive causal command @ 10 Hz
```

Slow model:

```text
[a_t ... a_t+9] → slow action encoder → l_t
```

where `l_t` is the macro-action representation used by the slow dynamics.

This lets the high-level planner reason over seconds while the low-level
controller retains 100 ms timing.

## Planning validity criterion

A candidate action channel is useful for planning only when all of the
following hold:

1. the action is actually controllable by the agent;
2. changing only the candidate action changes predicted future states;
3. the difference persists at horizons relevant to the goal;
4. a measurable goal objective can distinguish the resulting trajectories.

V1 satisfied only part of (2) for event-conditioned counterfactuals. V2 must
establish the full contract with causal commands.

## Non-goals

The following are not agent actions and must not be smuggled into the control
channel:

- future human onset/offset labels;
- future joint-speech-state labels;
- diarization or ASR outputs that reveal future behavior;
- lexical/syntactic completion labels;
- analysis-only profiles or probes.

No causal-control claim is made merely because a latent changes under an
out-of-distribution token intervention.
