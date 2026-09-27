# ADR 0002 — V2 uses hierarchical temporal world modelling for planning

- Status: **Accepted**
- Date: 2026-09-27

## Context

The project objective is planning in conversational interaction, not TurnBench
classification. V1 established that a compact latent can retain useful
conversation structure, that autoregressive latent dynamics can beat
persistence, and that the predictor reacts to its conditioning sequence.

The remaining problem is temporal planning: reason over seconds without
searching directly over a very long 10 Hz primitive-action sequence, while
retaining precise sub-second control.

## Decision

V2 introduces a two-level temporal hierarchy over a shared latent state.

```text
observations
    ↓
 encoder / projector
    ↓
 shared latent z_t @ 10 Hz
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
FAST WORLD MODEL              SLOW WORLD MODEL
Δ = 100 ms                    Δ ≈ 1 s
primitive control             macro-actions
short-horizon dynamics        long-horizon dynamics
    │                              │
    └──────────────┬───────────────┘
                   ↓
          hierarchical MPC
```

A forced speaker-factorized latent such as `z_scene + z_A + z_B` is **not**
part of this decision. It was considered after TurnBench, but TurnBench is not
the project objective and does not justify making speaker factorization the V2
core. The default remains a shared interaction latent; factorization may be a
later ablation only if planning evidence motivates it.

## Temporal contract

The canonical decision grid remains 10 Hz:

```text
Δ = 100 ms
```

This is distinct from audio sample rate or video FPS.

### Fast level

Default planning/model scale:

- stride: 1 grid step
- temporal resolution: 100 ms
- default learned/planning horizon: up to 10 steps ≈ 1 s
- role: precise reactive control, onset/offset timing, waiting, continuation
  and short-horizon consequence prediction

The V1 fast dynamics are the starting baseline; V2 does not discard the
validated 10 Hz dynamics.

### Slow level

Default scale:

- stride: 10 grid steps
- temporal resolution: about 1 s
- default horizon: 5 slow steps = about 5 s
- conceptual reasoning range: roughly 1–10 s
- role: interaction-level trajectory, longer-term consequences and latent
  subgoal generation

The dataset/model-ready window therefore needs at least 50 future 10 Hz grid
steps for the default 5 s slow horizon.

A 20 Hz base grid is not the V2 default. It remains an ablation only; changing
the base grid would require changing context lengths and all temporal contracts.

## Macro-actions

The slow level does not directly search every 100 ms primitive token.

A slow action encoder compresses a block of fast actions into a macro-action:

```text
[a_t, ..., a_t+9]
        ↓
 slow action encoder
        ↓
   macro-action l_t
```

The exact macro-action representation is an implementation choice to validate
in V2, but the abstraction boundary is accepted: primitive actions live on the
10 Hz grid and slow planning operates on a coarser action representation.

## Latent subgoals

The slow planner predicts/selects a desirable future interaction state. That
future latent becomes a subgoal for the fast planner.

```text
slow:
z_t + macro-action sequence
        ↓
slow rollout
        ↓
desired latent subgoal z*

fast:
z_t + primitive candidate sequence
        ↓
fast rollout
        ↓
trajectory approaching z*
```

The fast controller therefore answers "how do I reach the selected interaction
state?" while the slow level answers "which interaction state should I aim for
over the next seconds?"

## Hierarchical MPC

Planning is receding-horizon MPC at inference time; it is not a monolithic
planner jointly trained inside the world model.

```text
observe z_t
   ↓
generate candidate macro-action sequences
   ↓
slow-world-model rollouts
   ↓
score with goal-dependent objective J_g
   ↓
select latent subgoal / macro trajectory
   ↓
generate primitive action sequences
   ↓
fast-world-model rollouts
   ↓
score against the subgoal and local constraints
   ↓
execute first primitive action
   ↓
observe again and replan
```

Only the first primitive action is committed before replanning.

## Goal-dependent objective

The planner must not hard-code TurnBench EOT/INT as the project objective.

For a goal `g`, candidate trajectories are scored by an explicit objective of
the form:

```text
J_g(context, action_sequence, predicted_rollout)
```

The cost/reward may use direct action properties and/or measurable properties
of the predicted rollout. A planning goal is valid only if its desired outcome
is time-indexed/measurable and candidate action sequences can change the
predicted trajectory relevant to that outcome.

The exact family of planning goals and whether parts of `J_g` are analytic or
learned remain separate V2 experiment decisions.

## Training contract

Both temporal levels use the same basic world-model principles validated in V1:

- teacher forcing for local transition learning;
- autoregressive rollout using the model's own predicted latents;
- multi-horizon losses;
- SIGReg / latent-health regularization;
- persistence and trajectory diagnostics;
- no ground-truth future latent fed back during rollout.

For the fast model, V1's 0.1/0.5/1.0 s multi-horizon recipe is the starting
baseline. The slow model adds supervision at the coarser temporal scale rather
than replacing the fast objective.

## Data and representation constraints

V2 preserves the data contract:

- canonical source timestamps remain authoritative;
- the 10 Hz grid is a model/decision projection, not a replacement for native
  media timing;
- provenance and dataset revisions remain explicit;
- derived profiles and analysis labels are diagnostics, not prediction inputs;
- no label leakage from future turn-taking annotations into observations.

The first hierarchical-planning iteration remains ego/focal-agent centric.
Exocentric/passive/multi-agent transfer is deferred to a later iteration.

Multimodal observations remain an architectural direction, but adding a new
video representation is not a prerequisite for validating the hierarchy. The
V1 audio path remains the controlled baseline until any additional observation
path is independently validated.

## Consequence

V2 is not "V1 plus a better turn-taking head." It is V1's validated fast latent
dynamics extended with:

1. causal agent-control semantics;
2. a slow temporal abstraction;
3. macro-actions;
4. latent subgoals;
5. hierarchical receding-horizon planning.
