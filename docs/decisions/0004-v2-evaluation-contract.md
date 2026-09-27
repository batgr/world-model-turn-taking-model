# ADR 0004 — V2 evaluation is planning-first and layered

- Status: **Accepted**
- Date: 2026-09-27

## Purpose

V2 must be evaluated in layers so that representation quality, prediction
accuracy, action sensitivity and planning utility are not conflated.

TurnBench and other downstream classifiers remain useful diagnostics, but they
must not become the definition of success.

## Layer 1 — Representation health

Retain V1 diagnostics:

- latent standard deviation and norm;
- effective rank / spectrum;
- PCA only for visualization;
- label-conditioned structure and linear probes;
- corpus-conditioned analyses.

These are diagnostics and guardrails. They are not automatically optimization
targets and do not justify semantic claims by themselves.

## Layer 2 — Dynamics quality

For both fast and slow models, report:

- prediction error;
- persistence baseline;
- skill = 1 − error_model / error_persistence;
- displacement-direction alignment;
- predicted-vs-true movement magnitude;
- horizon-specific metrics;
- per-corpus metrics.

Fast V2 must preserve or improve the positive V1 skill at 0.1, 0.5 and 1 s.
Slow V2 must demonstrate useful prediction beyond the fast horizon rather than
merely copying the current latent.

## Layer 3 — Causal action use

Repeat the V1 ablation logic under the corrected action semantics:

- observed/chosen action vs neutral action;
- shuffled action sequence;
- counterfactual action changes from the same latent/context;
- action leverage by horizon and interaction state.

The critical difference from V1 is that the compared V2 actions must be
agent-controllable commands, not future observed conversation events.

## Layer 4 — Temporal hierarchy

Evaluate whether the slow abstraction adds value:

- slow model vs persistence;
- slow model vs a fast-only rollout projected to the same horizon;
- macro-action ablation/shuffling;
- latent-subgoal stability and reachability by the fast controller;
- error accumulation as the slow horizon grows.

A slow model that does not improve long-horizon planning evidence is not kept
merely because it is architecturally elegant.

## Layer 5 — Planning utility

Planning is a separate evaluation target from model prediction.

For each predeclared goal `g`, evaluate complete candidate trajectories using:

```text
J_g(context, action_sequence, predicted_rollout)
```

The planner must be compared with non-planning baselines appropriate to the
same action space, such as a fixed/neutral policy or a greedy one-step policy.
The exact benchmark protocol is a V2 experiment decision, but the comparison
must isolate the value added by multi-step planning.

Required claims are progressively stronger:

1. candidate actions lead to different predicted futures;
2. the objective ranks those futures differently;
3. hierarchical planning chooses different actions when the goal changes;
4. chosen actions improve measured goal outcomes relative to the declared
   baselines.

Only after (4) is demonstrated should the project claim planning utility.

## External diagnostics

TurnBench remains an external representation/turn-taking diagnostic.

The completed V1 DEV results are a baseline:

| representation | EOT recall | INT recall |
|---|---:|---:|
| Mimi | 0.595 | 0.553 |
| current z_t | 0.590 | 0.484 |
| predicted latent | 0.536 | 0.481 |

V2 does not need to maximize these numbers at the expense of planning. They are
used to detect whether a representation destroys useful conversational
information.

TurnBench TEST remains out of scope unless access and evaluation rules
explicitly permit it. TRAIN/DEV data are not redistributed.

## Checkpoint selection

V1 selected checkpoints by `val/loss`.

During V1 analysis, alternative selection signals were raised as hypotheses:
rollout skill, horizon-specific skill and latent-health guardrails. They are
**not yet accepted V2 checkpoint criteria**.

V2 must define checkpoint selection before the corresponding experiment. It
must not select retrospectively using the metric on which the final result is
reported.

## Data leakage and evaluation hygiene

- no test split during training or model selection;
- no future labels as observation inputs;
- no analysis-only profile columns as model inputs;
- whole-recording/corpus grouping must be respected where required;
- confidence intervals should resample recordings when anchors are temporally
  dependent;
- every report records checkpoint, dataset revision, code revision and seed.

## Decision rule for V2 changes

An architectural addition is retained only if it improves evidence relevant to
planning or fixes a demonstrated V1 limitation. A downstream diagnostic alone
is insufficient reason to redefine the architecture.
