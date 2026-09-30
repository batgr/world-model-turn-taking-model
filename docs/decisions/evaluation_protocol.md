# Evaluation protocol: decision record

**Status:** accepted for V2 evaluation.

## Evaluation target

The model is evaluated as an **action-conditioned model of future conversational activity** through three co-primary views:

1. **Ego/robot future activity and turn-taking role**: whether the ego speaks, continues/holds the floor, releases/yields the floor, starts or stops speaking, and participates in overlap/backchannel-related events when those events are operationally defined by the labels.
2. **Other-participant future activity**: aggregate activity of the other participants, and participant-level marginal activity / next-speaker prediction whenever stable local participant slots are identifiable.
3. **Joint future conversational configuration**: the simultaneous configuration of ego and other participants, including silence, exclusive speech, overlap, floor changes and richer participant-aware joint states when the interface supports them.

The ego/robot therefore has two roles: it is a **prediction target for turn-taking behavior** and the **source of controllable actions** for action-conditioned rollouts and later planning. Ego-only quality is never treated as sufficient evidence of a good multi-party world model.

For the current deterministic V2, evaluation covers one predicted future rollout per supplied ego-action sequence. Probabilistic metrics such as NLL, Brier score, calibration, or best-of-K trajectory metrics are out of scope until the model represents a distribution over futures.

## Metric families

| Family | Question |
| --- | --- |
| **Representation quality** | Does the latent retain task-relevant conversational information without collapsing? |
| **Dynamics / predictive quality** | Does the world model predict the future latent trajectory at useful horizons? |
| **Task readout / turn-taking relevance** | Can turn-taking variables be decoded from the representation or predicted rollout? |
| **Action-conditioned dynamics** | Does the predicted future change appropriately when the ego action changes? |
| **Planning performance** | Can world-model rollouts rank/select useful candidate ego-action trajectories? |
| **Reliability / robustness** | Do the same primary metrics remain stable across horizons, corpora and future stress conditions? |
| **Efficiency** | Can model rollouts and planning operate inside conversational latency and compute budgets? |

**Interaction-performance metrics are explicitly out of scope for this phase.** They require the complete interactive system, not the world model alone.

## Ego, marginal-other and joint evaluation

Multi-party evaluation keeps three complementary views and none replaces the others:

- **Ego:** future activity and turn-taking role of the robot/focal speaker.
- **Marginal other:** prediction quality for each identifiable non-ego participant separately.
- **Joint:** quality of the complete future conversational configuration across ego and the other participants.

When participant identities cannot be represented consistently, **aggregate-other** targets such as `others_active` remain first-class role-relative targets, but they are reported explicitly as aggregate-other rather than as participant-level marginal metrics.

A model can obtain good ego or marginal scores while producing an incoherent joint scene. Conversely, a coarse joint label can hide which participant was predicted incorrectly. The protocol therefore requires both marginal/aggregate and joint views whenever the data supports them.

The current audio-only V2 labels expose role-relative aggregate targets such as `others_active` and `future_joint_speech_state`, but the model evaluation interface does not yet expose stable participant slots across recordings. Therefore:

- `future_joint_speech_state` is a valid role-relative joint readout;
- `others_active` is a first-class aggregate-other readout;
- participant-level marginal metrics are required whenever stable local participant slots can be constructed;
- participant identities need only be stable within an interaction/window; they do not need a global identity vocabulary across recordings.

## Frozen turn-taking readouts

The following conversational readouts are in scope. They are added only when the underlying labels have explicit, reproducible semantics.

**Ego / focal-speaker readouts**
- future speaking/activity state;
- speech onset and offset;
- **Hold**: the current floor holder continues holding the floor across the evaluation window;
- **Shift / yield outcome**: the current floor holder releases the floor and another participant takes it. "Yield" is not interpreted as latent intention unless the dataset explicitly labels intention;
- overlap participation;
- backchannel occurrence/appropriateness when a reproducible backchannel label exists.

**Other-participant readouts**
- aggregate other activity;
- future other onset/offset;
- participant-level future activity when stable local slots exist;
- next-speaker prediction;
- current-speaker-continues versus floor-shift.

**Joint-scene readouts**
- joint speech state;
- silence / gap;
- overlap;
- floor-holder change;
- richer participant-aware joint activity configuration when local participant slots exist.

VAP-family Shift/Hold and backchannel tasks are direct turn-taking precedents. MuVAP adds a directly relevant multi-party precedent for Shift/Hold and next-speaker prediction.

## V2 metrics implemented now

### 1. Dynamics / predictive quality

The primary dynamics metric is standard autoregressive latent rollout MSE:

`MSE_h = mean(||z_hat[t+h] - z[t+h]||^2)`

Report it at the physical V2 horizons:

- 0.1 s (`h=1`);
- 0.5 s (`h=5`);
- 1.0 s (`h=10`).

Report **persistence MSE** at the same horizons as a separate baseline, where the last observed context latent is copied into the future. Persistence is a baseline, not folded into a custom headline skill metric.

The same rollout MSE may be stratified on windows containing an ego vocal transition versus windows without one. This is a shortcut/persistence diagnostic, not a separate metric family and not evidence of full multi-agent turn-taking quality.

Teacher-forcing MSE remains a training diagnostic. Autoregressive rollout MSE is the dynamics result.

### 2. Representation quality

Use:

- **effective rank** as a collapse/representation-health diagnostic;
- train-fitted, validation-evaluated linear readouts for task-relevant variables.

Effective rank is not a model-selection score: small differences in rank are not interpreted as better dynamics or better planning.

### 3. Task readout / turn-taking relevance

For categorical readouts use **balanced accuracy** as the primary score. The current continuous probes report **R²**. An interpretable **MAE** in seconds/milliseconds is planned for timing readouts before they are promoted to headline results.

Current role-relative readouts include, where labels are available:

- ego speaking state;
- aggregate activity of other participants (`others_active`);
- joint speech state;
- time to next speaker onset;
- future joint speech state.

The frozen protocol additionally requires future ego activity and turn-taking-event readouts (notably Hold/Shift or observable yield outcome), future aggregate-other activity, next-speaker prediction, overlap/gap-related readouts, and participant-level marginal activity whenever stable local participant slots are available.

For categorical readouts, balanced accuracy remains the primary score and macro-F1 / per-class recall are useful complementary diagnostics when event imbalance is material. For timing readouts, MAE in seconds or milliseconds is required before promotion to headline results.

### 4. Action-conditioned dynamics

Keep action evaluation separate from raw prediction error. The action ablation compares observed, state-preserving and shuffled ego-action sequences as a diagnostic that the predictor uses the conditioning sequence. It does not by itself establish correct off-policy counterfactual reactions.

A headline action-fidelity metric is intentionally not frozen yet. It will be selected when the counterfactual/off-policy protocol is defined.

### 5. Planning performance

Not implemented in V2 yet. When planning is added, evaluate candidate ego-action trajectories through world-model rollouts and task/goal costs. Candidate ranking, Hit@K-style reliability, task success and planning cost belong here.

Do not call latent MSE a planning metric.

### 6. Reliability / robustness

Reliability reuses the **same primary metrics** rather than inventing new scores. At minimum report dynamics/readout metrics by:

- prediction horizon;
- corpus.

Later controlled strata may include noise, reverberation, participant count, language, long within-turn pauses and self-voice echo.

### 7. Efficiency

Efficiency is split into two levels.

**World-model rollout efficiency — measurable before a planner exists:**
- rollout latency p50 / p95 at fixed horizon and batch/candidate count;
- candidate rollouts per second;
- latent prediction steps per second;
- peak accelerator memory;
- scaling with rollout horizon and candidate batch size.

**Planning efficiency — once a planner exists:**
- end-to-end decision/planning latency p50 / p95;
- candidate rollouts evaluated per decision;
- planning iterations when applicable;
- compute-budget versus planning-performance curve;
- peak memory.

Latency measurements must state hardware, precision, batch/candidate count, rollout horizon and warm-up protocol. Human-perceived response latency remains an interaction-performance metric and is still out of scope for this phase.

## Metrics that are not headline metrics

The following are removed from the main V2 validation comparison unless a later hypothesis specifically requires them:

- cosine similarity between predicted and target latent;
- latent norm and prediction norm;
- latent standard deviation;
- custom persistence-normalized skill score;
- teacher-forcing skill.

PCA and other exploratory analyses remain research diagnostics, not model-selection metrics.

## External precedents

The structure borrows only transferable principles from established benchmarks and papers:

- Ekstedt & Skantze, *Voice Activity Projection* (2022): future voice activity, Shift/Hold, turn-shift and backchannel readouts;
- Qi & Skantze, *MuVAP* (2026): multi-party Shift/Hold and next-speaker prediction with role-relative projection;
- Skantze, *Turn-taking in Conversational Systems and Human-Robot Interaction: A Review* (2021): turn-holding/yielding, gaps, overlaps, interruptions and multi-party floor management;
- Waymo Open Motion Dataset: interactive forecasting requires joint, not only single-agent, evaluation;
- Weng et al., *Joint Metrics Matter* (2023): marginal accuracy can hide incoherent multi-agent futures;
- AD-E2E-JEPA (arXiv:2609.34085): separate world-model prediction, zero-shot planning quality, reliability and planning efficiency;
- *Evaluating World Models* (arXiv:2607.06401): prediction quality and decision/control utility are distinct evaluation levels.

Domain-specific metrics are not copied when the measured quantity has no conversational analogue.

Exact formulas, implementation names and metric provenance are documented in [`../metrics.md`](../metrics.md).
