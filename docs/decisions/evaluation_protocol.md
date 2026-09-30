# Evaluation protocol: decision record

**Status:** accepted for V2 evaluation.

## Evaluation target

The model is evaluated primarily as an **action-conditioned model of the joint future conversational scene**. The ego/robot supplies controllable actions; it is not the main prediction target.

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

## Marginal and joint evaluation

Multi-party evaluation distinguishes **marginal** and **joint** prediction:

- **Marginal:** prediction quality for each non-ego participant separately.
- **Joint:** quality of the complete future conversational configuration.

A model can obtain good marginal scores while producing an incoherent joint scene. This is the transferable lesson from joint metrics in multi-agent trajectory forecasting.

The current audio-only V2 labels expose role-relative aggregate targets such as `others_active` and `future_joint_speech_state`, but the model evaluation interface does not yet expose stable participant slots across recordings. Therefore:

- `future_joint_speech_state` is a valid role-relative joint readout;
- `others_active` is an aggregate-other readout, **not** a per-participant marginal metric;
- true participant-level marginal metrics remain pending until the representation/evaluator has a stable participant identity or slot mechanism.

Do not report aggregate `others_active` as "multi-agent marginal accuracy".

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

- aggregate activity of other participants (`others_active`);
- joint speech state;
- time to next speaker onset;
- future joint speech state.

Participant-level future activity, next-speaker and other multi-party readouts should only be added when their participant identity semantics are explicit and stable.

### 4. Action-conditioned dynamics

Keep action evaluation separate from raw prediction error. The existing observed/no-event/shuffled action ablation is retained as a diagnostic that the predictor uses the conditioning sequence. It does not by itself establish correct off-policy counterfactual reactions.

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

When planning is implemented, report at least:

- decision/planning latency (p50 and p95);
- candidate rollouts evaluated per decision;
- compute-budget versus planning-performance curve;
- peak memory when relevant.

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

- VAP-family work: class-imbalance-aware turn-taking readouts such as Shift/Hold and future voice-activity tasks;
- Waymo Open Motion Dataset: interactive forecasting requires joint, not only single-agent, evaluation;
- Weng et al., *Joint Metrics Matter* (2023): marginal accuracy can hide incoherent multi-agent futures;
- AD-E2E-JEPA (arXiv:2609.34085): separate world-model prediction, zero-shot planning quality, reliability and planning efficiency;
- *Evaluating World Models* (arXiv:2607.06401): prediction quality and decision/control utility are distinct evaluation levels.

Domain-specific metrics are not copied when the measured quantity has no conversational analogue.

Exact formulas, implementation names and metric provenance are documented in [`../metrics.md`](../metrics.md).
