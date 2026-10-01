# Metrics

This document records the metrics currently implemented in the repository,
their exact formulas, what they are used for, and their literature provenance
when a meaningful source exists.

The evaluation design itself is fixed in
[`docs/decisions/evaluation_protocol.md`](decisions/evaluation_protocol.md).

## Principles

1. Evaluate three co-primary views of future conversational activity: **ego/robot**, **other participants**, and the **joint scene**.
2. Keep **prediction**, **representation**, **action conditioning** and
   **planning** as separate evaluation levels.
3. Prefer established metrics when they measure the same object.
4. Do not import a benchmark metric only because it has a similar name.
5. Keep the headline table small; diagnostic metrics remain available when
   they answer a specific hypothesis.

## Implemented primary and diagnostic metrics

### Autoregressive rollout MSE

**Code:** `src/turn_wm/training/metrics.py`  
**Names:** `val/rollout_{h}_mse`  
**Role:** primary dynamics metric.

For horizon (h), with (N_h) valid examples and latent dimension (D),

[
operatorname{MSE}_h =
rac{1}{N_h D}
sum_{i=1}^{N_h}
left|
hat z^{(i)}_{t+h}-z^{(i)}_{t+h}
ight|_2^2 .
]

V2 reports the configured physical horizons:

- (h=1): 0.1 s;
- (h=5): 0.5 s;
- (h=10): 1.0 s.

The prediction is autoregressive: later predictions are obtained after feeding
earlier model predictions back into the rollout.

**Provenance:** mean-squared error is a standard regression/prediction error;
there is no single paper whose metric definition is being adopted here.
AD-E2E-JEPA also uses squared latent distance as the cost between predicted and
goal latent states during zero-shot planning, but that does not make its
planning cost identical to this rollout metric.

Reference:
Haoran Zhu, Wancong Zhang, Yann LeCun, Anna Choromanska,
*AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End
Autonomous Driving*, arXiv:2609.34085, 2026.
https://arxiv.org/abs/2609.34085

### Persistence MSE

**Code:** `src/turn_wm/training/metrics.py`  
**Names:** `val/persistence_{h}_mse`  
**Role:** dynamics baseline.

The persistence prediction copies the final observed context latent:

[
hat z^{mathrm{pers}}_{t+h}=z_t .
]

Its error is

[
operatorname{MSE}^{mathrm{pers}}_h =
rac{1}{N_hD}
sum_i
left|
z^{(i)}_t-z^{(i)}_{t+h}
ight|_2^2 .
]

The model MSE and persistence MSE are computed on exactly the same valid
examples.

**Provenance:** this is a persistence / no-change baseline rather than a new
metric. It is deliberately reported as raw MSE instead of being folded into a
project-specific normalized score.

### Transition / stable stratified rollout MSE

**Code:** `src/turn_wm/training/metrics.py`  
**Names:** `val/transition_{h}_mse`, `val/stable_{h}_mse`, plus matching
persistence baselines and counts.  
**Role:** diagnostic slice of dynamics quality.

The metric is exactly the same rollout MSE, evaluated on two subsets.

For the target (z_{C+h-1}), the relevant action sequence is

[
a_{C-1},a_C,ldots,a_{C+h-2}.
]

A horizon is currently classified as:

[
	ext{transition}
iff
exists a 
eq 	exttt{NO\_EVENT}
]

within that required sequence; otherwise it is `stable`.

This definition correctly counts a round trip such as onset followed by
offset as a transition horizon even when the endpoint vocal state returns to
its initial value.

**Provenance:** project-specific stratification, not a new metric and not
claimed to originate from a benchmark. Its purpose is to detect whether a
good average MSE is dominated by easy persistence cases.

### Effective rank

**Code:** `src/turn_wm/training/metrics.py`  
**Name:** `val/effective_rank`  
**Role:** representation-collapse / effective-dimensionality diagnostic.

Let (sigma_1,ldots,sigma_D) be the singular values of the centred latent
matrix and

[
p_i=rac{sigma_i}{sum_jsigma_j}.
]

The repository computes

[
r_{mathrm{eff}}
=
expleft(
-sum_i p_ilog p_i
ight).
]

A rank-one representation has effective rank near 1; a representation whose
singular-value mass is broadly distributed has a larger effective rank.

**Source:** Olivier Roy and Martin Vetterli,
*The Effective Rank: A Measure of Effective Dimensionality*, EUSIPCO 2007,
pp. 606-610.

Reference:
https://dblp.org/rec/conf/eusipco/RoyV07

This is a diagnostic only. A higher effective rank is not automatically a
better world model.

### Balanced accuracy

**Code:** `src/turn_wm/evaluation/latent_analysis/probes.py`  
**Output:** categorical linear-probe score.  
**Role:** task readout / turn-taking relevance.

For (K) canonical classes,

[
operatorname{BalancedAccuracy}
=
rac{1}{K}
sum_{k=1}^{K}
rac{TP_k}{TP_k+FN_k}.
]

The implementation keeps the canonical (K)-class task fixed. If a class is
missing from an evaluation setting, that setting is marked unsupported rather
than silently becoming a (K-1)-class problem.

**Source:** Kay H. Brodersen, Cheng Soon Ong, Klaas E. Stephan,
Joachim M. Buhmann,
*The Balanced Accuracy and Its Posterior Distribution*, ICPR 2010,
pp. 3121-3124, DOI 10.1109/ICPR.2010.764.

Reference:
https://doi.org/10.1109/ICPR.2010.764

**Turn-taking precedent:** Voice Activity Projection work evaluates
turn-taking readouts such as Shift/Hold, Shift Prediction and Backchannel
Prediction, motivating class-balanced evaluation of conversational events.

Erik Ekstedt and Gabriel Skantze,
*Voice Activity Projection: Self-supervised Learning of Turn-taking Events*,
Interspeech 2022.
https://arxiv.org/abs/2205.09812

Balanced accuracy is used here because the current probe tasks can be
imbalanced; the VAP task definitions themselves are not copied blindly into
the multi-party setting.

### Coefficient of determination (R²)

**Code:** `src/turn_wm/evaluation/latent_analysis/probes.py`  
**Output:** continuous linear-probe score.  
**Role:** secondary task-readout diagnostic.

[
R^2
=
1-
rac{
sum_i (y_i-hat y_i)^2
}{
sum_i (y_i-ar y)^2
}.
]

A constant prediction equal to the evaluation-set mean has (R^2=0); a
perfect prediction has (R^2=1). Negative values are possible.

**Provenance:** classical regression statistic; no specific modern benchmark
paper is treated as its source here.

For timing variables, an interpretable MAE in seconds/milliseconds is still
**planned** and is not yet reported by the probe pipeline.

## Implemented analyses that are not headline metrics

### Teacher-forcing MSE

**Name:** `val/tf_mse`.

[
operatorname{MSE}_{TF}
=
rac{1}{ND}
sum
left|
hat z_{t+1}(z_t,a_t)-z_{t+1}
ight|_2^2.
]

This measures one-step prediction under ground-truth state history. It is a
training diagnostic; autoregressive rollout MSE is the world-model dynamics
result.

### Legacy persistence-normalized skill

The helper remains available because existing rollout-analysis artifacts use
it:

[
operatorname{Skill}_h
=
1-
rac{
operatorname{MSE}_{model,h}
}{
operatorname{MSE}_{persistence,h}
}.
]

It is **not** part of the new training-time metric surface and is **not** a
checkpoint-selection metric. It is project-specific normalization, not an
established conversational benchmark metric.

### Displacement alignment and movement ratio

The existing `rollout_dynamics` analysis also contains exploratory latent
geometry diagnostics:

[
operatorname{Alignment}
=
cos(
hat z_{t+h}-z_t,
z_{t+h}-z_t
)
]

and an aggregate movement ratio comparing predicted and true latent
displacement magnitudes.

These are retained for hypothesis-driven analysis but are not headline model
metrics and have no claimed benchmark provenance.

### Observed / state-preserving / shuffled action ablation

The action-ablation analysis compares rollouts under the observed ego-action
sequence against a state-preserving future (WAIT if silent, HOLD if speaking)
and against shuffled future ego-action sequences.

This is an **experimental intervention on the input**, not a standalone
metric. The resulting rollout errors answer whether the predictor uses its
action/event conditioning. They do not establish that an unseen
counterfactual reaction is causally correct.

## Marginal versus joint evaluation

This distinction is frozen even though participant-level metrics are not fully
implemented yet.

- **Ego:** prediction quality for the robot/focal speaker's future activity and turn-taking role.
- **Aggregate other:** role-relative activity of the non-ego participants as a group.
- **Marginal other:** prediction quality for each identifiable non-ego participant separately.
- **Joint:** quality of the complete future conversational configuration across ego and the other participants.

The motivation is directly transferable from multi-agent forecasting:
single-agent/marginal errors can look good while the jointly predicted scene
is incoherent.

Source:
Erica Weng, Hana Hoshino, Deva Ramanan, Kris Kitani,
*Joint Metrics Matter: A Better Standard for Trajectory Forecasting*,
ICCV 2023.
https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html

The current V2 role-relative `future_joint_speech_state` probe is a useful
joint conversational readout. `others_active` is a first-class aggregate-other
readout, but it must not be described as a per-participant marginal metric.
Ego activity is also first-class: the frozen protocol includes future ego
speaking plus turn-taking readouts such as Hold, Shift / observable yield
outcome, onset/offset, overlap participation and backchannel events whenever
the label semantics support them.

The data layer already contains participant-aware speech primitives, including
future speaker activity, but the model-side evaluation does not yet expose a
stable participant-slot/identity interface across recordings. True participant-level marginal evaluation therefore remains pending. Stable
local participant slots are sufficient; they do not need global identities
across recordings.

## Planning metrics: frozen family, not implemented yet

Planning is a separate evaluation stage. Once a planner is implemented, it
should evaluate candidate ego-action sequences through world-model rollouts.

The relevant transferable precedent is AD-E2E-JEPA, which evaluates a world
model through goal-conditioned zero-shot planning and separates:

- task/driving performance;
- endpoint/geodesic accuracy;
- hit rate / reliability;
- planning efficiency.

For this project, those exact driving metrics are **not** copied. The
transferable principle is to measure whether latent rollouts actually support
candidate-action ranking and decision making.

Reference:
Haoran Zhu et al., arXiv:2609.34085, 2026.
https://arxiv.org/abs/2609.34085

The broader separation between predictive world-model quality and downstream
decision/control utility is also consistent with:

Xinyuan Chen et al.,
*A Definition and Roadmap for World Models*, arXiv:2607.06401, 2026.
https://arxiv.org/abs/2607.06401

## Current headline table

For the current deterministic V2, the minimal model-comparison table should
contain:

| Family | Metric |
| --- | --- |
| Dynamics | rollout MSE at 0.1 / 0.5 / 1.0 s |
| Baseline | persistence MSE at the same horizons |
| Dynamics slices | transition/stable rollout MSE |
| Representation health | effective rank |
| Task readout | balanced accuracy for categorical probes |
| Task readout | R² for current continuous probes |
| Reliability | the same metrics broken down by corpus |

Action-conditioned ablations are reported separately. Planning and
participant-level marginal metrics are not claimed until their corresponding
interfaces are implemented.
