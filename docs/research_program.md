# Research program: hypotheses, roadmap and architecture evolution

**Status:** living research document  
**Last updated:** 2026-09-29

This document is deliberately separate from `docs/decisions/`.

- **Decision records** state choices that have been accepted from project
  evidence.
- **This document** tracks hypotheses, unresolved questions, architecture
  evolution, experiment gates and future research.
- **Zotero** is the bibliographic source of truth; see
  `references/README.md`.
- **Run artifacts** (config, commit, dataset/feature revisions, checkpoint and
  analysis outputs) are the source of truth for experimental results.

A hypothesis may be motivated by literature and still remain unverified in
this project.

## Research objective

Build a predictive world model for multi-party conversational interaction that
eventually supports action selection and planning for a social robot.

The immediate problem is narrower: learn useful latent conversational dynamics
from audio while separating four questions that must not be conflated:

1. Is the latent representation healthy and informative?
2. Can the predictor model multi-step conversational dynamics?
3. Are predicted transitions meaningfully sensitive to actions/events?
4. Can those dynamics eventually support controllable planning?

Good prediction does not automatically imply action grounding or planning
utility.

## Architecture evolution

### V1 — conditional predictive dynamics

```text
frozen Mimi audio features
        ↓
projector
        ↓
       z_t
        ↓  + observed vocal event proxy
causal latent predictor
        ↓
    z_hat_(t+h)
```

Main role: establish that a JEPA-style model can beat latent persistence and
use the observed vocal-event conditioning channel.

Important limitation: future ONSET/OFFSET tokens are observed conversation
events, not actions chosen by an agent. V1/V2 therefore test
**event-conditioned dynamics**, not autonomous anticipation or planning.

### V2 — stronger audio-only dynamics and controlled ablations

Baseline V2 keeps the same conceptual architecture but changes the temporal
recipe and representation regularization setup:

```text
context C=30
rollout window W=30
future H=10 (10 Hz)
multi-horizon evaluation H={1,5,10}
CausalBN projectors
SIGReg lambda=0.09
transition-aware checkpoint metric
```

V2 is the stage where representation geometry, normalization, SIGReg strength,
event sensitivity, domain transfer and probe information are separated through
controlled experiments.

#### V2 baseline

Reference recipe for all controlled comparisons.

#### V2-A — lower SIGReg

Only change from V2:

```text
SIGReg lambda: 0.09 -> 0.03
```

Observed regime so far:

- effective rank falls from about 83 to about 41;
- H1/local skill falls strongly;
- H5/H10 and transition skill improve;
- medium/long-horizon displacement alignment improves;
- event-conditioning ablations show stronger dependence on the observed event
  sequence;
- pooled linear probes remain preserved or slightly improved;
- some cross-domain transfer degrades, especially
  `others_active: EgoCom -> Ego4D`;
- dataset/domain structure becomes stronger in the latent.

Interpretation: lower SIGReg does not simply "collapse" the representation.
It appears to move the model toward a more compressed, task/prediction-focused
regime while potentially sacrificing some domain invariance.

#### V2-B — standard BatchNorm

Only architectural change from V2:

```text
CausalBatchNorm1d -> torch.nn.BatchNorm1d
SIGReg remains 0.09
```

Observed regime so far:

- H1 and one-step displacement improve modestly;
- H5/H10 are almost unchanged from V2;
- effective rank remains about 82;
- label structure and pooled probes are almost indistinguishable from V2;
- cross-domain probe behavior is also very similar to V2.

Current implication: standard BatchNorm does not explain the major V2-A
behavioral change and is not currently justified as the new default.
Action-ablation remains a final comparison before closing this branch.

#### V2-D — positional CausalBN

Controlled normalization test:

```text
projector CausalBN running stats:
  shared -> per temporal position (40 positions)

prediction projector:
  remains per-position (30 positions)

SIGReg remains 0.09
```

Purpose: test whether the original projector has a train/eval running-statistic
mismatch rather than whether BatchNorm as a family is better.

Training has produced checkpoints; full analysis is pending. No scientific
conclusion should be recorded until matched V2 analyses exist.

### End of V2 — freeze the audio representation recipe

Planned gate:

```text
finish D analysis
      ↓
finish B action/event ablation
      ↓
analyse transition geometry Δz on V2/A/B/D
      ↓
choose normalization
      ↓
SIGReg lambda sweep
      ↓
freeze V2 recipe
```

Initial lambda sweep after normalization is fixed:

```text
{0.03, 0.05, 0.07, 0.09}
```

Zero/other values remain diagnostic options if the observed curve warrants
them.

If no lambda gives an acceptable prediction/generalization trade-off, test
temporally centered/residual SIGReg as a **new hypothesis**, not as an
uncontrolled patch.

### V3 — action grounding

V3 is a new architecture/research question, not another V2 ablation.

First explicit discrete conversational actions:

```text
if agent/focal speaker is SILENT:
  WAIT
  START

if agent/focal speaker is SPEAKING:
  CONTINUE
  STOP
```

These make the implicit semantics of NO_EVENT/ONSET/OFFSET explicit:

```text
SILENT   + NO_EVENT -> WAIT
SILENT   + ONSET    -> START
SPEAKING + NO_EVENT -> CONTINUE
SPEAKING + OFFSET   -> STOP
```

Invalid combinations must be masked instead of treated as meaningful controls.

Planned additions:

- inverse dynamics model (IDM);
- action classification from real transitions;
- action-conditioned `Δz` geometry;
- forward/inverse cycle consistency;
- explicit state/action validity masks;
- controlled comparison against the current event-proxy formulation.

#### Continuous / learned actions

Do not assume the final action space must remain discrete.

A parallel V3 research branch will test learned continuous/constrained actions:

```text
u_t in R^d
```

motivated by latent-action world-model work. The discrete model remains the
interpretable reference.

Questions include:

- Does a continuous action latent separate START/STOP/WAIT/CONTINUE while also
  capturing intensity/timing/style?
- Does it improve forward prediction?
- Can known conversational actions be mapped into/out of the latent action
  interface?
- Does it make planning easier or merely less interpretable?

### V3-P — planning

Planning starts only after V3 demonstrates that action-conditioned transitions
are identifiable and coherent.

The architecture must distinguish controllable robot action from exogenous
conversation events:

```text
z_t
 + controllable action u_t
 + exogenous/social event e_t
        ↓
forward world model
        ↓
z_hat_(t+1:t+H)
```

Planner research questions:

- discrete enumeration vs CEM/continuous optimization;
- goal latent vs state prototype vs learned cost;
- whether latent distance ranks candidate actions correctly;
- online re-planning after each observation;
- whether IDM consistency is a useful planner guardrail;
- how to penalize overlap/interruption/excessive floor-taking.

The first demo should expose candidate actions, predicted outcomes, planner
cost and selected action rather than hiding the decision process.

### V4 — hierarchical world model

Later architecture inspired by hierarchical latent planning:

```text
shared conversational latent
       ├── low-level dynamics
       │     ~100 ms
       │     WAIT / START / CONTINUE / STOP
       │
       └── high-level dynamics
             multi-second goals / macro-actions
             floor management / addressing / participation balance
```

Goal: test whether conversation benefits from explicit temporal hierarchy
rather than forcing one predictor/action space to cover every timescale.

## Current evidence snapshot

These numbers are diagnostics, not a model ranking.

### Matched rollout-dynamics analysis (10k validation anchors)

| model | H1 skill | H5 skill | H10 skill | effective rank |
| --- | ---: | ---: | ---: | ---: |
| V2 | 0.552 | 0.548 | 0.524 | ~83 |
| V2-A | 0.439 | 0.599 | 0.575 | ~41 |
| V2-B | 0.570 | 0.549 | 0.525 | ~82 |

V2-A therefore defines a qualitatively different compression/prediction regime;
V2-B stays close to V2 except at the shortest horizon.

### V2-A event ablation

On event-exposed anchors:

```text
H=0.5 s:
observed - no_event ≈ +0.237 skill

H=1.0 s:
observed - no_event ≈ +0.333 skill
```

This supports "the predictor materially uses the observed conditioning
channel." It does **not** establish causal action semantics.

### Probe interpretation

Linear probes show that low V2-A effective rank does not imply loss of the
tested conversational information. Rank is therefore a representation-health
guardrail, not a standalone objective.

Cross-domain probes reveal a separate concern: useful information may remain
decodable pooled/within-domain while being encoded in domain-specific ways.

Concept probes have also found linearly accessible multi-party/social
correlates (for example local speaker count), but recording/speaker identity
and acoustic nuisance are plausible shortcuts for some concepts. These results
need controls before becoming architectural claims.

## Hypothesis registry

Status vocabulary:

- **supported** — current project evidence is consistent and a predefined test
  supports the hypothesis;
- **partially supported** — some evidence supports it but a key alternative
  explanation remains;
- **open** — motivated but not adequately tested;
- **rejected** — a matched project experiment contradicted it;
- **literature-motivated** — external evidence motivates a project test but is
  not project evidence.

### V2 representation and optimization

| ID | Hypothesis | Status | Next discriminator |
| --- | --- | --- | --- |
| H-V2-01 | SIGReg strength controls a trade-off between compact predictive dynamics and domain-invariant/richer geometry. | partially supported | matched lambda sweep after normalization is fixed |
| H-V2-02 | Replacing CausalBN with standard BN materially improves the V2 representation/dynamics. | mostly rejected | B action-ablation; otherwise close branch |
| H-V2-03 | Shared projector CausalBN running statistics create a temporal train/eval mismatch; per-position stats improve it. | open | V2-D matched analysis |
| H-V2-04 | Low effective rank alone is not evidence of harmful collapse. | supported | continue using probes/skills as guardrails |
| H-V2-05 | The observed event-conditioning channel materially contributes to medium/long-horizon rollout. | supported | repeat matched ablation on B/D |
| H-V2-06 | Action/event information is more directly expressed in transition geometry `Δz` than in static `z_t`. | open | action probe + spectrum on `Δz` |
| H-V2-07 | Increased dataset separation partly represents meaningful interaction-setting differences rather than only nuisance. | open | nuisance/speaker/domain controls |
| H-V2-08 | Future-state linear accessibility correlates with useful long-horizon dynamics. | open | measure across V2/A/B/D/lambda sweep |
| H-V2-09 | Temporally centered SIGReg can preserve useful task/domain structure if the ordinary-lambda trade-off is irreducible. | literature-motivated | test only if lambda sweep fails |

### V3 action grounding

| ID | Hypothesis | Status | Planned test |
| --- | --- | --- | --- |
| H-V3-01 | WAIT/START/CONTINUE/STOP form a valid first controllable conversational action space. | open | derive labels, audit support/validity, retrain |
| H-V3-02 | An IDM makes the latent transition more action-identifiable and improves action-grounded dynamics. | literature-motivated | real-transition IDM + forward/IDM cycle consistency |
| H-V3-03 | `Δz`-based action decoding is more informative than endpoint-only decoding. | literature-motivated | matched IDM vs delta decoder |
| H-V3-04 | A continuous constrained action latent captures useful conversational variation beyond four discrete actions. | literature-motivated | discrete vs continuous latent-action comparison |
| H-V3-05 | Invalid conversational actions must be explicitly masked; otherwise a planner can exploit OOD transitions. | partially supported | current counterfactual stress tests + V3 validity experiment |

### Planning

| ID | Hypothesis | Status | Planned test |
| --- | --- | --- | --- |
| H-P-01 | A predictor can be accurate while latent-distance action ranking is wrong. | literature-motivated | candidate ranking vs realized outcome |
| H-P-02 | IDM/transition consistency can serve as a useful planning guardrail. | open | planning success conditioned on cycle consistency |
| H-P-03 | Controllable action `u_t` and exogenous event `e_t` must be represented separately for conversational planning. | open | explicit two-channel V3-P model |
| H-P-04 | Receding-horizon replanning is preferable to open-loop conversational plans. | open | offline replay then live demo |

### Hierarchical / multimodal

| ID | Hypothesis | Status | Planned test |
| --- | --- | --- | --- |
| H-V4-01 | Conversation benefits from separate low-level turn actions and higher-level social/floor-management subgoals. | literature-motivated | HWM-inspired architecture |
| H-MM-01 | Addressee and social-target information will require visual/gaze cues beyond audio-only V2. | partially supported | multimodal extension and matched audio-vs-AV probes |
| H-MM-02 | Multimodal cues improve multi-party transition prediction without sacrificing audio timing precision. | open | AV encoder/fusion ablation |

## Decision gates

### Gate 1 — close normalization

Required before choosing V2 normalization:

- matched V2-D rollout/spectrum/probes;
- B action-ablation;
- inspect train/eval SIGReg and projector statistics.

Output: one normalization recipe for the lambda sweep.

### Gate 2 — close SIGReg

Run the matched lambda sweep with the chosen normalization.

Do not select lambda from one scalar. Compare the Pareto surface over:

- H1/H5/H10 skill;
- transition and stable skill;
- displacement alignment;
- movement ratio;
- effective rank and spectrum;
- future/current conversational probes;
- cross-domain transfer;
- dataset separation;
- event-conditioning ablation.

Output: frozen V2 audio recipe or evidence that centered/residual SIGReg needs a
new experiment.

### Gate 3 — transition geometry

Before V3 implementation, analyse:

```text
Δz_t = z_(t+1) - z_t
```

by action/event:

- spectrum and effective transition dimension;
- r90/r95/r99;
- action-conditioned centroids/within-class variance;
- linear action readout;
- valid vs invalid counterfactual transitions.

Output: baseline action-sensitivity diagnostics that V3/IDM must beat.

### Gate 4 — start planning

Do not start the planner until V3 demonstrates:

- supported discrete/continuous actions;
- valid-state/action masking;
- action-identifiable real transitions;
- coherent counterfactual transitions;
- acceptable forward/IDM consistency;
- no obvious action shortcut that bypasses conversational state.

## Research questions backlog

### Representation

- What amount of effective rank is sufficient for planning rather than only
  prediction/probing?
- Does transition effective dimension correlate with action sensitivity better
  than state effective rank?
- Which parts of EgoCom/Ego4D domain separation are conversational vs acoustic
  nuisance?
- Does lower SIGReg remove nuisance variance or merely encode domains in fewer,
  stronger directions?

### Prediction

- Why does V2-A sacrifice H1 while improving H5/H10?
- Is this caused by objective weighting, representation geometry or greater
  reliance on event conditioning?
- Would another horizon weighting reproduce the same effect without reducing
  rank?

### Actions

- Is a four-action discrete vocabulary sufficient?
- Which action distinctions are genuinely supported by observational data?
- How should backchannel, interruption, gaze and addressee actions enter later?
- Can a continuous action space discover timing/intensity distinctions that
  discrete labels hide?

### Planning

- What is the right conversational goal representation?
- Is Euclidean latent distance a meaningful planning cost?
- Which planning objective discourages socially undesirable interruption or
  monopolising the floor?
- How should uncertainty over human/exogenous events enter the rollout?

### Evaluation

- Which probe/control tasks distinguish conversation structure from speaker or
  recording identity?
- Should future-state probe quality be monitored during training or remain an
  offline diagnostic?
- Which metric should be a checkpoint monitor vs a guardrail?
- How should planning-grounded evaluation differ from prediction metrics?

## Literature-to-experiment map

The Zotero library stores full metadata and notes. The map below records only
why a source matters to the current roadmap.

| Source | Project question |
| --- | --- |
| LeJEPA | SIGReg, identifiability and collapse prevention |
| LeWorldModel | core JEPA world-model architecture and latent rollout |
| Temporally Centered SIGReg | follow-up if ordinary SIGReg creates an irreducible trade-off |
| Delta-JEPA | action-sensitive `Δz` and inverse/action decoding |
| Physically Grounded JEPA + IDM/SA | transition subspace, IDM and state grounding |
| Learning Latent Action World Models In The Wild | continuous/constrained learned action spaces |
| D-JEPA | prediction accuracy vs decision-aligned action ranking |
| V-JEPA 2 / DINO-WM / FF-JEPA | planning protocols and latent goal objectives |
| Hierarchical Planning with Latent World Models | V4 multi-timescale planning |
| VAP / turn-taking HRI literature | conversational timing, events and action semantics |
| probing/control-task literature | avoid overinterpreting linear probes |

## Update rule

After every meaningful experiment:

1. record the exact run/artifact evidence;
2. update the relevant hypothesis status;
3. write the alternative explanation if one remains;
4. state the next discriminator;
5. update the architecture roadmap only if the result changes it;
6. update a decision record only when a choice is actually accepted;
7. attach/tag the relevant Zotero sources to the hypothesis.

This document should make it possible to answer, at any point:

> What do we currently believe, why do we believe it, what could falsify it,
> and which experiment changes the architecture next?
