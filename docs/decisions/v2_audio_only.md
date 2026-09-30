# V2 audio-only model: decision record

**Status:** V2 pilot recipe implemented locally; batch 512 still needs a Colab
memory check, and there are no V2 results yet.

## Question and V1 evidence

Can a model trained on frozen Mimi audio features preserve useful turn-taking
information while improving **conditional** latent dynamics, especially at
vocal transitions? V1 uses 15 observed 10 Hz states, a 10-state rollout
window, and a 10-step future target. In the [paired V1 rollout
analysis](https://drive.google.com/file/d/1nuid0ZZt8meLkZ7-R4aHHsJ_SZrFpwEu/view),
the selected step-9000 checkpoint beats persistence at 1 s, including on
transitions, but underpredicts displacement. The [event-conditioning
ablation](https://drive.google.com/file/d/1Huf82F2jcucofoo84P0kgzI4j_0Ugyu_/view)
shows sensitivity to the previous recorded future ONSET/OFFSET event tokens; the new action interface converts these to explicit START/STOP transitions and splits NO_EVENT into WAIT/HOLD. Neither result
establishes anticipation without supplied events or a controllable action
interface. The [representation
analysis](https://drive.google.com/file/d/1cSxIobd4i6k5LPgVlaIAbbxz0HjJdUUx/view)
finds a concentrated spectrum; the [downstream TurnBench DEV
analysis](https://drive.google.com/file/d/10UECTuYcP59PXoymZhU-r1N51ejLPfvs/view)
does not establish a robust gain over Mimi. These observations motivate the
next controlled study, but cannot identify a single cause of the V1 gap.
Plausible alternatives include insufficient retained history, projector
normalization, underrepresented transitions, the action-proxy vocabulary, or
the balance between prediction and regularization. The comparisons below
separate some of these explanations; the first combined V2 run does not.

## Accepted initial V2 design

| Choice | Decision and reason |
| --- | --- |
| Scope | Continue audio-only until the result is convincing. The first controlled V2 model trains and validates on **EgoCom only**; larger-corpus training is a separate later scaling experiment. Multimodal work has no fixed version boundary. |
| Context/history/target | `C=30`, training and inference rollout window `W=30`, target and rollout horizon `H=10`, all at 10 Hz. The coupled C/W choice retains the added observed history; H remains 1 s while testing it. |
| Projection | Use **causally safe BatchNorm in both** the Mimi-to-latent and predictor-output projectors; keep transformer AdaLN separate. No training-time normalization may mix future positions into a prefix. The Mimi projector uses shared running statistics; the predictor projector keeps per-position running statistics because causal receptive fields differ by position even under RoPE. |
| Position encoding | Replace learned absolute predictor positions with standard **RoPE** on Q/K ([Su et al., 2021](https://arxiv.org/abs/2104.09864)), base `10000`. RoPE is the default; `model=lewm_learned_pos` preserves the pre-RoPE learned-position predictor for reproduction/ablation. |
| Latent regularization | Retain 192 dimensions and raw-latent SIGReg at `lambda=0.09` for the first recipe. These are controls, not claimed optima. Log unweighted/weighted terms and their gradient contributions. |
| Optimizer schedule | Keep warmup plus cosine decay. Its planned duration in examples/epochs is distinct from the stopping rule; no 10,000-step training cutoff. |
| Training length | Count optimizer steps, samples and epochs from the actual train set and physical batch. A run must reach at least one complete epoch and give H=10 a full epoch of training before automatic early stopping can end it. |
| Action semantics | Use four explicit controllable ego actions derived from the audited vocal-action grid: `WAIT` = SILENT+NO_EVENT, `START` = ONSET, `HOLD` = SPEAKING+NO_EVENT, `STOP` = OFFSET. `TAKE/BACKCHANNEL/YIELD/INTERRUPT` remain contextual outcomes/readouts, not conditioning actions. `MASKED` and `PAD` are technical tokens only. |
| Architecture tracks | V2 studies the single-rate multi-horizon rollout; V2-bis is a separate multi-timescale alternative, compared on matched data and physical horizons. |

The first V2-versus-V1 result compares **whole recipes**. It cannot attribute
any difference to context or BatchNorm alone. The accepted attribution design
compares V1 (15/10, LayerNorm), context/history only (30/30, LayerNorm),
normalization only (15/10, BatchNorm), and V2 (30/30, BatchNorm) at fixed
`H=10`, on common admissible validation anchors. Match the batch, checkpoint
selection and example budget across comparison arms, and report their compute.
Test prefix invariance in training, fixed-statistic inference, batch-size-one
streaming, and train/eval behavior before using the BatchNorm result.
Set `loader.drop_last=true` for V2 training: the last training batch must not
contain a single example. Keep all validation rows (`drop_last=false`) and
report the exact number of training examples consumed per epoch.
The reference [LeWM implementation](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)
flattens batch and time before both projectors. This model instead keeps time
separate during training normalization so future positions cannot change an
earlier output. The Mimi-to-latent projector keeps one shared set of running
statistics. The predictor projector keeps running statistics per sequence
position: RoPE removes the learned absolute-position vector, but a causal
predictor can still have position-dependent output statistics because earlier
positions see shorter receptive fields. This preserves train/eval normalization
semantics while changing only the position encoding. `model=lewm_positional_cbn`
extends per-position statistics to the Mimi projector as an explicit ablation;
`model=lewm_learned_pos` restores the pre-RoPE learned absolute embeddings
while holding the rest of the current architecture fixed. `model=lewm_ln`
is the LayerNorm-projector ablation, and `model=lewm_bn` remains a
compatibility alias for the default causal-BatchNorm design.

## SIGReg reference and later tuning

The V1 [implementation](../../src/turn_wm/models/lewm/sigreg.py) computes an
Epps-Pulley statistic over the batch at each time step, then averages over
steps and random projections. A falling scalar has no useful interpretation
without its finite-sample reference. Under independent standard Gaussian
latents, its 17-point quadrature has expected value approximately **1.0525**;
with every latent fixed to zero it gives approximately **51.46** at physical
batch 128, or **205.85** at batch 512. The factor of the batch size is part
of the Epps-Pulley statistic as LeJEPA defines it, and LeJEPA keeps the
coefficient fixed from batch 128 to 1024 (its Table 1c); the same `lambda`
therefore weighs a given distribution mismatch four times more at batch 512
than at 128, one more reason comparison arms match the batch. These are
calculation-specific
references, not universal optimum values or a task-quality threshold.
Repeated synthetic evaluations with the actual batch, number of steps and
projection sampling will provide variation bands. Report the raw statistic,
its weighted contribution and task metrics alongside these references.

After the first V2 design is fixed, vary **only** the raw SIGReg coefficient
over `{0, 0.01, 0.03, 0.09, 0.2}` at matched settings. Zero is diagnostic.
Choose using conditional transition and stable-state prediction, rank and
matched audio readouts, not the lowest SIGReg scalar. Centered/residual
SIGReg addresses a separate temporal-information hypothesis; it is not the
fix for interpreting the raw loss and has not been adopted for the first V2.

## Evaluation protocol

The V2 evaluation protocol is now frozen separately in
[`evaluation_protocol.md`](evaluation_protocol.md). The primary world-model
dynamics result is autoregressive latent rollout MSE at 0.1, 0.5 and 1.0 s,
reported beside the persistence MSE baseline. Transition/stable subsets use
the same metric and remain diagnostics rather than defining a new score.

Checkpoint selection and early stopping now monitor
`val/rollout_10_mse` in `min` mode. The previous
`transition_skill_5_10` pilot criterion is superseded: it mixed the primary
prediction error with a project-specific normalization and over-weighted one
slice of the data.

Representation health is tracked with effective rank, while conversational
meaning is evaluated separately through train-fitted, validation-evaluated
readouts. The evaluation distinguishes role-relative aggregate/joint targets
from true participant-level marginal metrics; the latter are not claimed
until the model/evaluator exposes a stable participant identity or slot
interface.

Action-conditioned evaluation remains separate from raw dynamics. The
observed/state-preserving/shuffled ablation can show that the model uses its conditioning
sequence, but it does not establish correct off-policy counterfactual
responses. Planning metrics are deferred until candidate ego-action rollouts
and a planner are implemented.

The horizon curriculum is measured against steps in the **first epoch**,
irrespective of the eight-epoch cosine duration. H=5 activates after 20% of
the first epoch's optimizer steps, H=10 after 50%, then stays active. At least
one complete H=10 epoch precedes any early stop.

## Evidence and bibliography contract

Zotero is the bibliographic source of truth for selected papers, with DOI or
arXiv version and notes linking claim, limitation and experiment. A future
`references/library.bib` is an export, not a second hand-maintained source.
Relevant starting papers include
[RoFormer / RoPE](https://arxiv.org/abs/2104.09864),
[LeJEPA](https://arxiv.org/abs/2511.08544),
[LeWorldModel](https://arxiv.org/abs/2603.19312),
[the frozen-encoder window study](https://arxiv.org/abs/2512.24497),
[VAP](https://arxiv.org/abs/2205.09812), and
[hierarchical latent planning](https://arxiv.org/abs/2604.03208).
The robot/video findings motivate tests here; they do not establish audio
hyperparameters. Future decision updates must link the exact code/config,
data and feature revisions, run/checkpoint hashes and saved analysis artifacts.
