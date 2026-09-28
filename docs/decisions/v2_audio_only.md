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
shows sensitivity to recorded future ONSET/OFFSET tokens. Neither result
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
| Scope | Continue audio-only until the result is convincing. Multimodal work has no fixed version boundary. |
| Context/history/target | `C=30`, training and inference rollout window `W=30`, target and rollout horizon `H=10`, all at 10 Hz. The coupled C/W choice retains the added observed history; H remains 1 s while testing it. |
| Projection | Replace LayerNorm with **causally safe BatchNorm in both** the Mimi-to-latent and predictor-output projectors for V2; keep transformer AdaLN separate. No training-time normalization may mix future positions into a prefix. |
| Latent regularization | Retain 192 dimensions and raw-latent SIGReg at `lambda=0.09` for the first recipe. These are controls, not claimed optima. Log unweighted/weighted terms and their gradient contributions. |
| Optimizer schedule | Keep warmup plus cosine decay. Its planned duration in examples/epochs is distinct from the stopping rule; no 10,000-step training cutoff. |
| Training length | Count optimizer steps, samples and epochs from the actual train set and physical batch. A run must reach at least one complete epoch and give H=10 a full epoch of training before automatic early stopping can end it. |
| Action semantics | Preserve v0 `NO_EVENT/ONSET/OFFSET` observed vocal-action/event proxies as the first comparison. `TAKE/BACKCHANNEL/HOLD/YIELD` are outcomes, not conditioning actions. A deeper V2 action-utility study is required before revising the vocabulary. |
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
flattens batch and time before both projectors. The V2 variant instead keeps
time separate during training normalization so future positions cannot change
an earlier output. Evaluation must then use the statistics training used. The
Mimi-to-latent projector's inputs do not depend on the window position, so it
keeps one set of running statistics. The predictor-output projector's inputs
carry the predictor's absolute position embedding, and their mean varies by
position; pooling its running statistics over positions would make evaluation
normalize differently from training. It therefore keeps running statistics
per window position (`num_positions` = `model.predictor.num_frames`), which
stays causal and matches training position by position. This departs from
the reference implementation, which pools batch and time in both modes and
so lets later window positions enter the training statistics of earlier
ones. `configs/model/lewm_bn.yaml` changes only the projector
normalization; the observed context is selected separately in the training
configuration or notebook. The V1 `model=lewm` path remains available.

## SIGReg reference and later tuning

The V1 [implementation](../../src/turn_wm/models/lewm/sigreg.py) computes an
Epps-Pulley statistic over the batch at each time step, then averages over
steps and random projections. A falling scalar has no useful interpretation
without its finite-sample reference. Under independent standard Gaussian
latents, its 17-point quadrature has expected value approximately **1.0525**;
with every latent fixed to zero it gives approximately **51.46** at physical
batch 128, or **205.85** at batch 512. These are calculation-specific
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

## Evaluation and decisions still open

Report logged-event-conditioned rollout skill versus persistence at 0.1,
0.5 and 1 s, with transition and stable subsets, per corpus and paired
recording-level uncertainty. Check learned-state motion, rank, and matched
Mimi/latent/predicted-latent readouts. Keep the TEST split closed; external
TurnBench DEV is a separate downstream check. The checkpoint score should
reflect 0.5/1 s transitions rather than `val/loss` alone, with stable-state
and readout guardrails. For the first run, `configs/train/lewm_v2.yaml` uses
a physical batch of 512 after a synthetic GPU memory/gradient check, validates
at each epoch, plans eight epochs for the cosine schedule and disables the
10,000-step limit. Checkpoint selection and early stopping use the mean of
validation skill against persistence on known state transitions at 0.5 and
1 s. These are observed state changes, not proof of anticipation without
future events. `min_delta=0` counts any measured improvement, with patience
two validation epochs. The first monitored validation is at the end of epoch
two, when H=10 has had a complete epoch. These are pilot choices to revisit
using validation noise and the transition/stable diagnostics; gradient
accumulation does not reproduce physical-batch BN and SIGReg statistics.

The horizon curriculum is measured against steps in the **first epoch**,
irrespective of the eight-epoch cosine duration. H=5 activates after 20% of
the first epoch's optimizer steps, H=10 after 50%, then stays active. The
notebook reports the thresholds in optimizer steps and seen examples from the
actual train dataset after cache filtering and dropping the incomplete batch.
At least one complete H=10 epoch precedes any early stop. V1's 10,000-step cap
is not carried over.

The deeper action study will audit token frequency and reliability, compare
action-conditioned against matched simpler baselines, and substitute only
valid state/action combinations. A revised action grid needs supported,
selectable distinctions and a matched retraining comparison against v0.
Good conditional prediction alone does not prove intervention or planning.
An independent context-only future-event anticipation test was discussed
and **deferred**. Longer context (including 50 states), H>10 and the exact
V2-bis architecture require separate decisions after relevant evidence.

## Evidence and bibliography contract

Zotero is the bibliographic source of truth for selected papers, with DOI or
arXiv version and notes linking claim, limitation and experiment. A future
`references/library.bib` is an export, not a second hand-maintained source.
Relevant starting papers include
[LeJEPA](https://arxiv.org/abs/2511.08544),
[LeWorldModel](https://arxiv.org/abs/2603.19312),
[the frozen-encoder window study](https://arxiv.org/abs/2512.24497),
[VAP](https://arxiv.org/abs/2205.09812), and
[hierarchical latent planning](https://arxiv.org/abs/2604.03208).
The robot/video findings motivate tests here; they do not establish audio
hyperparameters. Future decision updates must link the exact code/config,
data and feature revisions, run/checkpoint hashes and saved analysis artifacts.
