# ADR 0001 — Freeze V1 as the scientific baseline

- Status: **Frozen**
- Date: 2026-09-27
- Scope: V1 model, training recipe, selected checkpoint and scientific claims

## Decision

Freeze the current audio-only LeWM V1. Do not continue modifying V1 to improve
individual downstream metrics. V1 becomes the reference baseline against which
V2 is compared.

The freeze applies to the model and training recipe. Evaluation tooling may
receive correctness fixes, but any recomputed result must identify the
evaluation-code revision and must use the frozen checkpoint.

## Frozen identity

Training run:

```text
/content/drive/MyDrive/turn-wm-runs/lewm-v1-audio-full/lewm/20260926-010750-4216838a
```

Training metadata:

- repository commit: `1cab3245a6e5283d5200e8203a820f064c81fcbb`
- dataset: `full`
- dataset revision: `c2e99a6067f66610e86bf48a179caa82612f84ad`
- seed: `3072`
- observation source: precomputed Mimi cache
- Mimi: `kyutai/mimi`
- Mimi resolved revision: `89091b3e466eb6a9d11e537bf26b144f194978f7`
- decision grid: 10 Hz
- feature dimension: 512
- latent dimension: 192
- config hash: `4216838a402798ab9efef112d21645875f6ac404cb69b6a97151865eff8b3d95`

Frozen checkpoint:

```text
checkpoints/epoch=000-step=9000.ckpt
```

The same checkpoint is used by the completed analysis suite and TurnBench
evaluation. Its SHA-256 is:

```text
69cf37824725390f3f4dea3c3cc5a5c455a8c07339dd7ab0e2e57515638b32e8
```

## Frozen architecture

```text
frozen / precomputed Mimi, 512-d @ 10 Hz
                ↓
        trainable projector
             512 → 192
                ↓
               z_t
              /   \
         SIGReg    causal AR predictor
                    action conditioning
                    via AdaLN-zero
                       ↓
              predicted future latents
```

The V1 action vocabulary in the dataset is
`NO_EVENT / ONSET / OFFSET / MASKED / PAD`. These are treated in V1 as the
conditioning sequence supplied to the predictor.

## Frozen training recipe

- context: 15 steps = 1.5 s
- future: 10 steps = 1.0 s
- rollout context: 10 steps
- rollout horizons: 1, 5 and 10 steps = 0.1, 0.5 and 1.0 s
- teacher forcing weight: 1.0
- rollout weight: 1.0
- SIGReg weight: 0.09
- rollout stop-gradient: enabled
- rollout horizon loss: weighted mean, equal horizon weights
- curriculum:
  - 0–20%: [1]
  - 20–50%: [1, 5]
  - 50–100%: [1, 5, 10]
- validation always evaluates [1, 5, 10]
- AdamW, learning rate 1e-4, weight decay 1e-3
- 5% linear warmup from 1e-6 to 1e-4, then cosine decay to 1e-6
- batch size 128
- BF16 mixed precision
- gradient clipping 1.0
- maximum 10,000 optimizer steps
- validation every 1,000 optimizer steps
- checkpoint selection: minimum `val/loss`

## What V1 established

### 1. The latent is not a trivial copy of Mimi

The label-conditioned analyses show that the 192-d latent retains current
conversation state and increases structure associated with several temporal and
future variables relative to the 512-d Mimi features. Examples include
time-to-next-speaker-onset, silence duration, time-since-floor-change and
future joint speech state at 0.5–1 s.

This is evidence of useful representation structure, not evidence that the
latent is disentangled or causally interpretable.

### 2. The learned dynamics beat persistence

On 10,000 deterministic validation anchors, autoregressive V1 rollouts beat
copy-last persistence at every evaluated horizon.

Transition-row skill:

| horizon | skill vs persistence |
|---|---:|
| 0.1 s | 0.474 [0.451, 0.493] |
| 0.5 s | 0.603 [0.581, 0.623] |
| 1.0 s | 0.560 [0.534, 0.586] |

Predicted displacement is also directionally aligned with true latent motion,
especially on transitions. V1 nevertheless systematically undershoots movement
magnitude; movement ratios remain roughly 0.72–0.76 rather than 1.

### 3. The predictor materially uses the conditioning sequence

The action/event ablation showed that replacing the observed future event
sequence with NO_EVENT or a shuffled sequence degrades rollout quality,
especially when a real conversational event is exposed.

At 1 s on event-exposed rows:

- observed − NO_EVENT skill: +0.231 [0.183, 0.274]
- observed − shuffled skill: +0.208 [0.168, 0.244]

Counterfactual one-step predictions also move when only the conditioning token
changes. For SILENT anchors, ONSET vs NO_EVENT changes the predicted latent by
a mean Euclidean distance of about 3.884.

This establishes action/event sensitivity. It does **not** establish autonomous
planning or causal intervention semantics.

### 4. External turn-taking information is present, but predicted latents are
not automatically better decision representations

TurnBench DEV, with matched causal heads and each head restored at its best
internal validation epoch:

| representation | EOT recall | EOT FP | EOT p50 | INT recall | INT FP | INT p50 |
|---|---:|---:|---:|---:|---:|---:|
| Mimi | 0.595 | 0.097 | 1444 ms | 0.553 | 0.089 | 291 ms |
| current z_t | 0.590 | 0.098 | 1559 ms | 0.484 | 0.097 | 316 ms |
| predicted latent | 0.536 | 0.098 | 1696 ms | 0.481 | 0.091 | 267 ms |

The current latent nearly preserves Mimi's EOT recall, loses more information
for INT, and the predicted latent is not a universally superior direct readout.

TurnBench therefore remains an external diagnostic, not the optimization target
for V2.

## What V1 did not establish

V1 does **not** establish:

- planning utility;
- causal controllability by an agent;
- autonomous anticipation of future conversation events;
- a high-level / low-level temporal hierarchy;
- macro-actions or latent subgoals;
- hierarchical MPC;
- multimodal or exocentric planning.

The most important limitation is semantic: V1 rollouts consume ground-truth
future `NO_EVENT / ONSET / OFFSET` event tokens. These are observed
conversation events, not a clean intervention channel for an agent.

## Freeze rule

From this point onward:

1. V1 architecture, recipe and checkpoint do not change.
2. V1 remains the baseline for V2 ablations.
3. Evaluation correctness fixes are allowed and must be revisioned.
4. New action semantics, temporal hierarchy, planning objectives and
   multimodal extensions belong to V2.
5. No downstream benchmark is allowed to redefine the primary project goal.
