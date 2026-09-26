# Training

This page describes the training recipe of `configs/train/lewm.yaml`, run by
`uv run turn-wm train` (see the README for data, media roots, runs and
checkpoints). The objective lives in `turn_wm.training.lewm`, the learning-rate
schedule in `turn_wm.training.scheduler`.

## Baseline V1

### Architecture

```text
frozen / precomputed Mimi (512-d, 10 Hz)
        ↓
trainable projector (512 -> 192)
        ↓
        z_t
       /   \
  SIGReg    AR predictor (actions via AdaLN-zero)
                ↓
      teacher forcing + autoregressive rollout
                ↓
          horizon curriculum
```

A trajectory is `data.context_steps` ground-truth context steps followed by
`data.future_steps` future steps on the 10 Hz action grid.

- **Teacher forcing** is dense over the context: from `z0 … z(C-1)` and their
  actions the predictor predicts `z1 … zC`, one step ahead at every position.
- **Rollout** starts at the context/future boundary from the ground-truth
  context, then feeds its own predictions back (without gradient:
  `rollout_stop_gradient: true`), never a ground-truth future latent, with the
  real future actions. Before each prediction it keeps only the latest
  `prediction.rollout_context_size` states and actions.
- **SIGReg** keeps the latents close to an isotropic Gaussian.

### Observation sources

The projector always receives Mimi's causal 10 Hz × 512 features; only where
they come from changes (`data.observation_source`):

```text
raw_audio    audio -> frozen Mimi (12.5 Hz) -> causal alignment -> 10 Hz × 512 -> projector
mimi_cache   precomputed 10 Hz × 512 (turn-wm precompute-mimi)         -> projector
```

Both feed Mimi features of the same shape to the same projector, row `k` of
a recording being its grid step `start_index + k`. They differ in one
respect: the cache encodes each **whole recording** continuously (streamed
Mimi matches one-shot Mimi to float32 precision), so every row has Mimi's full
causal history, while `raw_audio` encodes each training window **from its
first sample**, without the audio before it; its first rows therefore differ
from the cache's. The cache is the more faithful input. The projector, the
predictor and SIGReg (applied to the projected latents) are trained
identically; the cache does not change the recipe.

**The cache is the recommended mode for real training.** Mimi is frozen, and
training windows overlap heavily, so the raw path would decode, resample and
encode the same audio again at every step. With the cache Mimi is never loaded
(`build_model` builds the model without an encoder), GPU memory holds only
the trainable model, ablations run faster, and no raw media is needed: a
machine with the published dataset and the cache can train (e.g. Colab).
`raw_audio` stays available for debugging and end-to-end checks; it needs the
media roots.

```yaml
data:
  observation_source: mimi_cache
```

```bash
uv run turn-wm train data.mimi_cache.root=/path/to/cache                 # cached
uv run turn-wm train data.observation_source=raw_audio                   # raw audio
```

`data.mimi_cache.root` is either one corpus cache (its `manifest.json`) or a
release root (`release_manifest.json` and one cache per corpus, as on the
Hub), which serves every corpus of `data.dataset=full`.

Before training, the runner refuses a cache whose feature rate is not 10 Hz,
whose dimension differs from `model.projector.input_dim`, or that does not
cover a loaded corpus. A cache computed from the loaded dataset revision is
accepted as is; otherwise (the release's EgoCom cache comes from the public
EgoCom repository, while `full` loads EgoCom from the private one) each cached
recording must have exactly the loaded grid's `start_index`, steps and
`start_time_s`, and every loaded recording must be cached or excluded. The
run's `metadata.json` records the observation source and each cache's identity
(schema, Mimi model and revisions, source dataset revision, rate, dim).

### Optimization

```text
AdamW, lr = 1e-4, weight decay = 1e-3 (every trainable parameter)

first 5 % of optimizer steps:   linear warmup   1e-6 -> 1e-4
remaining 95 %:                 cosine decay    1e-4 -> 1e-6

scheduler interval:   every optimizer step
precision:            bf16-mixed
gradient clipping:    1.0
```

The number of optimizer steps is Lightning's
`trainer.estimated_stepping_batches`, which accounts for gradient
accumulation, batch limits, `max_steps` and devices. With a logger (e.g.
`logging.wandb.enabled=true`) the LR is logged at every optimizer step by
Lightning's `LearningRateMonitor`. The scheduler is a
PyTorch `LambdaLR`; its step count is saved in checkpoints, so resuming from
`last.ckpt` continues the schedule where it stopped (as long as the run length
is unchanged). Frozen Mimi weights are not optimized.

### Loss

```text
L = 1.0 * L_teacher_forcing + 1.0 * L_rollout + 0.09 * L_SIGReg
```

`L_rollout` is the **weighted mean** of the per-horizon losses over the
active horizons:

```text
L_rollout = Σ_h w_h · L_h / Σ_h w_h        (h active)
```

With equal weights this is `L1`, then `(L1 + L5) / 2`, then
`(L1 + L5 + L10) / 3`. A plain sum would roughly triple the rollout term as
horizons are activated, mixing two changes (a longer horizon and a heavier
loss); the mean keeps `loss.rollout.weight` meaning the same thing through the
whole run, so the curriculum only changes the temporal difficulty.

### Horizon curriculum

On the 10 Hz grid, `h=1` is 100 ms ahead, `h=5` 500 ms and `h=10` 1 s.

```text
training progress     active horizons
   0 % – 20 %         [1]
  20 % – 50 %         [1, 5]
  50 % – 100 %        [1, 5, 10]
```

Progress is `global_step / (total optimizer steps - 1)`, clamped to [0, 1]:
optimizer steps, not epochs, so gradient accumulation, another batch size or
more devices keep the same recipe. The rollout only runs up to the largest
active horizon, which saves compute early on. Training logs
`train/curriculum_progress` and `train/max_rollout_horizon` at every step;
`train/rollout_<h>_loss` exists only for active horizons.

**Validation always evaluates `[1, 5, 10]`**, whatever the training stage, so
`val/*` metrics (and the checkpoints selected on `val/loss`) stay comparable
from the first epoch to the last.

### Configuration

The recipe as it appears in `configs/train/lewm.yaml` (comments omitted; a
unit test checks these excerpts against the real configuration):

```yaml
optimizer:
  type: AdamW
  lr: 1e-4
  weight_decay: 1e-3

scheduler:
  type: warmup_cosine
  interval: step
  warmup_ratio: 0.05
  min_lr: 1e-6

prediction:
  rollout_context_size: 10
  rollout_horizons: [1, 5, 10]
  rollout_stop_gradient: true
  curriculum:
    enabled: true
    stages:
      - until: 0.20
        horizons: [1]
      - until: 0.50
        horizons: [1, 5]
      - until: 1.00
        horizons: [1, 5, 10]

loss:
  teacher_forcing:
    weight: 1.0
  rollout:
    weight: 1.0
    horizon_weights:
      "1": 1.0
      "5": 1.0
      "10": 1.0
  sigreg:
    weight: 0.09

trainer:
  precision: bf16-mixed
  gradient_clip_val: 1.0
```

`validate_config` checks the recipe before any data is read: scheduler bounds
(`0 <= warmup_ratio < 1`, `0 < min_lr <= lr`, `interval: step`), and a
curriculum whose stages end at increasing `until` values in (0, 1] with the
last at 1.0, each using configured horizons that have a positive weight and
including every horizon of the stage before (the curriculum is cumulative).

### Validation metrics

Validation evaluates every `prediction.rollout_horizons` (the curriculum only
affects training) and, besides the losses, logs diagnostics computed from the
latents and predictions of the losses (`turn_wm.training.metrics`; no second
forward). All are accumulated as sums over the whole validation epoch, so they
do not depend on the batch size. Training logs only the losses, the
curriculum state and the learning rate.

1. **Prediction MSE** — `val/tf_mse` (one-step, on exactly the teacher-forced
   positions) and `val/rollout_{h}_mse`.
2. **Persistence baseline** — the error of copying a latent forward:
   `z_t` for the teacher-forced step `z_t -> z_{t+1}` (`val/tf_persistence_mse`)
   and the last ground-truth context latent `z_{C-1}` for every rollout
   horizon (`val/persistence_{h}_mse`). Model and baseline use the same
   targets and positions.
3. **Skill score** —

   ```text
   skill = 1 - MSE_model / MSE_persistence
   ```

   from the epoch's aggregated errors (never an average of per-batch skills):
   `> 0` better than persistence, `0` as good, `< 0` worse (unclipped).
   `val/tf_skill`, `val/skill_{h}` and `val/skill_mean` (mean over horizons).
4. **Cosine similarity** — `val/cosine_{h}` between predicted and target
   latents, a geometric complement to the MSE.
5. **Delta dynamics** — `val/target_delta_norm_{h}` = mean `||z_target -
   z_{C-1}||` and `val/prediction_delta_norm_{h}` = mean `||z_pred -
   z_{C-1}||`. A predicted delta near 0 while the target delta is large means
   the model stays close to persistence rather than predicting a change.
6. **Latent health** — on the projected target latents: `val/latent_std`
   (per-dimension std, averaged over dimensions), `val/latent_norm`
   (mean L2 norm), `val/prediction_norm` (mean L2 norm of the rollout
   predictions at every horizon) and `val/effective_rank`
   (`exp(entropy)` of the normalized singular values of the centred latents,
   1 for a collapsed direction up to `embed_dim`), computed on a deterministic
   random sample of at most `evaluation.latent_rank_samples` latents per epoch.
7. **Global vs per corpus** — prediction, persistence and skill metrics are
   also logged per corpus (`val/egocom/...`, `val/ego4d/...`, from each
   sample's `dataset`). Global metrics pool every element of every corpus;
   they are not averages of the corpus metrics.

```yaml
evaluation:
  persistence_baseline: true
  cosine_similarity: true
  latent_health: true
  latent_rank_samples: 8192
```

Checkpoints are still selected on `val/loss`; these metrics are diagnostics.

**The test split is not used during training or model selection.** `run()`
builds only the train and validation splits, and the Lightning module has no
test step. The test split is evaluated once, after the final checkpoint has
been selected.

### Not in V1

Left for later ablations: other optimizers or schedules, weight-decay
exclusions for norms and biases, EMA or target encoders, scheduled sampling,
backpropagation through the rollout, and curricula on the context length or
the rollout weight.
