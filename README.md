# world-model-turn-taking-models

## Data

Inspect the structure of the first batch built from the public EgoCom dataset
([`batgre/conversational-dynamics-egocom`](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom)).
This loads the dataset through the same data package used for training:

```bash
uv run turn-wm inspect-data --dataset egocom
uv run turn-wm inspect-data --dataset egocom --split validation --batch-size 4
```

`--dataset full` loads every corpus of the private release (EgoCom + Ego4D)
into one dataset; a split includes only the corpora that publish it. Add
`--shuffle` to inspect a seeded shuffled (mixed-corpus) batch.

```bash
uv run turn-wm inspect-data --dataset full --batch-size 8 --shuffle
```

In code, one or many corpora go through the same path:

```python
loaded = load_data(DATASETS["full"])
dataset = build_dataset(loaded, split="train", window=WindowConfig(), training=True)
loader = build_dataloader(dataset, loader=DataLoaderConfig(batch_size=32))
```

Raw audio/video is never downloaded. To decode it from a local copy of the
corpus, pass its root (the directory the published `media_manifest` paths are
relative to):

```bash
uv run turn-wm inspect-data --dataset egocom --batch-size 1 --media-root /path/to/EgoCom
uv run turn-wm inspect-data --dataset full --media-root egocom=/path/to/EgoCom --media-root ego4d=/path/to/Ego4D
```

## Model

The world model follows [LeWM](https://github.com/lucas-maes/le-wm): a JEPA
that predicts the next latent observation from past latents and turn-taking
actions.

- **Encoder** (`models/encoders/mimi.py`): frozen
  [Mimi](https://huggingface.co/kyutai/mimi). Audio is down-mixed to mono,
  resampled to 24 kHz and encoded to Mimi's continuous pre-quantization
  latents (12.5 Hz, 512-d), then causally aligned to the 10 Hz action grid:
  grid step `k` takes the latest Mimi frame available by the end of its
  interval, so no step sees future audio. A batch may mix windows of
  different lengths and sample rates (EgoCom and Ego4D).
- **Projector / prediction head** (`lewm/mlp.py`): MLPs to and from the
  `embed_dim` latent space.
- **Action embedder** (`lewm/embedder.py`): the five action ids of the
  dataset (`NO_EVENT`, `ONSET`, `OFFSET`, `MASKED`, `PAD`, the last one
  zeroed).
- **Predictor** (`lewm/predictor.py`, `lewm/transformer.py`): causal
  transformer conditioned on actions through AdaLN-zero, with one learned
  position per step of the teacher-forced context (`num_frames =
  data.context_steps`). Its attention is plain causal attention; how much
  history it sees is set by the inputs it is given.
- **SIGReg** (`lewm/sigreg.py`): regularizer keeping latents close to an
  isotropic Gaussian, which prevents collapse.

## Configuration

Experiments are configured with [Hydra](https://hydra.cc) in the spirit of
le-wm. `configs/config.yaml` holds the shared `embed_dim` and selects one
model and one training recipe:

```text
configs/
  config.yaml       embed_dim; defaults: model, train
  model/lewm.yaml   JEPA and its nested sub-modules (_target_), encoder included
  train/lewm.yaml   seed, data, prediction, loss, optimizer, loader, trainer
```

The training recipe is merged at the root of the composed config, so its keys
read `cfg.trainer.max_epochs`, `cfg.loss.sigreg.weight`, etc. Compose and
build from code, with Hydra override syntax:

```python
from turn_wm.config import load_config
from turn_wm.models.build import build_model

cfg = load_config(["embed_dim=256", "data.context_steps=20", "optimizer.lr=1e-4"])
model = build_model(cfg)
```

A new model or recipe is a new file in its group (`configs/train/xxx.yaml`,
selected with `train=xxx`).

## Training

`turn_wm.training.lewm` holds the objective and a Lightning module.

- **Trajectories.** A sample's context window followed by its future window
  forms one trajectory of `data.context_steps + data.future_steps` steps.
  Training uses a fixed context (`training_window(cfg)`), so every trajectory
  in a batch has the same length; anchors too close to a recording's start
  for that context are left out by the dataset. Context and future audio are
  encoded together, once per trajectory.
- **Teacher forcing** is dense over the ground-truth context
  (`C = data.context_steps`): from `z0 … z(C-1)` and their actions the
  predictor predicts `z1 … zC`, one step ahead at every position. The first
  future latent `zC` is only a target, never an input.
- **Rollout** starts at the context/future boundary from the ground-truth
  context, then feeds its own predictions back, never a ground-truth future
  latent (without gradient when `prediction.rollout_stop_gradient`), with
  the real future actions. Before each prediction it keeps only the latest
  `prediction.rollout_context_size` states and actions
  (`<= data.context_steps`). It is supervised at
  `prediction.rollout_horizons`, each weighted in `loss.rollout`.
- `validate_config` checks these sizes against each other and against the
  predictor before any data is read; `seed` seeds the model initialization
  (`LeWMModule`) and should also seed the data loader.
- The total loss weights teacher forcing, rollout and SIGReg
  (`loss.*.weight`). Latent targets are defined at every step, including
  steps whose annotation is `UNKNOWN`.

```python
import lightning as L

from turn_wm.config import load_config
from turn_wm.data.build import build_dataset
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.source import DATASETS, load_data
from turn_wm.training.lewm import LeWMModule, training_window

cfg = load_config()
dataset = build_dataset(
    load_data(DATASETS["full"]),
    split="train",
    window=training_window(cfg),
    training=True,
    media_roots={"egocom": ..., "ego4d": ...},
    modalities=tuple(cfg.data.modalities),
)
loader = build_dataloader(
    dataset,
    loader=DataLoaderConfig(batch_size=cfg.loader.batch_size, seed=cfg.seed),
)

L.Trainer(**cfg.trainer).fit(LeWMModule(cfg), train_dataloaders=loader)
```

There is no training entry point yet; the snippet above is the intended
wiring.

## Tests

```bash
uv run pytest                  # offline unit tests (default)
uv run pytest -m integration   # real-data smoke test; downloads EgoCom from the Hub
```

Tests marked `integration` need network access (or a warm Hugging Face cache),
so the default run skips them. The raw-media smoke tests also need local media
and are skipped unless their corpus root is set:

```bash
EGOCOM_MEDIA_ROOT=/path/to/EgoCom uv run pytest -m integration
EGO4D_MEDIA_ROOT=/path/to/Ego4D uv run pytest -m integration   # private dataset
```

The Mimi and training smoke tests also download the Mimi weights from the Hub
on first use.
