# World model for conversational turn-taking

A JEPA-style world model for learning the latent dynamics of multi-party
conversation for social-robot turn-taking. 
**Explore:** [public EgoCom dataset](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom) · [training design](docs/training.md) · [V2 research program](https://github.com/batgr/world-model-turn-taking-model/blob/v2/audio-only/docs/research_program.md) · [research decisions](https://github.com/batgr/world-model-turn-taking-model/tree/v2/audio-only/docs/decisions)

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

Implementation map:

- `src/turn_wm/models/encoders/mimi.py` — frozen Mimi encoder
- `src/turn_wm/models/lewm/mlp.py` — latent projector and prediction head
- `src/turn_wm/models/lewm/embedder.py` — action embedding
- `src/turn_wm/models/lewm/predictor.py` — action-conditioned predictor
- `src/turn_wm/models/lewm/transformer.py` — causal Transformer backbone
- `src/turn_wm/models/lewm/sigreg.py` — SIGReg regularizer
- `src/turn_wm/training/lewm.py` — Lightning module and training objective

See [docs/training.md](docs/training.md) for architecture and training details.

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

Use precomputed Mimi features for normal training runs:

```bash
uv run turn-wm train \
  data.dataset=full \
  data.observation_source=mimi_cache \
  data.mimi_cache.root=/path/to/mimi-features
```

Override the Hydra configuration directly from the CLI:

```bash
uv run turn-wm train \
  data.dataset=egocom \
  data.mimi_cache.root=/path/to/cache \
  data.context_steps=20 \
  prediction.rollout_context_size=10 \
  loader.batch_size=16 \
  optimizer.lr=1e-4
```

For raw-audio debugging instead of the feature cache:

```bash
export EGOCOM_MEDIA_ROOT=/path/to/EgoCom
export EGO4D_MEDIA_ROOT=/path/to/Ego4D

uv run turn-wm train data.observation_source=raw_audio
```

Invalid Hydra overrides or configurations rejected by `validate_config` fail
before data loading. See [docs/training.md](docs/training.md) for the objective,
rollout semantics, horizon curriculum, validation protocol, and observation
sources.

### Runs

Each run gets its own directory, created only once the configuration, the
dataset and the media roots have been checked (a rejected run leaves
nothing behind):

```text
outputs/<experiment.name>/<UTC timestamp>-<config hash>/
  config.yaml      the fully resolved configuration
  metadata.json    run id, seed, config hash, git commit and dirty flag,
                   dataset and its resolved revision
  checkpoints/     best checkpoints by val/loss and last.ckpt
  wandb/           Weights & Biases files, when logging.wandb.enabled
```

`experiment.output_root` (default `outputs/`, relative to the working
directory and git-ignored) and `experiment.name` choose where runs go. The
hash covers the whole resolved configuration, so runs of the same
configuration share its suffix.

Checkpoints follow `checkpoint.*`: by default the three best by `val/loss`
plus `last.ckpt` (`checkpoint.enabled=false` turns them off). Resume from a
checkpoint, with its optimizer state, epoch and step; the resumed run gets a
new directory (quote paths starting with `~` for Hydra):

```bash
uv run turn-wm train checkpoint.resume_from=outputs/lewm/<run id>/checkpoints/last.ckpt
```

Weights & Biases logging is optional and off by default:

```bash
uv sync --extra wandb
uv run turn-wm train logging.wandb.enabled=true logging.wandb.entity=<entity>
```

It logs to `logging.wandb.project` (default `turn-wm`) under the run id, or
`logging.wandb.name`, with the resolved configuration. Without it, the
Trainer runs without a logger: metrics are not recorded anywhere and only
drive checkpointing.

## Precompute Mimi features

Mimi is frozen, so its features can be computed once per recording instead
of at every training step:

```bash
uv run turn-wm precompute-mimi \
    --dataset egocom \
    --media-root /path/to/egocom \
    --output /path/to/mimi-cache \
    --device cuda
```

- Each recording of the action grid is read once from its local media
  (respecting `media_offset_s`), resampled once to 24 kHz and streamed
  through Mimi with its convolution and attention caches kept across chunks,
  so the features do not depend on `--chunk-seconds`.
- Mimi's continuous 12.5 Hz features are causally aligned to the 10 Hz
  action grid: one 512-d row per grid step, row 0 being the recording's
  first `decision_index`. The learned 512 -> 192 projector is not
  precomputed; it stays part of the model.
- The cache holds one float16 safetensors file per recording and a
  `manifest.json` recording the Mimi model and revisions, the source dataset
  revision, the feature rate, dim and dtype, and for every recording its
  `start_index`, `start_time_s`, number of steps, and detected `audio_gaps`.
- `--media-root` works as for `inspect-data` (`DATASET=PATH`, repeated, for
  `--dataset full`). `--revision` pins Mimi, ideally to a commit SHA. The
  output directory must be new or empty; grids and media are checked for
  every recording before encoding starts.

### Audio timeline notes

Synchronization, gap handling, causal alignment, and excluded-recording details
are documented in [docs/training.md](docs/training.md).

After the per-corpus caches are in a shared release directory, deterministic
release metadata can be rebuilt with:

```bash
uv run python scripts/build_mimi_release.py /path/to/mimi/v1
```

The features will then feed training in place of the raw audio.

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
