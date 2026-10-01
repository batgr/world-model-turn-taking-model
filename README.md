# World model for conversational turn-taking

A JEPA-style latent world model for planning-based multi-party turn-taking in
social robots. Frozen Mimi features encode speech into latent observations, and
an action-conditioned causal predictor learns how the conversational state
evolves over time. Its rollouts are designed to serve as the predictive model
for a downstream planner deciding when the robot should speak, wait, or yield
the floor.
**Explore:** [public EgoCom dataset](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom) · [training design](docs/training.md) · [research program](docs/research_program.md) · [research decisions](docs/decisions/README.md)

## Data

Inspect the structure of the first batch built from a dataset. This loads the
dataset through the same data package used for training:

```bash
uv run turn-wm inspect-data --dataset dataset
uv run turn-wm inspect-data --dataset dataset --split validation --batch-size 4
```

`--dataset full` loads every configured corpus into one dataset; a split
includes only the corpora that publish it. Add `--shuffle` to inspect a
seeded shuffled mixed-corpus batch.

```bash
uv run turn-wm inspect-data --dataset full --batch-size 8 --shuffle
```

In code, one or many corpora go through the same path:

```python
loaded = load_data(DATASETS["full"])
dataset = build_dataset(loaded, split="train", window=WindowConfig(), training=True)
loader = build_dataloader(dataset, loader=DataLoaderConfig(batch_size=32))
```

Raw media is not downloaded automatically. To decode it from a local copy,
pass the root directory referenced by the dataset's media manifest:

```bash
uv run turn-wm inspect-data --dataset dataset --batch-size 1 --media-root /path/to/media
```

## Model

Implementation map:

- `src/turn_wm/models/encoders/mimi.py` — frozen Mimi encoder
- `src/turn_wm/models/lewm/mlp.py` — latent projector and prediction head
- `src/turn_wm/models/lewm/embedder.py` — action embedding
- `src/turn_wm/models/lewm/predictor.py` — action-conditioned predictor
- `src/turn_wm/models/lewm/transformer.py` — causal Transformer backbone
- `src/turn_wm/models/lewm/sigreg.py` — SIGReg regularizer
- `src/turn_wm/training/objective.py` — training objective (teacher forcing, rollout, SIGReg)
- `src/turn_wm/training/lewm.py` — Lightning module

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
selected with `train=xxx`). The default model uses a RoPE predictor and causal
BatchNorm in both projectors; `model=lewm_ln`, `lewm_standard_bn`,
`lewm_positional_cbn` and `lewm_learned_pos` each change one component.
`train=lewm_v2` is the current recipe: a 30-step context and rollout window,
horizons up to 10 steps and checkpoint selection on `val/rollout_10_mse`.

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

For raw-audio debugging instead of the feature cache, set the media-root
environment variable for the selected dataset. Replace `<dataset>` with the
dataset key configured in `DATASETS`:

```bash
export <DATASET>_MEDIA_ROOT=/path/to/media

uv run turn-wm train \
  data.dataset=<dataset> \
  data.observation_source=raw_audio
```

For example, a dataset key `my_corpus` uses `MY_CORPUS_MEDIA_ROOT`.

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

Mimi is frozen, so its features can be precomputed once and reused across
training runs:

```bash
uv run turn-wm precompute-mimi \
  --dataset dataset \
  --media-root /path/to/media \
  --output /path/to/mimi-cache \
  --device cuda
```

For multiple corpora, repeat `--media-root` with the `DATASET=PATH` form:

```bash
uv run turn-wm precompute-mimi \
  --dataset full \
  --media-root dataset_a=/path/to/dataset_a \
  --media-root dataset_b=/path/to/dataset_b \
  --output /path/to/mimi-cache \
  --device cuda
```

The output directory must be new or empty. Use `--revision` to pin the Mimi
checkpoint revision.

The cache contains one feature file per recording plus a `manifest.json`
describing the cache and source revisions. See [docs/training.md](docs/training.md)
for feature alignment, cache format, synchronization handling, and validation
details.

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
