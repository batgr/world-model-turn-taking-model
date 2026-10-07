# World model for conversational turn-taking

A JEPA-style latent world model for multi-party turn-taking in social robots.
A frozen encoder (Mimi by default) turns the recording into latent
observations, and an action-conditioned causal predictor learns how the
conversational state evolves. Its rollouts are meant to serve a downstream
planner deciding when the robot should speak, wait or yield the floor.

[Training design](docs/training.md) · [research program](docs/research_program.md) ·
[decisions](docs/decisions/README.md) · [data pipeline](https://github.com/batgr/world-model-turn-taking-data)

## Data

Runs load a Hugging Face release listed in `turn_wm.data.source.DATASETS`
(`data.dataset`): `egocom` (public), `ego4d` and `full` (private, both corpora)
on a 10 Hz grid, and `egocom_12.5hz`, `ego4d_12.5hz`, `full_12.5hz` on a
12.5 Hz grid, one step per Mimi frame (train them with `data.grid_rate_hz=12.5`).
A split includes only the corpora that publish it.

## Configuration

[Hydra](https://hydra.cc) composes `configs/config.yaml` from one model and one
training recipe:

```text
configs/
  config.yaml       embed_dim; defaults: model, train
  model/lewm*.yaml  the JEPA and its sub-modules (_target_)
  model/encoder/    the frozen encoder (mimi, logmel) and the feature_dim it gives
  train/lewm*.yaml  seed, data, prediction, loss, optimizer, loader, trainer
```

The recipe sits at the root of the composed config (`cfg.trainer.max_epochs`,
`cfg.loss.sigreg.weight`). `train=lewm_v2` is the current recipe; each
`model=lewm_*` variant changes one component of the default model.

## Training

The encoder is frozen, so its features are computed once per dataset:

```bash
uv run turn-wm precompute-features --encoder mimi --dataset egocom_12.5hz \
  --media-root /path/to/EgoCom --output /path/to/cache --device cuda
```

`--media-root` takes `DATASET=PATH` once per corpus for `full`; trailing Hydra
overrides configure the encoder (e.g. `model.encoder.revision=<sha>`). Then
train on the cache, with any Hydra override:

```bash
uv run turn-wm train train=lewm_v2 data.dataset=egocom_12.5hz data.grid_rate_hz=12.5 \
  data.feature_cache.root=/path/to/cache loader.batch_size=16
```

`data.observation_source=raw_audio` encodes the media during training instead,
for debugging; it reads `<DATASET>_MEDIA_ROOT` (e.g. `EGOCOM_MEDIA_ROOT`).
A configuration rejected by `validate_config` fails before any data is read.
[docs/training.md](docs/training.md) covers the objective, the curriculum,
the encoders and the cache.

### Runs

Each run gets a directory, created once the configuration, the dataset and
the media roots have been checked:

```text
outputs/<experiment.name>/<UTC timestamp>-<config hash>/
  config.yaml      the resolved configuration
  metadata.json    seed, git commit and dirty flag, host, dataset revision, resumptions
  train.log        Lightning's messages (validation verdicts, resumption, warnings)
  tensorboard/     losses, LR, throughput
  fit-profile.txt  time per training hook (e.g. train_dataloader_next: waiting for data)
  checkpoints/     best checkpoints by checkpoint.monitor, and last.ckpt
```

```bash
uv run tensorboard --logdir outputs/
uv run turn-wm train --restore outputs/lewm/<run id>     # resume in place
uv run turn-wm train checkpoint.resume_from=<ckpt>        # new run from these weights
```

Weights & Biases is optional: `uv sync --extra wandb`, then
`logging.wandb.enabled=true logging.wandb.entity=<entity>`.

## Tests

```bash
uv run pytest                  # offline unit tests
uv run pytest -m integration   # downloads EgoCom (and Mimi) from the Hub
```

Raw-media smoke tests also need `EGOCOM_MEDIA_ROOT` or `EGO4D_MEDIA_ROOT`.
