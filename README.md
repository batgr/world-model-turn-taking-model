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
