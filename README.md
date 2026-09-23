# world-model-turn-taking-models

## Data

Inspect the structure of the first batch built from the public EgoCom dataset
([`batgre/conversational-dynamics-egocom`](https://huggingface.co/datasets/batgre/conversational-dynamics-egocom)).
This loads the dataset through the same data package used for training:

```bash
uv run turn-wm inspect-data --dataset egocom
uv run turn-wm inspect-data --dataset egocom --split validation --batch-size 4
```

## Tests

```bash
uv run pytest                  # offline unit tests (default)
uv run pytest -m integration   # real-data smoke test; downloads EgoCom from the Hub
```

Tests marked `integration` need network access (or a warm Hugging Face cache),
so the default run skips them.
