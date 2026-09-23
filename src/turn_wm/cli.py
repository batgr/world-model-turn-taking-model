"""
Command-line entry point for the turn-taking modelling repository.

`inspect-data` runs the same data path that training will use (source loading,
TurnTakingDataset, DataLoader) and prints a structural summary of one batch.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import httpx
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    PAD_ACTION_ID,
    PAD_STATE_ID,
    STATE_TO_ID,
    TurnTakingDataset,
    WindowConfig,
)
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.source import DATASETS, HuggingFaceSource, LoadedData, load_data

SPLITS = ("train", "validation", "test")

_CONTEXT_KEYS = ("context_state", "context_action", "context_valid", "context_mask")
_FUTURE_KEYS = ("future_state", "future_action", "future_valid")

_DEFAULT_WINDOW = WindowConfig()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    return args.handler(args, parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turn-wm",
        description="Turn-taking world-model tools.",
    )
    commands = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    inspect = commands.add_parser(
        "inspect-data",
        help="Load a published dataset and summarize one batch.",
        description=(
            "Load a published dataset through the modelling data package and "
            "print the structure of its first batch. Uses natural sampling "
            "and deterministic (evaluation-style) context lengths."
        ),
    )
    inspect.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="egocom",
        help="Published dataset to inspect (default: egocom).",
    )
    inspect.add_argument(
        "--split",
        choices=SPLITS,
        default="train",
        help="Model-ready split to inspect (default: train).",
    )
    inspect.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        help="Number of samples in the inspected batch (default: 32).",
    )
    inspect.add_argument(
        "--context-min",
        type=_positive_int,
        default=_DEFAULT_WINDOW.min_context_steps,
        help="Minimum context steps (default: %(default)s).",
    )
    inspect.add_argument(
        "--context-max",
        type=_positive_int,
        default=_DEFAULT_WINDOW.max_context_steps,
        help="Maximum context steps (default: %(default)s).",
    )
    inspect.add_argument(
        "--future-steps",
        type=_positive_int,
        default=_DEFAULT_WINDOW.future_steps,
        help="Future steps to predict (default: %(default)s).",
    )
    inspect.set_defaults(handler=_inspect_data)

    return parser


def _inspect_data(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        window = WindowConfig(
            min_context_steps=args.context_min,
            max_context_steps=args.context_max,
            future_steps=args.future_steps,
        )
    except ValueError as error:
        parser.error(str(error))

    source = DATASETS[args.dataset]
    data = _load(source)

    if args.split not in data.model_ready:
        raise SystemExit(
            f"turn-wm: error: split {args.split!r} is not published for "
            f"{source.repo_id}; available: {sorted(data.model_ready)}"
        )

    anchors = data.model_ready[args.split]

    if len(anchors) == 0:
        raise SystemExit(f"turn-wm: error: split {args.split!r} is empty")

    if len(data.action_grid) == 0:
        raise SystemExit("turn-wm: error: action grid is empty")

    dataset = TurnTakingDataset(
        anchors=anchors,
        action_grid=data.action_grid,
        window=window,
        training=False,
    )

    if len(dataset) == 0:
        raise SystemExit(
            f"turn-wm: error: split {args.split!r} has no usable (is_trainable) anchors"
        )

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(batch_size=args.batch_size, num_workers=0),
    )
    batch = next(iter(loader))

    print(
        format_summary(
            source=source,
            split=args.split,
            data=data,
            anchor_count=len(anchors),
            usable_count=len(dataset),
            window=window,
            batch=batch,
        )
    )

    return 0


def _load(source: HuggingFaceSource) -> LoadedData:
    """Load a source, turning common Hub failures into clear CLI errors."""

    repo = source.repo_id

    try:
        return load_data(source)
    except GatedRepoError as error:
        raise SystemExit(
            f"turn-wm: error: {repo} requires authentication or accepted "
            f"access terms (run `hf auth login`): {error}"
        ) from error
    except (RepositoryNotFoundError, DatasetNotFoundError) as error:
        raise SystemExit(
            f"turn-wm: error: dataset {repo} was not found or is not "
            "accessible; private datasets require `hf auth login`: "
            f"{error}"
        ) from error
    except HfHubHTTPError as error:
        raise SystemExit(
            f"turn-wm: error: Hugging Face request for {repo} failed: {error}"
        ) from error
    except (httpx.TransportError, ConnectionError) as error:
        raise SystemExit(
            f"turn-wm: error: could not reach Hugging Face to load {repo} "
            f"(network unavailable?): {error}"
        ) from error
    except ValueError as error:
        raise SystemExit(
            f"turn-wm: error: {repo} does not match the modelling data "
            f"contract: {error}"
        ) from error


def format_summary(
    *,
    source: HuggingFaceSource,
    split: str,
    data: LoadedData,
    anchor_count: int,
    usable_count: int,
    window: WindowConfig,
    batch: dict[str, Any],
) -> str:
    """Render a concise structural summary of one inspected batch."""

    metadata = data.metadata
    recordings = _metadata_value(metadata, "splits", "recordings", split)
    total_recordings = _metadata_value(metadata, "counts", "recordings")
    frequency_hz = _metadata_value(metadata, "grid", "frequency_hz")
    lengths = batch["context_lengths"]

    lines = [
        *_section("Dataset"),
        f"source: {source.repo_id}",
        f"revision: {source.revision or 'default branch (unpinned)'}",
        f"split: {split}",
        "",
        *_section("Model-ready"),
        f"samples: {anchor_count:,}",
        f"usable (is_trainable): {usable_count:,}",
        f"recordings: {_count(recordings)}",
        "",
        *_section("Action grid"),
        f"rows: {len(data.action_grid):,}",
        f"recordings: {_count(total_recordings)}",
        "",
        *_section("Window"),
        f"context: {window.min_context_steps}–{window.max_context_steps} steps",
        f"future: {window.future_steps} steps",
        f"grid: {_frequency(frequency_hz)}",
        "",
        *_section("Batch"),
        f"batch size: {batch['context_state'].shape[0]}",
        *_shape_lines(batch, _CONTEXT_KEYS),
        "",
        *_shape_lines(batch, _FUTURE_KEYS),
        "",
        "context lengths:",
        f"min: {int(lengths.min())}",
        f"max: {int(lengths.max())}",
        "",
        *_section("First sample"),
        f"sample_id: {batch['sample_id'][0]}",
        f"recording_id: {batch['recording_id'][0]}",
        f"anchor_idx: {int(batch['anchor_idx'][0])}",
        f"sample_class: {batch['sample_class'][0]}",
        f"context_length: {int(lengths[0])}",
        "",
        *_section("States"),
        *_vocabulary_lines({**STATE_TO_ID, "PAD": PAD_STATE_ID}),
        "",
        *_section("Actions"),
        *_vocabulary_lines(
            {**ACTION_TO_ID, "MASKED": MASKED_ACTION_ID, "PAD": PAD_ACTION_ID}
        ),
    ]

    return "\n".join(lines)


def _section(title: str) -> list[str]:
    return [title, "-" * len(title)]


def _shape_lines(batch: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    width = max(len(key) for key in _CONTEXT_KEYS + _FUTURE_KEYS) + 1

    return [f"{key + ':':<{width}} {tuple(batch[key].shape)}" for key in keys]


def _vocabulary_lines(vocabulary: dict[str, int]) -> list[str]:
    return [
        f"{index} {name}"
        for name, index in sorted(vocabulary.items(), key=lambda item: item[1])
    ]


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    value: Any = metadata

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None

        value = value[key]

    return value


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "unknown (not in metadata)"


def _frequency(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:g} Hz"

    return "unknown (not in metadata)"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")

    return number
