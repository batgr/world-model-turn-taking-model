"""
Command-line entry point for the turn-taking modelling repository.

`inspect-data` runs the same data path that training will use (source loading,
TurnTakingDataset, DataLoader) and prints a structural summary of one batch.
With `--media-root`, raw media is decoded from a local corpus copy; media is
never downloaded. `--modalities` restricts what is decoded. `train` composes
the Hydra configuration from its overrides and runs
`turn_wm.training.train.run`. `precompute-mimi` writes the frozen Mimi
features of every recording, aligned to the action grid, to a local cache.
`extract-latents` writes a finished run's anchor representations for offline
analysis (`turn_wm.evaluation.latent_analysis`), and `analyze-latents`
analyzes such a snapshot (spectral geometry, globally and per corpus).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from datasets import concatenate_datasets
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from hydra.errors import HydraException

from turn_wm.config import load_config
from turn_wm.data.build import build_dataset
from turn_wm.data.dataset import (
    ACTION_TO_ID,
    MASKED_ACTION_ID,
    PAD_ACTION_ID,
    PAD_STATE_ID,
    STATE_TO_ID,
    WindowConfig,
)
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaIndex,
    MediaModality,
    MediaPaths,
    validate_modalities,
)
from turn_wm.data.mimi_precompute import RecordingSpan, precompute_mimi_cache
from turn_wm.data.multi import MultiCorpusDataset
from turn_wm.data.reader import MediaWindow
from turn_wm.data.source import (
    DATASETS,
    HuggingFaceSource,
    LoadedCorpus,
    LoadedData,
    load_data,
)
from turn_wm.evaluation.latent_analysis.analyze import ANALYSES, analyze_snapshot
from turn_wm.evaluation.latent_analysis.run import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SPLIT,
    extract_run,
)
from turn_wm.training.train import run as run_training

SPLITS = ("train", "validation", "test")

_CONTEXT_KEYS = ("context_state", "context_action", "context_valid", "context_mask")
_FUTURE_KEYS = ("future_state", "future_action", "future_valid")

_DEFAULT_WINDOW = WindowConfig()

_ROOT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


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
        help=(
            "Published source to inspect: one corpus, or 'full' for every "
            "corpus of the private release (default: egocom)."
        ),
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
    inspect.add_argument(
        "--shuffle",
        action="store_true",
        help="Inspect a seeded shuffled batch instead of the first samples.",
    )
    inspect.add_argument(
        "--media-root",
        action="append",
        type=_media_root,
        metavar="[DATASET=]PATH",
        help=(
            "Decode raw media from a local corpus root. Use DATASET=PATH, "
            "repeated, when the media manifest covers several datasets."
        ),
    )
    inspect.add_argument(
        "--modalities",
        type=_modalities,
        metavar="MODALITY[,MODALITY]",
        help=(
            "Media to decode with --media-root, comma-separated among "
            f"{', '.join(MEDIA_MODALITIES)} (default: all)."
        ),
    )
    inspect.set_defaults(handler=_inspect_data)

    train = commands.add_parser(
        "train",
        help="Train a turn-taking world model.",
        description=(
            "Train a world model from the configured dataset and local media. "
            "Additional arguments are Hydra overrides."
        ),
    )

    train.add_argument(
        "overrides",
        nargs="*",
        metavar="OVERRIDE",
        help=(
            "Hydra configuration overrides, for example "
            "'data.dataset=egocom' or 'trainer.max_epochs=10'."
        ),
    )

    train.set_defaults(handler=_train)

    precompute = commands.add_parser(
        "precompute-mimi",
        help="Precompute frozen Mimi features for every recording.",
        description=(
            "Encode every recording of a published dataset with frozen Mimi "
            "from local raw media, align the features to the 10 Hz action "
            "grid and write one safetensors file per recording plus "
            "manifest.json."
        ),
    )
    precompute.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="egocom",
        help="Published source to encode (default: egocom).",
    )
    precompute.add_argument(
        "--media-root",
        action="append",
        type=_media_root,
        required=True,
        metavar="[DATASET=]PATH",
        help=(
            "Local corpus root the media manifest paths are relative to. Use "
            "DATASET=PATH, repeated, when the source has several corpora."
        ),
    )
    precompute.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Cache directory to create; must not exist or be empty.",
    )
    precompute.add_argument(
        "--device",
        default="cpu",
        help="Torch device for Mimi, e.g. cpu, cuda, mps (default: cpu).",
    )
    precompute.add_argument(
        "--chunk-seconds",
        type=_positive_float,
        default=20.0,
        help=(
            "Audio streamed through Mimi per call, rounded down to whole Mimi "
            "frames; features do not depend on it (default: %(default)s)."
        ),
    )
    precompute.add_argument(
        "--model",
        default="kyutai/mimi",
        help="Mimi checkpoint on the Hub (default: %(default)s).",
    )
    precompute.add_argument(
        "--revision",
        help="Mimi revision to pin, ideally a commit SHA (default: latest).",
    )
    precompute.set_defaults(handler=_precompute_mimi)

    extract = commands.add_parser(
        "extract-latents",
        help="Extract a trained run's representations for analysis.",
        description=(
            "Rebuild a training run's data (config, dataset revision, "
            "observation source, window) and write the encoder features and "
            "projected latents at each anchor of a seeded sample of one split."
        ),
    )
    extract.add_argument(
        "run_dir",
        type=Path,
        help="Run directory holding config.yaml, metadata.json and checkpoints/.",
    )
    extract.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="File under checkpoints/, or a path (default: %(default)s).",
    )
    extract.add_argument(
        "--split",
        choices=SPLITS,
        default=DEFAULT_SPLIT,
        help="Split to extract (default: %(default)s).",
    )
    extract.add_argument(
        "--max-samples",
        type=_positive_int,
        help="Samples to keep from the seeded order (default: all).",
    )
    extract.add_argument(
        "--seed",
        type=int,
        help="Seed of the sample order (default: the run's seed).",
    )
    extract.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Batch size; does not change the samples (default: the run's).",
    )
    extract.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loader workers (default: %(default)s).",
    )
    extract.add_argument(
        "--device",
        default="cpu",
        help="Torch device, e.g. cpu, cuda, mps (default: %(default)s).",
    )
    extract.add_argument(
        "--mimi-cache-root",
        type=Path,
        help="Where the run's Mimi cache now lives, if it moved.",
    )
    extract.add_argument(
        "--output",
        type=Path,
        help=(
            "Directory to create; must not exist or be empty (default: "
            "RUN_DIR/latent_analysis/<checkpoint>-<split>)."
        ),
    )
    extract.set_defaults(handler=_extract_latents)

    analyze = commands.add_parser(
        "analyze-latents",
        help="Analyze an extracted representation snapshot.",
        description=(
            "Analyze every representation of a snapshot written by "
            "extract-latents, globally and per corpus. Reads the snapshot "
            "only: no checkpoint, dataset or cache."
        ),
    )
    analyze.add_argument(
        "snapshot",
        type=Path,
        help="Snapshot directory holding representations.safetensors.",
    )
    analyze.add_argument(
        "--analysis",
        action="append",
        choices=ANALYSES,
        help="Analysis to run, repeatable (default: all).",
    )
    analyze.add_argument(
        "--output",
        type=Path,
        help=(
            "Root of the results, one directory per analysis "
            "(default: SNAPSHOT/analysis)."
        ),
    )
    analyze.set_defaults(handler=_analyze_latents)

    return parser


def _train(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        cfg = load_config(args.overrides)
    except HydraException as error:
        parser.error(f"invalid configuration override: {error}")

    try:
        run_training(cfg)
    except ValueError as error:
        # Rejected configuration, unknown dataset or missing media root.
        raise SystemExit(f"turn-wm: error: {error}") from error

    return 0


def _extract_latents(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        output = extract_run(
            args.run_dir,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            mimi_cache_root=args.mimi_cache_root,
        )
    except (ValueError, FileNotFoundError) as error:
        # Not a run directory, edited config, other data revision or cache,
        # missing checkpoint or media, or a non-empty output directory.
        raise SystemExit(f"turn-wm: error: {error}") from error

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    print(f"Representations written to {output}")
    print(f"samples: {manifest['samples']}")

    for name, spec in manifest["representations"].items():
        print(f"{name}: {spec['shape']}")

    return 0


def _analyze_latents(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    try:
        outputs = analyze_snapshot(
            args.snapshot,
            analyses=args.analysis or ANALYSES,
            output_root=args.output,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        # Not a snapshot, a non-empty output directory or no matplotlib.
        raise SystemExit(f"turn-wm: error: {error}") from error

    for name, output in outputs.items():
        print(f"{name}: {output}")

    return 0


def _precompute_mimi(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    data = _load(DATASETS[args.dataset])
    media_roots = _media_roots(data, args.media_root)

    def report(index: int, total: int, span: RecordingSpan) -> None:
        print(f"[{index}/{total}] {span.dataset} / {span.recording_id}", flush=True)

    try:
        manifest_path = precompute_mimi_cache(
            data,
            media_roots=media_roots,
            output_root=args.output,
            model_name=args.model,
            model_revision=args.revision,
            chunk_seconds=args.chunk_seconds,
            device=args.device,
            progress=report,
        )
    except (ValueError, OSError) as error:
        # Output not empty, missing media or root, inconsistent grid or
        # audio, or a Mimi model/revision the Hub cannot provide.
        raise SystemExit(f"turn-wm: error: {error}") from error

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["features"]

    print(f"Mimi cache written to {manifest_path.parent}")
    print(f"recordings: {len(manifest['recordings'])}")
    print(f"feature rate: {features['rate_hz']:g} Hz")
    print(f"feature dim: {features['dim']}")

    return 0


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

    if args.modalities is not None and not args.media_root:
        parser.error("--modalities requires --media-root")

    modalities = args.modalities or MEDIA_MODALITIES

    source = DATASETS[args.dataset]
    data = _load(source)
    media_roots = _media_roots(data, args.media_root) if args.media_root else None

    try:
        dataset = build_dataset(
            data,
            split=args.split,
            window=window,
            training=False,
            media_roots=media_roots,
            modalities=modalities,
        )
    except ValueError as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    loader = build_dataloader(
        dataset,
        loader=DataLoaderConfig(
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=args.shuffle,
        ),
    )

    try:
        batch = next(iter(loader))
    except FileNotFoundError as error:
        if media_roots is None:
            raise

        raise SystemExit(
            f"turn-wm: error: raw media for the first batch is missing "
            f"under the configured media root: {error}"
        ) from error

    print(
        format_summary(
            source=source,
            split=args.split,
            data=data,
            dataset=dataset,
            window=window,
            batch=batch,
            media_index=(
                None if media_roots is None else _display_index(data, media_roots)
            ),
            modalities=modalities,
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


def _media_roots(
    data: LoadedData,
    media_roots: list[tuple[str | None, Path]],
) -> dict[str, Path]:
    manifests = [c.media_manifest for c in data.corpora if c.media_manifest is not None]

    if not manifests:
        raise SystemExit(
            "turn-wm: error: this dataset source does not publish a media "
            "manifest, so --media-root cannot be used"
        )

    manifest_datasets = sorted(
        {name for manifest in manifests for name in manifest.unique("dataset")}
    )
    roots: dict[str, Path] = {}

    for name, path in media_roots:
        if name is None:
            if len(media_roots) > 1 or len(manifest_datasets) != 1:
                raise SystemExit(
                    "turn-wm: error: the media manifests cover datasets "
                    f"{manifest_datasets}; pass --media-root DATASET=PATH "
                    "for each"
                )

            name = manifest_datasets[0]

        if name in roots:
            raise SystemExit(f"turn-wm: error: duplicate --media-root for {name!r}")

        roots[name] = path

    return roots


def _display_index(data: LoadedData, roots: dict[str, Path]) -> MediaIndex:
    """Lookup of every loaded media record, for describing the first sample."""

    manifests = [c.media_manifest for c in data.corpora if c.media_manifest is not None]

    return MediaIndex.from_manifest(concatenate_datasets(manifests), roots)


def format_summary(
    *,
    source: HuggingFaceSource,
    split: str,
    data: LoadedData,
    dataset: MultiCorpusDataset,
    window: WindowConfig,
    batch: dict[str, Any],
    media_index: MediaIndex | None = None,
    modalities: tuple[MediaModality, ...] = MEDIA_MODALITIES,
) -> str:
    """Render a concise structural summary of one inspected batch."""

    lengths = batch["context_lengths"]
    usable = dataset.corpus_sizes()
    not_in_split = [name for name in data.names if name not in usable]

    lines = [
        *_section("Dataset"),
        f"source: {source.repo_id}",
        f"revision: {source.revision or 'default branch (unpinned)'}",
        f"resolved commit: {data.revision or 'unknown'}",
        f"split: {split}",
        "corpora:",
        *[f"  - {name}" for name in dataset.corpora],
        *([f"not in this split: {', '.join(not_in_split)}"] if not_in_split else []),
        f"usable samples: {len(dataset):,}",
        "",
    ]

    for name in dataset.corpora:
        lines += _corpus_lines(data.corpus(name), split=split, usable=usable[name])

    lines += [
        *_section("Window"),
        f"context: {window.min_context_steps}–{window.max_context_steps} steps",
        f"future: {window.future_steps} steps",
        "",
        *_section("Batch"),
        f"batch size: {batch['context_state'].shape[0]}",
        "datasets:",
        *[
            f"  {name}: {count}"
            for name, count in Counter(batch["dataset"]).most_common()
        ],
        *_shape_lines(batch, _CONTEXT_KEYS),
        "",
        *_shape_lines(batch, _FUTURE_KEYS),
        "",
        "context lengths:",
        f"min: {int(lengths.min())}",
        f"max: {int(lengths.max())}",
        "",
        *_section("First sample"),
        f"dataset: {batch['dataset'][0]}",
        f"sample_id: {batch['sample_id'][0]}",
        f"recording_id: {batch['recording_id'][0]}",
        f"anchor_idx: {int(batch['anchor_idx'][0])}",
        f"sample_class: {batch['sample_class'][0]}",
        f"context_length: {int(lengths[0])}",
        "",
        *_media_lines(batch, media_index, modalities),
        *_section("States"),
        *_vocabulary_lines({**STATE_TO_ID, "PAD": PAD_STATE_ID}),
        "",
        *_section("Actions"),
        *_vocabulary_lines(
            {**ACTION_TO_ID, "MASKED": MASKED_ACTION_ID, "PAD": PAD_ACTION_ID}
        ),
    ]

    return "\n".join(lines)


def _corpus_lines(corpus: LoadedCorpus, *, split: str, usable: int) -> list[str]:
    metadata = corpus.metadata

    return [
        *_section(f"Corpus: {corpus.name}"),
        f"samples: {len(corpus.model_ready[split]):,}",
        f"usable (is_trainable): {usable:,}",
        f"recordings: {_count(_metadata_value(metadata, 'splits', 'recordings', split))}",
        f"action grid rows: {len(corpus.action_grid):,}",
        f"grid: {_frequency(_metadata_value(metadata, 'grid', 'frequency_hz'))}",
        "",
    ]


def _media_lines(
    batch: dict[str, Any],
    media_index: MediaIndex | None,
    modalities: tuple[MediaModality, ...],
) -> list[str]:
    if media_index is None or "context_media" not in batch:
        return []

    media = media_index.get(
        dataset=batch["dataset"][0],
        recording_id=batch["recording_id"][0],
    )

    return [
        *_section("Media"),
        f"dataset: {media.dataset}",
        f"recording: {media.recording_id}",
        f"modalities: {', '.join(modalities)}",
        f"video file: {media.video_path or 'none'}",
        f"audio file: {media.audio_path or 'none'}",
        f"media offset: {media.media_offset_s:g} s",
        "",
        "context:",
        *_window_lines(batch["context_media"][0], media),
        "",
        "future:",
        *_window_lines(batch["future_media"][0], media),
        "",
    ]


def _window_lines(window: MediaWindow, media: MediaPaths) -> list[str]:
    lines = [
        (
            "  canonical time: "
            f"{_seconds(window.canonical_start_time_s)} → "
            f"{_seconds(window.canonical_end_time_s)}"
        ),
        (
            "  physical media time: "
            f"{_seconds(window.start_time_s)} → {_seconds(window.end_time_s)}"
        ),
        "  audio:",
    ]

    audio = window.audio

    if audio is None:
        lines.append("    present: no")
    else:
        lines += [
            "    present: yes",
            f"    source: {_audio_source(media)}",
            f"    sample rate: {audio.sample_rate} Hz",
            f"    shape (channels, samples): {tuple(audio.waveform.shape)}",
        ]

    lines.append("  video:")

    video = window.video

    if video is None:
        lines.append("    present: no")
    else:
        timestamps = video.timestamps_s
        lines += [
            "    present: yes",
            f"    frames: {video.frames.shape[0]}",
            f"    shape (T, C, H, W): {tuple(video.frames.shape)}",
            f"    first timestamp: {_seconds(float(timestamps[0]))}",
            f"    last timestamp: {_seconds(float(timestamps[-1]))}",
        ]

    return lines


def _audio_source(media: MediaPaths) -> str:
    if media.audio_path is not None:
        return "dedicated audio file"

    if media.video_has_audio is None:
        return "embedded video (unprobed)"

    return "embedded video"


def _seconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f} s"


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


def _media_root(value: str) -> tuple[str | None, Path]:
    name: str | None = None
    head, separator, tail = value.partition("=")

    if separator and _ROOT_NAME.match(head):
        name, value = head, tail

    path = Path(value).expanduser()

    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"media root is not a directory: {path}")

    return name, path


def _modalities(value: str) -> tuple[MediaModality, ...]:
    try:
        # validate_modalities rejects anything that is not a MediaModality.
        parts = cast(list[MediaModality], [part.strip() for part in value.split(",")])
        return validate_modalities(parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None

    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")

    return number


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
