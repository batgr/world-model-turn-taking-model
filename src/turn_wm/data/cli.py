"""Dataset inspection and encoder feature precomputation commands."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import cast

import httpx
from datasets import concatenate_datasets
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from turn_wm.config import CONFIG_DIR
from turn_wm.data.build import build_dataset
from turn_wm.data.dataset import WindowConfig
from turn_wm.data.feature_precompute import RecordingSpan, precompute_features
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.data.media import (
    MEDIA_MODALITIES,
    MediaIndex,
    MediaModality,
    validate_modalities,
)
from turn_wm.data.source import DATASETS, HuggingFaceSource, LoadedData, load_data
from turn_wm.data.summary import format_summary
from turn_wm.models.encoders.base import Encoder
from turn_wm.progress import log, progress

SPLITS = ("train", "validation", "test")
_DEFAULT_WINDOW = WindowConfig()
_ROOT_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def add_data_commands(commands: argparse._SubParsersAction) -> None:
    _add_inspect_data(commands)
    _add_precompute_features(commands)


def _add_precompute_features(commands: argparse._SubParsersAction) -> None:
    precompute = commands.add_parser(
        "precompute-features",
        help="Precompute a frozen encoder's features for every recording.",
        description=(
            "Encode every recording of a published dataset with a frozen "
            "encoder (configs/model/encoder) from local raw media, align the "
            "features to the dataset's decision grid and write one "
            "safetensors file per recording plus manifest.json."
        ),
    )
    precompute.add_argument(
        "--encoder",
        choices=ENCODERS,
        default="mimi",
        help="Encoder config in configs/model/encoder (default: %(default)s).",
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
        help="Torch device for the encoder, e.g. cpu, cuda, mps (default: cpu).",
    )
    precompute.add_argument(
        "--chunk-seconds",
        type=positive_float,
        default=20.0,
        help=(
            "Audio per call for encoders that stream (Mimi: rounded down to "
            "whole frames); features do not depend on it (default: %(default)s)."
        ),
    )
    precompute.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "Hydra overrides of the encoder config, e.g. "
            "model.encoder.revision=<commit SHA> to pin Mimi's weights."
        ),
    )
    precompute.set_defaults(handler=_precompute_features)


def _precompute_features(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    start = time.perf_counter()
    log(f"precompute-features: dataset {args.dataset}")
    data = _load(DATASETS[args.dataset])
    media_roots = _media_roots(data, args.media_root)
    encoder = _encoder(args.encoder, args.overrides, data, parser)
    log(
        f"precompute-features: device {args.device}, encoder {encoder.name} "
        f"({encoder.frame_rate:g} Hz frames)"
    )
    log(f"precompute-features: output {args.output}")

    with progress(desc="precompute-features", unit="recording") as bar:

        def report(index: int, total: int, span: RecordingSpan) -> None:
            # Called before each recording: the ones before it are done.
            if bar.total is None:
                log(f"precompute-features: {total} recordings")
                bar.total = total

            bar.n = index - 1
            bar.set_postfix_str(f"{span.dataset} / {span.recording_id}")

        try:
            manifest_path = precompute_features(
                data,
                encoder=encoder,
                media_roots=media_roots,
                output_root=args.output,
                chunk_seconds=args.chunk_seconds,
                device=args.device,
                progress=report,
            )
        except (ValueError, OSError) as error:
            # Output not empty, missing media or root, inconsistent grid or
            # audio, or encoder weights the Hub cannot provide.
            raise SystemExit(f"turn-wm: error: {error}") from error

        bar.n = bar.total or 0
        bar.refresh()

    log(f"precompute-features: done in {time.perf_counter() - start:.0f}s")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["features"]

    print(f"Feature cache written to {manifest_path.parent}")
    print(f"manifest: {manifest_path}")
    print(f"recordings: {len(manifest['recordings'])}")
    print(f"feature rate: {features['rate_hz']:g} Hz")
    print(f"feature dim: {features['dim']}")

    return 0


ENCODERS = tuple(
    sorted(path.stem for path in (CONFIG_DIR / "model" / "encoder").glob("*.yaml"))
)
"""The encoder configs `--encoder` offers."""


def _encoder(
    name: str,
    overrides: list[str],
    data: LoadedData,
    parser: argparse.ArgumentParser,
) -> Encoder:
    """The `name` encoder, which must run at the decision grid rate of `data`."""

    from turn_wm.config import load_config
    from turn_wm.models.build import build_encoder

    try:
        cfg = load_config(
            [
                f"model/encoder={name}",
                f"data.grid_rate_hz={data.grid_rate_hz}",
                *overrides,
            ]
        )
        return build_encoder(cfg)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


def _add_inspect_data(commands: argparse._SubParsersAction) -> None:
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
        type=positive_int,
        default=32,
        help="Number of samples in the inspected batch (default: 32).",
    )
    inspect.add_argument(
        "--context-min",
        type=positive_int,
        default=_DEFAULT_WINDOW.min_context_steps,
        help="Minimum context steps (default: %(default)s).",
    )
    inspect.add_argument(
        "--context-max",
        type=positive_int,
        default=_DEFAULT_WINDOW.max_context_steps,
        help="Maximum context steps (default: %(default)s).",
    )
    inspect.add_argument(
        "--future-steps",
        type=positive_int,
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


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from None

    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")

    return number


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None

    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")

    return number
