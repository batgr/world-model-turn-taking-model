"""Training command for the project CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra.errors import HydraException

from turn_wm.config import load_config
from turn_wm.training.train import restore
from turn_wm.training.train import run as run_training


def add_train_command(commands: argparse._SubParsersAction) -> None:
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

    train.add_argument(
        "--restore",
        type=Path,
        metavar="RUN_DIR",
        help=(
            "Resume this run directory from its config.yaml and "
            "checkpoints/last.ckpt, in place; takes no overrides."
        ),
    )

    train.set_defaults(handler=_train)


def _train(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.restore is not None and args.overrides:
        parser.error("--restore resumes the run as configured; it takes no overrides")

    try:
        if args.restore is not None:
            restore(args.restore)
            return 0

        cfg = load_config(args.overrides)
    except HydraException as error:
        parser.error(f"invalid configuration override: {error}")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"turn-wm: error: {error}") from error

    try:
        run_training(cfg)
    except ValueError as error:
        # Rejected configuration, unknown dataset or missing media root.
        raise SystemExit(f"turn-wm: error: {error}") from error

    return 0
