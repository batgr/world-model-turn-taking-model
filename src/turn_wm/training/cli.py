"""Training command for the project CLI."""

from __future__ import annotations

import argparse

from hydra.errors import HydraException

from turn_wm.config import load_config
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

    train.set_defaults(handler=_train)


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
