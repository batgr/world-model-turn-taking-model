"""Small command-line dispatcher for the turn-taking tools."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from turn_wm.data.cli import add_data_commands
from turn_wm.evaluation.cli import add_analysis_commands
from turn_wm.training.cli import add_train_command


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

    add_data_commands(commands)
    add_train_command(commands)
    add_analysis_commands(commands)

    return parser
