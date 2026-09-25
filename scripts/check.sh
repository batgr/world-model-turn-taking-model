#!/usr/bin/env bash
# Lint, format, type-check and unit-test the repository. Every step runs even
# if an earlier one fails; the script exits non-zero if any step failed.
# Integration tests (real data, Mimi weights) run separately:
#   uv run pytest -m integration
set -uo pipefail

cd "$(dirname "$0")/.."

status=0

step() {
    echo "==> $*"
    "$@" || status=1
}

step uv run ruff check .
step uv run ruff format --check .
step uv run pyright src
step uv run pytest

exit "$status"
