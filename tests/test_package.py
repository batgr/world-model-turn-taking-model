import subprocess
import sys

import pytest

_IMPORT = (
    "import os, turn_wm, matplotlib; "
    "print(os.environ['MPLBACKEND'], matplotlib.get_backend())"
)


def _backend(value: str) -> str:
    return (
        subprocess.run(
            [sys.executable, "-c", _IMPORT],
            env={"MPLBACKEND": value, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .lower()
    )


def test_a_missing_notebook_backend_falls_back_to_agg():
    # What Colab exports to shell commands.
    assert _backend("module://matplotlib_inline.backend_inline") == "agg agg"


@pytest.mark.parametrize("value", ["pdf", "module://matplotlib.backends.backend_svg"])
def test_an_available_backend_is_kept(value):
    assert _backend(value).split()[0] == value.lower()
