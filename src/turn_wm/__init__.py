"""Turn-taking world models."""

import importlib.util
import os


def _usable_matplotlib_backend() -> None:
    """Fall back to Agg when MPLBACKEND names a module this env lacks.

    Notebooks (e.g. Colab) export `module://matplotlib_inline...` to their
    shell commands, whose environment may not have it; matplotlib then fails
    at import, and Lightning imports it through torchmetrics. Nothing here
    needs an interactive backend.
    """

    backend = os.environ.get("MPLBACKEND", "")

    if not backend.startswith("module://"):
        return

    module = backend.removeprefix("module://").split(".")[0]

    if importlib.util.find_spec(module) is None:
        os.environ["MPLBACKEND"] = "Agg"


_usable_matplotlib_backend()
