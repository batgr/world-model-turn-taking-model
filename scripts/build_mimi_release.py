"""Write README.md and release_manifest.json for a local Mimi feature release.

uv run python scripts/build_mimi_release.py /path/to/mimi/v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from turn_wm.data.mimi_release import write_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()

    manifest = write_release(args.release_root)

    print(args.release_root / "README.md")
    print(manifest)


if __name__ == "__main__":
    main()
