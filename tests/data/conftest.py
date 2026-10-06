from pathlib import Path

import pytest

from turn_wm.data import reader


@pytest.fixture
def decode_spies(monkeypatch):
    """Record every call to the per-modality decode paths."""

    calls: dict[str, list[Path]] = {"audio": [], "video": []}

    for modality, recorded in calls.items():
        name = f"_read_{modality}"
        original = getattr(reader, name)

        def spy(path, *args, _original=original, _calls=recorded):
            _calls.append(path)
            return _original(path, *args)

        monkeypatch.setattr(reader, name, spy)

    return calls
