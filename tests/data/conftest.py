from pathlib import Path

import pytest

from turn_wm.data.reader import MediaReader


@pytest.fixture
def decode_spies(monkeypatch):
    """Record every call to the per-modality decode paths."""

    calls: dict[str, list[Path]] = {"audio": [], "video": []}

    for modality, recorded in calls.items():
        name = f"_read_{modality}"
        original = getattr(MediaReader, name)

        def spy(self, path, *, _original=original, _calls=recorded, **kwargs):
            _calls.append(path)
            return _original(self, path, **kwargs)

        monkeypatch.setattr(MediaReader, name, spy)

    return calls
