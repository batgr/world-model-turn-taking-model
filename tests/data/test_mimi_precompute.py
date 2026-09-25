"""Mimi cache precomputation with a fake encoder and reader (no weights, no media)."""

import math
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from corpora import make_corpus, make_grid, make_manifest

from turn_wm.data import mimi_precompute
from turn_wm.data.media import MediaPaths
from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.mimi_precompute import (
    RecordingSpan,
    _load_recording_audio,
    _recording_spans,
    precompute_mimi_cache,
)
from turn_wm.data.reader import DecodedAudio, MediaWindow
from turn_wm.data.source import LoadedData

MEDIA_RATE = 16_000
DIM = 4


class FrameIndexEncoder:
    """Stand-in for FrozenMimiEncoder: native frame i has every feature = i."""

    sample_rate = 2_400
    source_rate = 12.5
    output_dim = DIM
    resolved_revision = "resolved-sha"
    instances: ClassVar[list["FrameIndexEncoder"]] = []

    def __init__(self, *, model_name, revision, target_rate):
        self.kwargs = {
            "model_name": model_name,
            "revision": revision,
            "target_rate": target_rate,
        }
        self.streamed: list[int] = []
        FrameIndexEncoder.instances.append(self)

    def to(self, device):
        return self

    def eval(self):
        return self

    def stream_native_features(self, audio, *, chunk_seconds):
        self.streamed.append(audio.shape[-1])
        frames = math.ceil(audio.shape[-1] / (self.sample_rate / self.source_rate))
        steps = torch.arange(frames, dtype=torch.float32)

        return steps.view(1, -1, 1).expand(1, frames, DIM).clone()


class FakeReader:
    """Returns `duration_scale` times the requested audio, recording requests."""

    def __init__(self, duration_scale: float = 1.0) -> None:
        self.duration_scale = duration_scale
        self.calls: list[dict] = []

    def read_window(self, media, *, start_time_s, end_time_s, modalities):
        self.calls.append(
            {
                "key": media.key,
                "interval": (start_time_s, end_time_s),
                "modalities": modalities,
            }
        )
        samples = round((end_time_s - start_time_s) * MEDIA_RATE * self.duration_scale)

        return MediaWindow(
            start_time_s=start_time_s,
            end_time_s=end_time_s,
            audio=DecodedAudio(waveform=torch.ones(1, samples), sample_rate=MEDIA_RATE),
            video=None,
        )


def media(offset: float = 0.0) -> MediaPaths:
    return MediaPaths(
        dataset="a",
        recording_id="r1",
        video_path=Path("/fake/r1.mp4"),
        media_offset_s=offset,
        video_has_audio=True,
    )


def span(steps: int = 40, start_time_s: float = 0.0) -> RecordingSpan:
    return RecordingSpan(
        dataset="a",
        recording_id="r1",
        start_index=0,
        start_time_s=start_time_s,
        steps=steps,
    )


def test_recording_span_covers_the_grid():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})

    assert _recording_spans(corpus, target_rate=10.0) == [
        RecordingSpan(
            dataset="a",
            recording_id="r1",
            start_index=0,
            start_time_s=0.0,
            steps=40,
        )
    ]
    assert span().end_time_s == pytest.approx(4.0)


def test_recording_with_a_gap_is_rejected():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] != 7)

    with pytest.raises(ValueError, match="Non-contiguous action grid for 'r1'"):
        _recording_spans(replace(corpus, action_grid=grid), target_rate=10.0)


def test_recording_with_irregular_timing_is_rejected():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})
    grid = make_grid(dataset="a", state="SILENT").map(
        lambda row: {"decision_time_s": row["decision_index"] / 8}
    )

    with pytest.raises(ValueError, match="timing mismatch for 'r1'"):
        _recording_spans(replace(corpus, action_grid=grid), target_rate=10.0)


def load(reader, **kwargs):
    return _load_recording_audio(
        reader=reader,
        media=kwargs.pop("media", media()),
        span=kwargs.pop("span", span()),
        encoder=FrameIndexEncoder(model_name="m", revision=None, target_rate=10.0),
        target_rate=10.0,
    )


def test_audio_is_read_on_the_media_timeline_as_audio_only():
    reader = FakeReader()

    load(reader, media=media(offset=300.0), span=span(start_time_s=1.0))

    assert reader.calls == [
        {"key": ("a", "r1"), "interval": (301.0, 305.0), "modalities": ("audio",)}
    ]


@pytest.mark.parametrize("duration_scale", [0.99, 1.0, 1.1])
def test_audio_has_the_exact_canonical_duration(duration_scale):
    audio = load(FakeReader(duration_scale))

    # 40 steps = 4 s at the encoder's 2.4 kHz.
    assert audio.shape == (1, 1, 9_600)


def test_slightly_short_audio_is_padded_with_silence_at_the_end():
    # 0.08 s short of 4 s: under one grid step.
    audio = load(FakeReader(0.98))

    assert audio[0, 0, 8_000:9_000].mean() == pytest.approx(1.0, abs=0.05)
    assert torch.all(audio[0, 0, 9_450:] == 0)


def test_much_shorter_audio_is_rejected():
    with pytest.raises(ValueError, match="0.400 s less audio than its 40 grid"):
        load(FakeReader(0.9))


def test_audio_slightly_before_the_media_start_is_silence():
    reader = FakeReader()

    audio = load(reader, media=media(offset=-0.05))

    # The reader never gets a negative time; 0.05 s of silence comes first.
    assert reader.calls[0]["interval"] == (0.0, pytest.approx(3.95))
    assert torch.all(audio[0, 0, :100] == 0)
    assert audio[0, 0, 200:1_000].mean() == pytest.approx(1.0, abs=0.05)
    assert audio.shape == (1, 1, 9_600)


def test_span_far_before_the_media_start_is_rejected():
    with pytest.raises(ValueError, match="starts 0.500 s before its media"):
        load(FakeReader(), media=media(offset=-0.5))


@pytest.fixture
def fake_models(monkeypatch):
    FrameIndexEncoder.instances.clear()
    reader = FakeReader()
    monkeypatch.setattr(mimi_precompute, "FrozenMimiEncoder", FrameIndexEncoder)
    monkeypatch.setattr(mimi_precompute, "MediaReader", lambda: reader)

    return reader


@pytest.fixture
def media_roots(tmp_path):
    roots = {}

    for name in ("a", "b"):
        root = tmp_path / "media" / name
        (root / "videos").mkdir(parents=True)
        (root / "videos/r1.mp4").touch()
        roots[name] = root

    return roots


def test_precompute_writes_aligned_features_and_manifest(
    fake_models, media_roots, tmp_path
):
    loaded = LoadedData(
        corpora=(
            make_corpus(
                "a",
                state="SILENT",
                splits={"train": 1},
                media_manifest=make_manifest(dataset="a"),
            ),
            make_corpus(
                "b",
                state="SPEAKING",
                splits={"train": 1},
                media_manifest=make_manifest(dataset="b", media_offset_s=300.0),
            ),
        ),
        revision="dataset-rev",
    )
    output = tmp_path / "cache"

    manifest = precompute_mimi_cache(
        loaded,
        media_roots=media_roots,
        output_root=output,
        model_revision="mimi-sha",
    )

    assert manifest == output / "manifest.json"
    assert [call["interval"] for call in fake_models.calls] == [
        (0.0, 4.0),
        (300.0, 304.0),
    ]

    [encoder] = FrameIndexEncoder.instances
    assert encoder.kwargs == {
        "model_name": "kyutai/mimi",
        "revision": "mimi-sha",
        "target_rate": 10.0,
    }

    store = MimiFeatureStore(output)

    assert store.metadata["model"] == {
        "name": "kyutai/mimi",
        "revision": "mimi-sha",
        "resolved_revision": "resolved-sha",
    }
    assert store.metadata["source_dataset_revision"] == "dataset-rev"
    assert store.metadata["features"]["rate_hz"] == 10.0
    assert store.metadata["features"]["dim"] == DIM

    for dataset in ("a", "b"):
        features = store.get(dataset=dataset, recording_id="r1", start=0, end=40)

        # Row k is the latest native frame available by the end of step k.
        expected = [math.floor(1.25 * (k + 1) + 1e-8) - 1 for k in range(40)]
        assert features.shape == (40, DIM)
        assert features[:, 0].tolist() == expected


def test_cache_rate_is_the_action_grid_rate(fake_models, media_roots, tmp_path):
    with pytest.raises(ValueError, match="10 Hz action grid"):
        precompute_mimi_cache(
            LoadedData(
                corpora=(make_corpus("a", state="SILENT", splits={"train": 1}),)
            ),
            media_roots=media_roots,
            output_root=tmp_path,
            target_rate=12.5,
        )


def test_corpus_without_media_manifest_is_rejected(fake_models, media_roots, tmp_path):
    loaded = LoadedData(
        corpora=(make_corpus("a", state="SILENT", splits={"train": 1}),)
    )

    with pytest.raises(ValueError, match="'a' has no media manifest"):
        precompute_mimi_cache(
            loaded, media_roots=media_roots, output_root=tmp_path / "cache"
        )


def one_corpus(**manifest):
    return LoadedData(
        corpora=(
            make_corpus(
                "a",
                state="SILENT",
                splits={"train": 1},
                media_manifest=make_manifest(dataset="a", **manifest),
            ),
        )
    )


def test_progress_is_reported_per_recording(fake_models, media_roots, tmp_path):
    seen = []

    precompute_mimi_cache(
        one_corpus(),
        media_roots=media_roots,
        output_root=tmp_path / "cache",
        progress=lambda index, total, span: seen.append(
            (index, total, span.recording_id)
        ),
    )

    assert seen == [(1, 1, "r1")]


def test_non_empty_output_is_rejected(fake_models, media_roots, tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    (output / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="not an empty directory"):
        precompute_mimi_cache(one_corpus(), media_roots=media_roots, output_root=output)

    assert FrameIndexEncoder.instances == []


def test_missing_media_fails_before_any_encoding(fake_models, media_roots, tmp_path):
    (media_roots["a"] / "videos/r1.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="r1.mp4"):
        precompute_mimi_cache(
            one_corpus(), media_roots=media_roots, output_root=tmp_path / "cache"
        )

    # Checked before the (slow) encoder is even built.
    assert FrameIndexEncoder.instances == []
    assert fake_models.calls == []


def test_non_contiguous_grid_fails_before_any_encoding(
    fake_models, media_roots, tmp_path
):
    loaded = one_corpus()
    [corpus] = loaded.corpora
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] != 3)

    with pytest.raises(ValueError, match="Non-contiguous"):
        precompute_mimi_cache(
            LoadedData(corpora=(replace(corpus, action_grid=grid),)),
            media_roots=media_roots,
            output_root=tmp_path / "cache",
        )

    assert FrameIndexEncoder.instances == []


def test_recording_without_audio_is_rejected(fake_models, media_roots, tmp_path):
    with pytest.raises(ValueError, match="No audio source"):
        precompute_mimi_cache(
            one_corpus(video_has_audio=False),
            media_roots=media_roots,
            output_root=tmp_path / "cache",
        )


def test_features_follow_the_first_decision_index(fake_models, media_roots, tmp_path):
    # The grid need not start at decision_index 0 or time 0.
    loaded = one_corpus()
    [corpus] = loaded.corpora
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] >= 5)

    precompute_mimi_cache(
        LoadedData(corpora=(replace(corpus, action_grid=grid),)),
        media_roots=media_roots,
        output_root=tmp_path / "cache",
    )

    [record] = MimiFeatureStore(tmp_path / "cache")._records.values()

    assert (record.start_index, record.start_time_s, record.steps) == (5, 0.5, 35)
    assert fake_models.calls[0]["interval"] == (0.5, 4.0)
