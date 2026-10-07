"""Feature cache precomputation with a fake encoder and reader (no weights, no media)."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from corpora import make_corpus, make_grid, make_manifest

from turn_wm.data import feature_precompute
from turn_wm.data.feature_cache import CacheExclusion, FeatureStore
from turn_wm.data.feature_precompute import (
    EGO4D_V1_EXCLUSIONS,
    RecordingSpan,
    _load_recording_audio,
    _recording_spans,
    precompute_features,
)
from turn_wm.data.media import MediaPaths
from turn_wm.data.reader import AudioGap, DecodedAudio, MediaWindow
from turn_wm.data.source import LoadedData
from turn_wm.models.encoders.base import Encoder

MEDIA_RATE = 16_000
DIM = 4


class FrameIndexEncoder(Encoder):
    """One frame per started 1 / frame_rate s of audio; frame i has every feature = i."""

    name = "frame-index"
    modality = "audio"
    output_dim = DIM
    revision = "encoder-sha"
    resolved_revision = "resolved-sha"

    def __init__(self, frame_rate: float = 10.0) -> None:
        super().__init__()
        self.frame_rate = frame_rate
        self.streamed: list[tuple[int, float]] = []

    def encode_recording(self, input, rate, *, chunk_seconds):
        self.streamed.append((input.shape[-1], rate))
        frames = math.ceil(input.shape[-1] * self.frame_rate / rate)
        steps = torch.arange(frames, dtype=torch.float32)

        return steps.view(-1, 1).expand(frames, DIM).clone()


class FakeReader:
    """Returns `duration_scale` times the requested audio, recording requests."""

    def __init__(
        self, duration_scale: float = 1.0, audio_gaps: tuple[AudioGap, ...] = ()
    ) -> None:
        self.duration_scale = duration_scale
        self.audio_gaps = audio_gaps
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
            audio=DecodedAudio(
                waveform=torch.ones(1, samples),
                sample_rate=MEDIA_RATE,
                audio_gaps=self.audio_gaps,
            ),
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
        grid_rate_hz=10.0,
    )


def load(reader, **kwargs):
    return prepare(reader, **kwargs).waveform


def test_recording_span_covers_the_grid():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})

    assert _recording_spans(corpus, grid_rate=10.0) == [
        RecordingSpan(
            dataset="a",
            recording_id="r1",
            start_index=0,
            start_time_s=0.0,
            steps=40,
            grid_rate_hz=10.0,
        )
    ]
    assert span().end_time_s == pytest.approx(4.0)


def test_recording_with_a_gap_is_rejected():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] != 7)

    with pytest.raises(ValueError, match="Non-contiguous action grid for 'r1'"):
        _recording_spans(replace(corpus, action_grid=grid), grid_rate=10.0)


def test_recording_with_irregular_timing_is_rejected():
    corpus = make_corpus("a", state="SILENT", splits={"train": 1})
    grid = make_grid(dataset="a", state="SILENT").map(
        lambda row: {"decision_time_s": row["decision_index"] / 8}
    )

    with pytest.raises(ValueError, match="timing mismatch for 'r1'"):
        _recording_spans(replace(corpus, action_grid=grid), grid_rate=10.0)


def prepare(reader, **kwargs):
    return _load_recording_audio(
        reader=reader,
        media=kwargs.pop("media", media()),
        span=kwargs.pop("span", span()),
        grid_rate=10.0,
    )


def test_audio_is_read_on_the_media_timeline_as_audio_only():
    reader = FakeReader()

    load(reader, media=media(offset=300.0), span=span(start_time_s=1.0))

    assert reader.calls == [
        {"key": ("a", "r1"), "interval": (301.0, 305.0), "modalities": ("audio",)}
    ]


def test_audio_gaps_are_mapped_to_the_canonical_cache_timeline():
    prepared = prepare(
        FakeReader(audio_gaps=(AudioGap(301.5, 301.75),)),
        media=media(offset=300.0),
        span=span(start_time_s=1.0),
    )

    [gap] = prepared.audio_gaps
    assert (gap.start_time_s, gap.end_time_s, gap.duration_s) == pytest.approx(
        (1.5, 1.75, 0.25)
    )


@pytest.mark.parametrize("duration_scale", [0.99, 1.0, 1.1])
def test_audio_has_the_exact_canonical_duration(duration_scale):
    audio = load(FakeReader(duration_scale))

    # 40 steps = 4 s at the media's own 16 kHz.
    assert audio.shape == (1, 64_000)


def test_slightly_short_audio_is_padded_with_silence_at_the_end():
    # 0.08 s short of 4 s: under one grid step.
    audio = load(FakeReader(0.98))

    assert audio[0, 53_000:60_000].mean() == 1.0
    assert torch.all(audio[0, 62_720:] == 0)


def test_audio_up_to_a_second_short_is_padded_with_silence():
    # Grids run to the next whole second past the media: up to 1 s of tail.
    audio = load(FakeReader(0.76))

    assert audio.shape == (1, 64_000)
    assert torch.all(audio[0, 48_640:] == 0)


def test_much_shorter_audio_is_rejected():
    with pytest.raises(ValueError, match="1.200 s less audio than its 40 grid"):
        load(FakeReader(0.7))


def test_audio_slightly_before_the_media_start_is_silence():
    reader = FakeReader()

    audio = load(reader, media=media(offset=-0.05))

    # The reader never gets a negative time; 0.05 s of silence comes first.
    assert reader.calls[0]["interval"] == (0.0, pytest.approx(3.95))
    assert torch.all(audio[0, :800] == 0)
    assert audio[0, 800:5_000].mean() == 1.0
    assert audio.shape == (1, 64_000)


def test_span_far_before_the_media_start_is_rejected():
    with pytest.raises(ValueError, match="starts 0.500 s before its media"):
        load(FakeReader(), media=media(offset=-0.5))


@pytest.fixture
def fake_models(monkeypatch):
    reader = FakeReader()
    monkeypatch.setattr(feature_precompute, "MediaReader", lambda: reader)

    return reader


@pytest.fixture
def encoder():
    return FrameIndexEncoder()


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
    fake_models, encoder, media_roots, tmp_path
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

    manifest = precompute_features(
        loaded, encoder=encoder, media_roots=media_roots, output_root=output
    )

    assert manifest == output / "manifest.json"
    assert [call["interval"] for call in fake_models.calls] == [
        (0.0, 4.0),
        (300.0, 304.0),
    ]

    store = FeatureStore(output)

    assert store.metadata["model"] == {
        "name": "frame-index",
        "revision": "encoder-sha",
        "resolved_revision": "resolved-sha",
    }
    assert store.metadata["source_dataset_revision"] == "dataset-rev"
    assert store.metadata["features"]["rate_hz"] == 10.0
    assert store.metadata["features"]["dim"] == DIM

    for dataset in ("a", "b"):
        features = store.get(dataset=dataset, recording_id="r1", start=0, end=40)

        # Row k is frame k: the encoder runs at the grid's rate.
        expected = list(range(40))
        assert features.shape == (40, DIM)
        assert features[:, 0].tolist() == expected


def test_evidence_backed_ego4d_v1_exclusions_are_explicit():
    assert [exclusion.recording_id for exclusion in EGO4D_V1_EXCLUSIONS] == [
        "85506322-449a-45c0-b77a-48a8077b4bbd",
        "b3ef3563-ecc0-4a15-9a7b-4feb4558d953",
        "ba5b1882-c9d7-48e7-85fe-2c7b10494fac",
    ]
    assert [exclusion.max_drift_s for exclusion in EGO4D_V1_EXCLUSIONS] == [
        pytest.approx(0.44265625),
        pytest.approx(0.476),
        pytest.approx(0.52534375),
    ]
    assert {exclusion.reason for exclusion in EGO4D_V1_EXCLUSIONS} == {
        "audio_annotation_clock_drift"
    }


def test_precompute_omits_explicit_exclusion_and_records_it(
    fake_models, encoder, media_roots, tmp_path
):
    exclusion = CacheExclusion(
        dataset="a",
        recording_id="r1",
        reason="audio_annotation_clock_drift",
        max_drift_s=0.4,
    )
    output = tmp_path / "cache"

    precompute_features(
        one_corpus(),
        encoder=encoder,
        media_roots=media_roots,
        output_root=output,
        excluded_recordings=(exclusion,),
    )

    store = FeatureStore(output)
    assert store.records == ()
    assert store.exclusions == (exclusion,)
    assert fake_models.calls == []


def test_precompute_rejects_exclusion_outside_canonical_grid(
    fake_models, encoder, media_roots, tmp_path
):
    exclusion = CacheExclusion(
        dataset="a",
        recording_id="missing",
        reason="audio_annotation_clock_drift",
        max_drift_s=0.4,
    )

    with pytest.raises(ValueError, match="not in the canonical action grid"):
        precompute_features(
            one_corpus(),
            encoder=encoder,
            media_roots=media_roots,
            output_root=tmp_path / "cache",
            excluded_recordings=(exclusion,),
        )

    assert encoder.streamed == []


def test_the_encoder_must_run_at_the_grid_rate(fake_models, media_roots, tmp_path):
    with pytest.raises(ValueError, match="decision grid is 10 Hz"):
        precompute_features(
            LoadedData(
                corpora=(make_corpus("a", state="SILENT", splits={"train": 1}),)
            ),
            encoder=FrameIndexEncoder(frame_rate=12.5),
            media_roots=media_roots,
            output_root=tmp_path,
        )


def test_corpus_without_media_manifest_is_rejected(
    fake_models, encoder, media_roots, tmp_path
):
    loaded = LoadedData(
        corpora=(make_corpus("a", state="SILENT", splits={"train": 1}),)
    )

    with pytest.raises(ValueError, match="'a' has no media manifest"):
        precompute_features(
            loaded,
            encoder=encoder,
            media_roots=media_roots,
            output_root=tmp_path / "cache",
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


def test_progress_is_reported_per_recording(
    fake_models, encoder, media_roots, tmp_path
):
    seen = []

    precompute_features(
        one_corpus(),
        encoder=encoder,
        media_roots=media_roots,
        output_root=tmp_path / "cache",
        progress=lambda index, total, span: seen.append(
            (index, total, span.recording_id)
        ),
    )

    assert seen == [(1, 1, "r1")]


def test_non_empty_output_is_rejected(fake_models, encoder, media_roots, tmp_path):
    output = tmp_path / "cache"
    output.mkdir()
    (output / "manifest.json").write_text("{}")

    with pytest.raises(ValueError, match="not an empty directory"):
        precompute_features(
            one_corpus(), encoder=encoder, media_roots=media_roots, output_root=output
        )

    assert encoder.streamed == []


def test_missing_media_fails_before_any_encoding(
    fake_models, encoder, media_roots, tmp_path
):
    (media_roots["a"] / "videos/r1.mp4").unlink()

    with pytest.raises(FileNotFoundError, match="r1.mp4"):
        precompute_features(
            one_corpus(),
            encoder=encoder,
            media_roots=media_roots,
            output_root=tmp_path / "cache",
        )

    # Checked before anything is encoded.
    assert encoder.streamed == []
    assert fake_models.calls == []


def test_non_contiguous_grid_fails_before_any_encoding(
    fake_models, encoder, media_roots, tmp_path
):
    loaded = one_corpus()
    [corpus] = loaded.corpora
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] != 3)

    with pytest.raises(ValueError, match="Non-contiguous"):
        precompute_features(
            LoadedData(corpora=(replace(corpus, action_grid=grid),)),
            encoder=encoder,
            media_roots=media_roots,
            output_root=tmp_path / "cache",
        )

    assert encoder.streamed == []


def test_recording_without_audio_is_rejected(
    fake_models, encoder, media_roots, tmp_path
):
    with pytest.raises(ValueError, match="No audio source"):
        precompute_features(
            one_corpus(video_has_audio=False),
            encoder=encoder,
            media_roots=media_roots,
            output_root=tmp_path / "cache",
        )


def test_features_follow_the_first_decision_index(
    fake_models, encoder, media_roots, tmp_path
):
    # The grid need not start at decision_index 0 or time 0.
    loaded = one_corpus()
    [corpus] = loaded.corpora
    grid = corpus.action_grid.filter(lambda row: row["decision_index"] >= 5)

    precompute_features(
        LoadedData(corpora=(replace(corpus, action_grid=grid),)),
        encoder=encoder,
        media_roots=media_roots,
        output_root=tmp_path / "cache",
    )

    [record] = FeatureStore(tmp_path / "cache")._records.values()

    assert (record.start_index, record.start_time_s, record.steps) == (5, 0.5, 35)
    assert fake_models.calls[0]["interval"] == (0.5, 4.0)


def test_any_encoder_precomputes_features_on_a_grid_at_its_own_rate(
    fake_models, media_roots, tmp_path
):
    """Log-mel at 12.5 Hz on a 12.5 Hz grid: one row per 80 ms step."""

    from turn_wm.models.encoders.logmel import LogMelEncoder

    [corpus] = one_corpus().corpora
    grid = corpus.action_grid.map(
        lambda row: {"decision_time_s": row["decision_index"] / 12.5}
    )
    loaded = LoadedData(corpora=(replace(corpus, action_grid=grid),))
    output = tmp_path / "cache"

    precompute_features(
        loaded,
        encoder=LogMelEncoder(frame_rate=12.5),
        media_roots=media_roots,
        output_root=output,
    )

    # 40 steps of 80 ms: the audio read is 3.2 s.
    assert fake_models.calls[0]["interval"] == (0.0, pytest.approx(3.2))
    store = FeatureStore(output)
    assert store.model_name == "logmel-80"
    assert store.feature_rate_hz == 12.5
    features = store.get(dataset="a", recording_id="r1", start=0, end=40)
    assert features.shape == (40, 80)
