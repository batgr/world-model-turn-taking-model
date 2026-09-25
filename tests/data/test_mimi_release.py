import hashlib
import json
from dataclasses import replace

import pytest

from turn_wm.data.mimi_cache import (
    MimiAudioGap,
    MimiCacheExclusion,
    MimiFeatureStore,
    write_manifest,
)
from turn_wm.data.mimi_release import (
    build_release_manifest,
    release_files,
    write_release,
)

REPOS = {"egocom": "org/egocom", "ego4d": "org/full"}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(make_mimi_cache, tmp_path):
    """EgoCom-like cache and an Ego4D-like one with a gap and an exclusion."""

    root = tmp_path / "release"
    make_mimi_cache({("egocom", "r1"): (0, 20)}, root=root / "egocom")
    ego4d_root = make_mimi_cache(
        {("ego4d", "kept"): (10, 30)},
        root=root / "ego4d",
        source_dataset_revision="rev-full",
    )
    store = MimiFeatureStore(ego4d_root)
    [record] = store.records
    write_manifest(
        ego4d_root,
        recordings=[
            replace(
                record,
                audio_gaps=(MimiAudioGap(start_time_s=1.0, end_time_s=1.25),),
            )
        ],
        model_name=store.model_name,
        model_revision=store.model_revision,
        model_resolved_revision=store.model_resolved_revision,
        source_dataset_revision=store.source_dataset_revision,
        feature_rate_hz=store.feature_rate_hz,
        feature_dim=store.feature_dim,
        excluded_recordings=(
            MimiCacheExclusion(
                dataset="ego4d",
                recording_id="excluded",
                reason="audio_annotation_clock_drift",
                max_drift_s=0.4,
            ),
        ),
    )

    return root


def test_manifest_identifies_mimi_corpora_and_every_file(release):
    payload = build_release_manifest(release, source_repos=REPOS)

    assert payload["release_schema_version"] == 1
    assert payload["mimi"] == {
        "model": "kyutai/mimi",
        "requested_revision": "requested-sha",
        "resolved_revision": "resolved-sha",
        "feature_rate_hz": 10.0,
        "feature_dim": 512,
        "dtype": "float16",
        "alignment": "causal",
    }

    egocom = payload["corpora"]["egocom"]
    assert egocom["source_dataset_repo"] == "org/egocom"
    assert egocom["source_dataset_revision"] == "rev-123"
    assert (egocom["recordings"], egocom["total_steps"]) == (1, 20)
    assert egocom["total_duration_seconds"] == 2.0
    assert egocom["manifest_sha256"] == sha256(release / "egocom" / "manifest.json")

    files = payload["files"]
    assert set(files) == set(release_files(release))
    assert all(files[path] == sha256(release / path) for path in files)
    assert egocom["total_size_bytes"] == sum(
        (release / path).stat().st_size for path in files if path.startswith("egocom/")
    )


def test_manifest_keeps_gap_and_exclusion_statistics(release):
    ego4d = build_release_manifest(release, source_repos=REPOS)["corpora"]["ego4d"]

    assert ego4d["source_dataset_revision"] == "rev-full"
    assert ego4d["canonical_recordings"] == 2
    assert ego4d["recordings"] == 1
    assert ego4d["excluded_recordings"] == 1
    assert ego4d["recordings_with_audio_gaps"] == 1
    assert ego4d["total_audio_gaps"] == 1
    assert ego4d["total_gap_duration_s"] == pytest.approx(0.25)


def test_release_is_reproducible_and_hashes_its_readme(release):
    first = write_release(release, source_repos=REPOS).read_bytes()
    payload = json.loads(first)

    assert write_release(release, source_repos=REPOS).read_bytes() == first
    assert payload["files"]["README.md"] == sha256(release / "README.md")
    assert str(release) not in first.decode()
    assert "generated_at" not in payload


def test_macos_metadata_is_ignored(release):
    (release / "._egocom").write_bytes(b"x")
    (release / "egocom" / "._manifest.json").write_bytes(b"x")
    (release / ".DS_Store").write_bytes(b"x")

    files = build_release_manifest(release, source_repos=REPOS)["files"]

    assert not [path for path in files if "._" in path or "DS_Store" in path]


@pytest.mark.parametrize("stray", ["raw/audio.wav", "egocom/notes.log", "extra.bin"])
def test_unexpected_files_are_refused(release, stray):
    path = release / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="Unexpected files in the release"):
        build_release_manifest(release, source_repos=REPOS)


def test_missing_feature_file_is_refused(release):
    [record] = MimiFeatureStore(release / "egocom").records
    (release / "egocom" / record.path).unlink()

    with pytest.raises(ValueError, match="missing"):
        build_release_manifest(release, source_repos=REPOS)


def test_corpora_must_share_mimi_settings(release, make_mimi_cache):
    make_mimi_cache({("ego4d", "kept"): (10, 30)}, root=release / "ego4d", dim=256)

    with pytest.raises(ValueError, match="disagree on Mimi settings"):
        build_release_manifest(release, source_repos=REPOS)


def test_readme_documents_the_release(release):
    write_release(release, source_repos=REPOS)
    readme = (release / "README.md").read_text()

    for expected in (
        "kyutai/mimi` at revision `resolved-sha`",
        "continuous pre-quantization latents",
        "native 12.5 Hz",
        "| 512 |",
        "float16",
        "is **not**\nincluded",
        "model-derived features, not canonical annotations",
        "`org/egocom` | `rev-123`",
        "`org/full` | `rev-full`",
        "under 100 ms",
        "at least 100 ms",
        "`ego4d` / `excluded`: audio_annotation_clock_drift, max drift 0.400 s",
        "no raw audio or video",
        "private",
    ):
        assert expected in readme, expected
