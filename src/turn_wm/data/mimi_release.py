"""
Build the deterministic release of the per-corpus Mimi feature caches.

A release directory holds one cache per corpus, as written by
`turn-wm precompute-mimi`, plus a README and `release_manifest.json`:

    <release>/
        README.md
        release_manifest.json
        <corpus>/manifest.json
        <corpus>/<corpus>/*.safetensors

The manifest identifies the release exactly (Mimi settings, per-corpus source
dataset revisions and totals, the SHA-256 of every released file). It holds
no timestamp, absolute path, user or host name, so rebuilding it from the same
caches reproduces it byte for byte. macOS metadata files (`._*`,
`.DS_Store`, created on exFAT drives) are ignored and never released.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from turn_wm.data.mimi_cache import MimiFeatureStore
from turn_wm.data.source import DATASETS

RELEASE_SCHEMA_VERSION = 1
CORPORA = ("egocom", "ego4d")
README = "README.md"
RELEASE_MANIFEST = "release_manifest.json"


def is_macos_metadata(path: Path) -> bool:
    """AppleDouble (`._*`) and `.DS_Store` files, never part of a release."""

    return path.name.startswith("._") or path.name == ".DS_Store"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def release_files(
    release_root: Path,
    *,
    corpus_names: tuple[str, ...] = CORPORA,
) -> list[str]:
    """Released corpus files (manifests and features), as sorted POSIX paths.

    Refuses any other file (raw media, logs, stray caches): only what the
    corpus manifests reference, the manifests, the README and the release
    manifest may be in the release directory.
    """

    root = Path(release_root)
    expected: set[str] = set()

    for name in corpus_names:
        store = MimiFeatureStore(root / name)
        expected.add(f"{name}/manifest.json")
        expected.update(
            str(PurePosixPath(name) / record.path) for record in store.records
        )

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not is_macos_metadata(path)
    }
    unexpected = sorted(present - expected - {README, RELEASE_MANIFEST})
    missing = sorted(expected - present)

    if unexpected:
        raise ValueError(f"Unexpected files in the release: {unexpected}")

    if missing:
        raise ValueError(f"Files referenced by the manifests are missing: {missing}")

    return sorted(expected)


def _mimi_settings(stores: Mapping[str, MimiFeatureStore]) -> dict[str, Any]:
    settings = {
        name: {
            "model": store.model_name,
            "requested_revision": store.model_revision,
            "resolved_revision": store.model_resolved_revision,
            "feature_rate_hz": store.feature_rate_hz,
            "feature_dim": store.feature_dim,
            "dtype": store.dtype,
            "alignment": store.metadata["features"]["alignment"],
        }
        for name, store in stores.items()
    }
    distinct = {json.dumps(value, sort_keys=True) for value in settings.values()}

    if len(distinct) != 1:
        raise ValueError(f"Corpus caches disagree on Mimi settings: {settings}")

    [value] = {json.dumps(value, sort_keys=True) for value in settings.values()}

    return json.loads(value)


def _corpus_summary(
    name: str,
    store: MimiFeatureStore,
    *,
    root: Path,
    source_repo: str,
) -> dict[str, Any]:
    records = store.records
    gaps = [gap for record in records for gap in record.audio_gaps]
    gap_durations = [gap.duration_s for gap in gaps]
    steps = sum(record.steps for record in records)
    manifest = root / name / "manifest.json"
    size = manifest.stat().st_size + sum(
        (store.root / record.path).stat().st_size for record in records
    )

    return {
        "source_dataset_repo": source_repo,
        "source_dataset_revision": store.source_dataset_revision,
        "cache_schema_version": store.schema_version,
        "recordings": len(records),
        "canonical_recordings": len(records) + len(store.exclusions),
        "excluded_recordings": len(store.exclusions),
        "total_steps": steps,
        "total_duration_seconds": steps / store.feature_rate_hz,
        "total_size_bytes": size,
        "manifest_sha256": sha256_file(manifest),
        "recordings_with_audio_gaps": sum(
            bool(record.audio_gaps) for record in records
        ),
        "total_audio_gaps": len(gaps),
        "total_gap_duration_s": sum(gap_durations),
        "gap_duration_s": {
            "min": min(gap_durations) if gap_durations else None,
            "median": statistics.median(gap_durations) if gap_durations else None,
            "max": max(gap_durations) if gap_durations else None,
        },
    }


def build_release_manifest(
    release_root: Path,
    *,
    corpus_names: tuple[str, ...] = CORPORA,
    source_repos: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Release manifest from the corpus caches, without hardcoded counts.

    `source_repos` maps each corpus to its source dataset repository; by
    default the repository of `DATASETS[corpus]`.
    """

    root = Path(release_root)
    files = release_files(root, corpus_names=corpus_names)
    stores = {name: MimiFeatureStore(root / name) for name in corpus_names}
    repos = source_repos or {name: DATASETS[name].repo_id for name in corpus_names}
    exclusions = sorted(
        (exclusion for store in stores.values() for exclusion in store.exclusions),
        key=lambda exclusion: (exclusion.dataset, exclusion.recording_id),
    )
    hashed = {path: sha256_file(root / path) for path in files}

    if (root / README).is_file():
        hashed[README] = sha256_file(root / README)

    return {
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "mimi": _mimi_settings(stores),
        "corpora": {
            name: _corpus_summary(name, store, root=root, source_repo=repos[name])
            for name, store in sorted(stores.items())
        },
        "exclusions": [
            {
                "dataset": exclusion.dataset,
                "recording_id": exclusion.recording_id,
                "reason": exclusion.reason,
                "max_drift_s": exclusion.max_drift_s,
            }
            for exclusion in exclusions
        ],
        "files": dict(sorted(hashed.items())),
    }


def write_release_manifest(
    release_root: Path,
    *,
    corpus_names: tuple[str, ...] = CORPORA,
    source_repos: Mapping[str, str] | None = None,
) -> Path:
    """Write `release_manifest.json` deterministically and return its path."""

    root = Path(release_root)
    payload = build_release_manifest(
        root, corpus_names=corpus_names, source_repos=source_repos
    )
    path = root / RELEASE_MANIFEST
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return path


def release_readme(manifest: Mapping[str, Any]) -> str:
    """The release card, derived from the manifest's values."""

    mimi = manifest["mimi"]
    rows = "".join(
        f"| {name} | `{corpus['source_dataset_repo']}` | "
        f"`{corpus['source_dataset_revision']}` | {corpus['recordings']} | "
        f"{corpus['excluded_recordings']} | {corpus['total_steps']:,} | "
        f"{corpus['total_duration_seconds'] / 3600:.1f} |\n"
        for name, corpus in manifest["corpora"].items()
    )
    exclusions = "".join(
        f"- `{e['dataset']}` / `{e['recording_id']}`: {e['reason']}, "
        f"max drift {e['max_drift_s']:.3f} s\n"
        for e in manifest["exclusions"]
    )
    gaps = "".join(
        f"- {name}: {corpus['total_audio_gaps']} gaps in "
        f"{corpus['recordings_with_audio_gaps']} recordings, "
        f"{corpus['total_gap_duration_s']:.3f} s in total\n"
        for name, corpus in manifest["corpora"].items()
    )
    layout = "".join(
        f"{name}/manifest.json\n{name}/{name}/*.safetensors\n"
        for name in manifest["corpora"]
    )

    return f"""---
license: other
pretty_name: Turn-taking Mimi features
tags:
- audio
- turn-taking
- world-model
- mimi
---

# Turn-taking Mimi features

Precomputed continuous [Mimi](https://huggingface.co/{mimi["model"]}) features
for turn-taking world-model experiments, aligned to the canonical
{mimi["feature_rate_hz"]:g} Hz turn-taking action grid of the source datasets.
They stand in for the frozen Mimi encoder during training: no audio is decoded
and Mimi is never loaded.

| | |
|---|---|
| Model | `{mimi["model"]}` at revision `{mimi["resolved_revision"]}` |
| Representation | continuous pre-quantization latents (encoder, encoder transformer, downsample) |
| Encoding | causal, streamed over each whole recording with Mimi's convolution and attention caches |
| Rate | native 12.5 Hz, {mimi["alignment"]}ly aligned to the {mimi["feature_rate_hz"]:g} Hz grid (step `k` uses the latest Mimi frame available by the end of its slot) |
| Dimension | {mimi["feature_dim"]} |
| Storage | {mimi["dtype"]}, one safetensors tensor `features` of shape `[steps, {mimi["feature_dim"]}]` per recording |

The world model's trainable projector ({mimi["feature_dim"]} -> 192) is **not**
included; it is trained on top of these features:

```text
cached Mimi [T, {mimi["feature_dim"]}] -> trainable projector -> world-model latent
```

These are **model-derived features, not canonical annotations**.

## Contents

| Corpus | Source dataset | Revision | Recordings | Excluded | Grid steps | Hours |
|---|---|---|---|---|---|---|
{rows}
```text
README.md
release_manifest.json     Mimi settings, per-corpus totals, SHA-256 of every file
{layout}```

Each corpus `manifest.json` lists, per recording, its `dataset`,
`recording_id`, `path` (relative to the manifest's directory), `steps`,
`start_index` and `start_time_s`: row `k` of a recording's tensor is its
action-grid step `decision_index = start_index + k`, covering
`start_time_s + k / 10` to `start_time_s + (k + 1) / 10` on the canonical
timeline. It also records the source dataset revision and the Mimi revision
requested and actually loaded.

## Audio timeline caveats

- Local audio timestamp irregularities under 100 ms are timestamp jitter:
  decoded frames are concatenated, without inserted silence.
- Local gaps of at least 100 ms (missing audio in the source media) are
  encoded as silence of their duration at their place on the timeline, and
  listed per recording in `audio_gaps`:
{gaps}- Where a grid runs slightly past the end of the audio (EgoCom grids end on
  the next whole second, up to 1 s after the audio), the missing tail is
  encoded as silence; the source dataset labels those steps `UNKNOWN`.
- Canonical `SPEAKING`, `SILENT` and `UNKNOWN` labels are not modified.
- Recordings whose audio clock drifts slowly away from its timestamps by
  several grid steps are excluded, as their audio/annotation synchronization
  is uncertain (not because the media is corrupted):
{exclusions}
## Access and licensing

This repository is private and contains **no raw audio or video**. Access to
these features does not replace the terms of the source datasets (EgoCom,
Ego4D): use them only under the licenses and agreements that govern the
source data.
"""


def write_release(
    release_root: Path,
    *,
    corpus_names: tuple[str, ...] = CORPORA,
    source_repos: Mapping[str, str] | None = None,
) -> Path:
    """Write the README, then the release manifest (which hashes the README)."""

    root = Path(release_root)
    readme = release_readme(
        build_release_manifest(
            root, corpus_names=corpus_names, source_repos=source_repos
        )
    )
    (root / README).write_text(readme, encoding="utf-8")

    return write_release_manifest(
        root, corpus_names=corpus_names, source_repos=source_repos
    )
