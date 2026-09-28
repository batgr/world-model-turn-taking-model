"""
Concept probes: derivations from the release's tables, conversation-grouped
CV, unsupported concepts and written artifacts, on synthetic snapshots and
label tables (no network).
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_latent_labels import GRID_SHA, entry

from turn_wm.cli import main
from turn_wm.data.labels import local_store
from turn_wm.evaluation.latent_analysis.concepts import (
    CONCEPTS,
    LOCAL_WINDOW_CELLS,
    any_event,
    grouped_cv,
    local_speakers,
    turn_rate,
    voices,
    write_concepts,
)
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.label_source import CorpusLabelSource
from turn_wm.evaluation.latent_analysis.probes import recording_folds
from turn_wm.evaluation.latent_analysis.show import show_concepts

STEPS = 120
ANCHORS = range(0, STEPS, 4)
TRAIN = tuple(f"t{k}" for k in range(12))
VALIDATION = tuple(f"v{k}" for k in range(12))

# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def test_voices_counts_the_loudest_subframe_and_refuses_unknowns():
    assert voices([0, 0, 0]) == "0"
    assert voices([0, 1, 0]) == "1"
    assert voices([1, 3, 2]) == "2+"
    assert voices([1, None, 1]) is None
    assert voices(None) is None


def test_any_event_is_tri_state():
    assert any_event([False, True, None]) == "true"  # one known true suffices
    assert any_event([False, False, False]) == "false"
    assert any_event([False, None, False]) is None  # false needs every value
    # A value flagged invalid is unknown, even if it reads true.
    assert any_event([True, False, False], [False, True, True]) is None
    assert any_event([False, False], [True, True]) == "false"


def test_local_speakers_needs_a_full_known_window():
    one = [[True, False, False]] * LOCAL_WINDOW_CELLS
    assert local_speakers(one) == "0-1"
    assert local_speakers([[False, False, False]] * LOCAL_WINDOW_CELLS) == "0-1"
    two = [[True, False, False]] * 50 + [[False, True, False]] * 50
    assert local_speakers(two) == "2"
    three = two[:-1] + [[False, False, True]]
    assert local_speakers(three) == "3+"
    assert local_speakers(one[:-1]) is None  # window starts before the recording
    unknown = [[None, False, False]] * 51 + [[True, False, False]] * 49
    assert local_speakers(unknown) is None


def test_turn_rate_reads_the_turn_containing_the_anchor():
    turns = [(1.0, 2.0, 3.0), (4.0, 6.0, None), (7.0, 8.0, 2.5)]
    assert turn_rate(turns, 0.5) is None
    assert turn_rate(turns, 1.0) == 3.0
    assert turn_rate(turns, 2.0) is None  # [start, end)
    assert turn_rate(turns, 5.0) is None  # invalid rate
    assert turn_rate(turns, 7.5) == 2.5


# ---------------------------------------------------------------------------
# Conversation-grouped CV
# ---------------------------------------------------------------------------


def test_outer_folds_never_split_a_conversation():
    groups = [f"egocom/c{k // 6}" for k in range(120)]  # 6 rows (3 views) each
    fold, _ = recording_folds(groups, folds=5, seed=1)

    for group in set(groups):
        assert len({int(fold[k]) for k, g in enumerate(groups) if g == group}) == 1


def _grouped_rows(values, groups, signal):
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(len(values), 4, generator=generator)
    features = noise.clone()
    features[:, 0] = torch.tensor(signal) + 0.1 * features[:, 0]

    return {
        "values": values,
        "groups": groups,
        "corpora": ["egocom"] * len(values),
        "representations": {"features": features, "latent": noise},
    }


def test_grouped_cv_recovers_a_recording_trait_and_leaves_noise_at_chance():
    concept = next(c for c in CONCEPTS if c.name == "background_music")
    groups = [f"egocom/c{k // 10}" for k in range(200)]  # 20 conversations
    values = ["true" if (k // 10) % 2 else "false" for k in range(200)]
    rows = _grouped_rows(values, groups, [2.0 if v == "true" else -2.0 for v in values])

    score = grouped_cv(concept, rows, bootstrap=50, seed=0)

    assert score["skipped"] is None
    assert score["n_eval_recordings"] == 20
    assert score["features_score"] > 0.95
    assert score["features_ci"][0] > 0.5
    assert abs(score["latent_score"] - 0.5) < 0.2


def test_a_class_in_too_few_conversations_is_not_evaluable():
    concept = next(c for c in CONCEPTS if c.name == "background_fan")
    groups = [f"egocom/c{k // 10}" for k in range(200)]
    # Only 3 conversations have a fan, as in EgoCom.
    values = ["true" if k < 30 else "false" for k in range(200)]
    rows = _grouped_rows(values, groups, [0.0] * 200)

    score = grouped_cv(concept, rows, bootstrap=20, seed=0)

    assert score["skipped"].startswith("unsupported")
    assert "true in 3 conversation(s)" in score["skipped"]
    assert score["features_score"] is None and score["delta_score"] is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _state(corpus, recording, k):
    """Deterministic per-cell labels; the recording's traits from its index."""

    n = int(recording[1:]) + (100 if recording[0] == "v" else 0)
    voices_now = (k // 7 + n) % 3
    return {
        "counts": [voices_now] * 3,
        "onset": [k % 11 == 0, False, False],
        "activity": [voices_now >= 1, voices_now >= 2, n % 2 == 1 and k % 3 == 0],
        "ego": k % 2 == 0,
        "addressed": [k % 5 == 0, False, False],
    }


def _traits(recording):
    n = int(recording[1:]) + (100 if recording[0] == "v" else 0)
    return {
        "conversation": f"c{recording[0]}{n // 2}",  # two views per conversation
        "participants": 2 if n % 2 == 0 else 3 + n % 3,
        "music": (n // 2) % 2 == 1,
        "fan": n in (0, 1),
        "native": n % 3 != 0,
        "host": n % 2 == 0,
    }


def _registry(corpus):
    labels = [
        entry(
            "instantaneous.active_speaker_count_subframes", "fixed_size_list<int8, S>"
        ),
        entry("events.other_onset_subframes", "fixed_size_list<bool, S>"),
        entry("instantaneous.speaker_activity", "list<bool>", "[participant]"),
        entry("instantaneous.ego_speaking", "bool"),
        entry("metadata.participants", "scalar columns"),
        entry("metadata.participant_native_speaker", "scalar columns"),
        entry("metadata.participant_is_host", "scalar columns"),
        entry("metadata.background_conditions", "scalar columns"),
        entry(
            "social_native.anyone_talking_to_wearer_subframes",
            "fixed_size_list<bool, S>",
            extractor="social",
        ),
        entry("text.speech_rate", "table rows", extractor="text"),
    ]
    return {"registry_version": 1, "families": {}, "labels": labels}


def _manifest(names, tables):
    return {
        "materialized_labels": names,
        "unavailable_labels": {},
        "tables": {kind: {"file": f"{kind}.parquet"} for kind in tables},
        "inputs": {"action_grid": {"sha256": GRID_SHA}},
    }


def _sources(tmp_path):
    sources = {}

    for corpus in ("egocom", "ego4d"):
        root = tmp_path / "labels" / corpus
        recordings = TRAIN + VALIDATION
        registry = _registry(corpus)
        speech = [e["name"] for e in registry["labels"] if e["extractor"] == "speech"]
        (root / "speech").mkdir(parents=True)
        grid = []

        for recording in recordings:
            for k in range(STEPS):
                s = _state(corpus, recording, k)
                grid.append(
                    {
                        "recording_id": recording,
                        "decision_index": k,
                        "decision_time_s": k / 10,
                        "active_speaker_count_subframes": s["counts"],
                        "other_onset_subframes": s["onset"],
                        "speaker_activity": s["activity"],
                        "ego_speaking": s["ego"],
                    }
                )

        pq.write_table(pa.Table.from_pylist(grid), root / "speech" / "grid.parquet")
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "recording_id": r,
                        "conversation_id": _traits(r)["conversation"],
                        "wearer_index": 0,
                        "participant_count": _traits(r)["participants"],
                        "background_fan": _traits(r)["fan"],
                        "background_music": _traits(r)["music"],
                    }
                    for r in recordings
                ]
            ),
            root / "speech" / "recordings.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "recording_id": r,
                        "participant_index": p,
                        "is_ego": p == 0,
                        "native_speaker": _traits(r)["native"] if p == 0 else True,
                        "is_host": _traits(r)["host"] if p == 0 else False,
                    }
                    for r in recordings
                    for p in range(3)
                ]
            ),
            root / "speech" / "participants.parquet",
        )
        (root / "speech" / "manifest.json").write_text(
            json.dumps(_manifest(speech, ("grid", "recordings", "participants")))
        )

        if corpus == "ego4d":
            (root / "social").mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "recording_id": r,
                            "decision_index": k,
                            "decision_time_s": k / 10,
                            "anyone_talking_to_wearer_subframes": _state(corpus, r, k)[
                                "addressed"
                            ],
                            "anyone_talking_to_wearer_valid_subframes": [True] * 3,
                        }
                        for r in recordings
                        for k in range(STEPS)
                    ]
                ),
                root / "social" / "grid.parquet",
            )
            (root / "social" / "manifest.json").write_text(
                json.dumps(
                    _manifest(
                        ["social_native.anyone_talking_to_wearer_subframes"], ("grid",)
                    )
                )
            )
        else:
            (root / "text").mkdir()
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "recording_id": r,
                            "participant_index": 0,
                            "start_s": start / 10,
                            "end_s": (start + 20) / 10,
                            "words_per_second": 2.0 + (start // 30),
                            "speech_rate_valid": True,
                        }
                        for r in recordings
                        for start in range(0, STEPS, 30)
                    ]
                ),
                root / "text" / "segments.parquet",
            )
            (root / "text" / "manifest.json").write_text(
                json.dumps(_manifest(["text.speech_rate"], ("segments",)))
            )

        (root / "registry.json").write_text(json.dumps(registry))
        sources[corpus] = CorpusLabelSource(
            corpus=corpus,
            fetch=local_store(root),
            grid_sha256=GRID_SHA,
            provenance={"repo_id": "local", "labels_revision": "rev"},
        )

    return sources


def _snapshot(tmp_path, name, split, recordings, *, seed, shift_time=0.0):
    """Features carry the number of voices (dim 0) and of participants (dim 1)."""

    generator = torch.Generator().manual_seed(seed)
    metadata = {
        k: []
        for k in ("sample_id", "dataset", "recording_id", "anchor_idx", "anchor_time")
    }
    features, latent = [], []

    for corpus in ("egocom", "ego4d"):
        for recording in recordings:
            for k in ANCHORS:
                metadata["sample_id"].append(f"{corpus}:{recording}#{k}")
                metadata["dataset"].append(corpus)
                metadata["recording_id"].append(recording)
                metadata["anchor_idx"].append(k)
                metadata["anchor_time"].append(k / 10 + shift_time)
                row = torch.randn(8, generator=generator)
                row[0] = 2.0 * _state(corpus, recording, k)["counts"][0] + 0.1 * row[0]
                row[1] = _traits(recording)["participants"] + 0.05 * row[1]
                features.append(row)
                latent.append(torch.randn(4, generator=generator))

    provenance = {
        "run": {"run_id": "run-1", "config_hash": "cfg"},
        "data": {"dataset": "full", "dataset_revision": "rev", "split": split},
        "sampling": {"order": "fixed_permutation", "seed": 3072},
        "checkpoint": {"filename": "e.ckpt", "global_step": 1, "sha256": "c" * 64},
    }

    return write_snapshot(
        RepresentationSnapshot(
            representations={
                "features": torch.stack(features),
                "latent": torch.stack(latent),
            },
            metadata=metadata,
        ),
        tmp_path / name,
        provenance=provenance,
    )


@pytest.fixture
def snapshots(tmp_path):
    return (
        _snapshot(tmp_path, "train-snapshot", "train", TRAIN, seed=1),
        _snapshot(tmp_path, "validation-snapshot", "validation", VALIDATION, seed=2),
    )


def test_every_concept_is_scored_or_reported_with_a_reason(tmp_path, snapshots):
    output = write_concepts(*snapshots, label_sources=_sources(tmp_path), bootstrap=30)

    for name in ("summary.json", "scores.parquet", "report.md", "figures/concepts.png"):
        assert (output / name).is_file()

    summary = json.loads((output / "summary.json").read_text())
    scores = {s["task"]: s for s in summary["scores"]}
    assert set(scores) == {c.name for c in CONCEPTS}
    assert all(s["skipped"] or s["features_score"] is not None for s in scores.values())

    # Signal in the features, noise in the latent.
    voices_now = scores["voices_now"]
    assert voices_now["skipped"] is None
    assert voices_now["features_ci"][0] > 1 / 3
    assert voices_now["delta_ci"][1] < 0
    assert scores["participant_count"]["features_score"] > 0.8
    # Ego4D only; grouped CV counts conversations (two views each here).
    assert scores["participant_count"]["n_eval_recordings"] == 12

    # Corpus-specific concepts are scored where defined, and say where not.
    assert summary["concepts"]["addressed_to_wearer"]["unavailable"] == {
        "egocom": "not defined for egocom"
    }
    assert scores["wearer_speech_rate"]["corpora"] == ["egocom"]
    # Two fan recordings in one conversation: reported, not forced.
    assert scores["background_fan"]["skipped"].startswith("unsupported")
    assert "background_fan" in (output / "report.md").read_text()


def test_a_shifted_timeline_is_refused(tmp_path):
    train = _snapshot(tmp_path, "train", "train", TRAIN, seed=1)
    validation = _snapshot(
        tmp_path, "validation", "validation", VALIDATION, seed=2, shift_time=0.05
    )

    with pytest.raises(ValueError, match="differs from the snapshot's anchor_time"):
        write_concepts(train, validation, label_sources=_sources(tmp_path), bootstrap=5)


def test_cli_and_show_change_no_artifact(tmp_path, snapshots, monkeypatch, capsys):
    sources = _sources(tmp_path)
    monkeypatch.setattr(
        "turn_wm.evaluation.latent_analysis.concepts.hub_label_sources",
        lambda provenance, labels_revision=None: sources,
    )
    output = tmp_path / "concepts"

    main(
        [
            "probe-concepts",
            *map(str, snapshots),
            "--bootstrap",
            "10",
            "--output",
            str(output),
        ]
    )
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    show_concepts(output)

    printed = capsys.readouterr().out
    assert f"concepts: {output}" in printed
    assert "voices_now" in printed and "multi_party" in printed
    assert {p: p.read_bytes() for p in output.rglob("*") if p.is_file()} == before
