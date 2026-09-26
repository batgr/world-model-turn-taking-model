"""
Linear probes: protocol checks, probes and scores, settings, bootstrap and
written results, on synthetic snapshots and label sidecars (no network).
"""

import hashlib
import json
import math
import sys
import types
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import load_file
from test_latent_labels import GRID_SHA, entry

from turn_wm.cli import main
from turn_wm.data.labels import local_store
from turn_wm.evaluation.latent_analysis.analyze import Snapshot
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    CATEGORICAL,
    CONTINUOUS,
    CorpusLabelSource,
)
from turn_wm.evaluation.latent_analysis.probes import (
    LOGISTIC_C_GRID,
    RIDGE_ALPHA_GRID,
    ProbeData,
    Setting,
    Standardizer,
    analyze_probes,
    balanced_accuracy,
    check_no_recording_leakage,
    choose_regularization,
    fit_probe,
    probe_predictions,
    probe_settings,
    r2,
    recording_folds,
    run_probe,
    select_regularization,
    write_probes,
)
from turn_wm.evaluation.latent_analysis.show import show_probes

CORPORA = ("ego4d", "egocom")
STATES = ("silence", "ego_only", "others_only", "both")
STEPS = 60
HORIZONS = (0.5, 1.0)

REGISTRY = {
    "registry_version": 1,
    "families": {},
    "labels": [
        entry("instantaneous.ego_speaking", "bool"),
        entry("instantaneous.others_active", "bool"),
        entry(
            "instantaneous.joint_speech_state_occupancy",
            "list<float32> (4 values)",
            "[joint_state]",
        ),
        entry(
            "timing.time_to_next_speaker_onset",
            "float32 seconds (+ bool valid)",
            columns=["time_to_next_speaker_onset", "time_to_next_speaker_onset_valid"],
        ),
        entry("timing.silence_duration", "float32"),
        entry(
            "future.future_joint_speech_state", "fixed_size_list<int8, H>", "[horizon]"
        ),
    ],
}


def _labels():
    """Labels of every (corpus, recording, decision_index), seeded."""

    generator = torch.Generator().manual_seed(0)
    labels = {}

    for corpus in CORPORA:
        for recording in ("t0", "t1", "t2", "t3", "v0", "v1", "v2", "v3"):
            for k in range(STEPS):
                state = int(torch.randint(4, (1,), generator=generator))
                labels[(corpus, recording, k)] = {
                    "ego_speaking": bool(torch.rand(1, generator=generator) < 0.4),
                    "others_active": bool(torch.rand(1, generator=generator) < 0.5),
                    "state": state,
                    "onset": float(torch.rand(1, generator=generator) * 3),
                    "silence": float(torch.rand(1, generator=generator)),
                    "future": [int(torch.randint(4, (1,), generator=generator))] * 2,
                }

    return labels


LABELS = _labels()


def _sources(tmp_path):
    sources = {}

    for corpus in CORPORA:
        root = tmp_path / "labels" / corpus
        (root / "speech").mkdir(parents=True)
        rows = [
            {
                "recording_id": recording,
                "decision_index": k,
                "decision_time_s": k / 10,
                "ego_speaking": v["ego_speaking"],
                "others_active": v["others_active"],
                "joint_speech_state_occupancy": [
                    1.0 if s == v["state"] else 0.0 for s in range(4)
                ],
                "time_to_next_speaker_onset": v["onset"],
                # Every tenth onset is censored: missing, not zero.
                "time_to_next_speaker_onset_valid": k % 10 != 0,
                "silence_duration": v["silence"],
                "future_joint_speech_state": v["future"],
            }
            for (c, recording, k), v in LABELS.items()
            if c == corpus
        ]
        pq.write_table(pa.Table.from_pylist(rows), root / "speech" / "grid.parquet")
        (root / "registry.json").write_text(json.dumps(REGISTRY))
        (root / "speech" / "manifest.json").write_text(
            json.dumps(
                {
                    "materialized_labels": [e["name"] for e in REGISTRY["labels"]],
                    "unavailable_labels": {},
                    "tables": {"grid": {"file": "grid.parquet"}},
                    "inputs": {"action_grid": {"sha256": GRID_SHA}},
                    "config": {"future_horizons_s": list(HORIZONS)},
                }
            )
        )
        sources[corpus] = CorpusLabelSource(
            corpus=corpus,
            fetch=local_store(root),
            grid_sha256=GRID_SHA,
            provenance={"repo_id": "local", "labels_revision": "rev"},
        )

    return sources


def _snapshot_parts(split, recordings, *, seed):
    """Features carry ego_speaking (dim 0) and onset time (dim 1); latent is noise."""

    generator = torch.Generator().manual_seed(seed)
    metadata = {k: [] for k in ("sample_id", "dataset", "recording_id", "anchor_idx")}
    metadata["anchor_time"] = []
    features, latent = [], []

    for corpus in CORPORA:
        for recording in recordings:
            for k in range(STEPS):
                v = LABELS[(corpus, recording, k)]
                metadata["sample_id"].append(f"{corpus}:{recording}#{k}")
                metadata["dataset"].append(corpus)
                metadata["recording_id"].append(recording)
                metadata["anchor_idx"].append(k)
                metadata["anchor_time"].append(float(torch.tensor(k / 10)))
                row = torch.randn(8, generator=generator)
                row[0] = (2.0 if v["ego_speaking"] else -2.0) + 0.1 * row[0]
                row[1] = v["onset"] + 0.05 * row[1]
                features.append(row)
                latent.append(torch.randn(4, generator=generator))

    representations = {
        "features": torch.stack(features),
        "latent": torch.stack(latent),
    }
    provenance = {
        "run": {"run_id": "run-1", "config_hash": "cfg"},
        "data": {"dataset": "full", "dataset_revision": "rev", "split": split},
        "sampling": {"order": "fixed_permutation", "seed": 3072},
        "checkpoint": {
            "filename": "step-9000.ckpt",
            "global_step": 9000,
            "sha256": "c" * 64,
        },
    }

    return representations, metadata, provenance


def _write(tmp_path, name, split, recordings, *, seed, **changes):
    representations, metadata, provenance = _snapshot_parts(
        split, recordings, seed=seed
    )

    for key, value in changes.items():
        section, field = key.split("__")
        provenance[section][field] = value

    return write_snapshot(
        RepresentationSnapshot(representations=representations, metadata=metadata),
        tmp_path / name,
        provenance=provenance,
    )


TRAIN = ("t0", "t1", "t2", "t3")
VALIDATION = ("v0", "v1", "v2", "v3")


@pytest.fixture
def snapshots(tmp_path):
    return (
        _write(tmp_path, "train-snapshot", "train", TRAIN, seed=1),
        _write(tmp_path, "validation-snapshot", "validation", VALIDATION, seed=2),
    )


def _hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _score(summary, task, setting):
    return next(
        s for s in summary["scores"] if s["task"] == task and s["setting"] == setting
    )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_recording_leakage_is_refused(tmp_path):
    train = _write(tmp_path, "train", "train", ("t0", "t1", "v2"), seed=1)
    validation = _write(tmp_path, "validation", "validation", VALIDATION, seed=2)

    with pytest.raises(ValueError, match="occur in both probe-train"):
        write_probes(train, validation, label_sources=_sources(tmp_path))

    # The corpus is part of the recording key.
    check_no_recording_leakage(
        {"dataset": ["egocom"], "recording_id": ["r1"]},
        {"dataset": ["ego4d"], "recording_id": ["r1"]},
    )


def test_the_test_split_and_other_checkpoints_are_refused(tmp_path):
    train = _write(tmp_path, "train", "train", TRAIN, seed=1)
    test = _write(tmp_path, "test", "test", VALIDATION, seed=2)
    other = _write(
        tmp_path, "other", "validation", VALIDATION, seed=2, checkpoint__sha256="d" * 64
    )
    swapped = _write(tmp_path, "swapped", "validation", TRAIN, seed=1)

    with pytest.raises(ValueError, match="never read the test split"):
        write_probes(train, test, label_sources={})
    with pytest.raises(ValueError, match="checkpoint.sha256"):
        write_probes(train, other, label_sources={})
    with pytest.raises(ValueError, match="train-split snapshot"):
        write_probes(swapped, other, label_sources={})


# ---------------------------------------------------------------------------
# Scores and probes
# ---------------------------------------------------------------------------


def test_trivial_references():
    y = torch.tensor([0, 0, 0, 0, 1, 2])
    # A constant prediction: one class fully right, the others wrong.
    assert balanced_accuracy(y, torch.zeros_like(y), 3) == pytest.approx(1 / 3)

    target = torch.tensor([1.0, 2.0, 4.0, 7.0])
    assert r2(target, torch.full_like(target, float(target.mean()))) == pytest.approx(
        0.0
    )
    assert r2(target, target) == pytest.approx(1.0)


def test_preprocessing_is_fitted_on_train_only():
    generator = torch.Generator().manual_seed(0)
    train_x = torch.randn(200, 3, generator=generator) * 5 + 2
    train_y = (train_x[:, 0] > 2).long()
    eval_x = torch.randn(50, 3, generator=generator) * 5 + 2

    standardizer = Standardizer.fit(train_x)
    assert torch.allclose(standardizer.mean, train_x.double().mean(0))

    recordings = [f"x/r{i // 20}" for i in range(200)]
    alone = probe_predictions(
        CATEGORICAL, train_x, train_y, eval_x, recordings=recordings, classes=2
    )
    # Far-off extra evaluation rows would move eval-fitted statistics or an
    # eval-tuned penalty.
    more = probe_predictions(
        CATEGORICAL,
        train_x,
        train_y,
        torch.cat([eval_x, eval_x + 1_000]),
        recordings=recordings,
        classes=2,
    )
    assert torch.equal(alone, more[:50])

    ridge_alone = probe_predictions(
        CONTINUOUS, train_x, train_x[:, 1], eval_x, recordings=recordings
    )
    ridge_more = probe_predictions(
        CONTINUOUS,
        train_x,
        train_x[:, 1],
        torch.cat([eval_x, eval_x * 100]),
        recordings=recordings,
    )
    assert torch.allclose(ridge_alone, ridge_more[:50])


def test_settings():
    settings = {
        s.name: s for s in probe_settings(["egocom", "ego4d"], cross_domain=True)
    }

    assert settings == {
        "pooled": Setting("pooled", "pooled", ("ego4d", "egocom"), ("ego4d", "egocom")),
        "within:ego4d": Setting("within:ego4d", "within", ("ego4d",), ("ego4d",)),
        "within:egocom": Setting("within:egocom", "within", ("egocom",), ("egocom",)),
        "ego4d->egocom": Setting("ego4d->egocom", "cross", ("ego4d",), ("egocom",)),
        "egocom->ego4d": Setting("egocom->ego4d", "cross", ("egocom",), ("ego4d",)),
    }
    assert [
        s.name for s in probe_settings(["egocom", "ego4d"], cross_domain=False)
    ] == [
        "pooled",
        "within:ego4d",
        "within:egocom",
    ]


def _data(values, corpora, recordings, x):
    return ProbeData(
        representations={"features": x, "latent": x},
        values=values,
        corpora=corpora,
        recordings=recordings,
        sample_ids=[f"s{i:05d}" for i in range(len(values))],
    )


def test_a_class_missing_in_one_domain_makes_the_setting_unsupported():
    generator = torch.Generator().manual_seed(0)
    # "both" never occurs in ego4d.
    values = [STATES[i % 4] for i in range(400)]
    corpora = ["egocom" if i < 200 else "ego4d" for i in range(400)]
    values = [
        "silence" if c == "ego4d" and v == "both" else v
        for v, c in zip(values, corpora, strict=True)
    ]
    recordings = [f"{c}/r{i % 5}" for i, c in enumerate(corpora)]
    x = torch.randn(400, 3, generator=generator)
    data = _data(values, corpora, recordings, x)

    within = run_probe(
        CATEGORICAL,
        STATES,
        Setting("within:ego4d", "within", ("ego4d",), ("ego4d",)),
        data,
        data,
        bootstrap=20,
        seed=0,
    )
    # Never silently reduced to a 3-class task with a 1/3 reference.
    assert within["classes"] == list(STATES)
    assert within["reference"] == pytest.approx(0.25)
    assert within["unsupported_classes"] == {"both": {"train": 0, "eval": 0}}
    assert within["skipped"].startswith("unsupported: both (train 0, eval 0)")
    assert within["features_score"] is None and within["features_ci"] is None

    pooled = run_probe(
        CATEGORICAL,
        STATES,
        Setting("pooled", "pooled", ("ego4d", "egocom"), ("ego4d", "egocom")),
        data,
        data,
        bootstrap=20,
        seed=0,
    )
    assert pooled["classes"] == list(STATES)
    assert pooled["reference"] == pytest.approx(0.25)


def test_intervals_resample_recordings():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(300, 2, generator=generator)
    values = [bool(v > 0) for v in x[:, 0]]
    values = ["true" if v else "false" for v in values]
    setting = Setting("pooled", "pooled", ("egocom",), ("egocom",))

    # Training rows from ten recordings, so the grouped CV can choose C.
    train_recordings = [f"egocom/t{i // 30}" for i in range(300)]
    one = run_probe(
        CATEGORICAL,
        ("false", "true"),
        setting,
        _data(values, ["egocom"] * 300, train_recordings, x),
        _data(values, ["egocom"] * 300, ["egocom/r1"] * 300, x),
        bootstrap=50,
        seed=0,
    )
    # One validation recording: nothing to resample, however many anchors.
    assert one["n_eval_recordings"] == 1 and one["features_ci"] is None

    many = run_probe(
        CATEGORICAL,
        ("false", "true"),
        setting,
        _data(values, ["egocom"] * 300, train_recordings, x),
        _data(values, ["egocom"] * 300, [f"egocom/r{i // 30}" for i in range(300)], x),
        bootstrap=50,
        seed=0,
    )
    assert many["n_eval_recordings"] == 10
    low, high = many["features_ci"]
    assert low <= many["features_score"] <= high


# ---------------------------------------------------------------------------
# The whole analysis on synthetic snapshots
# ---------------------------------------------------------------------------


def test_signal_is_recovered_and_noise_stays_at_the_reference(tmp_path, snapshots):
    output = write_probes(*snapshots, label_sources=_sources(tmp_path), bootstrap=100)
    summary = json.loads((output / "summary.json").read_text())

    ego = _score(summary, "instantaneous.ego_speaking", "pooled")
    assert ego["reference"] == 0.5
    assert ego["features_score"] > 0.95
    assert abs(ego["latent_score"] - 0.5) < 0.08
    assert ego["latent_ci"][0] < 0.5 < ego["latent_ci"][1]
    assert ego["delta_score"] == pytest.approx(
        ego["latent_score"] - ego["features_score"]
    )
    assert ego["delta_ci"][1] < 0

    onset = _score(summary, "timing.time_to_next_speaker_onset", "pooled")
    assert onset["reference"] == 0.0
    assert onset["features_score"] > 0.95
    assert abs(onset["latent_score"]) < 0.08
    # Censored onsets are missing, not probed.
    assert onset["n_eval"] == len(CORPORA) * len(VALIDATION) * STEPS * 9 // 10

    # Continuous labels: pooled and within-domain only.
    assert {s["setting"] for s in summary["scores"] if s["task"] == onset["task"]} == {
        "pooled",
        "within:ego4d",
        "within:egocom",
    }
    cross = _score(summary, "instantaneous.ego_speaking", "egocom->ego4d")
    assert cross["train_corpora"] == ["egocom"] and cross["eval_corpora"] == ["ego4d"]
    assert cross["n_eval"] == len(VALIDATION) * STEPS
    assert cross["features_score"] > 0.95

    future = _score(summary, "future.future_joint_speech_state@1s", "pooled")
    assert future["reference"] == 0.25

    assert sorted(p.name for p in output.iterdir()) == [
        "figures",
        "report.md",
        "scores.parquet",
        "summary.json",
    ]
    assert sorted(p.name for p in (output / "figures").iterdir()) == [
        "cross_domain.png",
        "current_state.png",
        "future_state.png",
        "temporal_state.png",
    ]
    report = (output / "report.md").read_text()

    for heading in (
        "# Linear probe evaluation",
        "## Purpose",
        "## Current conversational state",
        "## Temporal state",
        "## Future conversational state",
        "## Within-domain representation",
        "## Cross-domain transfer",
        "## Mimi vs WM representation",
        "## What this does NOT show",
        "## Metric hypotheses",
        "## Literature questions",
    ):
        assert heading in report

    assert "**Degrades** linear accessibility: ego speaking (pooled" in report
    assert "transfers: information present in both corpora" in report


def _in_memory(path, reorder):
    snapshot_dir = path
    representations = load_file(snapshot_dir / "representations.safetensors")
    metadata = pq.read_table(snapshot_dir / "metadata.parquet").to_pydict()
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    order = reorder(len(metadata["sample_id"]))

    return Snapshot(
        path=snapshot_dir,
        representations={k: v[order] for k, v in representations.items()},
        metadata={k: [v[i] for i in order] for k, v in metadata.items()},
        manifest=manifest,
    )


def test_row_order_does_not_change_the_scores(tmp_path, snapshots):
    sources = _sources(tmp_path)
    identity = _in_memory(snapshots[0], lambda n: list(range(n)))
    validation = _in_memory(snapshots[1], lambda n: list(range(n)))
    shuffled_train = _in_memory(
        snapshots[0],
        lambda n: torch.randperm(
            n, generator=torch.Generator().manual_seed(5)
        ).tolist(),
    )
    shuffled_validation = _in_memory(
        snapshots[1],
        lambda n: torch.randperm(
            n, generator=torch.Generator().manual_seed(6)
        ).tolist(),
    )

    a = analyze_probes(identity, validation, sources, bootstrap=50)["scores"]
    b = analyze_probes(shuffled_train, shuffled_validation, sources, bootstrap=50)[
        "scores"
    ]

    for x, y in zip(a, b, strict=True):
        assert x["setting"] == y["setting"] and x["task"] == y["task"]

        for key in ("features_score", "latent_score", "delta_score"):
            assert x[key] == pytest.approx(y[key], abs=1e-6)

        for key in ("features_ci", "delta_ci"):
            assert (x[key] is None) == (y[key] is None)
            if x[key] is not None:
                assert x[key] == pytest.approx(y[key], abs=1e-6)


# ---------------------------------------------------------------------------
# Showing
# ---------------------------------------------------------------------------


def test_show_changes_no_artifact(tmp_path, snapshots, monkeypatch, capsys):
    sources = _sources(tmp_path)
    monkeypatch.setattr(
        "turn_wm.evaluation.latent_analysis.probes.hub_label_sources",
        lambda provenance, labels_revision=None: sources,
    )
    train, validation = map(str, snapshots)

    main(["probe-latents", train, validation, "--bootstrap", "20"])
    main(
        [
            "probe-latents",
            train,
            validation,
            "--bootstrap",
            "20",
            "--show",
            "--output",
            str(tmp_path / "shown"),
        ]
    )

    plain = snapshots[1] / "analysis" / "probes"
    assert _hashes(plain) == _hashes(tmp_path / "shown")
    printed = capsys.readouterr().out
    assert "Key deltas (pooled, latent − features):" in printed
    assert "instantaneous.ego_speaking: delta" in printed

    # In a notebook: the table, the deltas, then the four figures.
    before = _hashes(plain)
    shown = []
    ipython: Any = types.ModuleType("IPython")
    ipython.get_ipython = lambda: object()
    display: Any = types.ModuleType("IPython.display")
    display.display = shown.append
    display.HTML = lambda text: ("html", text)
    display.Image = lambda filename: ("image", filename.rsplit("/", 1)[-1])
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)

    show_probes(plain)

    assert "Probe scores" in shown[0][1] and "Key deltas" in shown[1][1]
    assert [item[1] for item in shown[2:]] == [
        "current_state.png",
        "temporal_state.png",
        "future_state.png",
        "cross_domain.png",
    ]
    assert _hashes(plain) == before


# ---------------------------------------------------------------------------
# Regularization: predeclared grid, recording-grouped CV inside probe-train
# ---------------------------------------------------------------------------


def test_cv_folds_hold_whole_recordings_in_a_seeded_order():
    recordings = [f"x/r{i % 7}" for i in range(70)]

    fold, count = recording_folds(recordings, folds=5, seed=1)

    assert count == 5
    for r in set(recordings):
        assert len({int(fold[i]) for i in range(70) if recordings[i] == r}) == 1
    # Each fold's recordings do not depend on the row order.
    reordered, _ = recording_folds(list(reversed(recordings)), folds=5, seed=1)
    assert torch.equal(reordered, fold.flip(0))
    assert recording_folds(recordings, folds=5, seed=2)[0].tolist() != fold.tolist()
    # Fewer recordings than folds: one fold per recording.
    assert recording_folds(["a", "b", "a"], folds=5, seed=0)[1] == 2


def test_ties_go_to_the_stronger_regularization():
    equal = dict.fromkeys(LOGISTIC_C_GRID, 0.7)
    # Logistic C: smaller C is stronger L2.
    assert choose_regularization(equal, kind=CATEGORICAL) == min(LOGISTIC_C_GRID)

    equal = dict.fromkeys(RIDGE_ALPHA_GRID, 0.3)
    # Ridge alpha: larger alpha is stronger.
    assert choose_regularization(equal, kind=CONTINUOUS) == max(RIDGE_ALPHA_GRID)

    # Within numerical tolerance is a tie; beyond it, the better score wins.
    near = {1e-4: 0.5, 1e-2: 0.5 + 1e-12, 1.0: 0.4}
    assert choose_regularization(near, kind=CATEGORICAL) == 1e-4
    better = {1e-4: 0.5, 1e-2: 0.6, 1.0: 0.4}
    assert choose_regularization(better, kind=CATEGORICAL) == 1e-2


def test_ridge_grid_spans_several_orders_of_magnitude():
    assert max(RIDGE_ALPHA_GRID) / min(RIDGE_ALPHA_GRID) >= 1e8
    assert list(LOGISTIC_C_GRID) == [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def test_ridge_cv_picks_strong_regularization_on_noise_and_weak_on_signal():
    generator = torch.Generator().manual_seed(0)
    recordings = [f"x/r{i // 20}" for i in range(400)]
    signal = torch.randn(400, 1, generator=generator)
    y = signal[:, 0] * 3

    noise = select_regularization(
        CONTINUOUS,
        torch.randn(400, 30, generator=generator),
        y,
        recordings,
        classes=None,
        seed=0,
    )
    assert noise.selected == max(RIDGE_ALPHA_GRID)
    assert noise.valid_folds == [0, 1, 2, 3, 4]

    clean = fit_probe(
        CONTINUOUS,
        torch.cat([signal, 0.01 * torch.randn(400, 3, generator=generator)], 1),
        y,
        recordings,
    )
    assert clean.cv.selected is not None
    assert clean.cv.selected < max(RIDGE_ALPHA_GRID)
    assert max(clean.cv.mean_scores.values()) > 0.99


def _cv_labels(recordings, present_in_folds, seed=0):
    """Class 2 only in recordings of `present_in_folds`; classes 0, 1 everywhere."""

    fold, _ = recording_folds(recordings, folds=5, seed=seed)

    return torch.tensor(
        [
            2 if int(fold[i]) in present_in_folds and i % 3 == 0 else i % 2
            for i in range(len(recordings))
        ]
    )


def test_a_fold_missing_a_canonical_class_is_invalid_not_k_minus_1():
    generator = torch.Generator().manual_seed(0)
    recordings = [f"x/r{i // 30}" for i in range(600)]
    y = _cv_labels(recordings, {0, 1, 2, 3})
    x = torch.randn(600, 4, generator=generator)

    cv = select_regularization(CATEGORICAL, x, y, recordings, classes=3, seed=0)

    # Fold 4's held-out part has no class 2: invalid, never scored on 2 classes.
    assert cv.invalid_folds == {
        4: "a canonical class is absent from the fold's train or held-out part"
    }
    assert cv.fold_class_counts is not None
    assert cv.fold_class_counts[4]["held_out"][2] == 0
    assert cv.valid_folds == [0, 1, 2, 3]
    # Every candidate C is scored on exactly the same valid folds.
    assert {tuple(sorted(f)) for f in cv.fold_scores.values()} == {(0, 1, 2, 3)}
    assert cv.selected in LOGISTIC_C_GRID
    provenance = cv.provenance()
    assert provenance["requested_cv_folds"] == 5
    assert provenance["valid_cv_folds"] == [0, 1, 2, 3]
    assert provenance["regularization_parameter"] == "C"
    assert set(provenance["fold_scores_by_c"]) == {f"{v:g}" for v in LOGISTIC_C_GRID}

    # A K-class balanced accuracy never falls back to K - 1.
    assert math.isnan(balanced_accuracy(torch.tensor([0, 1]), torch.tensor([0, 1]), 3))


def test_too_few_valid_folds_make_the_setting_unsupported():
    generator = torch.Generator().manual_seed(0)
    recordings = [f"egocom/r{i // 30}" for i in range(600)]
    # Class "c" in two recordings only (20 rows): whatever the fold
    # assignment, at most two grouped folds see it on both sides.
    values = ["c" if i < 60 and i % 3 == 0 else ("a", "b")[i % 2] for i in range(600)]
    x = torch.randn(600, 4, generator=generator)
    data = _data(values, ["egocom"] * 600, recordings, x)
    evaluation = _data(
        values, ["egocom"] * 600, [f"egocom/v{i // 30}" for i in range(600)], x
    )

    result = run_probe(
        CATEGORICAL,
        ("a", "b", "c"),
        Setting("pooled", "pooled", ("egocom",), ("egocom",)),
        data,
        evaluation,
        bootstrap=20,
        seed=0,
    )

    # Globally supported (>= 20 rows of every class on both sides) ...
    assert not result["unsupported_classes"]
    # ... but not selectable by grouped CV: no validation performance at all.
    assert result["skipped"].startswith("insufficient_grouped_cv_class_support")
    for name in ("features", "latent", "delta"):
        assert result[f"{name}_score"] is None and result[f"{name}_ci"] is None
    assert result["features_selected_regularization"] is None


def test_selected_regularization_is_recorded_and_shared_by_settings(
    tmp_path, snapshots
):
    output = write_probes(*snapshots, label_sources=_sources(tmp_path), bootstrap=20)
    summary = json.loads((output / "summary.json").read_text())
    within = _score(summary, "instantaneous.ego_speaking", "within:egocom")
    cross = _score(summary, "instantaneous.ego_speaking", "egocom->ego4d")

    # within:egocom and egocom->ego4d share the probe trained on egocom.
    assert within["regularization_parameter"] == "C"
    for name in ("features", "latent"):
        assert within[f"{name}_selected_regularization"] in LOGISTIC_C_GRID
        assert (
            within[f"{name}_selected_regularization"]
            == cross[f"{name}_selected_regularization"]
        )

    # One fitted probe per (task, representation, training domain).
    probes = [
        p
        for p in summary["fitted_probes"]
        if p["task"] == "instantaneous.ego_speaking"
        and p["training_domain"] == ["egocom"]
    ]
    assert sorted(p["representation"] for p in probes) == ["features", "latent"]
    cv = probes[0]["cv"]
    assert cv["grouping"] == "(dataset, recording_id)"
    assert cv["candidate_c"] == list(LOGISTIC_C_GRID)
    assert cv["valid_cv_folds"] == [0, 1, 2, 3]  # four training recordings
    assert cv["invalid_cv_folds"] == {
        "4": "no recording: fewer training recordings than folds"
    }

    ridge = next(
        p for p in summary["fitted_probes"] if p["task"] == "timing.silence_duration"
    )["cv"]
    assert ridge["candidate_alphas"] == list(RIDGE_ALPHA_GRID)
    assert ridge["selected_alpha"] in RIDGE_ALPHA_GRID
    assert set(ridge["fold_scores_by_alpha"]) == {f"{v:g}" for v in RIDGE_ALPHA_GRID}

    report = (output / "report.md").read_text()
    assert "smaller values mean stronger L2 regularization" in report
    assert "| ego speaking | egocom | WM latent | logistic | C=" in report
    assert (
        "| silence duration | ego4d + egocom | Mimi features | ridge | alpha=" in report
    )
    assert "4/5 |" in report


def test_balanced_or_unordered_snapshots_are_refused(tmp_path):
    train = _write(
        tmp_path, "train", "train", TRAIN, seed=1, sampling__order="balanced"
    )
    validation = _write(tmp_path, "validation", "validation", VALIDATION, seed=2)

    with pytest.raises(ValueError, match="not a seeded fixed permutation"):
        write_probes(train, validation, label_sources={})
