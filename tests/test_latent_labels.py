"""
Label-conditioned analysis: audit, exact join, metrics and written results,
on synthetic label sidecars stored locally (no network).
"""

import hashlib
import json
import types

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from turn_wm.cli import main
from turn_wm.data.labels import local_store
from turn_wm.evaluation.latent_analysis import label_source as label_source_module
from turn_wm.evaluation.latent_analysis.analyze import analyze_snapshot
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    CorpusLabelSource,
    audit_corpus,
    hub_label_sources,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.label_structure import (
    analyze_labels,
    balanced_silhouette,
    continuous_metrics,
    domain_structure,
    spearman,
)
from turn_wm.evaluation.latent_analysis.pca import sample_keys
from turn_wm.evaluation.latent_analysis.show import show_labels

GRID_SHA = "a" * 64
FIRST_INDEX = 5  # decision_index of the grid's first row: rows are not indices
STEPS = 60


def entry(name, dtype, shape="[]", columns=None, extractor="speech"):
    return {
        "name": name,
        "family": name.split(".")[0],
        "role": "context",
        "level": "grid",
        "source_kind": "native_annotation",
        "modalities": ["audio"],
        "availability": "available",
        "unsupported_reason": None,
        "extractor": extractor,
        "table": "grid",
        "columns": columns or [name.split(".")[1]],
        "dtype": dtype,
        "shape": shape,
    }


REGISTRY = {
    "registry_version": 1,
    "families": {},
    "labels": [
        entry("instantaneous.ego_speaking", "bool"),
        entry(
            "instantaneous.joint_speech_state_occupancy",
            "list<float32> (4 values)",
            "[joint_state]",
        ),
        entry(
            "timing.time_to_next_floor_change",
            "float32 seconds (+ bool valid)",
            columns=["time_to_next_floor_change", "time_to_next_floor_change_valid"],
        ),
        entry("next_speaker.next_unique_speaker", "int16"),
        entry("future.future_ego_onset", "fixed_size_list<bool, H>", "[horizon]"),
        entry("nuisance.global_audio_rms", "float32", extractor="audio"),
    ],
}


def write_store(root, corpus, *, grid_sha=GRID_SHA, horizons=(0.1, 0.5), drop=()):
    """One recording r1 (and r2, never in the snapshot) of `corpus`."""

    rows = []
    for recording in ("r1", "r2"):
        for k in range(FIRST_INDEX, FIRST_INDEX + STEPS):
            if (recording, k) in drop:
                continue
            speaking = k % 2 == 0 if corpus == "egocom" else k % 3 == 0
            rows.append(
                {
                    "recording_id": recording,
                    "decision_index": k,
                    "decision_time_s": k / 10,
                    "ego_speaking": speaking,
                    "joint_speech_state_occupancy": (
                        [0.0, 1.0, 0.0, 0.0] if speaking else [0.7, 0.0, 0.3, 0.0]
                    ),
                    "time_to_next_floor_change": float(k),
                    "time_to_next_floor_change_valid": k % 10 != 0,
                    "next_unique_speaker": 1,
                    "future_ego_onset": [speaking, not speaking][: len(horizons)],
                }
            )

    (root / "speech").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "speech" / "grid.parquet")
    names = [e["name"] for e in REGISTRY["labels"] if e["extractor"] == "speech"]
    (root / "registry.json").write_text(json.dumps(REGISTRY))
    (root / "speech" / "manifest.json").write_text(
        json.dumps(
            {
                "materialized_labels": names,
                "unavailable_labels": {},
                "tables": {"grid": {"file": "grid.parquet"}},
                "inputs": {"action_grid": {"sha256": grid_sha}},
                "config": {"future_horizons_s": list(horizons)},
                "extractor_version": "speech-rules-v1",
                "code_revision": {"git_commit": "abc"},
            }
        )
    )

    return root


@pytest.fixture
def sources(tmp_path):
    return {
        corpus: CorpusLabelSource(
            corpus=corpus,
            fetch=local_store(write_store(tmp_path / corpus / "labels", corpus)),
            grid_sha256=GRID_SHA,
            provenance={
                "repo_id": "local",
                "snapshot_revision": "rev",
                "labels_revision": "rev",
                "action_grid_sha256": GRID_SHA,
            },
        )
        for corpus in ("egocom", "ego4d")
    }


def snapshot_rows(indices=range(FIRST_INDEX, FIRST_INDEX + STEPS)):
    """The same recording id r1 in both corpora: the corpus is part of the key."""

    metadata = {
        k: []
        for k in (
            "sample_id",
            "dataset",
            "recording_id",
            "anchor_idx",
            "anchor_time",
            "sample_class",
            "action_id",
            "action",
        )
    }

    for corpus in ("egocom", "ego4d"):
        for k in indices:
            metadata["sample_id"].append(f"{corpus}:r1#{k}")
            metadata["dataset"].append(corpus)
            metadata["recording_id"].append("r1")
            metadata["anchor_idx"].append(k)
            metadata["anchor_time"].append(
                float(torch.tensor(k / 10, dtype=torch.float32))
            )
            metadata["sample_class"].append("event")
            metadata["action_id"].append(k % 2)
            metadata["action"].append(["NO_EVENT", "ONSET"][k % 2])

    return metadata


def audits(sources):
    return {corpus: audit_corpus(source) for corpus, source in sources.items()}


def join(sources, metadata=None):
    return join_labels(metadata or snapshot_rows(), audits(sources), sources)


def variable(joined, name):
    return next(v for v in joined.variables if v.name == name)


# ---------------------------------------------------------------------------
# Audit and exact join
# ---------------------------------------------------------------------------


def test_labels_join_exactly_on_corpus_recording_and_decision_index(sources):
    metadata = snapshot_rows()
    joined = join(sources, metadata)

    speaking = variable(joined, "instantaneous.ego_speaking").values
    for i, (corpus, k) in enumerate(
        zip(metadata["dataset"], metadata["anchor_idx"], strict=True)
    ):
        expected = k % 2 == 0 if corpus == "egocom" else k % 3 == 0
        assert speaking[i] == ("true" if expected else "false")

    assert joined.alignment["joined_rows_checked"] == 2 * STEPS
    assert joined.alignment["max_abs_time_difference_s"] < 1e-5


def test_label_shapes_become_variables(sources):
    joined = join(sources)

    assert [v.name for v in joined.variables] == [
        "instantaneous.ego_speaking",
        "instantaneous.joint_speech_state_occupancy:dominant",
        "timing.time_to_next_floor_change",
        "future.future_ego_onset@0.1s",
        "future.future_ego_onset@0.5s",
    ]
    dominant = variable(joined, "instantaneous.joint_speech_state_occupancy:dominant")
    assert set(dominant.values) == {"ego_only", "silence"}
    # Horizons come from the manifest.
    assert variable(joined, "future.future_ego_onset@0.5s").horizon_s == 0.5
    assert joined.excluded["next_speaker.next_unique_speaker"]["egocom"].startswith(
        "integer index"
    )
    assert joined.excluded["nuisance.global_audio_rms"] == {
        "ego4d": "extractor audio not published",
        "egocom": "extractor audio not published",
    }


def test_invalid_values_are_null_and_counted(sources):
    joined = join(sources)

    times = variable(joined, "timing.time_to_next_floor_change").values
    assert all(
        (v is None) == (k % 10 == 0)
        for v, k in zip(times, snapshot_rows()["anchor_idx"], strict=True)
    )
    row = next(
        r
        for r in joined.coverage
        if r["variable"] == "timing.time_to_next_floor_change"
        and r["dataset"] == "egocom"
    )
    assert row["n_joined"] == STEPS
    assert row["n_valid"] == STEPS - 6
    assert row["null_fraction"] == pytest.approx(6 / STEPS)


def test_partial_coverage_is_reported_not_approximated(tmp_path):
    source = CorpusLabelSource(
        corpus="egocom",
        fetch=local_store(
            write_store(tmp_path / "labels", "egocom", drop={("r1", 7), ("r1", 8)})
        ),
        grid_sha256=GRID_SHA,
        provenance={},
    )
    metadata = snapshot_rows()
    metadata = {k: v[:STEPS] for k, v in metadata.items()}  # egocom only

    joined = join_labels(metadata, {"egocom": audit_corpus(source)}, {"egocom": source})

    row = joined.coverage[0]
    assert (row["n_snapshot"], row["n_joined"], row["n_missing"]) == (
        STEPS,
        STEPS - 2,
        2,
    )
    speaking = variable(joined, "instantaneous.ego_speaking").values
    assert speaking[2] is None and speaking[3] is None  # anchors 7 and 8


def test_labels_built_from_another_grid_are_refused(tmp_path):
    source = CorpusLabelSource(
        corpus="egocom",
        fetch=local_store(
            write_store(tmp_path / "labels", "egocom", grid_sha="b" * 64)
        ),
        grid_sha256=GRID_SHA,
        provenance={},
    )
    metadata = {k: v[:STEPS] for k, v in snapshot_rows().items()}

    with pytest.raises(ValueError, match="built from action grid bbbb"):
        join_labels(metadata, {"egocom": audit_corpus(source)}, {"egocom": source})


def test_another_timeline_is_refused(sources):
    metadata = snapshot_rows()
    metadata["anchor_time"][0] += 0.05

    with pytest.raises(ValueError, match="not the same timeline"):
        join(sources, metadata)


def test_a_missing_label_sidecar_is_refused(tmp_path, sources):
    empty = CorpusLabelSource(
        corpus="egocom",
        fetch=local_store(tmp_path / "nothing"),
        grid_sha256=GRID_SHA,
        provenance={},
    )
    both = {**sources, "egocom": empty}

    with pytest.raises(ValueError, match="no labels/registry.json"):
        join_labels(snapshot_rows(), audits(both), both)


def test_horizons_must_match_the_manifest(tmp_path):
    root = write_store(tmp_path / "labels", "egocom", horizons=(0.1, 0.5))
    manifest = json.loads((root / "speech" / "manifest.json").read_text())
    manifest["config"]["future_horizons_s"] = [0.1, 0.5, 1.0]
    (root / "speech" / "manifest.json").write_text(json.dumps(manifest))
    source = CorpusLabelSource("egocom", local_store(root), GRID_SHA, {})
    metadata = {k: v[:STEPS] for k, v in snapshot_rows().items()}

    with pytest.raises(ValueError, match="has 2 horizons"):
        join_labels(metadata, {"egocom": audit_corpus(source)}, {"egocom": source})


def test_another_labels_revision_needs_the_same_grid(monkeypatch):
    shas = {"v1": "a" * 64, "later": "b" * 64}

    class Api:
        def list_repo_tree(self, repo, repo_type, revision, recursive):
            return [
                types.SimpleNamespace(
                    path=f"{corpus}/action_grid.parquet",
                    lfs=types.SimpleNamespace(sha256=shas[revision]),
                )
                for corpus in ("egocom", "ego4d")
            ]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    provenance = {"data": {"dataset": "full", "dataset_revision": "v1"}}

    with pytest.raises(ValueError, match="is not the snapshot's"):
        hub_label_sources(provenance, labels_revision="later")

    shas["later"] = shas["v1"]
    sources = hub_label_sources(provenance, labels_revision="later")
    assert sources["egocom"].provenance["labels_revision"] == "later"
    assert sources["egocom"].provenance["snapshot_revision"] == "v1"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def representations(metadata, seed=0):
    """features separate ego_speaking in egocom only; latent = a scaled slice."""

    generator = torch.Generator().manual_seed(seed)
    rows = len(metadata["dataset"])
    features = torch.randn(rows, 6, generator=generator)

    for i, (corpus, k) in enumerate(
        zip(metadata["dataset"], metadata["anchor_idx"], strict=True)
    ):
        if corpus == "egocom" and k % 2 == 0:
            features[i, 0] += 4.0
        features[i, 1] += k / 20  # follows the continuous label

    return {"features": features, "latent": features[:, :3] * 2}


def analysis(sources, metadata=None, **kwargs):
    metadata = metadata or snapshot_rows()
    joined = join(sources, metadata)

    return analyze_labels(
        representations(metadata),
        metadata,
        joined.variables,
        seed=0,
        silhouette_samples=kwargs.pop("silhouette_samples", 1_000),
        **kwargs,
    )


def test_binary_label_structure_per_corpus(sources):
    result = analysis(sources)
    speaking = result.metrics["instantaneous.ego_speaking"]["features"]

    assert list(speaking) == ["all", "ego4d", "egocom"]
    assert speaking["egocom"]["between_variance_fraction"] > 0.3
    assert speaking["ego4d"]["between_variance_fraction"] < 0.1
    assert (
        speaking["egocom"]["silhouette_natural"]
        > speaking["ego4d"]["silhouette_natural"]
    )
    assert speaking["egocom"]["counts"] == {"false": 30, "true": 30}
    assert result.domain["instantaneous.ego_speaking"]["features"]["class"] == (
        "strong_but_different"
    )


def test_categorical_label_structure(sources):
    dominant = analysis(sources).metrics[
        "instantaneous.joint_speech_state_occupancy:dominant"
    ]["latent"]["egocom"]

    assert dominant["counts"] == {"silence": 30, "ego_only": 30}
    assert dominant["fractions"] == {"silence": 0.5, "ego_only": 0.5}
    assert dominant["centroid_distance_over_pooled_within_rms"] > 1


def test_continuous_label_quantiles_and_spearman(sources):
    metrics = analysis(sources).metrics["timing.time_to_next_floor_change"]["features"][
        "all"
    ]

    valid = [float(k) for k in range(FIRST_INDEX, FIRST_INDEX + STEPS) if k % 10] * 2
    assert metrics["samples"] == len(valid)
    assert metrics["quantiles"]["0.5"] == pytest.approx(
        float(torch.tensor(valid).quantile(0.5))
    )
    assert metrics["bins"] == 10
    assert metrics["between_bin_variance_fraction"] > 0


def test_spearman_is_the_rank_correlation():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    # Ties take their average rank: ranks (1.5, 1.5, 3, 4) vs (1, 2, 3, 4).
    assert spearman(
        torch.tensor([1.0, 1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0, 4.0])
    ) == pytest.approx(0.9486833, rel=1e-6)


def test_continuous_bins_are_deterministic_quantiles():
    values = torch.arange(100, dtype=torch.float64)
    x = values.unsqueeze(1).repeat(1, 3)

    metrics = continuous_metrics(
        x,
        values,
        coordinates=x[:, :2],
        keys=torch.arange(100),
        bins=10,
        name="v",
        condition="all",
    )

    assert metrics["bins"] == 10
    assert metrics["spearman_pc1"] == pytest.approx(1.0)
    assert metrics["between_bin_variance_fraction"] > 0.95


def test_natural_and_balanced_silhouettes_differ_on_imbalanced_classes():
    generator = torch.Generator().manual_seed(1)
    x = torch.cat(
        [
            torch.randn(500, 2, generator=generator),
            torch.randn(20, 2, generator=generator) + 1.5,
        ]
    )
    labels = ["common"] * 500 + ["rare"] * 20
    keys = sample_keys([str(i) for i in range(520)], seed=0)

    balanced, per_class, reason = balanced_silhouette(x, labels, keys=keys, cap=100)

    assert per_class == 20 and reason is None
    from turn_wm.evaluation.latent_analysis.pca import silhouette

    natural, _, _ = silhouette(x, labels)
    assert balanced != pytest.approx(natural, abs=0.01)


def test_a_class_too_small_gives_null_with_a_reason():
    x = torch.randn(10, 2)
    keys = torch.arange(10)

    value, per_class, reason = balanced_silhouette(
        x, ["a"] * 9 + ["b"], keys=keys, cap=100
    )

    assert value is None and per_class == 1
    assert reason == "a class has fewer than 2 rows"


def test_domain_classes():
    assert domain_structure({"a": 0.001, "b": 0.002})["class"] == "weak_in_both"
    assert domain_structure({"a": 0.20, "b": 0.15})["class"] == "similar"
    assert domain_structure({"a": 0.30, "b": 0.02})["class"] == ("strong_but_different")
    assert domain_structure({"a": 0.3, "b": None})["class"] == "undetermined"


def test_features_to_latent_deltas(sources):
    result = analysis(sources)
    delta = result.deltas["instantaneous.ego_speaking"]["egocom"]
    features = result.metrics["instantaneous.ego_speaking"]["features"]["egocom"]
    latent = result.metrics["instantaneous.ego_speaking"]["latent"]["egocom"]

    assert delta["delta_between_variance_fraction"] == pytest.approx(
        latent["between_variance_fraction"] - features["between_variance_fraction"]
    )
    assert delta["delta_silhouette_natural"] == pytest.approx(
        latent["silhouette_natural"] - features["silhouette_natural"]
    )
    # Fewer noise dimensions: the latent slice concentrates the structure.
    assert delta["change"] == "stronger"
    continuous = result.deltas["timing.time_to_next_floor_change"]["all"]
    assert "delta_abs_spearman_pc1" in continuous


def test_row_order_does_not_change_the_metrics(sources):
    metadata = snapshot_rows()
    order = torch.randperm(
        2 * STEPS, generator=torch.Generator().manual_seed(2)
    ).tolist()
    shuffled = {k: [v[i] for i in order] for k, v in metadata.items()}
    joined, joined_shuffled = join(sources, metadata), join(sources, shuffled)
    reps = representations(metadata)

    first = analyze_labels(
        reps, metadata, joined.variables, seed=0, silhouette_samples=1_000
    )
    second = analyze_labels(
        {k: v[order] for k, v in reps.items()},
        shuffled,
        joined_shuffled.variables,
        seed=0,
        silhouette_samples=1_000,
    )

    for name in ("instantaneous.ego_speaking", "timing.time_to_next_floor_change"):
        for condition in ("all", "egocom"):
            a = first.metrics[name]["latent"][condition]
            b = second.metrics[name]["latent"][condition]
            for key in (
                "between_variance_fraction",
                "silhouette_natural",
                "spearman_pc1",
            ):
                if key in a:
                    assert a[key] == pytest.approx(b[key])


# ---------------------------------------------------------------------------
# Written results
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_dir(tmp_path):
    metadata = snapshot_rows()

    return write_snapshot(
        RepresentationSnapshot(
            representations=representations(metadata), metadata=metadata
        ),
        tmp_path / "latent-snapshot",
        provenance={
            "run": {"run_id": "run-1"},
            "data": {
                "dataset": "full",
                "dataset_revision": "rev",
                "split": "validation",
            },
            "sampling": {"seed": 3072},
        },
    )


def _hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_written_label_results(snapshot_dir, sources):
    before = _hashes(snapshot_dir)

    output = analyze_snapshot(
        snapshot_dir, analyses=["labels"], silhouette_samples=500, label_sources=sources
    )["labels"]

    assert sorted(p.name for p in output.iterdir()) == [
        "coverage.parquet",
        "figures",
        "joined_labels.parquet",
        "label_inventory.json",
        "label_inventory.md",
        "metrics.parquet",
        "report.md",
        "summary.json",
    ]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["labels"]["corpora"]["egocom"]["action_grid_sha256"] == GRID_SHA
    assert (
        summary["labels"]["joined_labels_sha256"]
        == hashlib.sha256((output / "joined_labels.parquet").read_bytes()).hexdigest()
    )
    assert sorted(
        p.name for p in (output / "figures" / "temporal_state").iterdir()
    ) == [
        "timing.time_to_next_floor_change_features.png",
        "timing.time_to_next_floor_change_latent.png",
    ]

    inventory = json.loads((output / "label_inventory.json").read_text())
    rms = [r for r in inventory if r["label"] == "nuisance.global_audio_rms"]
    assert {(r["dataset"], r["available"], r["materialized"]) for r in rms} == {
        ("egocom", True, False),
        ("ego4d", True, False),
    }

    joined = pq.read_table(output / "joined_labels.parquet").to_pydict()
    # Only the snapshot's rows: r2 (outside the snapshot) is never joined.
    assert set(joined["recording_id"]) == {"r1"}
    assert len(joined["sample_id"]) == 2 * STEPS

    report = (output / "report.md").read_text()
    for heading in (
        "# Label-conditioned representation analysis",
        "## Data and label provenance",
        "## Conversational state",
        "## Temporal state",
        "## Future structure",
        "## Nuisance controls",
        "## Feature-to-latent changes",
        "## Domain-conditioned structure",
        "## Metric hypotheses",
        "## Literature questions",
        "## Open questions",
    ):
        assert heading in report
    assert "may reflect genuine differences in interaction settings" in report
    assert "Hypothesis:" in report

    after = _hashes(snapshot_dir)
    assert {k: v for k, v in after.items() if not k.startswith("analysis/")} == before


def test_the_test_split_is_refused(tmp_path, sources):
    metadata = snapshot_rows()
    path = write_snapshot(
        RepresentationSnapshot(representations(metadata), metadata),
        tmp_path / "snapshot",
        provenance={"data": {"split": "test"}, "sampling": {"seed": 0}},
    )

    with pytest.raises(ValueError, match="never reads the test split"):
        analyze_snapshot(path, analyses=["labels"], label_sources=sources)


def test_show_changes_no_result(snapshot_dir, sources, capsys, monkeypatch):
    output = analyze_snapshot(
        snapshot_dir, analyses=["labels"], silhouette_samples=500, label_sources=sources
    )["labels"]
    before = _hashes(output)

    show_labels(output)

    printed = capsys.readouterr().out
    assert "Coverage:" in printed
    assert "instantaneous.ego_speaking" in printed
    assert _hashes(output) == before


def test_cli_passes_the_labels_revision(snapshot_dir, sources, monkeypatch):
    seen = {}

    def hub(provenance, *, labels_revision=None):
        seen["revision"] = labels_revision
        return sources

    monkeypatch.setattr(
        "turn_wm.evaluation.latent_analysis.analyze.hub_label_sources", hub
    )

    main(
        [
            "analyze-latents",
            str(snapshot_dir),
            "--analysis",
            "labels",
            "--silhouette-samples",
            "200",
            "--labels-revision",
            "later",
        ]
    )

    assert seen["revision"] == "later"
    assert (snapshot_dir / "analysis" / "labels" / "report.md").is_file()


def test_label_source_module_keeps_its_selection():
    sections = {section for section, _ in label_source_module.SELECTION}

    assert sections == set(label_source_module.SECTIONS)
