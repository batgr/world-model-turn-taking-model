"""
PCA analysis of extracted representations: the metrics on synthetic groups,
the written results, and their display.
"""

import hashlib
import json
import subprocess
import sys
import types

import pyarrow.parquet as pq
import pytest
import torch

from turn_wm.cli import main
from turn_wm.evaluation.latent_analysis.analyze import analyze_snapshot
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.pca import (
    analyze_pca,
    group_structure,
    pca_2d,
    plot_selection,
    sample_keys,
    silhouette,
)
from turn_wm.evaluation.latent_analysis.show import show_pca


def _normal(n: int, d: int, seed: int) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _ids(n: int) -> list[str]:
    return [f"s{i}" for i in range(n)]


def _structure(x, labels, *, silhouette_samples=10_000):
    return group_structure(
        x,
        labels,
        keys=sample_keys(_ids(len(labels)), seed=0),
        grouping="dataset",
        condition="all",
        silhouette_samples=silhouette_samples,
    )


# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------


def test_identical_domains_are_not_separated():
    x = _normal(4_000, 8, seed=0)
    labels = ["a", "b"] * 2_000

    summary = _structure(x, labels).summary()

    assert summary["centroid_distance_over_pooled_within_rms"] < 0.05
    assert summary["between_variance_fraction"] < 0.001
    assert abs(summary["silhouette"]) < 0.02


def test_shifted_domains_are_separated():
    shift = torch.zeros(8)
    shift[0] = 6.0
    x = torch.cat([_normal(1_000, 8, seed=1), _normal(1_000, 8, seed=2) + shift])
    labels = ["a"] * 1_000 + ["b"] * 1_000

    summary = _structure(x, labels).summary()

    # Pooled within variance ~8 (unit variance in 8 dims): distance ~6.
    assert summary["centroid_distance"] == pytest.approx(6.0, rel=0.05)
    assert summary["centroid_distance_over_pooled_within_rms"] == pytest.approx(
        6.0 / 8**0.5, rel=0.05
    )
    assert summary["between_to_within_variance_ratio"] == pytest.approx(9 / 8, rel=0.1)
    assert summary["silhouette"] > 0.3


def test_variance_decomposition_is_exact():
    x = _normal(300, 5, seed=3) * torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0])
    labels = ["a"] * 100 + ["b"] * 150 + ["c"] * 50

    summary = _structure(x, labels).summary()
    x64 = x.double()
    total = float((x64 - x64.mean(dim=0)).pow(2).sum(dim=1).mean())

    assert summary["pooled_within_variance"] + summary[
        "between_variance"
    ] == pytest.approx(total)
    # Three groups: every pair, no single distance.
    assert summary["centroid_distance"] is None
    assert len(summary["centroid_distances"]) == 3


def test_row_order_does_not_change_the_metrics():
    x = _normal(600, 6, seed=4) + torch.tensor([[2.0] + [0.0] * 5])
    datasets = ["a"] * 250 + ["b"] * 350
    actions = (["ONSET", "NO_EVENT", "NO_EVENT"] * 200)[:600]
    order = torch.randperm(600, generator=torch.Generator().manual_seed(5)).tolist()

    def run(rows):
        return analyze_pca(
            {"x": x[rows]},
            {
                "sample_id": [_ids(600)[i] for i in rows],
                "dataset": [datasets[i] for i in rows],
                "action": [actions[i] for i in rows],
            },
            seed=1,
            silhouette_samples=200,
        )

    first, second = run(list(range(600))), run(order)
    a, b = first.representations[0], second.representations[0]

    assert a.projection.summary() == pytest.approx(b.projection.summary())
    assert a.dataset.silhouette == pytest.approx(b.dataset.silhouette)
    assert a.dataset.between_variance == pytest.approx(b.dataset.between_variance)

    for condition in a.actions:
        assert a.actions[condition].silhouette == pytest.approx(
            b.actions[condition].silhouette
        )

    # The plotted rows are the same samples.
    plotted_first = {_ids(600)[i] for i in first.plotted.nonzero().flatten().tolist()}
    plotted_second = {
        _ids(600)[order[i]] for i in second.plotted.nonzero().flatten().tolist()
    }
    assert plotted_first == plotted_second


# ---------------------------------------------------------------------------
# Action structure
# ---------------------------------------------------------------------------


def _action_data():
    """Actions clustered in corpus a, mixed in corpus b; ONSET is rare."""

    generator = torch.Generator().manual_seed(6)
    rows, datasets, actions = [], [], []

    for dataset, spread in (("a", 0.2), ("b", 5.0)):
        for action, count, centre in (("NO_EVENT", 400, -1.0), ("ONSET", 40, 1.0)):
            centre_row = torch.zeros(4)
            centre_row[0] = centre
            rows.append(
                centre_row + spread * torch.randn(count, 4, generator=generator)
            )
            datasets += [dataset] * count
            actions += [action] * count

    return torch.cat(rows), datasets, actions


def test_action_silhouette_globally_and_within_each_dataset():
    x, datasets, actions = _action_data()

    analysis = analyze_pca(
        {"x": x},
        {"sample_id": _ids(len(x)), "dataset": datasets, "action": actions},
        seed=0,
    )
    structures = analysis.representations[0].actions

    assert list(structures) == ["all", "a", "b"]
    assert structures["a"].silhouette > 0.7
    assert structures["b"].silhouette < 0.2
    assert structures["b"].silhouette < structures["all"].silhouette
    assert structures["a"].samples == 440


def test_imbalanced_classes_are_kept_as_they_are():
    x, datasets, actions = _action_data()

    structure = (
        analyze_pca(
            {"x": x},
            {"sample_id": _ids(len(x)), "dataset": datasets, "action": actions},
            seed=0,
            silhouette_samples=300,
        )
        .representations[0]
        .actions["all"]
    )

    assert structure.counts == {"NO_EVENT": 800, "ONSET": 80}
    assert structure.fractions["ONSET"] == pytest.approx(80 / 880)
    assert structure.silhouette_samples == 300


def test_a_class_too_small_for_the_silhouette_is_reported():
    x = _normal(50, 3, seed=7)
    labels = ["NO_EVENT"] * 49 + ["ONSET"]

    structure = _structure(x, labels)

    assert structure.silhouette is None
    assert structure.silhouette_undefined_reason == (
        "classes with fewer than 2 samples: ['ONSET']"
    )
    assert structure.counts == {"NO_EVENT": 49, "ONSET": 1}


def test_a_single_class_has_no_silhouette():
    value, by_class, reason = silhouette(_normal(10, 2, seed=8), ["a"] * 10)

    assert value is None
    assert by_class == {}
    assert reason == "fewer than 2 classes"


def test_silhouette_matches_its_definition():
    x = _normal(40, 3, seed=9)
    labels = ["a"] * 15 + ["b"] * 10 + ["c"] * 15
    distances = torch.cdist(x.double(), x.double())
    expected = []

    for i, own in enumerate(labels):
        same = [j for j, label in enumerate(labels) if label == own and j != i]
        a = distances[i, same].mean()
        b = min(
            distances[i, [j for j, label in enumerate(labels) if label == other]].mean()
            for other in set(labels) - {own}
        )
        expected.append(float((b - a) / max(a, b)))

    value, _, _ = silhouette(x, labels)

    assert value == pytest.approx(sum(expected) / len(expected))


# ---------------------------------------------------------------------------
# PCA and plotting sample
# ---------------------------------------------------------------------------


def test_pca_explains_the_expected_variance():
    x = _normal(20_000, 4, seed=10) * torch.tensor([3.0, 2.0, 1.0, 1.0])

    summary = pca_2d(x).summary()

    assert summary["pc1_explained_variance"] == pytest.approx(9 / 15, abs=0.01)
    assert summary["pc2_explained_variance"] == pytest.approx(4 / 15, abs=0.01)
    assert summary["pc1_pc2_cumulative"] == pytest.approx(13 / 15, abs=0.01)


def test_pca_does_not_standardize_or_modify_the_rows():
    x = _normal(500, 3, seed=11) * torch.tensor([10.0, 1.0, 1.0]) + 5.0
    original = x.clone()

    projection = pca_2d(x)

    assert torch.equal(x, original)
    # The large-scale axis dominates: no per-dimension standardization.
    assert projection.components[:, 0].abs().argmax() == 0
    assert projection.coordinates.mean(dim=0).abs().max() < 1e-9


def test_rare_groups_stay_in_the_plotting_sample():
    keys = sample_keys(_ids(10_000), seed=0)
    strata = [("a", "ONSET")] * 30 + [("a", "NO_EVENT")] * 9_970

    plotted = plot_selection(keys, strata=strata, max_samples=1_000, group_floor=20)

    assert int(plotted[:30].sum()) >= 20
    assert 1_000 <= int(plotted.sum()) <= 1_020


# ---------------------------------------------------------------------------
# Written results and display
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_dir(tmp_path):
    x, datasets, actions = _action_data()
    n = len(x)

    return write_snapshot(
        RepresentationSnapshot(
            representations={
                "features": torch.cat([x, _normal(n, 12, seed=12)], dim=1),
                "latent": x,
                "other": _normal(n, 3, seed=13),
            },
            metadata={
                "sample_id": _ids(n),
                "dataset": datasets,
                "action_id": [0 if a == "NO_EVENT" else 1 for a in actions],
                "action": actions,
                "sample_class": ["event"] * n,
            },
        ),
        tmp_path / "latent-snapshot",
        provenance={"sampling": {"seed": 3072}, "run": {"run_id": "run-1"}},
    )


def _hashes(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_written_pca_results(snapshot_dir):
    snapshot_before = _hashes(snapshot_dir)

    output = analyze_snapshot(snapshot_dir, analyses=["pca"])["pca"]

    assert output == snapshot_dir / "analysis" / "pca"
    assert sorted(p.name for p in output.iterdir()) == [
        "action_distribution.parquet",
        "coordinates.parquet",
        "figures",
        "group_metrics.parquet",
        "report.md",
        "summary.json",
    ]
    assert sorted(p.name for p in (output / "figures").iterdir()) == sorted(
        f"{name}_{view}.png"
        for name in ("features", "latent", "other")
        for view in ("dataset", "action", "action_a", "action_b")
    )

    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["representations"]) == {"features", "latent", "other"}
    assert summary["settings"]["seed"] == 3072
    assert summary["settings"]["samples"] == 880
    latent = summary["representations"]["latent"]
    assert set(latent["action"]) == {"all", "a", "b"}
    assert latent["action"]["a"]["counts"] == {"NO_EVENT": 400, "ONSET": 40}
    assert summary["action_distribution"]["b"]["fractions"]["ONSET"] == 40 / 440

    coordinates = pq.read_table(output / "coordinates.parquet").to_pydict()
    assert len(coordinates["pc1"]) == 3 * 880
    assert {"sample_id", "dataset", "action", "sample_class", "plotted"} <= set(
        coordinates
    )

    report = (output / "report.md").read_text()
    assert report.startswith("# PCA analysis")
    assert "not interpreted as inherently undesirable" in report
    assert "stronger within a than within b" in report
    assert "## Open questions" in report

    # The snapshot itself is only read.
    assert {
        path: digest
        for path, digest in _hashes(snapshot_dir).items()
        if not path.startswith("analysis/")
    } == snapshot_before


def test_show_changes_no_result(snapshot_dir, tmp_path, capsys):
    main(["analyze-latents", str(snapshot_dir), "--analysis", "pca"])
    main(
        [
            "analyze-latents",
            str(snapshot_dir),
            "--analysis",
            "pca",
            "--show",
            "--output",
            str(tmp_path / "shown"),
        ]
    )

    plain = snapshot_dir / "analysis" / "pca"
    shown = tmp_path / "shown" / "pca"
    assert _hashes(plain) == _hashes(shown)

    # Without a notebook: the table as text, and where the figures are.
    printed = capsys.readouterr().out
    assert "action silhouette | a" in printed
    assert str(shown / "figures" / "latent_action_a.png") in printed


def test_show_in_a_notebook_displays_the_table_then_the_figures(
    snapshot_dir, monkeypatch
):
    output = analyze_snapshot(snapshot_dir, analyses=["pca"])["pca"]
    before = _hashes(output)
    shown = []

    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: object()
    display = types.ModuleType("IPython.display")
    display.display = shown.append
    display.HTML = lambda text: ("html", text)
    display.Image = lambda filename: ("image", filename.rsplit("/", 1)[-1])
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)

    show_pca(output)

    assert shown[0][0] == "html" and "dataset silhouette" in shown[0][1]
    images = [item[1] for item in shown if item[0] == "image"]
    assert images[:3] == [
        "features_dataset.png",
        "latent_dataset.png",
        "other_dataset.png",
    ]
    assert images[3:6] == [
        "features_action.png",
        "latent_action.png",
        "other_action.png",
    ]
    assert images[6] == "features_action_a.png"
    assert _hashes(output) == before


def test_analysis_runs_without_ipython(snapshot_dir):
    code = (
        "import sys; sys.modules['IPython'] = None\n"
        "from pathlib import Path\n"
        "from turn_wm.evaluation.latent_analysis.analyze import analyze_snapshot\n"
        f"analyze_snapshot(Path({str(snapshot_dir)!r}), analyses=['pca'])\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)

    assert (snapshot_dir / "analysis" / "pca" / "summary.json").is_file()


def test_the_display_layer_needs_no_training_dependencies():
    code = (
        "import sys\n"
        "for name in ('torch', 'lightning', 'matplotlib', 'IPython'):\n"
        "    sys.modules[name] = None\n"
        "from turn_wm.evaluation.latent_analysis.show import show_pca\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)
