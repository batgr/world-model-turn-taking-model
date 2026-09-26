"""
Spectral analysis of extracted representations: the metrics on known
spectra, and the analysis of a snapshot directory.
"""

import json

import pyarrow.parquet as pq
import pytest
import torch

from turn_wm.evaluation.latent_analysis.analyze import analyze_snapshot
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.spectrum import (
    ALL,
    analyze_spectra,
    spectrum,
)
from turn_wm.training.metrics import effective_rank


def _gaussian(n: int, d: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _rank_one(n: int = 400, d: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    direction = torch.randn(1, d, generator=generator)

    return torch.randn(n, 1, generator=generator) * direction + 3.0


def _with_variances(variances: list[float], n: int = 2_000) -> torch.Tensor:
    """Rows whose sample covariance is exactly diag(variances)."""

    x = _gaussian(n, len(variances), seed=2).double()
    x = x - x.mean(dim=0)
    # Whiten exactly, then scale each axis.
    covariance = x.T @ x / (n - 1)
    whitened = x @ torch.linalg.inv(torch.linalg.cholesky(covariance)).T

    return whitened * torch.tensor(variances, dtype=torch.float64).sqrt()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_rank_one_rows_have_rank_one():
    summary = spectrum(_rank_one(), representation="x").summary()

    assert summary["effective_rank_singular"] == pytest.approx(1.0, abs=1e-6)
    assert summary["spectral_entropy_rank"] == pytest.approx(1.0, abs=1e-6)
    assert summary["participation_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert summary["dimensions_for_99_percent"] == 1


def test_isotropic_rows_have_a_high_rank():
    summary = spectrum(_gaussian(20_000, 16), representation="x").summary()

    assert summary["effective_rank_singular_fraction"] > 0.95
    assert summary["spectral_entropy_rank_fraction"] > 0.95
    assert summary["participation_ratio_fraction"] > 0.9


def test_singular_rank_is_the_training_metric():
    rows = _gaussian(500, 12) * torch.linspace(0.1, 3.0, 12)

    summary = spectrum(rows, representation="x").summary()

    assert summary["effective_rank_singular"] == pytest.approx(
        effective_rank(rows), rel=1e-9
    )


def test_known_spectrum():
    # Eigenvalues 4, 2, 1, 1: PR = 8^2 / (16 + 4 + 1 + 1).
    result = spectrum(_with_variances([1.0, 4.0, 1.0, 2.0]), representation="x")
    summary = result.summary()

    torch.testing.assert_close(
        result.eigenvalues, torch.tensor([4.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.explained_variance_ratio,
        torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64),
    )
    assert summary["participation_ratio"] == pytest.approx(64 / 22)
    assert summary["participation_ratio_fraction"] == pytest.approx(64 / 22 / 4)
    assert summary["total_variance"] == pytest.approx(8.0)
    # The singular values of the centred rows.
    torch.testing.assert_close(
        result.singular_values,
        torch.linalg.svdvals(_with_variances([1.0, 4.0, 1.0, 2.0])),
    )
    assert summary["cumulative_explained_variance"] == {"1": pytest.approx(0.5)}
    assert summary["dimensions_for_50_percent"] == 1
    assert summary["dimensions_for_80_percent"] == 3
    assert summary["dimensions_for_95_percent"] == 4


def test_explained_variance_sums_to_one_and_accumulates_monotonically():
    result = spectrum(_gaussian(300, 40) * torch.rand(40) * 5, representation="x")

    assert float(result.explained_variance_ratio.sum()) == pytest.approx(1.0)
    cumulative = result.cumulative_explained_variance
    assert bool((cumulative[1:] >= cumulative[:-1]).all())
    assert float(cumulative[-1]) == pytest.approx(1.0)
    assert list(result.summary()["cumulative_explained_variance"]) == [
        "1",
        "5",
        "10",
        "25",
    ]


def test_norms_are_those_of_the_uncentred_rows():
    rows = torch.tensor([[3.0, 4.0], [3.0, 4.0], [6.0, 8.0]])

    result = spectrum(rows, representation="x")

    assert result.mean_norm == pytest.approx((5 + 5 + 10) / 3)
    assert result.mean_dimension_std == pytest.approx(
        float(rows.double().std(dim=0).mean())
    )


def test_input_rows_are_not_modified():
    rows = _gaussian(100, 8) + 2.0
    original = rows.clone()

    analyze_spectra({"x": rows}, groups=["a"] * 50 + ["b"] * 50)

    assert torch.equal(rows, original)
    assert rows.dtype == torch.float32


def test_row_order_does_not_matter():
    rows = _gaussian(300, 10) * torch.linspace(0.5, 2.0, 10)
    groups = ["a"] * 100 + ["b"] * 200
    order = torch.randperm(300, generator=torch.Generator().manual_seed(3))

    ordered = analyze_spectra({"x": rows}, groups=groups)
    shuffled = analyze_spectra(
        {"x": rows[order]}, groups=[groups[i] for i in order.tolist()]
    )

    for first, second in zip(ordered, shuffled, strict=True):
        assert first.group == second.group
        torch.testing.assert_close(first.eigenvalues, second.eigenvalues)
        first_summary, second_summary = first.summary(), second.summary()
        assert first_summary.pop("cumulative_explained_variance") == pytest.approx(
            second_summary.pop("cumulative_explained_variance")
        )
        assert first_summary == pytest.approx(second_summary)


# ---------------------------------------------------------------------------
# Several representations, per group
# ---------------------------------------------------------------------------


def test_every_representation_is_analyzed():
    spectra = analyze_spectra(
        {
            "features": _gaussian(50, 12),
            "latent": _gaussian(50, 4),
            "other": _rank_one(50, 3),
        }
    )

    assert [(s.representation, s.group, s.dim) for s in spectra] == [
        ("features", ALL, 12),
        ("latent", ALL, 4),
        ("other", ALL, 3),
    ]


def test_groups_are_centred_on_their_own_mean():
    # Two isotropic corpora far apart: the mix is dominated by the offset
    # between them, each corpus alone is not.
    a = _gaussian(500, 8, seed=4)
    b = _gaussian(300, 8, seed=5) + torch.tensor([20.0] + [0.0] * 7)

    spectra = analyze_spectra(
        {"x": torch.cat([a, b])}, groups=["a"] * 500 + ["b"] * 300
    )
    by_group = {s.group: s for s in spectra}

    assert [s.group for s in spectra] == [ALL, "a", "b"]
    assert by_group[ALL].samples == 800
    assert by_group["a"].samples == 500
    assert by_group["b"].samples == 300
    assert by_group[ALL].summary()["participation_ratio"] < 2
    assert by_group["a"].summary()["participation_ratio"] > 6
    torch.testing.assert_close(
        by_group["a"].eigenvalues, spectrum(a, representation="x").eigenvalues
    )


def test_group_labels_must_cover_every_row():
    with pytest.raises(ValueError, match="2 group labels for 3 rows"):
        analyze_spectra({"x": _gaussian(3, 2)}, groups=["a", "b"])


def test_too_few_rows_are_refused():
    with pytest.raises(ValueError, match="at least 2 rows"):
        spectrum(torch.zeros(1, 3), representation="x")


# ---------------------------------------------------------------------------
# A snapshot directory
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_dir(tmp_path):
    datasets = ["egocom"] * 60 + ["ego4d"] * 40

    return write_snapshot(
        RepresentationSnapshot(
            representations={
                "features": _gaussian(100, 32, seed=6),
                "latent": _gaussian(100, 8, seed=7),
            },
            metadata={"sample_id": [f"s{i}" for i in range(100)], "dataset": datasets},
        ),
        tmp_path / "latent-snapshot",
        provenance={"run": {"run_id": "run-1"}},
    )


def test_analysis_of_a_snapshot(snapshot_dir):
    outputs = analyze_snapshot(snapshot_dir)

    output = outputs["spectrum"]
    assert output == snapshot_dir / "analysis" / "spectrum"
    assert sorted(path.name for path in output.iterdir()) == [
        "cumulative_variance.png",
        "eigenvalue_spectrum.png",
        "spectrum.parquet",
        "summary.json",
    ]

    summary = json.loads((output / "summary.json").read_text())
    assert summary["source"]["snapshot_provenance"] == {"run": {"run_id": "run-1"}}
    assert len(summary["source"]["representations_sha256"]) == 64
    assert summary["settings"]["group_column"] == "dataset"
    assert set(summary["representations"]) == {"features", "latent"}
    assert {
        group: metrics["samples"]
        for group, metrics in summary["representations"]["latent"].items()
    } == {"all": 100, "ego4d": 40, "egocom": 60}
    assert summary["representations"]["features"]["all"]["dim"] == 32

    table = pq.read_table(output / "spectrum.parquet").to_pydict()
    # (32 + 8) components, for all + 2 corpora.
    assert len(table["component"]) == 3 * (32 + 8)
    assert set(table["group"]) == {"all", "egocom", "ego4d"}
    assert set(table) == {
        "representation",
        "group",
        "component",
        "eigenvalue",
        "singular_value",
        "explained_variance_ratio",
        "cumulative_explained_variance",
    }


def test_a_snapshot_without_datasets_is_analyzed_globally(tmp_path):
    path = write_snapshot(
        RepresentationSnapshot(
            representations={"latent": _gaussian(20, 4)},
            metadata={"sample_id": [str(i) for i in range(20)]},
        ),
        tmp_path / "snapshot",
    )

    output = analyze_snapshot(path, output_root=tmp_path / "results")["spectrum"]

    summary = json.loads((output / "summary.json").read_text())
    assert output == tmp_path / "results" / "spectrum"
    assert list(summary["representations"]["latent"]) == ["all"]


def test_existing_results_are_not_overwritten(snapshot_dir):
    analyze_snapshot(snapshot_dir)

    with pytest.raises(ValueError, match="Output directory is not empty"):
        analyze_snapshot(snapshot_dir)


def test_not_a_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a representation snapshot"):
        analyze_snapshot(tmp_path)
