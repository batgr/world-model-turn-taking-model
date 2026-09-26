"""
Rollout dynamics: the validation rollout written to a trajectory snapshot,
and its skill, displacement alignment and movement ratio on stable and
changing joint-speech states (synthetic run and label sidecars, no network).
"""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_latent_analysis as runs
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from test_latent_labels import GRID_SHA, entry

from turn_wm.cli import main
from turn_wm.data.labels import local_store
from turn_wm.data.loader import DataLoaderConfig, build_dataloader
from turn_wm.evaluation.latent_analysis.analyze import analyze_snapshot
from turn_wm.evaluation.latent_analysis.extract import (
    RepresentationSnapshot,
    write_snapshot,
)
from turn_wm.evaluation.latent_analysis.label_source import (
    CorpusLabelSource,
    audit_corpus,
    join_labels,
)
from turn_wm.evaluation.latent_analysis.pca import pca_2d
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
    extract_rollout_run,
    rollout_provenance,
    rollout_representations,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    MOVEMENT,
    SELECTION,
    SKILL,
    cluster_bootstrap_weights,
    condition_metrics,
    rollout_conditions,
    row_terms,
    trajectory_selection,
    true_latent_projection,
    write_rollout_dynamics,
)
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.lewm import LeWMModule, lejepa_forward, trajectories
from turn_wm.training.train import build_run_dataset, prepare_observations

REVISION = runs.REVISION
_config, _loaded, _make_run = runs._config, runs._loaded, runs._make_run
# The synthetic run's fixtures.
cache_root = runs.cache_root
loads = runs.loads

# ---------------------------------------------------------------------------
# The rollout is the validation forward
# ---------------------------------------------------------------------------


@pytest.fixture
def validation_batch(cache_root):
    cfg = _config(cache_root)
    loaded = _loaded()
    dataset = build_run_dataset(
        cfg,
        loaded,
        prepare_observations(cfg, loaded),
        split="validation",
        training=False,
    )
    batch = next(iter(build_dataloader(dataset, loader=DataLoaderConfig(batch_size=4))))
    torch.manual_seed(0)

    return cfg, LeWMModule(cfg).eval(), batch


def test_rollout_tensors_are_the_validation_forward(validation_batch):
    cfg, module, batch = validation_batch
    trajectory = trajectories(batch)

    with torch.no_grad():
        got = rollout_representations(
            module.model, trajectory, sigreg=module.sigreg, cfg=cfg
        )
        output = lejepa_forward(module.model, module.sigreg, trajectory, cfg)

    c = cfg.data.context_steps
    horizons = sorted(cfg.prediction.rollout_horizons)
    assert torch.equal(got[ANCHOR_LATENT], output.latents[:, c - 1])

    for k, h in enumerate(horizons):
        assert torch.equal(got[TRUE_FUTURE_LATENT][:, k], output.latents[:, c + h - 1])
        assert torch.equal(got[PRED_FUTURE_LATENT][:, k], output.rollout_predictions[h])

    # The actions the rollout read: trajectory steps 0 .. C + max(h) - 2.
    assert torch.equal(
        got[ROLLOUT_ACTION_IDS], trajectory.actions[:, : c + max(horizons) - 1]
    )


def test_future_actions_condition_exactly_the_documented_horizons(validation_batch):
    cfg, module, batch = validation_batch
    c = cfg.data.context_steps
    provenance = rollout_provenance(cfg)

    # AdaLN-zero starts with no action conditioning: give it some.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))

    def predictions(step):
        changed = dict(batch)
        actions = torch.cat([batch["context_action"], batch["future_action"]], 1)
        # Another action at trajectory step `step` (a future step: >= C).
        future = batch["future_action"].clone()
        future[:, step - c] = (actions[:, step] + 1) % 4
        changed["future_action"] = future

        with torch.no_grad():
            return rollout_representations(
                module.model, trajectories(changed), sigreg=SIGReg(), cfg=cfg
            )[PRED_FUTURE_LATENT]

    with torch.no_grad():
        base = rollout_representations(
            module.model, trajectories(batch), sigreg=SIGReg(), cfg=cfg
        )[PRED_FUTURE_LATENT]

    for step in (c, c + 4):
        changed = predictions(step)

        for k, h in enumerate(provenance["horizons_steps"]):
            start, stop = provenance["action_steps_by_horizon"][str(h)]
            read = start <= step < stop
            assert torch.equal(changed[:, k], base[:, k]) != read, (step, h)

    assert provenance["conditioned_on_ground_truth_future_actions"] == {
        "1": False,
        "5": True,
        "10": True,
    }
    assert provenance["future_action_tokens_by_horizon"] == {"1": 0, "5": 4, "10": 9}
    assert provenance["horizons_s"] == pytest.approx([0.1, 0.5, 1.0])


# ---------------------------------------------------------------------------
# The trajectory snapshot of a run
# ---------------------------------------------------------------------------


def _hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_run_writes_a_deterministic_validation_rollout(tmp_path, cache_root, loads):
    cfg = _config(cache_root)
    run_dir, _ = _make_run(tmp_path, cfg)

    output = extract_rollout_run(run_dir, max_samples=9, batch_size=4)
    again = extract_rollout_run(
        run_dir, output_dir=tmp_path / "again", max_samples=9, batch_size=2
    )

    assert output == run_dir / "latent_analysis" / "last-validation-rollout"
    tensors = load_file(output / "representations.safetensors")
    metadata = pq.read_table(output / "metadata.parquet").to_pydict()
    manifest = json.loads((output / "manifest.json").read_text())
    c = cfg.data.context_steps

    assert tensors[ANCHOR_LATENT].shape == (9, 192)
    assert tensors[TRUE_FUTURE_LATENT].shape == (9, 3, 192)
    assert tensors[PRED_FUTURE_LATENT].shape == (9, 3, 192)
    assert tensors[ROLLOUT_ACTION_IDS].shape == (9, c + 10 - 1)
    assert tensors[ROLLOUT_ACTION_IDS].dtype == torch.int64
    assert all(":validation#" in s for s in metadata["sample_id"])
    assert manifest["provenance"]["data"]["split"] == "validation"
    assert manifest["provenance"]["data"]["dataset_revision"] == REVISION
    assert manifest["provenance"]["rollout"]["horizons_steps"] == [1, 5, 10]
    # Same anchors, same values, whatever the batch size.
    for name in ("representations.safetensors", "metadata.parquet"):
        assert _hashes(again)[name] == _hashes(output)[name]

    # The anchor-latent analysis does not take a rollout snapshot.
    with pytest.raises(ValueError, match="analyze-rollouts"):
        analyze_snapshot(output)


# ---------------------------------------------------------------------------
# Analysis on a synthetic rollout snapshot
# ---------------------------------------------------------------------------

STATES = ("silence", "ego_only", "others_only", "both")
ROWS = 40
HORIZONS_S = (0.1, 0.5, 1.0)

LABEL_REGISTRY = {
    "registry_version": 1,
    "families": {},
    "labels": [
        entry(
            "instantaneous.joint_speech_state_occupancy",
            "list<float32> (4 values)",
            "[joint_state]",
        ),
        entry(
            "future.future_joint_speech_state",
            "fixed_size_list<int8, H>",
            "[horizon]",
        ),
    ],
}


def _recording(k: int) -> str:
    # Four recordings of ten consecutive anchors.
    return f"r{k // 10}"


def _label_sources(tmp_path, horizons=HORIZONS_S):
    root = tmp_path / "egocom" / "labels"
    (root / "speech").mkdir(parents=True)
    rows = []

    for k in range(ROWS):
        now = k % 4
        rows.append(
            {
                "recording_id": _recording(k),
                "decision_index": k,
                "decision_time_s": k / 10,
                "joint_speech_state_occupancy": [
                    1.0 if s == now else 0.0 for s in range(4)
                ],
                # Odd anchors change state at every horizon.
                "future_joint_speech_state": [
                    (now + 1) % 4 if k % 2 else now for _ in horizons
                ],
            }
        )

    pq.write_table(pa.Table.from_pylist(rows), root / "speech" / "grid.parquet")
    (root / "registry.json").write_text(json.dumps(LABEL_REGISTRY))
    (root / "speech" / "manifest.json").write_text(
        json.dumps(
            {
                "materialized_labels": [e["name"] for e in LABEL_REGISTRY["labels"]],
                "unavailable_labels": {},
                "tables": {"grid": {"file": "grid.parquet"}},
                "inputs": {"action_grid": {"sha256": GRID_SHA}},
                "config": {"future_horizons_s": list(horizons)},
            }
        )
    )

    return {
        "egocom": CorpusLabelSource(
            corpus="egocom",
            fetch=local_store(root),
            grid_sha256=GRID_SHA,
            provenance={"repo_id": "local", "labels_revision": "rev"},
        )
    }


def _rollout_snapshot(tmp_path, *, split="validation"):
    """Transitions: pred == true. Stable: pred == anchor (persistence)."""

    generator = torch.Generator().manual_seed(0)
    anchor = torch.randn(ROWS, 8, generator=generator)
    true = anchor[:, None] + torch.randn(ROWS, 3, 8, generator=generator)
    transition = torch.tensor([k % 2 == 1 for k in range(ROWS)])
    pred = torch.where(transition[:, None, None], true, anchor[:, None].expand_as(true))
    rollout = rollout_provenance(
        OmegaConf.create(
            {
                "data": {"context_steps": 15},
                "prediction": {
                    "rollout_horizons": [1, 5, 10],
                    "rollout_context_size": 10,
                    "rollout_stop_gradient": True,
                },
            }
        )
    )
    # An ONSET at t+1 (trajectory step 15) on every other transition row:
    # read for 0.5 s and 1 s, not for 0.1 s.
    actions = torch.zeros(ROWS, 24, dtype=torch.int64)
    actions[[k for k in range(ROWS) if k % 4 == 1], 15] = 1
    metadata = {
        "sample_id": [f"egocom:{_recording(k)}#{k}" for k in range(ROWS)],
        "dataset": ["egocom"] * ROWS,
        "recording_id": [_recording(k) for k in range(ROWS)],
        "anchor_idx": list(range(ROWS)),
        "anchor_time": [float(torch.tensor(k / 10)) for k in range(ROWS)],
    }

    return write_snapshot(
        RepresentationSnapshot(
            representations={
                ANCHOR_LATENT: anchor,
                TRUE_FUTURE_LATENT: true,
                PRED_FUTURE_LATENT: pred,
                ROLLOUT_ACTION_IDS: actions,
            },
            metadata=metadata,
        ),
        tmp_path / "rollout-snapshot",
        provenance={
            "run": {"run_id": "run-1"},
            "data": {"dataset": "egocom", "dataset_revision": "rev", "split": split},
            "sampling": {"seed": 3072},
            "checkpoint": {"filename": "last.ckpt", "global_step": 1},
            "extraction": {"precision": "float32"},
            "rollout": rollout,
        },
    )


def test_metrics_per_condition(tmp_path):
    snapshot = _rollout_snapshot(tmp_path)
    before = _hashes(snapshot)

    output = write_rollout_dynamics(
        snapshot, label_sources=_label_sources(tmp_path), bootstrap=200
    )

    # The snapshot itself is only read.
    after = _hashes(snapshot)
    assert {k: after[k] for k in before} == before
    assert sorted(p.name for p in output.iterdir()) == [
        "conditions.parquet",
        "figures",
        "metrics.parquet",
        "report.md",
        "summary.json",
        "trajectories.parquet",
    ]
    assert sorted(p.name for p in (output / "figures").iterdir()) == [
        "displacement_vs_horizon.png",
        "skill_vs_horizon.png",
        "transition_trajectories.png",
    ]
    summary = json.loads((output / "summary.json").read_text())

    for h in ("1", "5", "10"):
        metrics = summary["metrics"][h]
        assert metrics["all"]["n"] == ROWS
        assert metrics["transition"]["n"] == metrics["stable"]["n"] == ROWS // 2
        # Perfect on transitions.
        assert metrics["transition"][SKILL] == pytest.approx(1.0)
        assert metrics["transition"][ALIGNMENT] == pytest.approx(1.0)
        assert metrics["transition"][MOVEMENT] == pytest.approx(1.0)
        # Persistence on stable rows: no skill, no motion, no direction.
        assert metrics["stable"][SKILL] == pytest.approx(0.0)
        assert metrics["stable"][MOVEMENT] == pytest.approx(0.0)
        assert metrics["stable"][ALIGNMENT] is None
        assert metrics["stable"]["n_direction_defined"] == 0
        assert metrics["transition"]["direction_defined_fraction"] == 1.0
        low, high = metrics["all"]["skill_ci"]
        assert low <= metrics["all"][SKILL] <= high
        assert metrics["all"]["n_recordings"] == 4
        # Confounding diagnostic: the ONSET at t+1 is read from 0.5 s on.
        read = h != "1"
        assert metrics["transition"]["future_event_fraction"] == (0.5 if read else 0)
        assert metrics["stable"]["future_event_fraction"] == 0
        assert metrics["all"]["n_future_event"] == (10 if read else 0)

    assert summary["settings"]["cluster"] == "(dataset, recording_id)"

    conditions = pq.read_table(output / "conditions.parquet").to_pydict()
    assert conditions["condition@1s"] == [
        "transition" if k % 2 else "stable" for k in range(ROWS)
    ]
    assert conditions["current_state"][:4] == list(STATES)

    report = (output / "report.md").read_text()
    assert (
        "**Is the rollout conditioned on ground-truth future action/event tokens? "
        "Yes.**" in report
    )
    assert "on **transition** rows the rollout beats persistence" in report
    assert "evolve the latent state through the corresponding" in report
    assert "| 0.5 s | 4 | 25.0% (10/40) | 0.0% (0/20) | 50.0% (10/20) |" in report
    assert "recordings, not anchors, are resampled" in report

    # Deterministic: the same results again.
    again = write_rollout_dynamics(
        snapshot,
        output_dir=tmp_path / "again",
        label_sources=_label_sources(tmp_path / "labels-again"),
        bootstrap=200,
    )
    assert (
        json.loads((again / "summary.json").read_text())["metrics"]
        == summary["metrics"]
    )


def test_trajectory_pca_is_fitted_on_true_latents_only(tmp_path):
    snapshot = _rollout_snapshot(tmp_path)
    tensors = load_file(snapshot / "representations.safetensors")
    projection = true_latent_projection(tensors)
    true = torch.cat(
        [tensors[ANCHOR_LATENT], tensors[TRUE_FUTURE_LATENT].flatten(0, 1)]
    ).double()

    assert torch.allclose(projection.components, pca_2d(true).components)
    # Predictions do not move the projection.
    moved = dict(tensors, **{PRED_FUTURE_LATENT: tensors[PRED_FUTURE_LATENT] * 100})
    assert torch.equal(true_latent_projection(moved).components, projection.components)

    output = write_rollout_dynamics(
        snapshot, label_sources=_label_sources(tmp_path), bootstrap=10
    )
    table = pq.read_table(output / "trajectories.parquet").to_pydict()
    samples = set(table["sample_id"])
    # Only transitions are drawn, true and predicted paths of 4 points each.
    assert all(int(s.split("#")[1]) % 2 == 1 for s in samples)
    assert len(table["sample_id"]) == len(samples) * 2 * 4
    first = table["sample_id"][0]
    rows = [i for i, s in enumerate(table["sample_id"]) if s == first]
    point = true[int(first.split("#")[1])]
    expected = projection.transform(point[None])[0]
    anchor_row = next(
        i for i in rows if table["path"][i] == "true" and table["horizon_steps"][i] == 0
    )
    assert table["pc1"][anchor_row] == pytest.approx(float(expected[0]))


def test_a_missing_label_horizon_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no 0.5 s horizon"):
        write_rollout_dynamics(
            _rollout_snapshot(tmp_path),
            label_sources=_label_sources(tmp_path, horizons=(0.1, 1.0)),
        )


def test_only_validation_rollout_snapshots_are_analysed(tmp_path):
    with pytest.raises(ValueError, match="validation split only"):
        write_rollout_dynamics(
            _rollout_snapshot(tmp_path, split="test"),
            label_sources=_label_sources(tmp_path),
        )

    anchors = write_snapshot(
        RepresentationSnapshot(
            representations={"latent": torch.zeros(2, 3)},
            metadata={"sample_id": ["a", "b"]},
        ),
        tmp_path / "anchors",
    )
    with pytest.raises(ValueError, match="not a rollout snapshot"):
        write_rollout_dynamics(anchors, label_sources={})


def test_cli_analyze_rollouts(tmp_path, monkeypatch, capsys):
    snapshot = _rollout_snapshot(tmp_path)
    sources = _label_sources(tmp_path)
    monkeypatch.setattr(
        "turn_wm.evaluation.latent_analysis.rollout_dynamics.hub_label_sources",
        lambda provenance, labels_revision=None: sources,
    )

    assert main(["analyze-rollouts", str(snapshot), "--bootstrap", "20"]) == 0
    assert (snapshot / "analysis" / "rollout_dynamics" / "report.md").is_file()
    assert "rollout_dynamics:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_resamples_recordings_within_corpora():
    clusters = ["a/r1", "a/r2", "a/r3", "b/r1"]
    strata = ["a", "a", "a", "b"]

    weights = cluster_bootstrap_weights(
        clusters, strata, resamples=500, generator=torch.Generator().manual_seed(0)
    )

    # Each corpus keeps its number of recordings in every resample.
    assert torch.equal(weights[:, :3].sum(1), torch.full((500,), 3.0))
    assert torch.equal(weights[:, 3], torch.ones(500))
    assert (weights[:, :3] == 0).any()  # with replacement


def test_intervals_are_over_recordings_not_anchors():
    # One recording with many anchors: no between-recording variation to
    # resample, however many anchors.
    generator = torch.Generator().manual_seed(0)
    anchor = torch.randn(200, 4, generator=generator)
    true = anchor + torch.randn(200, 4, generator=generator)
    pred = anchor + 0.5 * (true - anchor)
    terms = row_terms(anchor, true, pred)
    members = torch.ones(200, dtype=torch.bool)

    single = condition_metrics(
        terms,
        members,
        recordings=["x/r1"] * 200,
        corpora=["x"] * 200,
        bootstrap=100,
        generator=generator,
    )
    assert single["n_recordings"] == 1 and single["skill_ci"] is None

    # Recordings of very different skill: wider than the i.i.d. interval.
    pred[:100] = true[:100]
    terms = row_terms(anchor, true, pred)
    clustered = condition_metrics(
        terms,
        members,
        recordings=[f"x/r{i // 100}" for i in range(200)],
        corpora=["x"] * 200,
        bootstrap=400,
        generator=torch.Generator().manual_seed(1),
    )
    rows = condition_metrics(
        terms,
        members,
        recordings=[f"x/r{i}" for i in range(200)],
        corpora=["x"] * 200,
        bootstrap=400,
        generator=torch.Generator().manual_seed(1),
    )
    width = lambda m: m["skill_ci"][1] - m["skill_ci"][0]
    assert clustered[SKILL] == pytest.approx(rows[SKILL])
    assert width(clustered) > 2 * width(rows)


def test_near_zero_true_displacements_are_excluded_from_the_alignment():
    anchor = torch.zeros(4, 2)
    true = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1e-4, 0.0]])
    # The last row points the wrong way, but its true motion is ~0.
    pred = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [-1.0, 0.0]])
    terms = row_terms(anchor, true, pred)

    values = condition_metrics(
        terms,
        torch.ones(4, dtype=torch.bool),
        recordings=["x/r1", "x/r2", "x/r3", "x/r4"],
        corpora=["x"] * 4,
        bootstrap=50,
        generator=torch.Generator().manual_seed(0),
    )

    assert values["n_direction_defined"] == 3
    assert values["direction_defined_fraction"] == 0.75
    # Excluded, not counted as 0 (which would give 0.75) or -1.
    assert values[ALIGNMENT] == pytest.approx(1.0)


def test_trajectory_selection_is_deterministic_and_order_free(tmp_path):
    sources = _label_sources(tmp_path)
    snapshot = _rollout_snapshot(tmp_path)
    output = write_rollout_dynamics(snapshot, label_sources=sources, bootstrap=10)
    summary = json.loads((output / "summary.json").read_text())
    chosen = summary["trajectories"]["samples"]

    joined = join_labels(
        pq.read_table(snapshot / "metadata.parquet").to_pydict(),
        {c: audit_corpus(s) for c, s in sources.items()},
        sources,
        selection=SELECTION,
    )
    conditions = rollout_conditions(joined.variables, {10: 1.0})
    ids = [f"egocom:{_recording(k)}#{k}" for k in range(ROWS)]
    order = list(reversed(range(ROWS)))
    reversed_conditions = type(conditions)(
        current=[conditions.current[i] for i in order],
        future={10: [conditions.future[10][i] for i in order]},
        condition={10: [conditions.condition[10][i] for i in order]},
    )

    first = trajectory_selection(ids, conditions, 10, count=6, seed=3072)
    again = trajectory_selection(
        [ids[i] for i in order], reversed_conditions, 10, count=6, seed=3072
    )

    # Same samples whatever the row order; the written figure used them.
    assert [ids[i] for i in first] == [ids[order[i]] for i in again] == chosen
    assert trajectory_selection(ids, conditions, 10, count=6, seed=1) != first
