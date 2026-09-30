"""
Action/event ablation: the conditions change only the action tensor of the
validation rollout, the shuffle is deterministic and group-preserving, and
the analysis compares conditions on identical rows (synthetic run, no
network).
"""

import json
import sys
import types

import pyarrow.parquet as pq
import pytest
import test_latent_analysis as runs
import test_rollout_dynamics as rollouts
import torch
from safetensors.torch import load_file

from turn_wm.cli import main
from turn_wm.data.dataset import ACTION_TO_ID, MASKED_ACTION_ID
from turn_wm.evaluation.latent_analysis.action_ablation import (
    CONDITIONS,
    COUNTERFACTUAL_NEXT,
    FOCAL_STATE,
    FORCED_ACTIONS,
    STATE_PRESERVING,
    OBSERVED,
    PRED,
    SHUFFLE_DONOR,
    SHUFFLED,
    SHUFFLED_ACTION_IDS,
    ablation_representations,
    extract_action_ablation_run,
    forced_anchor_action,
    state_preserving_actions,
    shuffle_donors,
    shuffled_actions,
)
from turn_wm.evaluation.latent_analysis.action_ablation_analysis import (
    paired_metrics,
    write_action_ablation,
)
from turn_wm.evaluation.latent_analysis.rollout import (
    ANCHOR_LATENT,
    PRED_FUTURE_LATENT,
    ROLLOUT_ACTION_IDS,
    TRUE_FUTURE_LATENT,
    rollout_representations,
)
from turn_wm.evaluation.latent_analysis.rollout_dynamics import (
    ALIGNMENT,
    row_terms,
)
from turn_wm.evaluation.latent_analysis.show import show_action_ablation
from turn_wm.models.lewm.sigreg import SIGReg
from turn_wm.training.lewm import trajectories

# The synthetic run's fixtures.
cache_root = runs.cache_root
loads = runs.loads
validation_batch = rollouts.validation_batch


def _conditioned(module):
    # AdaLN-zero starts with no action conditioning: give it some.
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))


def _ablation(cfg, module, batch, *, donor_future=None):
    trajectory = trajectories(batch)
    rows = len(batch["sample_id"])
    c = trajectory.context_steps

    with torch.no_grad():
        return trajectory, ablation_representations(
            module.model,
            trajectory,
            sigreg=SIGReg(),
            cfg=cfg,
            focal_state=batch["context_state"][:, c - 1],
            donor_future=(
                batch["future_action"].roll(1, dims=0)
                if donor_future is None
                else donor_future
            ),
            donor_rows=torch.arange(rows).roll(1),
        )


# ---------------------------------------------------------------------------
# Conditions on the rollout
# ---------------------------------------------------------------------------


def test_observed_is_the_existing_rollout_and_all_agree_at_one_step(validation_batch):
    cfg, module, batch = validation_batch
    _conditioned(module)
    trajectory, result = _ablation(cfg, module, batch)

    with torch.no_grad():
        existing = rollout_representations(
            module.model, trajectory, sigreg=SIGReg(), cfg=cfg
        )

    assert torch.equal(result[PRED[OBSERVED]], existing[PRED_FUTURE_LATENT])
    assert torch.equal(result[ANCHOR_LATENT], existing[ANCHOR_LATENT])
    assert torch.equal(result[TRUE_FUTURE_LATENT], existing[TRUE_FUTURE_LATENT])
    assert torch.equal(result[ROLLOUT_ACTION_IDS], existing[ROLLOUT_ACTION_IDS])

    # 0.1 s reads no future token: every condition identical there ...
    for condition in CONDITIONS:
        assert torch.equal(result[PRED[condition]][:, 0], result[PRED[OBSERVED]][:, 0])
    # ... and the ablations do reach the later horizons.
    for condition in (STATE_PRESERVING, SHUFFLED):
        assert not torch.equal(
            result[PRED[condition]][:, 1:], result[PRED[OBSERVED]][:, 1:]
        )


def test_state_preserving_changes_only_the_future_tokens(validation_batch):
    cfg, _, batch = validation_batch
    actions = trajectories(batch).actions
    c = cfg.data.context_steps

    ablated = state_preserving_actions(actions, c)

    assert torch.equal(ablated[:, :c], actions[:, :c])  # context and anchor action

    anchor = actions[:, c - 1]
    expected = torch.full_like(anchor, MASKED_ACTION_ID)
    silent = (anchor == ACTION_TO_ID["WAIT"]) | (anchor == ACTION_TO_ID["STOP"])
    speaking = (anchor == ACTION_TO_ID["START"]) | (anchor == ACTION_TO_ID["HOLD"])
    expected[silent] = ACTION_TO_ID["WAIT"]
    expected[speaking] = ACTION_TO_ID["HOLD"]

    assert torch.equal(ablated[:, c:], expected[:, None].expand_as(ablated[:, c:]))
    assert not torch.equal(ablated, actions)


def test_shuffled_replaces_whole_future_sequences_only(validation_batch):
    cfg, _, batch = validation_batch
    actions = trajectories(batch).actions
    c = cfg.data.context_steps
    donors = batch["future_action"].flip(0)

    shuffled = shuffled_actions(actions, c, donors)

    assert torch.equal(shuffled[:, :c], actions[:, :c])
    assert torch.equal(shuffled[:, c:], donors)  # complete sequences, in order
    with pytest.raises(ValueError, match="future length"):
        shuffled_actions(actions, c, donors[:, :-1])


def test_counterfactual_changes_only_the_forced_action(validation_batch):
    cfg, module, batch = validation_batch
    _conditioned(module)
    c = cfg.data.context_steps
    actions = trajectories(batch).actions

    for action in FORCED_ACTIONS:
        forced = forced_anchor_action(actions, c, action)
        changed = (forced != actions).any(dim=0).nonzero().flatten().tolist()
        assert changed in ([], [c - 1])
        assert bool((forced[:, c - 1] == ACTION_TO_ID[action]).all())

    _, result = _ablation(cfg, module, batch)
    counterfactual = result[COUNTERFACTUAL_NEXT]

    # Forcing the action that was observed reproduces the observed step.
    for row, observed in enumerate(actions[:, c - 1].tolist()):
        if observed in (ACTION_TO_ID[a] for a in FORCED_ACTIONS):
            k = [ACTION_TO_ID[a] for a in FORCED_ACTIONS].index(observed)
            assert torch.equal(counterfactual[row, k], result[PRED[OBSERVED]][row, 0])

    # Different actions from the same state give different predictions.
    assert not torch.equal(counterfactual[:, 1], counterfactual[:, 0])


# ---------------------------------------------------------------------------
# Shuffle assignment
# ---------------------------------------------------------------------------


def test_shuffle_is_deterministic_and_stays_within_groups():
    ids = [f"s{i}" for i in range(12)]
    groups = (
        [("egocom", i % 2, 10) for i in range(9)]
        + [("ego4d", 0, 10)] * 2
        + [("ego4d", 1, 10)]
    )

    donors = shuffle_donors(ids, groups, seed=3072)

    assert donors == shuffle_donors(ids, groups, seed=3072)
    assert donors != shuffle_donors(ids, groups, seed=1)
    # Row order changes nothing: the donor of each sample id is the same.
    order = list(reversed(range(12)))
    reordered = shuffle_donors([ids[i] for i in order], [groups[i] for i in order])
    assert {ids[order[r]]: ids[order[d]] for r, d in enumerate(reordered)} == {
        ids[r]: ids[d] for r, d in enumerate(donors)
    }

    for row, donor in enumerate(donors):
        assert groups[donor] == groups[row]  # corpus, focal state, length
        # Another sample whenever the group has more than one member.
        assert (donor != row) == (groups.count(groups[row]) > 1)

    assert donors[11] == 11  # the only ("ego4d", 1) sample keeps its own


# ---------------------------------------------------------------------------
# Run-level extraction and analysis
# ---------------------------------------------------------------------------


def test_run_extraction_and_analysis(tmp_path, cache_root, loads):
    cfg = runs._config(cache_root)
    run_dir, _ = runs._make_run(tmp_path, cfg)

    snapshot = extract_action_ablation_run(run_dir, max_samples=12, batch_size=5)
    tensors = load_file(snapshot / "representations.safetensors")
    metadata = pq.read_table(snapshot / "metadata.parquet").to_pydict()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    c = cfg.data.context_steps

    assert snapshot == run_dir / "latent_analysis" / "last-validation-action-ablation"
    assert manifest["provenance"]["data"]["split"] == "validation"
    assert all(":validation#" in s for s in metadata["sample_id"])
    assert tensors[COUNTERFACTUAL_NEXT].shape == (12, 4, 192)

    # Same anchors as extract-rollouts with the same seed and limit.
    from turn_wm.evaluation.latent_analysis.rollout import extract_rollout_run

    rollout = extract_rollout_run(run_dir, output_dir=tmp_path / "r", max_samples=12)
    rollout_meta = pq.read_table(rollout / "metadata.parquet").to_pydict()
    assert rollout_meta["sample_id"] == metadata["sample_id"]
    assert torch.equal(
        load_file(rollout / "representations.safetensors")[PRED_FUTURE_LATENT],
        tensors[PRED[OBSERVED]],
    )

    # Shuffled rows read their donor's whole future sequence, same group.
    donors = tensors[SHUFFLE_DONOR].tolist()
    for row, donor in enumerate(donors):
        assert metadata["dataset"][donor] == metadata["dataset"][row]
        assert tensors[ROLLOUT_ACTION_IDS][donor, c - 1] == tensors[ROLLOUT_ACTION_IDS][row, c - 1]
        assert torch.equal(
            tensors[SHUFFLED_ACTION_IDS][row, c:],
            tensors[ROLLOUT_ACTION_IDS][donor, c:],
        )
        assert torch.equal(
            tensors[SHUFFLED_ACTION_IDS][row, :c], tensors[ROLLOUT_ACTION_IDS][row, :c]
        )

    output = write_action_ablation(snapshot, bootstrap=20)
    summary = json.loads((output / "summary.json").read_text())

    assert sorted(p.name for p in output.iterdir()) == [
        "figures",
        "report.md",
        "scores.parquet",
        "summary.json",
    ]
    assert sorted(p.name for p in (output / "figures").iterdir()) == [
        "counterfactual_action_effect.png",
        "rollout_ablation.png",
    ]
    first = summary["rollout_ablation"]["1"]
    assert first["integrity"] == {
        "max_abs_difference_between_conditions": 0.0,
        "passed": True,
    }
    for entry in summary["rollout_ablation"].values():
        subsets = entry["subsets"]
        assert subsets["all"]["n"] == 12
        assert subsets["event_exposed"]["n"] + subsets["event_unexposed"]["n"] == 12
    assert summary["rollout_ablation"]["1"]["subsets"]["event_exposed"]["n"] == 0

    report = (output / "report.md").read_text()
    assert "passed" in report
    assert "useful use of the observed ego-action conditioning channel" in report
    assert "does not establish planner controllability" in report


def test_the_test_split_is_refused(tmp_path, cache_root, loads):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))
    snapshot = extract_action_ablation_run(run_dir, max_samples=6)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["provenance"]["data"]["split"] = "test"
    (snapshot / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="validation split only"):
        write_action_ablation(snapshot, bootstrap=10)


# ---------------------------------------------------------------------------
# Logs, progress and show
# ---------------------------------------------------------------------------


def test_extraction_logs_its_stages(tmp_path, cache_root, loads, capsys):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))

    snapshot = extract_action_ablation_run(run_dir, max_samples=6)
    stderr = capsys.readouterr().err

    assert f"run: {run_dir}" in stderr
    assert "checkpoint: " in stderr and "last.ckpt" in stderr
    assert "split: validation" in stderr
    assert "device cpu" in stderr and f"output {snapshot}" in stderr
    assert "action ablation: written in" in stderr


def test_cli_show_prints_the_results_and_changes_no_result(
    tmp_path, cache_root, loads, capsys
):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))
    snapshot = extract_action_ablation_run(run_dir, max_samples=12, batch_size=5)
    common = ["analyze-action-ablation", str(snapshot), "--bootstrap", "20"]
    capsys.readouterr()

    main(common)
    first = capsys.readouterr()
    main([*common, "--show", "--output", str(tmp_path / "shown")])
    shown = capsys.readouterr()

    assert rollouts._hashes(
        snapshot / "analysis" / "action_ablation"
    ) == rollouts._hashes(tmp_path / "shown")
    # Stage logs and a bar over the bootstraps; the final path on stdout.
    assert "horizons x 3 subsets + 2 focal states" in first.err
    assert "bootstrap" in first.err and "action ablation: done in" in first.err
    assert f"action_ablation: {snapshot / 'analysis' / 'action_ablation'}" in first.out
    printed = shown.out
    assert "Integrity check (no future token read): passed" in printed
    assert "Skill vs persistence (primary)" in printed
    assert "observed − state_preserving" in printed
    assert "Forced one-step action effect" in printed
    # The first horizon reads no future token: its exposed subset is empty.
    assert "Not evaluable: 0.1 s, event_exposed has no rows." in printed
    assert str(tmp_path / "shown" / "figures" / "rollout_ablation.png") in printed
    assert str(tmp_path / "shown" / "report.md") in printed


def test_show_action_ablation_in_a_notebook(tmp_path, cache_root, loads, monkeypatch):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))
    snapshot = extract_action_ablation_run(run_dir, max_samples=12, batch_size=5)
    output = write_action_ablation(snapshot, bootstrap=20)
    (output / "figures" / "counterfactual_action_effect.png").unlink()
    before = rollouts._hashes(output)
    shown = []
    ipython = types.ModuleType("IPython")
    vars(ipython).update(get_ipython=lambda: object())
    display = types.ModuleType("IPython.display")
    vars(display).update(
        display=shown.append,
        HTML=lambda text: ("html", text),
        Image=lambda filename: ("image", filename.rsplit("/", 1)[-1]),
    )
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)

    show_action_ablation(output)

    assert shown[0][0] == "html" and "Integrity check" in shown[0][1]
    titles = [item[1] for item in shown[1:5]]
    assert "Skill vs persistence (primary)" in titles[0]
    assert "Forced one-step action effect" in titles[3]
    # A missing figure is said, not silently skipped.
    assert shown[5] == ("image", "rollout_ablation.png")
    assert shown[6][0] == "html" and "Missing figure" in shown[6][1]
    assert rollouts._hashes(output) == before


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def test_conditions_are_compared_on_identical_rows():
    generator = torch.Generator().manual_seed(0)
    anchor = torch.randn(40, 4, generator=generator)
    true = anchor + torch.randn(40, 4, generator=generator)
    observed = anchor + 0.5 * (true - anchor)
    state_preserving = observed.clone()
    state_preserving[:10] = anchor[:10]  # no predicted motion: no direction there
    terms = {
        OBSERVED: row_terms(anchor, true, observed),
        STATE_PRESERVING: row_terms(anchor, true, state_preserving),
        SHUFFLED: row_terms(anchor, true, observed),
    }

    result = paired_metrics(
        terms,
        torch.ones(40, dtype=torch.bool),
        recordings=[f"x/r{i // 5}" for i in range(40)],
        corpora=["x"] * 40,
        bootstrap=50,
        generator=torch.Generator().manual_seed(0),
    )

    # The alignment uses only rows with a direction in every condition ...
    assert result["n_direction_defined_all_conditions"] == 30
    rows = slice(10, 40)
    expected = torch.nn.functional.cosine_similarity(
        (observed - anchor)[rows], (true - anchor)[rows], dim=-1
    ).mean()
    assert result["conditions"][OBSERVED][ALIGNMENT] == pytest.approx(float(expected))
    assert result["conditions"][SHUFFLED][ALIGNMENT] == pytest.approx(float(expected))
    # ... so identical predictions on identical rows differ by exactly 0.
    assert result["differences"]["observed-shuffled"][ALIGNMENT] == 0.0
    assert result["differences"]["observed-shuffled"]["skill_ci"] == [0.0, 0.0]
