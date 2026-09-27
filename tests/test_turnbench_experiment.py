"""
The V1 TurnBench experiment's scientific contracts: TRAIN labels follow the
official TurnBench semantics, the TRAIN split is conversation-disjoint, the
three heads read exactly their representation on the same frames, the
official sweep / scorer produce valid DEV predictions, and the pipeline
stops on a broken smoke run and refuses stale artifacts (synthetic data).
"""

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import test_latent_analysis as runs
import test_turnbench_extract as tx
import torch
from safetensors.torch import load_file, save_file
from turnbench.gold import CANONICAL, AnchorEvent, Interval
from turnbench.submission import load_submission

from turn_wm.evaluation.turnbench import extract as extract_module
from turn_wm.evaluation.turnbench import pipeline
from turn_wm.evaluation.turnbench.extract import (
    CONDITIONS,
    CURRENT,
    MIMI,
    PREDICTED,
    condition,
    extract_turnbench,
    extraction_identity,
    validate_extraction,
)
from turn_wm.evaluation.turnbench.heads import (
    CausalHead,
    Sequence_,
    TrainingConfig,
    evaluate,
    load_sequences,
    split_conversations,
    train_head,
)
from turn_wm.evaluation.turnbench.labels import (
    OUTPUTS,
    frame_targets,
    parse_srt,
    single_annotator_events,
)
from turn_wm.evaluation.turnbench.scoring import score_condition

cache_root = runs.cache_root
loads = runs.loads


def _srt(*entries):
    return "\n\n".join(
        f"{i}\n{_ts(start)} --> {_ts(end)}\n[{label}] {text}"
        for i, (start, end, label, text) in enumerate(entries, start=1)
    )


def _ts(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _events(speaker_1, speaker_2):
    return single_annotator_events(
        {1: parse_srt(_srt(*speaker_1)), 2: parse_srt(_srt(*speaker_2))}
    )


def _frames(slots):
    """Availability times of `slots` 100 ms frames (slot end, no lookahead)."""

    return torch.arange(1, slots + 1, dtype=torch.float64) / 10


# ---------------------------------------------------------------------------
# TRAIN labels: official TurnBench semantics, one annotator
# ---------------------------------------------------------------------------


def test_fine_labels_follow_the_turnbench_taxonomy():
    expected = {
        "Turn": [
            "Normal Turn",
            "Regular Turn",
            "Strong Floor Hold",
            "Bounded Response",
            "Filler",
            "Overlap",
        ],
        "Interruption": [
            "Floor-taking Competitive Interruption",
            "Floor-taking Cooperative Interruption",
        ],
        "NonFloorTakingInterruption": [
            "Non-floor Taking Competitive Interruption",
            "Non-floor Taking Cooperative Interruption",
        ],
        "Backchannel": [
            "Acknowledgement Backchannel",
            "Continuer Backchannel",
            "Reaction Backchannel",
        ],
        "NonContent": ["Non-Speech Noise", "Channel Bleed", "Speech, Non-Linguistic"],
    }

    for canonical, fines in expected.items():
        for fine in fines:
            assert CANONICAL[fine] == canonical

    parsed = parse_srt(_srt((12.34, 13.02, "Acknowledgement Backchannel", "Okay.")))
    assert parsed == [(12.34, 13.02, "Acknowledgement Backchannel", "Okay.")]
    with pytest.raises(ValueError, match="not in the TurnBench taxonomy"):
        parse_srt(_srt((0.0, 1.0, "Made-up Label", "")))


def test_a_true_floor_transfer_is_an_eot_positive():
    events = _events(
        [(0.5, 3.0, "Normal Turn", "hi")], [(3.4, 6.0, "Normal Turn", "yes")]
    )

    assert AnchorEvent(1, 3.0) in events.eot_positive_events
    targets, supervised = frame_targets(events, _frames(80))
    eot_1 = OUTPUTS.index("eot_speaker_1")
    positive = targets[:, eot_1] > 0
    # The first frame available at or after 3.0 s (index 29, at 3.0 s) and the
    # next one: 200 ms; negative again after them.
    assert torch.equal(positive.nonzero().flatten(), torch.tensor([29, 30]))
    assert supervised[31, eot_1] and targets[31, eot_1] == 0
    # The scorer's matching tolerance before the anchor is not supervised.
    assert not supervised[27, eot_1] and supervised[26, eot_1]


def test_the_positive_window_is_the_first_available_frame_and_the_next():
    events = _events(
        [(0.5, 3.05, "Normal Turn", "a")], [(3.4, 3.95, "Normal Turn", "b")]
    )
    # Real availability: slot end plus the resampler's lookahead.
    available = _frames(40) + 0.00025
    eot_1, eot_2 = OUTPUTS.index("eot_speaker_1"), OUTPUTS.index("eot_speaker_2")

    targets, supervised = frame_targets(events, available)

    # 3.05 s: first frame available at or after it is index 30 (3.10025 s).
    assert (targets[:, eot_1] > 0).nonzero().flatten().tolist() == [30, 31]
    # Frames available in [2.80 s, 3.05 s) stay ignored.
    assert supervised[:, eot_1].logical_not().nonzero().flatten().tolist() == [
        27,
        28,
        29,
    ]
    # 3.95 s ends the audio's frames: only the one remaining frame is positive.
    assert (targets[:, eot_2] > 0).nonzero().flatten().tolist() == [39]


def test_a_mid_turn_pause_is_not_an_eot_positive():
    events = _events(
        [(0.5, 3.0, "Normal Turn", "a"), (3.8, 6.0, "Normal Turn", "b")],
        [(7.0, 9.0, "Normal Turn", "c")],
    )

    assert AnchorEvent(1, 3.0) not in events.eot_positive_events
    assert AnchorEvent(1, 6.0) in events.eot_positive_events
    assert Interval(1, 3.0, 3.8) in events.eot_negative_spans
    targets, _ = frame_targets(events, _frames(100))
    eot_1 = OUTPUTS.index("eot_speaker_1")
    assert targets[29:38, eot_1].sum() == 0


def test_only_a_floor_taking_interruption_is_an_int_positive():
    floor = _events(
        [(0.0, 8.0, "Normal Turn", "a")],
        [(5.0, 9.0, "Floor-taking Competitive Interruption", "b")],
    )
    non_floor = _events(
        [(0.0, 8.0, "Normal Turn", "a")],
        [(5.0, 6.0, "Non-floor Taking Competitive Interruption", "b")],
    )
    int_2 = OUTPUTS.index("int_speaker_2")

    assert floor.int_positive_events == [AnchorEvent(2, 5.0)]
    targets, _ = frame_targets(floor, _frames(100))
    assert targets[48:52, int_2].tolist() == [0.0, 1.0, 1.0, 0.0]

    # Not a positive, not a negative: its extent is left unsupervised.
    assert non_floor.int_positive_events == []
    assert Interval(2, 5.0, 6.0) in non_floor.int_excluded
    targets, supervised = frame_targets(non_floor, _frames(100))
    assert targets[:, int_2].sum() == 0
    assert not supervised[49:60, int_2].any()
    assert supervised[70, int_2]


# ---------------------------------------------------------------------------
# TRAIN / validation split and matched heads
# ---------------------------------------------------------------------------


def test_the_train_validation_split_is_by_whole_conversation():
    ids = [str(i) for i in range(1, 421)]

    train, validation = split_conversations(ids, seed=0, validation_fraction=0.1)

    assert set(train).isdisjoint(validation)
    assert sorted(train + validation, key=int) == ids
    assert len(validation) == 42
    assert (train, validation) == split_conversations(
        ids, seed=0, validation_fraction=0.1
    )
    assert validation != split_conversations(ids, seed=1, validation_fraction=0.1)[1]


def _extraction(tmp_path, slots=40, window=10):
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "slot_start_s": torch.arange(slots) / 10,
        "available_s": _frames(slots),
        "mimi": torch.randn(slots, 512, generator=generator),
        "latent": torch.randn(slots, 192, generator=generator),
        "prediction_valid": torch.arange(slots) >= window - 1,
        "zpred_speaker_1": torch.randn(slots, 192, generator=generator),
        "zpred_speaker_2": torch.randn(slots, 192, generator=generator),
        "action_valid_speaker_1": torch.ones(slots, dtype=torch.bool),
        "action_valid_speaker_2": torch.ones(slots, dtype=torch.bool),
    }
    for speaker in ("speaker_1", "speaker_2"):
        tensors[f"zpred_{speaker}"][: window - 1] = float("nan")
    save_file(tensors, str(tmp_path / "c1.safetensors"))

    return tensors


def _labels(conversation_id, available_s):
    supervised = torch.ones(len(available_s), 4, dtype=torch.bool)
    supervised[20:25, 1] = False
    return torch.zeros(len(available_s), 4), supervised


def test_each_head_reads_exactly_its_representation_on_the_same_frames(tmp_path):
    tensors = _extraction(tmp_path)

    sequences = {
        name: load_sequences(tmp_path, ["c1"], _labels, name=name)[0]
        for name in CONDITIONS
    }

    assert {n: s.inputs.shape[1] for n, s in sequences.items()} == {
        MIMI: 512,
        CURRENT: 192,
        PREDICTED: 384,
    }
    for name, sequence in sequences.items():
        assert torch.equal(
            sequence.inputs, condition(tensors, name).nan_to_num(0.0).half()
        )
    # One eligible mask for all: prediction_valid and labels, never the input.
    masks = [s.mask for s in sequences.values()]
    assert all(torch.equal(mask, masks[0]) for mask in masks)
    assert not masks[0][:9].any() and not masks[0][20:25, 1].any()
    assert masks[0][9:20].all()


def test_the_predicted_head_reads_only_the_two_predictions(tmp_path):
    tensors = _extraction(tmp_path)
    before = load_sequences(tmp_path, ["c1"], _labels, name=PREDICTED)[0].inputs

    # Change everything but the two predictions: z_t, Mimi, action validity.
    tensors["latent"] = torch.zeros_like(tensors["latent"])
    tensors["mimi"] = torch.zeros_like(tensors["mimi"])
    tensors["action_valid_speaker_1"] = torch.zeros_like(
        tensors["action_valid_speaker_1"]
    )
    save_file(tensors, str(tmp_path / "c1.safetensors"))
    after = load_sequences(tmp_path, ["c1"], _labels, name=PREDICTED)[0].inputs

    assert torch.equal(before, after)
    assert torch.equal(
        after[9:].float(),
        torch.cat([tensors["zpred_speaker_1"], tensors["zpred_speaker_2"]], -1)[9:]
        .half()
        .float(),
    )


ACCELERATORS = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


def _sequences(count, frames=24):
    generator = torch.Generator().manual_seed(0)
    sequences = []

    for i in range(count):
        targets = torch.zeros(frames, 4)
        targets[frames // 2 : frames // 2 + 2] = 1.0
        sequences.append(
            Sequence_(
                f"c{i}",
                torch.randn(frames, 16, generator=generator).half(),
                targets,
                torch.ones(frames, 4, dtype=torch.bool),
            )
        )

    return sequences


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
def test_training_and_validation_run_on_one_device(device):
    config = TrainingConfig(batch_size=2, crop_frames=8, max_epochs=2, patience=5)

    head, record = train_head(
        _sequences(4), _sequences(2), config=config, device=device
    )

    assert all(np.isfinite(epoch["val/loss"]) for epoch in record["history"])
    # Validation metrics do not depend on the device the head was trained on.
    pos_weight = torch.tensor(record["pos_weight"])
    on_device = evaluate(head.to(device), _sequences(2), pos_weight.to(device), device)
    on_cpu = evaluate(head.cpu(), _sequences(2), pos_weight, "cpu")
    assert on_device["val/loss"] == pytest.approx(on_cpu["val/loss"], rel=1e-4)


def test_the_head_is_causal():
    head = CausalHead(8).eval()
    x = torch.randn(1, 30, 8)
    later = x.clone()
    later[:, 20:] = 5.0

    with torch.no_grad():
        assert torch.equal(head(x)[:, :20], head(later)[:, :20])


# ---------------------------------------------------------------------------
# Official DEV scoring
# ---------------------------------------------------------------------------


def _wav(duration_s, rate=48_000):
    buffer = io.BytesIO()
    sf.write(
        buffer, np.zeros(int(duration_s * rate), dtype=np.float32), rate, format="WAV"
    )
    return buffer.getvalue()


def _local_dev(tmp_path, tracks, duration_s=10.0):
    """A one-conversation dataset in TurnBench's parquet layout (3 annotators)."""

    event = pa.struct(
        [
            ("start_s", pa.float64()),
            ("end_s", pa.float64()),
            ("label", pa.string()),
            ("text", pa.string()),
        ]
    )
    audio = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    columns = {"conversation_id": pa.array(["7"])}

    for speaker in (1, 2):
        for annotator in "abc":
            columns[f"speaker_{speaker}_annotation_{annotator}"] = pa.array(
                [
                    [
                        {"start_s": s, "end_s": e, "label": l, "text": ""}
                        for s, e, l in tracks[speaker]
                    ]
                ],
                type=pa.list_(event),
            )
        columns[f"speaker_{speaker}_audio"] = pa.array(
            [{"bytes": _wav(duration_s), "path": f"s{speaker}.wav"}], type=audio
        )

    root = tmp_path / "dev"
    root.mkdir()
    pq.write_table(pa.table(columns), root / "dev-00000.parquet")

    from turnbench.data import resolve_dataset

    return resolve_dataset(str(root))


def test_official_sweep_and_scorer_produce_valid_dev_predictions(tmp_path):
    dataset = _local_dev(
        tmp_path,
        {
            1: [
                (0.5, 1.5, "Normal Turn"),
                (2.0, 3.0, "Normal Turn"),
                (6.5, 9.0, "Normal Turn"),
            ],
            2: [(3.4, 6.0, "Normal Turn")],
        },
    )
    probabilities = torch.full((100, 4), 0.01)
    # EOT scores rise right after each real turn end (3.0 and 9.0 on s1, 6.0 on s2).
    probabilities[30:33, 0] = 0.9
    probabilities[90:93, 0] = 0.9
    probabilities[60:63, 1] = 0.9
    probabilities[15:18, 0] = 0.3  # a weaker bump in the mid-turn pause
    probabilities[:9] = 0.0  # no prediction yet

    result = score_condition(
        {"7": probabilities},
        lookahead_s={"7": 0.00025},
        dataset=dataset,
        durations={"7": 10.0},
        output_dir=tmp_path / "scoring",
    )

    submission = load_submission(tmp_path / "scoring" / "predictions-dev.json")
    [prediction] = submission.predictions
    assert result["eot"]["recall"] == 1.0 and result["eot"]["fp_rate"] == 0.0
    assert 0.3 <= result["eot"]["threshold"] < 0.9
    # Committed at the frame end plus the resampler's lookahead, no +0.1 s.
    assert prediction.speaker_1.eot == pytest.approx([3.10025, 9.10025])
    assert prediction.speaker_2.eot == pytest.approx([6.10025])
    assert result["eot"]["latency_ms"]["p50"] == pytest.approx(100.25)
    # No INT event in the gold: no operating point, nothing committed.
    assert (
        result["int"]["threshold"] is None and prediction.speaker_1.interruption == []
    )
    for name in ("probs-eot.json", "probs-int.json", "sweep-eot.json", "scores.json"):
        assert (tmp_path / "scoring" / name).is_file()


# ---------------------------------------------------------------------------
# Pipeline: smoke gate, resume, stale artifacts
# ---------------------------------------------------------------------------


def test_the_pipeline_stops_after_a_failed_smoke_run(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(pipeline, "extraction_identity", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "scorer_revision", lambda: "rev")
    monkeypatch.setattr(pipeline, "load_durations", lambda split: {"1": 10.0})
    monkeypatch.setattr(pipeline, "train_conversation_ids", lambda: ["1"])

    def stage(config, output, *args, **kwargs):
        calls.append(output.name)
        return {}

    def smoke(output, manifest):
        raise pipeline.SmokeTestFailed(f"{output}: broken")

    monkeypatch.setattr(pipeline, "_extraction_stage", stage)
    monkeypatch.setattr(pipeline, "check_smoke", smoke)

    with pytest.raises(pipeline.SmokeTestFailed):
        pipeline.run_pipeline(
            pipeline.PipelineConfig(
                run_dir=tmp_path, work_dir=tmp_path / "w", checkpoint="x"
            )
        )

    assert calls == ["smoke-dev"]


@pytest.fixture
def synthetic_run(tmp_path, cache_root, loads, monkeypatch):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))
    monkeypatch.setattr(extract_module, "FrozenMimiEncoder", tx.FakeMimi)

    return run_dir


def _extract(run_dir, output, conversations, **kwargs):
    return extract_turnbench(
        lambda done: (c for c in conversations if c.conversation_id not in done),
        run_dir=run_dir,
        output_dir=output,
        **kwargs,
    )


def test_the_smoke_check_passes_a_sound_extraction_and_catches_a_broken_one(
    tmp_path, synthetic_run
):
    output = _extract(synthetic_run, tmp_path / "smoke", [tx._conversation()], limit=1)
    identity = extraction_identity(synthetic_run, checkpoint="last.ckpt", split="dev")
    manifest = validate_extraction(output, identity=identity)

    pipeline.check_smoke(output, manifest)

    tensors = load_file(str(output / "c1.safetensors"))
    tensors["available_s"] = tensors["available_s"] - 0.1  # timestamped at t_k
    save_file(tensors, str(output / "c1.safetensors"))
    with pytest.raises(pipeline.SmokeTestFailed, match="available_s"):
        pipeline.check_smoke(output, manifest)


def test_stale_or_changed_artifacts_are_refused(tmp_path, synthetic_run):
    output = _extract(synthetic_run, tmp_path / "dev", [tx._conversation()])
    identity = extraction_identity(synthetic_run, checkpoint="last.ckpt", split="dev")

    validate_extraction(output, identity=identity, expected_ids=["c1"])
    with pytest.raises(ValueError, match="Stale"):
        validate_extraction(output, identity={**identity, "window": 5})
    with pytest.raises(ValueError, match="does not cover"):
        validate_extraction(output, identity=identity, expected_ids=["c1", "c2"])

    (output / "c1.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_extraction(output, identity=identity)

    stage = tmp_path / "stage"
    stage.mkdir()
    pipeline._write_stage(stage, {"condition": "mimi"}, {}, {})
    with pytest.raises(ValueError, match="Stale stage"):
        pipeline._reuse(stage, {"condition": "current"})


def test_an_interrupted_extraction_resumes_only_its_own_work(tmp_path, synthetic_run):
    first = tx._conversation()
    second = tx.Conversation("c2", tx.RATE, first.speaker_2, first.speaker_1)
    output = tmp_path / "train"

    def crashing(done):
        yield first
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        extract_turnbench(crashing, run_dir=synthetic_run, output_dir=output)

    partial = json.loads((output / "partial.json").read_text())
    seen = []

    def remaining(done):
        seen.append(set(done))
        yield from (c for c in (first, second) if c.conversation_id not in done)

    extract_turnbench(remaining, run_dir=synthetic_run, output_dir=output)

    # Resumed: only the missing conversation was asked for, the done one kept.
    assert set(partial["files"]) == {"c1"}
    assert seen == [{"c1"}]
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {"c1", "c2"}
    assert manifest["files"]["c1"]["sha256"] == partial["files"]["c1"]["sha256"]
    assert not (output / "partial.json").exists()

    # A partial run made for another model is refused, not resumed.
    stale = tmp_path / "stale"
    with pytest.raises(RuntimeError):
        extract_turnbench(crashing, run_dir=synthetic_run, output_dir=stale)
    content = json.loads((stale / "partial.json").read_text())
    content["identity"]["checkpoint_sha256"] = "other"
    (stale / "partial.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match="Stale partial"):
        _extract(synthetic_run, stale, [first])


def test_the_whole_pipeline_runs_and_then_reuses_every_stage(
    tmp_path, synthetic_run, monkeypatch
):
    from turn_wm.evaluation.turnbench import scoring
    from turn_wm.evaluation.turnbench.heads import TrainingConfig

    base = tx._conversation()  # 16 slots, 1.6 s at 48 kHz
    dev = [tx.Conversation("7", tx.RATE, base.speaker_1, base.speaker_2)]
    train = [
        tx.Conversation(
            f"t{i}", tx.RATE, *(base.speaker_1, base.speaker_2)[:: 1 if i % 2 else -1]
        )
        for i in range(1, 5)
    ]
    dataset = _local_dev(
        tmp_path,
        {1: [(0.2, 1.0, "Normal Turn")], 2: [(1.1, 1.5, "Normal Turn")]},
        duration_s=1.6,
    )
    extracted = []

    def source(conversations):
        def load(skip=(), **kwargs):
            for conversation in conversations:
                if conversation.conversation_id not in skip:
                    extracted.append(conversation.conversation_id)
                    yield conversation

        return load

    def annotations(conversation_id, destination):
        paths = {}
        for speaker, text in (
            (1, _srt((0.2, 0.9, "Normal Turn", "a"))),
            (2, _srt((1.0, 1.4, "Floor-taking Competitive Interruption", "b"))),
        ):
            path = destination / conversation_id / f"s{speaker}.srt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            paths[speaker] = path
        return paths

    monkeypatch.setattr(pipeline, "load_dev", source(dev))
    monkeypatch.setattr(pipeline, "load_train", source(train))
    monkeypatch.setattr(pipeline, "load_durations", lambda split: {"7": 1.6})
    monkeypatch.setattr(scoring, "load_durations", lambda split: {"7": 1.6})
    monkeypatch.setattr(
        pipeline, "train_conversation_ids", lambda: [c.conversation_id for c in train]
    )
    monkeypatch.setattr(pipeline, "download_annotations", annotations)
    monkeypatch.setattr("turnbench.data.resolve_dataset", lambda *a, **k: dataset)
    config = pipeline.PipelineConfig(
        run_dir=synthetic_run,
        work_dir=tmp_path / "work",
        checkpoint="last.ckpt",
        device="cpu",
        wandb_mode="disabled",
        training=TrainingConfig(
            batch_size=2, crop_frames=16, max_epochs=2, validation_fraction=0.25
        ),
    )

    report = pipeline.run_pipeline(config)

    text = report.read_text()
    for heading in ("## Observation", "## Interpretation", "## Limitations"):
        assert heading in text
    for name in CONDITIONS:
        manifest = json.loads(
            (tmp_path / "work" / "heads" / name / "manifest.json").read_text()
        )
        assert set(manifest["train_conversations"]).isdisjoint(
            manifest["validation_conversations"]
        )
        assert manifest["input_dim"] == {MIMI: 512, CURRENT: 192, PREDICTED: 384}[name]
        scores = json.loads(
            (tmp_path / "work" / "scoring" / name / "scores.json").read_text()
        )
        assert set(scores) >= {"eot", "int"}
        load_submission(tmp_path / "work" / "scoring" / name / "predictions-dev.json")

    # A second run validates and reuses every stage: nothing is extracted again.
    extracted.clear()
    pipeline.run_pipeline(config)
    assert extracted == []
