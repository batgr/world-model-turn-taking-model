"""
Frozen V1 representations of TurnBench conversations: the three conditions,
one predictor step per speaker, availability times and the manifest
(synthetic run and a stand-in Mimi encoder; no network).
"""

import hashlib
import json
from typing import cast

import numpy as np
import pytest
import test_latent_analysis as runs
import test_turnbench_inputs as inputs
import torch
from safetensors.torch import load_file

from turn_wm.cli import main
from turn_wm.evaluation.turnbench import extract as extract_module
from turn_wm.evaluation.turnbench.actions import conversation_actions
from turn_wm.evaluation.turnbench.data import Conversation, conversation_from_row
from turn_wm.evaluation.turnbench.extract import (
    CURRENT,
    MIMI,
    PREDICTED,
    action_ids,
    condition,
    conversation_representations,
    one_step_predictions,
)
from turn_wm.evaluation.turnbench.timing import resample_lookahead_s
from turn_wm.models.encoders.mimi import FrozenMimiEncoder
from turn_wm.training.lewm import LeWMModule

cache_root = runs.cache_root
loads = runs.loads
RATE = inputs.RATE
SLOTS = 16


class FakeMimi:
    """Causal stand-in: frame i summarizes its own 1920 samples only."""

    sample_rate = 24_000
    source_rate = 12.5
    output_dim = 512
    resolved_revision = "resolved-sha"

    def __init__(self, model_name="kyutai/mimi", revision=None):
        self.revision = revision

    def to(self, device):
        return self

    def eval(self):
        return self

    def stream_native_features(self, waveform, *, chunk_seconds):
        frames = waveform[0, 0, : waveform.shape[-1] // 1920 * 1920].reshape(-1, 1920)
        features = torch.zeros(1, len(frames), self.output_dim)
        features[0, :, 0] = frames.pow(2).mean(dim=1).sqrt() * 10
        features[0, :, 1] = frames.mean(dim=1) * 10

        return features


@pytest.fixture
def model(cache_root):
    cfg = runs._config(cache_root)
    module = LeWMModule(cfg)
    # AdaLN-zero starts with no action conditioning: give it some.
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))

    return module.model.eval(), int(cfg.prediction.rollout_context_size)


def _same(a, b):
    """Equal, rows without a prediction (NaN) included."""

    return torch.equal(a.nan_to_num(), b.nan_to_num()) and torch.equal(
        a.isnan(), b.isnan()
    )


def _conversation():
    one = inputs._speech([False, True, True, False, False, True, True, True] * 2)
    two = inputs._speech([True, False, False, False, True, False, True, False] * 2)

    return Conversation("c1", RATE, one, two)


def _representations(conversation, model, window):
    return conversation_representations(
        conversation,
        model=model,
        encoder=cast(FrozenMimiEncoder, FakeMimi()),
        window=window,
        lookahead_s=resample_lookahead_s(RATE, FakeMimi.sample_rate),
    )


def test_each_prediction_is_the_same_predictor_with_that_speakers_actions(model):
    model, window = model
    conversation = _conversation()
    tensors = _representations(conversation, model, window)
    actions = conversation_actions(conversation)
    valid = tensors["prediction_valid"]

    for speaker in ("speaker_1", "speaker_2"):
        expected = one_step_predictions(
            model, tensors["latent"], action_ids(actions[speaker]), window=window
        )
        assert torch.allclose(tensors[f"zpred_{speaker}"][valid], expected[valid])

    # Different actions, different predictions from the same latent history.
    assert not torch.allclose(
        tensors["zpred_speaker_1"][valid], tensors["zpred_speaker_2"][valid]
    )

    # Swapping the channels keeps the scene, hence the latents, and swaps
    # only the actions: the two predictions swap.
    swapped = _representations(
        Conversation("c1", RATE, conversation.speaker_2, conversation.speaker_1),
        model,
        window,
    )
    assert torch.equal(swapped[MIMI], tensors[MIMI])
    assert torch.equal(swapped["latent"], tensors["latent"])
    for a, b in (("speaker_1", "speaker_2"), ("speaker_2", "speaker_1")):
        assert _same(swapped[f"zpred_{a}"], tensors[f"zpred_{b}"])


def test_exactly_one_predictor_step_from_the_past_window(model):
    model, window = model
    latents = torch.randn(SLOTS, model.projector(torch.zeros(1, 512)).shape[-1])
    actions = torch.randint(0, 4, (SLOTS,))

    predictions = one_step_predictions(model, latents, actions, window=window)

    assert predictions[: window - 1].isnan().all()
    k = SLOTS - 3
    with torch.inference_mode():
        manual = model.predict(
            latents[None, k - window + 1 : k + 1],
            model.encode_actions(actions[None, k - window + 1 : k + 1]),
        )[0, -1]
    assert torch.allclose(predictions[k], manual)

    # Row k reads nothing after slot k: no rollout, no future latent.
    later = latents.clone()
    later[k + 1 :] = 100.0
    assert _same(
        one_step_predictions(model, later, actions, window=window)[: k + 1],
        predictions[: k + 1],
    )


def test_the_predicted_condition_is_only_the_two_predictions(model):
    model, window = model
    tensors = _representations(_conversation(), model, window)

    predicted = condition(tensors, PREDICTED)

    assert _same(
        predicted,
        torch.cat([tensors["zpred_speaker_1"], tensors["zpred_speaker_2"]], dim=-1),
    )
    # Nothing else reaches it: not z_k, not the Mimi features, not actions.
    others = {
        name: torch.full_like(tensor.float(), 7.0)
        for name, tensor in tensors.items()
        if not name.startswith("zpred_")
    }
    assert _same(condition({**tensors, **others}, PREDICTED), predicted)
    assert torch.equal(condition(tensors, CURRENT), tensors["latent"])
    assert torch.equal(condition(tensors, MIMI), tensors[MIMI])
    with pytest.raises(ValueError, match="Unknown condition"):
        condition(tensors, "latent_and_actions")


def test_rows_are_stamped_with_their_availability(model):
    model, window = model
    tensors = _representations(_conversation(), model, window)
    lookahead = resample_lookahead_s(RATE, 24_000)

    assert 0 < lookahead < 0.001
    assert resample_lookahead_s(24_000, 24_000) == 0.0
    # Slot end plus the resampler's lookahead; not the +0.1 s horizon.
    assert torch.allclose(
        tensors["available_s"],
        torch.arange(1, SLOTS + 1, dtype=torch.float64) / 10 + lookahead,
    )
    assert tensors["prediction_valid"].tolist() == [
        k >= window - 1 for k in range(SLOTS)
    ]
    assert tensors["action_valid_speaker_1"][0].item() is False


def test_representations_need_no_annotation(model):
    model, window = model
    one = inputs._speech([False, True] * 6).numpy()
    row = inputs._InferenceOnlyRow(
        conversation_id="c1",
        speaker_1_audio={"bytes": inputs._flac(one)},
        speaker_2_audio={"bytes": inputs._flac(np.zeros_like(one))},
        speaker_1_annotation_a=[{"start_s": 0.0, "end_s": 1.0, "label": "EOT"}],
    )

    tensors = _representations(conversation_from_row(row), model, window)

    assert len(tensors["latent"]) == 12


def test_extraction_writes_the_conditions_and_a_manifest(
    tmp_path, cache_root, loads, monkeypatch, capsys
):
    run_dir, _ = runs._make_run(tmp_path, runs._config(cache_root))
    created = []

    def fake_mimi(**kwargs):
        created.append(kwargs)
        return FakeMimi(**kwargs)

    monkeypatch.setattr(extract_module, "FrozenMimiEncoder", fake_mimi)
    monkeypatch.setattr("turn_wm.cli.load_dev", lambda skip=(): iter([_conversation()]))
    monkeypatch.setattr("turn_wm.cli.split_size", lambda split: 38)
    output = tmp_path / "turnbench"

    main(
        [
            "extract-turnbench",
            str(run_dir),
            "--output",
            str(output),
            "--max-conversations",
            "1",
        ]
    )
    captured = capsys.readouterr()

    manifest = json.loads((output / "manifest.json").read_text())
    tensors = load_file(output / "c1.safetensors")
    entry = manifest["files"]["c1"]

    # The Mimi model of the cache V1 trained on, never a guessed one.
    assert created == [{"model_name": "kyutai/mimi", "revision": "resolved-sha"}]
    assert (
        entry["sha256"]
        == hashlib.sha256((output / "c1.safetensors").read_bytes()).hexdigest()
    )
    assert entry["slots"] == SLOTS
    assert set(tensors) == {
        "slot_start_s",
        "available_s",
        "mimi",
        "latent",
        "prediction_valid",
        "zpred_speaker_1",
        "zpred_speaker_2",
        "action_valid_speaker_1",
        "action_valid_speaker_2",
    }
    assert manifest["model"]["checkpoint_sha256"]
    assert manifest["inputs"]["turnbench_dataset"]["revision"]
    assert manifest["timing"]["resample_lookahead_s"] > 0
    assert set(manifest["conditions"]) == {MIMI, CURRENT, PREDICTED}
    assert f"turnbench: {output}" in captured.out
    # What runs, on what, from which revisions, how far, and where it went.
    stderr = captured.err
    assert "turnbench: split dev, mundo-ai/turn-benchmark-dev @ " in stderr
    assert "checkpoint " in stderr and "(step " in stderr
    assert "turnbench: Mimi kyutai/mimi @ resolved-sha" in stderr
    assert "turnbench: device cpu" in stderr
    assert f"turnbench: 1 conversations to extract -> {output}" in stderr
    assert "1/1" in stderr
    assert f"manifest {output / 'manifest.json'}" in stderr
