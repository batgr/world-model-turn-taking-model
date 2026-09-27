"""
TurnBench inputs for V1: the scene, causal activity and the 100 ms action
grid, on synthetic audio only (the real DEV split is gated).
"""

import io

import av
import numpy as np
import pytest
import torch

from turn_wm.data.mimi_precompute import GRID_RATE_HZ
from turn_wm.evaluation.turnbench.actions import (
    COMPOUND_TRANSITION,
    RECORDING_START,
    conversation_actions,
    slot_actions,
)
from turn_wm.evaluation.turnbench.activity import rms_activity
from turn_wm.evaluation.turnbench.data import (
    INFERENCE_COLUMNS,
    Conversation,
    conversation_from_row,
    decode_audio,
    input_provenance,
    load_train,
)
from turn_wm.evaluation.turnbench.timing import (
    CONTROL_RATE_HZ,
    commit_time_s,
    slot_count,
)

RATE = 48_000
WINDOW = 960  # 20 ms at 48 kHz
SLOT = 4_800  # 100 ms


def _speech(slots_active, *, amplitude=0.1):
    """A channel whose 100 ms slots are loud (True) or silent (False)."""

    audio = torch.zeros(len(slots_active) * SLOT)

    for k, active in enumerate(slots_active):
        if active:
            audio[k * SLOT : (k + 1) * SLOT] = amplitude

    return audio


def _conversation(speaker_1, speaker_2, conversation_id="c1"):
    return Conversation(conversation_id, RATE, speaker_1, speaker_2)


def _flac(samples, rate=RATE):
    buffer = io.BytesIO()

    with av.open(buffer, "w", format="flac") as container:
        stream = container.add_stream("flac", rate=rate, layout="mono")
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.rate = rate

        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)

    return buffer.getvalue()


def _wav(path, samples, rate=RATE):
    path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(path), "w", format="wav") as container:
        stream = container.add_stream("pcm_s16le", rate=rate, layout="mono")
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.rate = rate

        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)


def _activity(*windows):
    return torch.tensor(windows, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Scene and channels
# ---------------------------------------------------------------------------


def test_speaker_channels_stay_separately_accessible():
    one, two = torch.randn(SLOT), torch.randn(SLOT)
    conversation = _conversation(one, two)

    assert conversation.channel("speaker_1") is one
    assert conversation.channel("speaker_2") is two
    assert conversation.duration_s == pytest.approx(0.1)
    with pytest.raises(KeyError):
        conversation.channel("speaker_3")


def test_scene_is_the_exact_sum_and_ignores_channel_order():
    one, two = torch.randn(3 * SLOT) * 0.8, torch.randn(3 * SLOT) * 0.8

    scene = _conversation(one, two).scene()

    assert torch.equal(scene, one + two)
    assert torch.equal(_conversation(two, one).scene(), scene)
    # No normalization: the sum may leave [-1, 1] and is kept as is.
    assert scene.abs().max() > 1


def test_inconsistent_channels_fail_instead_of_being_repaired():
    with pytest.raises(ValueError, match="differ in length"):
        _conversation(torch.zeros(SLOT), torch.zeros(SLOT + 1))

    row = {
        "conversation_id": "c1",
        "speaker_1_audio": {"bytes": _flac(np.zeros(SLOT), 48_000)},
        "speaker_2_audio": {"bytes": _flac(np.zeros(SLOT // 3), 16_000)},
    }
    with pytest.raises(ValueError, match="sample rates differ"):
        conversation_from_row(row)


def test_flac_is_decoded_at_its_native_rate():
    samples = 0.5 * np.sin(np.arange(SLOT) / 10)

    decoded, rate = decode_audio(_flac(samples))

    assert rate == RATE
    assert decoded.dtype == torch.float32 and decoded.shape == (SLOT,)
    assert torch.allclose(
        decoded, torch.tensor(samples, dtype=torch.float32), atol=1e-4
    )


class _InferenceOnlyRow(dict):
    """A row whose annotation fields cannot be read."""

    def __getitem__(self, key):
        if key not in INFERENCE_COLUMNS:
            raise AssertionError(f"inference read the non-inference field {key!r}")

        return super().__getitem__(key)


def test_no_annotation_field_is_needed_for_inference():
    row = _InferenceOnlyRow(
        conversation_id="c1",
        speaker_1_audio={"bytes": _flac(np.full(3 * SLOT, 0.1))},
        speaker_2_audio={"bytes": _flac(np.zeros(3 * SLOT))},
        speaker_1_annotation_a=[{"start_s": 0.0, "end_s": 1.0, "label": "EOT"}],
    )

    conversation = conversation_from_row(row)
    actions = conversation_actions(conversation)

    assert conversation.conversation_id == "c1"
    assert actions["speaker_1"].state_before == ["UNKNOWN", "SPEAKING", "SPEAKING"]
    assert actions["speaker_2"].state_before == ["UNKNOWN", "SILENT", "SILENT"]


def test_train_conversations_come_from_the_two_speaker_files_only(
    tmp_path, monkeypatch
):
    import huggingface_hub

    names = []

    for conversation_id in ("10", "2"):
        for name in (
            "speaker_1_audio.wav",
            "speaker_2_audio.wav",
            "combined_audio.wav",
            "speaker_1_annotation_a.srt",
            "metadata.json",
        ):
            names.append(f"{conversation_id}/{name}")
            _wav(tmp_path / conversation_id / name, np.full(SLOT, 0.1))

    downloaded = []

    def download(repo, name, **kwargs):
        downloaded.append(name)
        return str(tmp_path / name)

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "list_repo_files",
        lambda self, repo, **kwargs: [*names, "LICENSE.md"],
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)

    conversations = list(load_train())

    assert [c.conversation_id for c in conversations] == ["2", "10"]
    assert downloaded == [
        "2/speaker_1_audio.wav",
        "2/speaker_2_audio.wav",
        "10/speaker_1_audio.wav",
        "10/speaker_2_audio.wav",
    ]
    assert conversations[0].sample_rate == RATE
    assert torch.equal(
        conversations[0].scene(),
        conversations[0].speaker_1 + conversations[0].speaker_2,
    )


# ---------------------------------------------------------------------------
# Activity
# ---------------------------------------------------------------------------


def test_activity_is_rms_above_threshold_per_20_ms_window():
    audio = torch.zeros(4 * WINDOW + 100)  # the partial last window is dropped
    audio[WINDOW : 2 * WINDOW] = 0.02  # RMS 0.02 > 0.01
    audio[2 * WINDOW : 3 * WINDOW] = 0.005  # RMS 0.005 < 0.01

    assert rms_activity(audio, RATE).tolist() == [False, True, False, False]


def test_activity_reads_no_sample_after_its_window():
    audio = torch.randn(10 * WINDOW) * 0.02
    later = audio.clone()
    later[6 * WINDOW :] = 1.0  # change only the future of window 5

    assert torch.equal(rms_activity(audio, RATE)[:6], rms_activity(later, RATE)[:6])


# ---------------------------------------------------------------------------
# Actions on the 100 ms grid
# ---------------------------------------------------------------------------


def test_transitions_map_to_the_v1_vocabulary():
    silent, loud = [False] * 5, [True] * 5

    actions = slot_actions(_activity(*silent, *silent, *loud, *loud, *silent))

    assert actions.state_before == [
        "UNKNOWN",
        "SILENT",
        "SILENT",
        "SPEAKING",
        "SPEAKING",
    ]
    assert actions.action == [None, "NO_EVENT", "ONSET", "NO_EVENT", "OFFSET"]
    assert actions.action_valid == [False, True, True, True, True]


def test_a_flip_at_a_slot_boundary_belongs_to_the_new_slot():
    # Silent up to t = 0.2 s exactly, loud from the first window of slot 2.
    actions = slot_actions(_activity(*[False] * 10, *[True] * 5))

    assert actions.action[1:] == ["NO_EVENT", "ONSET"]


def test_several_flips_in_one_slot_are_masked_not_no_event():
    # SILENT -> SPEAKING -> SILENT inside slot 1: same state at both ends.
    actions = slot_actions(_activity(*[False] * 5, False, True, True, False, False))

    assert actions.state_before[1] == "SILENT"
    assert actions.action[1] is None
    assert actions.action_valid[1] is False
    assert actions.mask_reason[1] == COMPOUND_TRANSITION


def test_the_first_slot_has_no_known_state_and_is_masked():
    actions = slot_actions(_activity(*[True] * 5, *[True] * 5))

    assert actions.state_before[0] == "UNKNOWN"
    assert actions.action[0] is None
    assert actions.mask_reason[0] == RECORDING_START
    # Slot 1 starts from the state the first slot ended in.
    assert (actions.state_before[1], actions.action[1]) == ("SPEAKING", "NO_EVENT")


def test_actions_are_available_at_the_slot_end():
    actions = slot_actions(_activity(*[False] * 15))

    assert actions.slot_start_s == [0.0, 0.1, 0.2]
    assert actions.slot_end_s == [0.1, 0.2, 0.3]
    # zhat_(k+1) from (z_k, a_k): committed after the slot end, never at t_k.
    assert commit_time_s(2, preprocessing_lookahead_s=0.0) == 0.3
    assert commit_time_s(2, preprocessing_lookahead_s=0.02) == pytest.approx(0.32)
    with pytest.raises(ValueError):
        commit_time_s(2, preprocessing_lookahead_s=-0.01)


def test_actions_consume_no_future_audio():
    one = _speech([False, True, True, False, True, False, False, True])
    two = _speech([True, False, False, True, True, True, False, False])
    full = conversation_actions(_conversation(one, two))

    for slots in range(1, 8):
        prefix = conversation_actions(
            _conversation(one[: slots * SLOT], two[: slots * SLOT])
        )

        for speaker, actions in prefix.items():
            assert actions.action == full[speaker].action[:slots]
            assert actions.state_before == full[speaker].state_before[:slots]


def test_each_speaker_is_read_from_its_own_channel():
    one = _speech([False, True, True])
    actions = conversation_actions(_conversation(one, torch.zeros_like(one)))

    assert actions["speaker_1"].action == [None, "ONSET", "NO_EVENT"]
    assert actions["speaker_2"].action == [None, "NO_EVENT", "NO_EVENT"]


def test_only_complete_slots_exist():
    assert slot_count(3 * SLOT + SLOT - 1, RATE) == 3
    assert len(slot_actions(_activity(*[False] * 14)).action) == 2


def test_windows_must_tile_samples_and_slots():
    with pytest.raises(ValueError, match="whole number of samples"):
        rms_activity(torch.zeros(1000), 11_025)
    with pytest.raises(ValueError, match="do not tile"):
        slot_actions(_activity(*[False] * 10), window_s=0.03)


# ---------------------------------------------------------------------------
# Contract and provenance
# ---------------------------------------------------------------------------


def test_the_control_grid_is_v1_grid():
    assert CONTROL_RATE_HZ == GRID_RATE_HZ


def test_provenance_records_the_input_contract():
    provenance = input_provenance(split="dev", source_sample_rate=RATE)

    assert provenance["turnbench_dataset"]["revision"]
    assert str(list(INFERENCE_COLUMNS)) in provenance["turnbench_dataset"]["read"]
    assert input_provenance(split="train", source_sample_rate=RATE)[
        "turnbench_dataset"
    ]["repo"].startswith("otoearth/")
    with pytest.raises(ValueError, match="Unknown TurnBench split"):
        input_provenance(split="test", source_sample_rate=RATE)
    assert provenance["turnbench_reference"]["revision"]
    assert provenance["scene_mix_policy"].startswith("sum")
    assert provenance["source_sample_rate"] == RATE
    assert provenance["mimi_target_sample_rate"] == 24_000
    assert provenance["activity"]["window_s"] == 0.02
    assert provenance["activity"]["threshold"] == 0.01
    assert provenance["control_rate_hz"] == 10.0
    # No zero lookahead is claimed for the resampled Mimi path.
    assert provenance["preprocessing_lookahead_s"]["mimi_scene"] != 0
    assert "never at t_k" in provenance["commit_time_convention"]
