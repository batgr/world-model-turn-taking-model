"""Synthetic canonical corpora shared by multi-corpus tests."""

from datasets import Dataset, DatasetDict

from turn_wm.data.source import LoadedCorpus

GRID_LENGTH = 40


def make_grid(*, dataset: str, state: str, recording_id: str = "r1") -> Dataset:
    """Grid whose rows all carry `state`, so corpora are distinguishable."""

    return Dataset.from_dict(
        {
            "dataset": [dataset] * GRID_LENGTH,
            "recording_id": [recording_id] * GRID_LENGTH,
            "decision_index": list(range(GRID_LENGTH)),
            "decision_time_s": [i / 10 for i in range(GRID_LENGTH)],
            "focal_state_before": [state] * GRID_LENGTH,
            "action": ["NO_EVENT"] * GRID_LENGTH,
            "action_valid": [True] * GRID_LENGTH,
        }
    )


def make_anchors(
    *,
    dataset: str,
    split: str,
    count: int,
    sample_class: str = "event",
    is_trainable: bool = True,
    recording_id: str = "r1",
) -> Dataset:
    # Every corpus uses the same local rows (19, 20, ...).
    rows = [19 + i for i in range(count)]

    return Dataset.from_dict(
        {
            "sample_id": [f"{dataset}:{split}#{row}" for row in rows],
            "dataset": [dataset] * count,
            "recording_id": [recording_id] * count,
            "split": [split] * count,
            "anchor_idx": rows,
            "anchor_row": rows,
            "anchor_time": [row / 10 for row in rows],
            "max_context_steps": [10] * count,
            "future_steps": [10] * count,
            "sample_class": [sample_class] * count,
            "is_trainable": [is_trainable] * count,
        }
    )


def make_corpus(
    name: str,
    *,
    state: str,
    splits: dict[str, int],
    sample_class: str = "event",
    is_trainable: bool = True,
    media_manifest: Dataset | None = None,
) -> LoadedCorpus:
    return LoadedCorpus(
        name=name,
        model_ready=DatasetDict(
            {
                split: make_anchors(
                    dataset=name,
                    split=split,
                    count=count,
                    sample_class=sample_class,
                    is_trainable=is_trainable,
                )
                for split, count in splits.items()
            }
        ),
        action_grid=make_grid(dataset=name, state=state),
        metadata={},
        media_manifest=media_manifest,
    )


def make_manifest(*, dataset: str, media_offset_s: float = 0.0) -> Dataset:
    return Dataset.from_list(
        [
            {
                "dataset": dataset,
                "recording_id": "r1",
                "video_path": "videos/r1.mp4",
                "audio_path": None,
                "media_offset_s": media_offset_s,
                "video_has_audio": True,
            }
        ]
    )
