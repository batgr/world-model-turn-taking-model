import pytest

from turn_wm.data.window import build_window, validate_against_anchor


def test_anchor_is_last_context_step():
    window = build_window(
        anchor_idx=100,
        context_steps=30,
        future_steps=10,
    )

    assert window.context_start == 71
    assert window.context_end == 100
    assert window.future_start == 101
    assert window.future_end == 110


def test_window_reports_correct_lengths():
    window = build_window(
        anchor_idx=100,
        context_steps=30,
        future_steps=10,
    )

    assert window.context_steps == 30
    assert window.future_steps == 10


@pytest.mark.parametrize(
    ("anchor_idx", "context_steps", "future_steps"),
    [
        (-1, 10, 10),
        (10, 0, 10),
        (10, 10, 0),
    ],
)
def test_invalid_window_arguments_raise(
    anchor_idx,
    context_steps,
    future_steps,
):
    with pytest.raises(ValueError):
        build_window(
            anchor_idx=anchor_idx,
            context_steps=context_steps,
            future_steps=future_steps,
        )


def test_context_cannot_start_before_recording():
    with pytest.raises(ValueError):
        build_window(
            anchor_idx=5,
            context_steps=10,
            future_steps=10,
        )


def test_context_cannot_exceed_anchor_support():
    with pytest.raises(ValueError):
        validate_against_anchor(
            context_steps=51,
            future_steps=10,
            max_context_steps=50,
            available_future_steps=10,
        )


def test_future_cannot_exceed_anchor_support():
    with pytest.raises(ValueError):
        validate_against_anchor(
            context_steps=30,
            future_steps=11,
            max_context_steps=50,
            available_future_steps=10,
        )


def test_valid_requested_window_is_accepted():
    validate_against_anchor(
        context_steps=30,
        future_steps=10,
        max_context_steps=50,
        available_future_steps=10,
    )
