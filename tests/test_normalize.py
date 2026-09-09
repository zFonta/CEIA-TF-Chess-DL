"""Tests for the centipawn <-> value transforms (requirements 1.1 and 4.2)."""

from __future__ import annotations

import math

import chess
import chess.engine
import pytest

from chessdl.normalize import (
    DEFAULT_CP_CLIP,
    DEFAULT_SCALE,
    Label,
    clip_cp,
    cp_to_value,
    label_from_povscore,
    score_to_cp,
    value_to_cp,
)


def test_equal_position_maps_to_zero():
    assert cp_to_value(0) == 0.0
    assert value_to_cp(0.0) == 0.0


def test_sign_follows_the_advantage():
    assert cp_to_value(300) > 0
    assert cp_to_value(-300) < 0


def test_values_stay_inside_the_unit_range():
    for cp in (-100_000, -DEFAULT_CP_CLIP, -1, 0, 1, DEFAULT_CP_CLIP, 100_000):
        assert -1.0 < cp_to_value(cp) < 1.0


@pytest.mark.parametrize("cp", [-2000, -1500, -400, -50, 0, 50, 400, 1500, 2000])
def test_round_trip_is_lossless_inside_the_clip_range(cp):
    """Requirement 4.2: the inverse must recover the original centipawns."""
    assert value_to_cp(cp_to_value(cp)) == pytest.approx(cp, abs=1e-6)


def test_scores_beyond_the_clip_saturate():
    assert clip_cp(5000) == DEFAULT_CP_CLIP
    assert clip_cp(-5000) == -DEFAULT_CP_CLIP
    assert cp_to_value(5000) == cp_to_value(DEFAULT_CP_CLIP)


def test_inverse_is_finite_for_a_saturated_prediction():
    """A network that outputs exactly +/-1 must not produce an infinite score."""
    for value in (-1.0, 1.0, -1.5, 1.5):
        cp = value_to_cp(value)
        assert math.isfinite(cp)
        assert abs(cp) == pytest.approx(DEFAULT_CP_CLIP, abs=1e-6)


def test_transform_is_strictly_monotonic():
    values = [cp_to_value(cp) for cp in range(-DEFAULT_CP_CLIP, DEFAULT_CP_CLIP + 1, 100)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_scale_controls_the_steepness():
    """One pawn at the default scale sits where tanh(100/400) does."""
    assert cp_to_value(100) == pytest.approx(math.tanh(100 / DEFAULT_SCALE))


# --- score_to_cp -----------------------------------------------------------


def test_score_to_cp_passes_through_ordinary_scores():
    assert score_to_cp(chess.engine.Cp(250)) == (250, None)


def test_score_to_cp_pins_mate_to_the_clip_bound():
    cp, mate_in = score_to_cp(chess.engine.Mate(3))
    assert cp == DEFAULT_CP_CLIP
    assert mate_in == 3

    cp, mate_in = score_to_cp(chess.engine.Mate(-2))
    assert cp == -DEFAULT_CP_CLIP
    assert mate_in == -2


# --- label_from_povscore ---------------------------------------------------


def test_label_with_white_to_move_keeps_the_sign():
    label = label_from_povscore(
        chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE), chess.WHITE
    )
    assert label.cp_white == 120
    assert label.cp_stm == 120
    assert label.value_white == pytest.approx(label.value_stm)
    assert not label.is_mate
    assert label.mate_in == 0


def test_label_with_black_to_move_flips_the_sign():
    """The engine score is relative to the mover; the stored white view is not."""
    # Black to move and black is a pawn up: +100 for the mover, -100 for white.
    label = label_from_povscore(
        chess.engine.PovScore(chess.engine.Cp(100), chess.BLACK), chess.BLACK
    )
    assert label.cp_white == -100
    assert label.cp_stm == 100
    assert label.value_white == pytest.approx(-label.value_stm)


@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
@pytest.mark.parametrize("cp", [-1800, -250, 0, 250, 1800])
def test_label_sign_invariant_holds(turn, cp):
    """The invariant the whole two-point-of-view design rests on."""
    label = label_from_povscore(
        chess.engine.PovScore(chess.engine.Cp(cp), turn), turn
    )
    expected = label.value_stm if turn == chess.WHITE else -label.value_stm
    assert label.value_white == pytest.approx(expected)
    assert label.cp_white == (label.cp_stm if turn == chess.WHITE else -label.cp_stm)


def test_label_records_mate_distance():
    label = label_from_povscore(
        chess.engine.PovScore(chess.engine.Mate(2), chess.BLACK), chess.BLACK
    )
    # Black mates in 2, so from white's point of view it is mate in -2.
    assert label.is_mate
    assert label.mate_in == -2
    assert label.cp_white == -DEFAULT_CP_CLIP
    assert label.value_white < -0.99


def test_label_is_immutable():
    label = label_from_povscore(
        chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE), chess.WHITE
    )
    assert isinstance(label, Label)
    with pytest.raises(Exception):
        label.cp_white = 1  # type: ignore[misc]
