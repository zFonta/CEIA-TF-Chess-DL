"""Tests for the side-to-move board encoding.

These are the most important tests in the data pipeline: the whole design rests
on the claim that mirroring the board when black is to move puts "my" pieces in
a fixed set of planes without corrupting castling rights or the en passant
square. That claim is verified here rather than assumed.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessdl.encoding import (
    N_PLANES,
    PIECE_PLANE_ORDER,
    PLANE_EN_PASSANT,
    PLANE_HALFMOVE_CLOCK,
    PLANE_OPP_CASTLE_KINGSIDE,
    PLANE_OPP_CASTLE_QUEENSIDE,
    PLANE_OPPONENT_PIECES,
    PLANE_OWN_CASTLE_KINGSIDE,
    PLANE_OWN_CASTLE_QUEENSIDE,
    PLANE_OWN_PIECES,
    TENSOR_SHAPE,
    board_to_tensor,
    fen_to_tensor,
    to_stm_perspective,
)

OWN_PIECE_PLANES = slice(PLANE_OWN_PIECES, PLANE_OWN_PIECES + 6)
OPPONENT_PIECE_PLANES = slice(PLANE_OPPONENT_PIECES, PLANE_OPPONENT_PIECES + 6)


def mirrored_square(square: int) -> int:
    """The square a vertical (rank) mirror maps ``square`` onto."""
    return chess.square(chess.square_file(square), 7 - chess.square_rank(square))


def en_passant_position() -> chess.Board:
    """A position with black to move and a legal en passant capture available.

    1. e4 d5  2. Nf3 d4  3. c4 -- black's d4 pawn may capture c3 en passant.
    """
    board = chess.Board()
    for san in ("e4", "d5", "Nf3", "d4", "c4"):
        board.push_san(san)
    assert board.turn == chess.BLACK
    assert board.has_legal_en_passant()
    return board


# --- basic shape -----------------------------------------------------------


def test_tensor_shape_and_dtype():
    tensor = board_to_tensor(chess.Board())
    assert tensor.shape == TENSOR_SHAPE == (N_PLANES, 8, 8)
    assert tensor.dtype == np.float32


def test_piece_plane_order_matches_python_chess():
    """Plane index is `piece_type - 1`, with no translation table in between."""
    assert PIECE_PLANE_ORDER == tuple(chess.PIECE_TYPES)
    for index, piece_type in enumerate(PIECE_PLANE_ORDER):
        assert piece_type - 1 == index


def test_starting_position_counts():
    tensor = board_to_tensor(chess.Board())
    assert tensor[OWN_PIECE_PLANES].sum() == 16
    assert tensor[OPPONENT_PIECE_PLANES].sum() == 16
    # Eight pawns each, on the second rank from each side's own point of view.
    assert tensor[PLANE_OWN_PIECES + chess.PAWN - 1, 1].sum() == 8
    assert tensor[PLANE_OPPONENT_PIECES + chess.PAWN - 1, 6].sum() == 8


def test_each_square_holds_at_most_one_piece():
    board = en_passant_position()
    occupancy = board_to_tensor(board)[0:12].sum(axis=0)
    assert occupancy.max() <= 1.0


# --- the mirroring itself --------------------------------------------------


def test_perspective_is_a_noop_for_white():
    board = chess.Board()
    assert to_stm_perspective(board).fen() == board.fen()


def test_perspective_makes_it_white_to_move():
    board = en_passant_position()
    assert to_stm_perspective(board).turn == chess.WHITE


def test_own_pieces_land_in_planes_0_to_5_for_both_colours():
    """The point of the whole design: 'my' pieces always occupy the same planes."""
    board = chess.Board("4k3/3p4/8/8/8/8/3P4/4K3 b - - 0 1")  # black to move
    tensor = board_to_tensor(board)

    # Black's king (e8) and pawn (d7) mirror onto e1 and d2 and become "own".
    own_king = tensor[PLANE_OWN_PIECES + chess.KING - 1]
    own_pawn = tensor[PLANE_OWN_PIECES + chess.PAWN - 1]
    assert own_king[chess.square_rank(chess.E1), chess.square_file(chess.E1)] == 1.0
    assert own_pawn[chess.square_rank(chess.D2), chess.square_file(chess.D2)] == 1.0

    # White's king (e1) and pawn (d2) become the opponent's, on e8 and d7.
    opp_king = tensor[PLANE_OPPONENT_PIECES + chess.KING - 1]
    opp_pawn = tensor[PLANE_OPPONENT_PIECES + chess.PAWN - 1]
    assert opp_king[chess.square_rank(chess.E8), chess.square_file(chess.E8)] == 1.0
    assert opp_pawn[chess.square_rank(chess.D7), chess.square_file(chess.D7)] == 1.0


def test_mirrored_position_encodes_identically():
    """A position and its colour-swapped mirror are the same thing to the network."""
    board = en_passant_position()
    np.testing.assert_array_equal(
        board_to_tensor(board), board_to_tensor(board.mirror())
    )


def test_symmetric_positions_with_opposite_turns_encode_identically():
    white_to_move = chess.Board("4k3/3p4/8/8/8/8/3P4/4K3 w - - 0 1")
    black_to_move = chess.Board("4k3/3p4/8/8/8/8/3P4/4K3 b - - 0 1")
    np.testing.assert_array_equal(
        board_to_tensor(white_to_move), board_to_tensor(black_to_move)
    )


# --- special cases the plan calls out --------------------------------------


def test_castling_rights_follow_the_mirror():
    """Requirement 1.5's special cases: black's own rights must not land in the
    opponent's planes after mirroring."""
    # Black to move; only black keeps kingside castling, only white queenside.
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Qk - 4 10")
    tensor = board_to_tensor(board)

    assert tensor[PLANE_OWN_CASTLE_KINGSIDE].all()
    assert not tensor[PLANE_OWN_CASTLE_QUEENSIDE].any()
    assert not tensor[PLANE_OPP_CASTLE_KINGSIDE].any()
    assert tensor[PLANE_OPP_CASTLE_QUEENSIDE].all()


def test_castling_rights_for_white_to_move_are_not_swapped():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 4 10")
    tensor = board_to_tensor(board)

    assert tensor[PLANE_OWN_CASTLE_KINGSIDE].all()
    assert not tensor[PLANE_OWN_CASTLE_QUEENSIDE].any()
    assert not tensor[PLANE_OPP_CASTLE_KINGSIDE].any()
    assert tensor[PLANE_OPP_CASTLE_QUEENSIDE].all()


def test_en_passant_square_is_mirrored_to_the_right_square():
    board = en_passant_position()
    tensor = board_to_tensor(board)

    assert board.ep_square == chess.C3
    expected = mirrored_square(chess.C3)
    assert expected == chess.C6

    ep_plane = tensor[PLANE_EN_PASSANT]
    assert ep_plane.sum() == 1.0
    assert ep_plane[chess.square_rank(expected), chess.square_file(expected)] == 1.0


def test_en_passant_plane_is_empty_without_a_legal_capture():
    """A double push that no pawn can answer must not set the plane.

    `Board.fen()` omits the en passant square unless the capture is legal, so
    keying off the legal capture is what keeps a live board and its round-tripped
    FEN encoding to the same tensor.
    """
    board = chess.Board()
    board.push_san("e4")  # nothing can capture en passant
    assert board.ep_square == chess.E3
    assert not board.has_legal_en_passant()
    assert board_to_tensor(board)[PLANE_EN_PASSANT].sum() == 0.0


def test_encoding_survives_a_fen_round_trip():
    """Whatever we store in Parquet must encode the same as the live board."""
    for board in (chess.Board(), en_passant_position()):
        np.testing.assert_array_equal(
            board_to_tensor(board), fen_to_tensor(board.fen())
        )


def test_promotion_is_encoded_as_the_promoted_piece():
    board = chess.Board("8/P6k/8/8/8/8/8/7K w - - 0 1")
    board.push_san("a8=Q")
    tensor = board_to_tensor(board)  # black to move now, so white is the opponent

    opp_queens = tensor[PLANE_OPPONENT_PIECES + chess.QUEEN - 1]
    opp_pawns = tensor[PLANE_OPPONENT_PIECES + chess.PAWN - 1]
    mirrored_a8 = mirrored_square(chess.A8)
    assert opp_queens[chess.square_rank(mirrored_a8), chess.square_file(mirrored_a8)] == 1.0
    assert opp_pawns.sum() == 0.0


# --- auxiliary planes ------------------------------------------------------


@pytest.mark.parametrize(
    "halfmove_clock, expected",
    [(0, 0.0), (25, 0.25), (100, 1.0), (150, 1.0)],
)
def test_halfmove_clock_plane_is_scaled_and_capped(halfmove_clock, expected):
    board = chess.Board(f"4k3/8/8/8/8/8/8/4K3 w - - {halfmove_clock} 1")
    tensor = board_to_tensor(board)
    assert tensor[PLANE_HALFMOVE_CLOCK].min() == pytest.approx(expected)
    assert tensor[PLANE_HALFMOVE_CLOCK].max() == pytest.approx(expected)


def test_no_side_to_move_plane_is_needed():
    """The orientation encodes the turn, so nothing else has to.

    If a plane were carrying the side to move, the two symmetric positions in
    `test_symmetric_positions_with_opposite_turns_encode_identically` could not
    be identical -- this test states the intent explicitly.
    """
    white_to_move = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    black_to_move = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")
    assert board_to_tensor(white_to_move).sum() == board_to_tensor(black_to_move).sum()
