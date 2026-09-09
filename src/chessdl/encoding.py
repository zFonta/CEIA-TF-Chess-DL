"""Board -> tensor encoding, always from the side-to-move's point of view.

The network always sees the board as if it were white's turn. When black is to
move the board is mirrored, so *my* pieces are always in planes 0-5 and on the
lower half of the board, and the opponent's are always in planes 6-11. The
network therefore never has to learn two mirror-image representations of the
same concept, which roughly doubles the effective value of every training
example.

Because the orientation already encodes whose turn it is, no side-to-move plane
is needed.

Plane layout -- 18 planes of 8x8:

    0-5    own pieces:      P, N, B, R, Q, K
    6-11   opponent pieces: P, N, B, R, Q, K
    12-13  own castling rights:      kingside, queenside
    14-15  opponent castling rights: kingside, queenside
    16     en passant target square
    17     halfmove clock (50-move rule), scaled to [0, 1]

Row 0 of every plane is the side-to-move's first rank, and column 0 is the
a-file (after mirroring). The piece order is arbitrary as far as the network is
concerned -- it only has to be identical across training, validation and
inference -- and matches ``chess.PIECE_TYPES`` so the plane index is simply
``piece_type - 1``, with no translation table to get wrong.
"""

from __future__ import annotations

import chess
import numpy as np

#: Piece planes in ``chess.PIECE_TYPES`` order, so plane index == piece_type - 1.
PIECE_PLANE_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

N_PIECE_TYPES = len(PIECE_PLANE_ORDER)

PLANE_OWN_PIECES = 0
PLANE_OPPONENT_PIECES = N_PIECE_TYPES
PLANE_OWN_CASTLE_KINGSIDE = 12
PLANE_OWN_CASTLE_QUEENSIDE = 13
PLANE_OPP_CASTLE_KINGSIDE = 14
PLANE_OPP_CASTLE_QUEENSIDE = 15
PLANE_EN_PASSANT = 16
PLANE_HALFMOVE_CLOCK = 17

N_PLANES = 18
BOARD_SIZE = 8
TENSOR_SHAPE = (N_PLANES, BOARD_SIZE, BOARD_SIZE)

#: The halfmove clock reaches 100 when the 50-move rule forces a draw.
HALFMOVE_CLOCK_MAX = 100


def to_stm_perspective(board: chess.Board) -> chess.Board:
    """Return the board as seen by the side to move, i.e. always white to move.

    ``chess.Board.mirror()`` flips the board vertically *and* swaps piece
    colours, the side to move, the castling rights and the en passant square.
    Mirroring is by rank, not by file, so castling geometry (king on the e-file)
    is preserved.
    """
    return board if board.turn == chess.WHITE else board.mirror()


def board_to_tensor(board: chess.Board, dtype: type = np.float32) -> np.ndarray:
    """Encode a position as a ``(18, 8, 8)`` tensor from the mover's perspective."""
    stm_board = to_stm_perspective(board)
    planes = np.zeros(TENSOR_SHAPE, dtype=dtype)

    for square, piece in stm_board.piece_map().items():
        # After `to_stm_perspective` the side to move is always white, so white
        # is "us" and black is "them".
        base = PLANE_OWN_PIECES if piece.color == chess.WHITE else PLANE_OPPONENT_PIECES
        plane = base + piece.piece_type - 1
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1.0

    if stm_board.has_kingside_castling_rights(chess.WHITE):
        planes[PLANE_OWN_CASTLE_KINGSIDE] = 1.0
    if stm_board.has_queenside_castling_rights(chess.WHITE):
        planes[PLANE_OWN_CASTLE_QUEENSIDE] = 1.0
    if stm_board.has_kingside_castling_rights(chess.BLACK):
        planes[PLANE_OPP_CASTLE_KINGSIDE] = 1.0
    if stm_board.has_queenside_castling_rights(chess.BLACK):
        planes[PLANE_OPP_CASTLE_QUEENSIDE] = 1.0

    # `has_legal_en_passant()` rather than a plain `ep_square` check: `Board.fen()`
    # only writes the en passant square when the capture is actually legal, so
    # keying off the legal capture is what makes encoding a live board and
    # encoding its round-tripped FEN produce the same tensor.
    if stm_board.has_legal_en_passant():
        ep = stm_board.ep_square
        planes[PLANE_EN_PASSANT, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    clock = min(stm_board.halfmove_clock, HALFMOVE_CLOCK_MAX) / HALFMOVE_CLOCK_MAX
    planes[PLANE_HALFMOVE_CLOCK] = clock

    return planes


def fen_to_tensor(fen: str, dtype: type = np.float32) -> np.ndarray:
    """Encode a position given as a FEN string."""
    return board_to_tensor(chess.Board(fen), dtype=dtype)
