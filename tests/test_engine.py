"""The engine: one-ply search over the value network (WBS 5, requirement 1.6).

These tests are about chess, not about software, and that is deliberate. The two
failure modes that matter here both produce an engine that *works* -- legal
moves, no exceptions, a confident evaluation printed next to each one -- and
plays badly:

* **The sign backwards.** The network scores the position after the move, which
  belongs to the opponent, so the move to play is the one that *minimises* it.
  Get that wrong and the engine picks the worst legal move every time.
* **Terminal positions sent to the network.** The dataset excluded them, so the
  network has never seen a checkmate. An engine that asks it about one ranks
  mate below some quiet move and misses mate in one.

Neither is visible in a shape assertion. What catches them is a position whose
right answer a chess player knows, so most of what follows is exactly that.

The tests run against a **stub evaluator** built on material counting rather
than a trained network. That is not a shortcut: it makes the expected move
unambiguous, the tests fast, and -- most importantly -- it means a failure points
at the search, which is what is under test, instead of at the quality of some
checkpoint.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.engine.evaluator import DRAW, LOSS, Evaluator, terminal_value  # noqa: E402
from chessdl.engine.search import (  # noqa: E402
    GameOverError,
    best_move,
    search,
    value_white,
)

PIEZAS = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
          chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


class MaterialNet(torch.nn.Module):
    """Scores a position by material, from the side to move's point of view.

    Reads the piece planes of the encoding the same way the real networks do, so
    it exercises the whole path -- encode, batch, device -- with an answer that
    is known in advance.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pesos = torch.tensor([PIEZAS[p] for p in chess.PIECE_TYPES], dtype=x.dtype)
        propias = x[:, 0:6].sum(dim=(2, 3)) @ pesos
        rivales = x[:, 6:12].sum(dim=(2, 3)) @ pesos
        return torch.tanh((propias - rivales) / 10.0)


@pytest.fixture
def evaluador() -> Evaluator:
    return Evaluator(MaterialNet(), device="cpu")


def tablero(*piezas: tuple[chess.Square, chess.PieceType, chess.Color],
            turno: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board()
    board.clear()
    board.turn = turno
    for casilla, tipo, color in piezas:
        board.set_piece_at(casilla, chess.Piece(tipo, color))
    return board


class TestTerminalValue:
    """Decided positions are resolved from the rules, never from the network."""

    def test_checkmate_is_the_worst_value_for_the_side_to_move(self):
        # Mate del loco: 1.f3 e5 2.g4 Dh4#
        board = chess.Board()
        for move in ["f3", "e5", "g4", "Qh4"]:
            board.push_san(move)
        assert board.is_checkmate()
        assert terminal_value(board) == LOSS

    def test_stalemate_is_a_draw(self):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        assert board.is_stalemate()
        assert terminal_value(board) == DRAW

    def test_insufficient_material_is_a_draw(self):
        assert terminal_value(chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")) == DRAW

    def test_a_normal_position_is_not_decided(self):
        assert terminal_value(chess.Board()) is None

    def test_a_threefold_repetition_counts_as_a_draw(self):
        """Otherwise a winning engine has no reason to stop shuffling."""
        board = chess.Board()
        for _ in range(2):
            for move in ["Nf3", "Nf6", "Ng1", "Ng8"]:
                board.push_san(move)
        assert terminal_value(board) == DRAW
        assert terminal_value(board, claim_draws=False) is None


class TestTheNetworkNeverSeesADecidedPosition:
    def test_a_mate_is_answered_from_the_rules(self, evaluador):
        board = chess.Board()
        for move in ["f3", "e5", "g4", "Qh4"]:
            board.push_san(move)

        assert evaluador.evaluate_board(board) == LOSS
        assert evaluador.network_calls == 0, "le pregunto a la red por un mate"
        assert evaluador.terminal_calls == 1

    def test_a_mixed_batch_routes_each_position_correctly(self, evaluador):
        mate = chess.Board()
        for move in ["f3", "e5", "g4", "Qh4"]:
            mate.push_san(move)

        valores = evaluador.evaluate([chess.Board(), mate, chess.Board()])

        assert valores[1] == LOSS
        assert evaluador.network_calls == 2
        assert evaluador.terminal_calls == 1


class TestItPlaysChess:
    """The answers a player knows, which is what catches a flipped sign."""

    def test_it_finds_mate_in_one(self, evaluador):
        """The single most important test in this file.

        It fails both ways: with the sign backwards the engine avoids the mate,
        and with terminal positions left to the network it never sees it.
        """
        # Rey negro en h8, dama blanca en g6, rey blanco en f6: Dg7#.
        board = tablero(
            (chess.H8, chess.KING, chess.BLACK),
            (chess.G6, chess.QUEEN, chess.WHITE),
            (chess.F6, chess.KING, chess.WHITE),
        )
        elegida = best_move(board, evaluador)

        board.push(elegida)
        assert board.is_checkmate(), f"jugo {elegida}, que no da mate"

    def test_it_captures_a_free_queen(self, evaluador):
        """Checked by what the move does, not by how it is written.

        The first version of this test compared against the string ``"Rxa8"``
        and failed on a correct engine: the move gives check, so its notation is
        ``"Rxa8+"``.
        """
        board = tablero(
            (chess.E1, chess.KING, chess.WHITE),
            (chess.A1, chess.ROOK, chess.WHITE),
            (chess.E8, chess.KING, chess.BLACK),
            (chess.A8, chess.QUEEN, chess.BLACK),
        )
        elegida = best_move(board, evaluador)

        assert elegida.to_square == chess.A8
        assert board.piece_at(elegida.to_square).piece_type == chess.QUEEN

    def test_it_avoids_stalemate_when_winning(self, evaluador):
        """A draw is worse than a winning position, and the engine must see it.

        Black has only the king on h8; White to move can play Qg6, which is
        stalemate, or keep the win. Handing the network the stalemate would give
        it whatever material says -- a huge advantage -- and the engine would
        throw the game away.
        """
        board = tablero(
            (chess.H8, chess.KING, chess.BLACK),
            (chess.F7, chess.QUEEN, chess.WHITE),
            (chess.F6, chess.KING, chess.WHITE),
        )
        elegida = best_move(board, evaluador)
        board.push(elegida)
        assert not board.is_stalemate(), f"jugo {elegida} y ahogo"

    def test_the_move_is_always_legal(self, evaluador):
        rng = np.random.default_rng(0)
        board = chess.Board()
        for _ in range(24):
            if board.is_game_over():
                break
            elegida = best_move(board, evaluador, rng)
            assert elegida in board.legal_moves
            board.push(elegida)


class TestWhatOnePlyCannotDo:
    """The limit of the method, pinned down rather than wished away.

    A one-ply search evaluates the position after *its own* move and stops. The
    opponent's reply is not in the tree at all, so nothing in the search can see
    a recapture. An earlier version of this file asserted the opposite -- that
    the engine would not hang its queen -- and failed against a perfectly
    correct engine.

    Worth being precise about where the limit actually falls, because it is not
    where it looks. The search cannot see the reply; the **evaluation function
    can**. These tests use a material stub that knows nothing but piece counts,
    so here the limitation is total. A network trained on Stockfish labels is a
    different matter: Stockfish's evaluation of the resulting position already
    accounts for what is hanging, so a trained network absorbs part of the
    tactics into the leaf value. How much of it survives is a question for the
    games of block 6, not something to assert here.
    """

    def test_a_recapture_is_invisible_to_the_search(self, evaluador):
        """Winning a rook and losing the queen to it reads as +4 at one ply."""
        board = tablero(
            (chess.E1, chess.KING, chess.WHITE),
            (chess.D1, chess.QUEEN, chess.WHITE),
            (chess.E8, chess.KING, chess.BLACK),
            (chess.D8, chess.ROOK, chess.BLACK),
        )
        elegida = best_move(board, evaluador)

        assert elegida.to_square == chess.D8, "con material puro, se lleva la torre"
        board.push(elegida)
        # Y el rey la recupera en la jugada siguiente, que la busqueda no miro.
        assert any(m.to_square == chess.D8 for m in board.legal_moves)


class TestSignConvention:
    def test_the_reported_value_belongs_to_the_side_that_moved(self, evaluador):
        """Positive means the engine came out ahead, whichever colour it is."""
        board = tablero(
            (chess.E1, chess.KING, chess.WHITE),
            (chess.A1, chess.ROOK, chess.WHITE),
            (chess.E8, chess.KING, chess.BLACK),
            (chess.A8, chess.QUEEN, chess.BLACK),
        )
        resultado = search(board, evaluador)
        assert resultado.value > 0, "capturar la dama tiene que dar valor positivo"

    def test_the_same_position_mirrored_gives_the_mirrored_value(self, evaluador):
        """The engine must not have a colour it prefers."""
        blancas = tablero(
            (chess.E1, chess.KING, chess.WHITE),
            (chess.D1, chess.QUEEN, chess.WHITE),
            (chess.E8, chess.KING, chess.BLACK),
        )
        negras = tablero(
            (chess.E8, chess.KING, chess.BLACK),
            (chess.D8, chess.QUEEN, chess.BLACK),
            (chess.E1, chess.KING, chess.WHITE),
            turno=chess.BLACK,
        )
        assert search(blancas, evaluador).value == pytest.approx(
            search(negras, evaluador).value, abs=1e-6
        )

    def test_white_perspective_flips_with_the_turn(self):
        blancas = chess.Board()
        negras = chess.Board()
        negras.turn = chess.BLACK
        assert value_white(blancas, 0.4) == pytest.approx(0.4)
        assert value_white(negras, 0.4) == pytest.approx(-0.4)


class TestSearchResult:
    def test_every_legal_move_is_ranked(self, evaluador):
        board = chess.Board()
        resultado = search(board, evaluador)
        assert len(resultado.ranked) == board.legal_moves.count() == 20

    def test_the_ranking_is_sorted_best_first(self, evaluador):
        valores = [v for _, v in search(chess.Board(), evaluador).ranked]
        assert valores == sorted(valores, reverse=True)

    def test_the_chosen_move_heads_the_ranking(self, evaluador):
        resultado = search(chess.Board(), evaluador)
        assert resultado.value == pytest.approx(resultado.ranked[0][1])

    def test_it_reports_centipawns(self, evaluador):
        """Requirement 4.2: the engine reports on the centipawn scale."""
        board = tablero(
            (chess.E1, chess.KING, chess.WHITE),
            (chess.A1, chess.ROOK, chess.WHITE),
            (chess.E8, chess.KING, chess.BLACK),
            (chess.A8, chess.QUEEN, chess.BLACK),
        )
        assert search(board, evaluador).centipawns > 0

    def test_the_table_renders_in_algebraic_notation(self, evaluador):
        board = chess.Board()
        assert "e4" in search(board, evaluador).table(board, top=20)


class TestOneBatchPerMove:
    def test_all_the_legal_moves_go_out_together(self, evaluador):
        """What keeps a move inside the budget of requirement 1.7."""
        board = chess.Board()
        search(board, evaluador)
        assert evaluador.network_calls == 20
        assert evaluador.terminal_calls == 0

    def test_the_board_is_left_exactly_as_it_was(self, evaluador):
        board = chess.Board()
        antes = board.fen()
        movimientos = len(board.move_stack)
        search(board, evaluador)
        assert board.fen() == antes
        assert len(board.move_stack) == movimientos


class TestDeterminism:
    def test_without_a_generator_the_choice_repeats(self, evaluador):
        board = chess.Board()
        assert best_move(board, evaluador) == best_move(board, evaluador)

    def test_a_finished_game_raises_instead_of_guessing(self, evaluador):
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        with pytest.raises(GameOverError, match="ya termino"):
            best_move(board, evaluador)
