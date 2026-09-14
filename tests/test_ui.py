"""The click-to-play board: what a pair of clicks means.

An interactive board fails differently from the rest of this project. Nothing
raises. The board keeps drawing, the pieces keep moving, and what is wrong is
that a click did something other than what the person meant -- a capture read as
a reselection, a promotion that silently became a queen, an undo that handed the
move back to the wrong side. None of that is visible from a screenshot and none
of it can be caught by driving widgets, so the click logic lives apart from the
widgets (:mod:`chessdl.ui.game`) and is tested here as ordinary function calls.

As in ``test_engine.py``, the engine underneath is a stub that counts material.
The question in this file is never "was that a good move" but "was that the move
the clicks named", and a network would only make the answer harder to read.
"""

from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.engine.evaluator import Evaluator  # noqa: E402
from chessdl.ui.game import Click, PlayState  # noqa: E402

PIEZAS = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
          chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


class MaterialNet(torch.nn.Module):
    """Material counting through the real encoding, as in ``test_engine.py``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pesos = torch.tensor([PIEZAS[p] for p in chess.PIECE_TYPES], dtype=x.dtype)
        propias = x[:, 0:6].sum(dim=(2, 3)) @ pesos
        rivales = x[:, 6:12].sum(dim=(2, 3)) @ pesos
        return torch.tanh((propias - rivales) / 10.0)


@pytest.fixture
def evaluator() -> Evaluator:
    return Evaluator(MaterialNet(), device="cpu")


@pytest.fixture
def partida(evaluator: Evaluator) -> PlayState:
    return PlayState(evaluator=evaluator, depth=1)


def jugar(estado: PlayState, origen: str, destino: str) -> Click:
    """Two clicks, the way a person makes them."""
    estado.click(chess.parse_square(origen))
    return estado.click(chess.parse_square(destino))


class TestSeleccion:
    """Picking a piece up, and putting it back down."""

    def test_clicking_an_own_piece_selects_it(self, partida: PlayState) -> None:
        assert partida.click(chess.E2) is Click.SELECTED
        assert partida.selected == chess.E2

    def test_the_selection_offers_exactly_the_legal_destinations(
        self, partida: PlayState
    ) -> None:
        partida.click(chess.G1)
        assert set(partida.destinations()) == {chess.F3, chess.H3}

    def test_clicking_the_selected_piece_again_drops_it(self, partida: PlayState) -> None:
        partida.click(chess.E2)
        assert partida.click(chess.E2) is Click.CLEARED
        assert partida.selected is None

    def test_a_piece_with_no_moves_is_not_a_selection(self, partida: PlayState) -> None:
        # The rook on a1 is boxed in at the start. Selecting it would leave the
        # board looking live with nothing at all to click.
        assert partida.click(chess.A1) is Click.IGNORED
        assert partida.selected is None

    def test_a_failed_selection_does_not_lose_the_current_one(
        self, partida: PlayState
    ) -> None:
        partida.click(chess.E2)
        partida.click(chess.A1)                 # the boxed-in rook
        assert partida.selected == chess.E2

    def test_clicking_an_enemy_piece_with_nothing_selected_does_nothing(
        self, partida: PlayState
    ) -> None:
        assert partida.click(chess.E7) is Click.IGNORED
        assert partida.selected is None

    def test_clicking_another_own_piece_reselects(self, partida: PlayState) -> None:
        partida.click(chess.E2)
        assert partida.click(chess.D2) is Click.RESELECTED
        assert partida.selected == chess.D2


class TestJugadas:
    """Completing a move, including the ones with a second meaning."""

    def test_two_clicks_play_the_move(self, partida: PlayState) -> None:
        assert jugar(partida, "e2", "e4") is Click.MOVED
        assert partida.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)
        assert partida.history == ["e4"]

    def test_the_history_records_algebraic_notation_of_the_move_played(
        self, evaluator: Evaluator
    ) -> None:
        # SAN depends on the position *before* the move; recording it afterwards
        # is a classic off-by-one that produces a plausible-looking wrong list.
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
        )
        jugar(estado, "e4", "d5")
        assert estado.history == ["exd5"]

    def test_a_capture_is_a_move_and_not_a_reselection(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("4k3/8/8/3r4/8/8/8/3RK3 w - - 0 1"),
        )
        assert jugar(estado, "d1", "d5") is Click.MOVED
        assert estado.board.piece_at(chess.D5).color == chess.WHITE

    def test_clicking_the_rook_castles(self, evaluator: Evaluator) -> None:
        # python-chess writes castling as e1g1, so the rook's square is not a
        # legal destination -- yet clicking the rook is how plenty of people
        # castle, and the intent is unambiguous.
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        )
        estado.click(chess.E1)
        assert estado.click(chess.H1) is Click.MOVED
        assert estado.history == ["O-O"]
        assert estado.board.piece_at(chess.G1).piece_type == chess.KING

    def test_clicking_the_other_rook_castles_the_other_way(
        self, evaluator: Evaluator
    ) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        )
        estado.click(chess.E1)
        estado.click(chess.A1)
        assert estado.history == ["O-O-O"]

    def test_a_rook_click_without_castling_rights_is_an_ordinary_reselection(
        self, evaluator: Evaluator
    ) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1"),
        )
        estado.click(chess.E1)
        assert estado.click(chess.H1) is Click.RESELECTED
        assert estado.selected == chess.H1


class TestCoronacion:
    """The move that cannot be finished by clicking a square."""

    @pytest.fixture
    def al_borde(self, evaluator: Evaluator) -> PlayState:
        return PlayState(
            evaluator=evaluator,
            board=chess.Board("8/4P3/8/8/8/8/8/4K1k1 w - - 0 1"),
            depth=1,
        )

    def test_a_promotion_asks_instead_of_assuming_a_queen(self, al_borde: PlayState) -> None:
        assert jugar(al_borde, "e7", "e8") is Click.PROMOTION
        assert al_borde.pending_promotion == (chess.E7, chess.E8)
        assert al_borde.board.piece_at(chess.E8) is None    # nothing played yet

    def test_the_chosen_piece_is_the_one_that_appears(self, al_borde: PlayState) -> None:
        jugar(al_borde, "e7", "e8")
        assert al_borde.promote(chess.KNIGHT) is Click.MOVED
        assert al_borde.board.piece_at(chess.E8) == chess.Piece(chess.KNIGHT, chess.WHITE)
        assert al_borde.history == ["e8=N"]

    def test_clicks_are_ignored_while_the_choice_is_pending(
        self, al_borde: PlayState
    ) -> None:
        jugar(al_borde, "e7", "e8")
        assert al_borde.click(chess.E1) is Click.IGNORED
        assert al_borde.pending_promotion is not None

    def test_the_choice_can_be_cancelled(self, al_borde: PlayState) -> None:
        jugar(al_borde, "e7", "e8")
        al_borde.cancel_promotion()
        assert al_borde.pending_promotion is None
        assert al_borde.click(chess.E1) is Click.SELECTED


class TestTurnos:
    """Whose move it is, and what that forbids."""

    def test_the_person_cannot_move_for_the_engine(self, evaluator: Evaluator) -> None:
        estado = PlayState(evaluator=evaluator, human_color=chess.BLACK)
        assert estado.click(chess.E2) is Click.IGNORED

    def test_playing_black_flips_the_board(self, evaluator: Evaluator) -> None:
        assert PlayState(evaluator=evaluator, human_color=chess.BLACK).flipped

    def test_the_engine_answers_with_a_legal_move(self, partida: PlayState) -> None:
        jugar(partida, "e2", "e4")
        resultado = partida.engine_move()
        assert resultado is not None
        assert partida.board.move_stack[-1] == resultado.move
        assert partida.last_search is resultado

    def test_the_engine_move_lands_in_the_history_and_returns_the_turn(
        self, partida: PlayState
    ) -> None:
        jugar(partida, "e2", "e4")
        partida.engine_move()
        assert len(partida.history) == 2
        assert partida.human_turn

    def test_a_finished_game_gets_no_engine_move(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),   # mate
        )
        assert estado.finished
        assert estado.engine_move() is None

    def test_the_hint_does_not_move(self, partida: PlayState) -> None:
        sugerencia = partida.hint()
        assert sugerencia is not None
        assert sugerencia.move in partida.board.legal_moves
        assert not partida.board.move_stack


class TestDeshacer:
    """Taking back, which has to give the move back to the person."""

    def test_undo_takes_back_both_plies(self, partida: PlayState) -> None:
        jugar(partida, "e2", "e4")
        partida.engine_move()
        partida.undo()
        assert partida.board.move_stack == []
        assert partida.history == []
        assert partida.human_turn

    def test_undo_before_the_engine_answered_takes_back_one(
        self, partida: PlayState
    ) -> None:
        jugar(partida, "e2", "e4")
        partida.undo()
        assert partida.board.move_stack == []
        assert partida.human_turn

    def test_undo_on_an_empty_board_is_harmless(self, partida: PlayState) -> None:
        partida.undo()
        assert partida.board.move_stack == []

    def test_undo_clears_a_pending_promotion(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("8/4P3/8/8/8/8/8/4K1k1 w - - 0 1"),
        )
        jugar(estado, "e7", "e8")
        estado.undo()
        assert estado.pending_promotion is None


class TestLecturas:
    """What the panel reads off the state."""

    def test_the_evaluation_is_reported_from_whites_side(self, evaluator: Evaluator) -> None:
        # A queen up for White has to read positive whoever is to move. Reading
        # it from the side to move would flip the bar every ply, and with a
        # material stub the sign is not a matter of opinion.
        blancas = PlayState(
            evaluator=evaluator, board=chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
        )
        negras = PlayState(
            evaluator=evaluator, board=chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
        )
        assert blancas.evaluation().value > 0
        assert negras.evaluation().value > 0

    def test_the_centipawn_reading_carries_the_same_sign(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator, board=chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1")
        )
        lectura = estado.evaluation()
        assert lectura.value < 0 and lectura.cp < 0

    def test_the_move_list_pairs_white_and_black(self, partida: PlayState) -> None:
        jugar(partida, "e2", "e4")
        partida.engine_move()
        jugar(partida, "d2", "d4")
        filas = partida.move_pairs()
        assert [n for n, _, _ in filas] == [1, 2]
        assert filas[0][1] == "e4"
        assert filas[1][2] == ""          # Black has not answered yet

    def test_a_game_that_starts_with_black_to_move_is_numbered_right(
        self, evaluator: Evaluator
    ) -> None:
        estado = PlayState(
            evaluator=evaluator,
            human_color=chess.BLACK,
            board=chess.Board(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ),
        )
        jugar(estado, "e7", "e5")
        numero, blancas, negras = estado.move_pairs()[0]
        assert (numero, blancas, negras) == (1, "...", "e5")

    def test_the_status_line_announces_mate(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator, board=chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        )
        assert "mate" in estado.status().lower()

    def test_the_status_line_announces_check(self, evaluator: Evaluator) -> None:
        estado = PlayState(
            evaluator=evaluator, board=chess.Board("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1")
        )
        assert "jaque" in estado.status().lower()


class TestVista:
    """The widget half: that it draws, and that it does not swallow failures.

    Constructing widgets works outside a kernel -- what does not work is
    *displaying* them -- so the board can be built and clicked in a test. The
    assertion that matters is the last one: an exception inside a widget
    callback goes nowhere by default, and a board that stops responding with no
    message looks exactly like a hung kernel.
    """

    @pytest.fixture
    def vista(self, evaluator: Evaluator):
        pytest.importorskip("ipywidgets", reason="requires the `ui` extra")
        from chessdl.ui import PlayUI

        return PlayUI({"A": evaluator, "B": evaluator}, depth=1)

    def test_the_board_has_sixty_four_squares(self, vista) -> None:
        assert len(vista.board_box.children) == 64

    def test_the_board_is_drawn_from_whites_side_by_default(self, vista) -> None:
        assert vista._order()[0] == chess.A8
        assert vista._order()[-1] == chess.H1

    def test_flipping_turns_the_board_around(self, vista) -> None:
        vista._on_flip()
        assert vista._order()[0] == chess.H1

    def test_the_selection_marks_its_destinations(self, vista) -> None:
        from chessdl.ui.board import DOT

        vista._on_square(chess.G1)()
        marcadas = {s for s, b in vista.squares.items() if b.description == DOT}
        assert marcadas == {chess.F3, chess.H3}

    def test_a_pair_of_clicks_plays_and_the_engine_answers(self, vista) -> None:
        vista._on_square(chess.E2)()
        vista._on_square(chess.E4)()
        assert len(vista.state.history) == 2
        assert "Partida" in vista.moves.value

    def test_nothing_is_swallowed(self, vista) -> None:
        vista._on_square(chess.E2)()
        vista._on_square(chess.E4)()
        vista._on_hint()
        vista._on_undo()
        vista._on_new()
        assert vista.errors.value == ""

    def test_no_square_is_ever_labelled_with_an_empty_string(self, vista) -> None:
        """An empty square carries a space, never ``""``, all the way through.

        This is the one bug of the board that a person reported and no test
        here could have caught by looking at the Python. `ButtonView.update` in
        ipywidgets 7 -- Colab's version -- only rewrites the button's text when
        the description or the icon is non-empty:

            if (description.length || icon.length) { el.textContent = ""; ... }

        So clearing a description to `""` leaves the previous label sitting in
        the DOM. Pieces stayed on the squares they had left and every move dot
        ever drawn stayed drawn, while the colours -- written by a separate,
        unguarded call -- updated correctly, which is what made it look like a
        refresh problem rather than a label one.

        Nothing in Python observes that, and ipywidgets 8 dropped the guard, so
        it cannot be reproduced here either. What *is* checkable is the
        invariant that avoids it, and that is what this pins.
        """
        from chessdl.ui.board import EMPTY

        def revisar(cuando: str) -> None:
            vacias = [
                chess.square_name(s) for s, b in vista.squares.items()
                if not b.description
            ]
            assert not vacias, f"{cuando}: descripcion vacia en {vacias}"
            assert EMPTY, "el relleno de casilla vacia no puede ser una cadena vacia"

        revisar("al construir")
        vista._on_square(chess.E2)()
        revisar("con una pieza levantada")
        vista._on_square(chess.E4)()
        revisar("despues de jugar")
        vista._on_undo()
        revisar("despues de deshacer")
        vista._on_flip()
        revisar("con el tablero dado vuelta")

    def test_a_preset_loads_on_the_side_the_position_is_waiting_for(
        self, evaluator: Evaluator
    ) -> None:
        # Black to move. Handing the position to the engine instead would turn a
        # puzzle into a demonstration of the engine solving it unwatched.
        pytest.importorskip("ipywidgets", reason="requires the `ui` extra")
        from chessdl.ui import PlayUI

        fen = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 1"
        vista = PlayUI(evaluator, depth=1, positions={"final": fen})
        vista.preset_buttons[0].click()
        assert vista.state.board.turn == chess.BLACK
        assert vista.state.human_color == chess.BLACK
        assert vista.state.human_turn
        assert vista.errors.value == ""

    def test_the_panel_names_the_source_of_each_evaluation(
        self, evaluator: Evaluator, stockfish_path: str
    ) -> None:
        """Whose number is whose has to be readable off the panel itself.

        This is what the board was missing: one unlabelled bar, and no way to
        tell whether it was the model talking or the reference.
        """
        pytest.importorskip("ipywidgets", reason="requires the `ui` extra")
        import chess.engine

        from chessdl.ui import PlayUI

        motor = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            vista = PlayUI(
                {"ResNet": evaluator}, depth=1, reference=motor, reference_depth=8
            )
            panel = vista.status.value
        finally:
            motor.quit()

        assert "ResNet" in panel
        assert "Stockfish" in panel
        assert "Diferencia" in panel
        assert vista.errors.value == ""

    def test_without_an_engine_the_panel_says_so_instead_of_guessing(
        self, vista
    ) -> None:
        assert "Sin Stockfish" in vista.status.value
        assert "Diferencia" not in vista.status.value

    def test_a_malformed_fen_is_reported_and_not_loaded(self, vista) -> None:
        vista.fen_field.value = "esto no es un FEN"
        vista._on_load()
        assert "no se puede leer" in vista.errors.value
        assert vista.state.board.fen() == chess.STARTING_FEN


class TestLasDosLecturas:
    """The network's evaluation next to the one it was trained to reproduce.

    Two numbers side by side are only worth showing if they are in the same
    units, and that is the whole risk here: Stockfish answers in centipawns and
    from whichever side is to move, the network answers in [-1, 1] from White's.
    Get either conversion wrong and the panel shows a disagreement on every
    position -- which looks like a bad model rather than a bad panel.
    """

    @pytest.fixture
    def stockfish(self, stockfish_path: str):
        import chess.engine

        motor = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        yield motor
        motor.quit()

    def test_without_an_engine_there_is_simply_no_second_reading(
        self, partida: PlayState
    ) -> None:
        assert partida.reference_evaluation() is None
        assert partida.evaluation() is not None

    def test_both_readings_agree_on_who_is_winning(self, evaluator, stockfish) -> None:
        # White a queen up. Any sign convention error shows up here, because the
        # right answer does not depend on the model being any good.
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1"),
            reference=stockfish,
            reference_depth=8,
        )
        red = estado.evaluation()
        referencia = estado.reference_evaluation()
        assert referencia is not None
        assert red.value > 0 and referencia.value > 0

    def test_the_reference_is_read_from_whites_side_whoever_moves(
        self, evaluator, stockfish
    ) -> None:
        """The bug this catches is the one that costs nothing to write.

        `analyse` hands back a score relative to the side to move, so taking it
        as-is makes the bar flip meaning every ply: the same position reads
        +queen for White and +queen for Black one move apart.
        """
        posicion = "4k3/8/8/8/8/8/8/3QK3"
        blancas = PlayState(
            evaluator=evaluator, board=chess.Board(f"{posicion} w - - 0 1"),
            reference=stockfish, reference_depth=8,
        )
        negras = PlayState(
            evaluator=evaluator, board=chess.Board(f"{posicion} b - - 0 1"),
            reference=stockfish, reference_depth=8,
        )
        assert blancas.reference_evaluation().value > 0
        assert negras.reference_evaluation().value > 0

    def test_the_reference_lands_on_the_same_scale_as_the_network(
        self, evaluator, stockfish
    ) -> None:
        """Both bounded by `tanh`, so neither can push the bar off the end."""
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1"),  # absurdly won
            reference=stockfish, reference_depth=8,
        )
        referencia = estado.reference_evaluation()
        assert -1.0 <= referencia.value <= 1.0

    def test_a_forced_mate_comes_back_marked_as_one(self, evaluator, stockfish) -> None:
        estado = PlayState(
            evaluator=evaluator,
            board=chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),   # Ra8#
            reference=stockfish, reference_depth=8,
        )
        referencia = estado.reference_evaluation()
        assert referencia.is_mate
        assert referencia.mate == 1

    def test_a_dead_engine_costs_the_comparison_and_not_the_game(
        self, evaluator, stockfish_path: str
    ) -> None:
        """The person is mid-game; losing the second bar beats losing the board."""
        import chess.engine

        motor = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        estado = PlayState(evaluator=evaluator, reference=motor, reference_depth=8)
        assert estado.reference_evaluation() is not None

        motor.quit()
        estado.invalidate_readings()
        assert estado.reference_evaluation() is None      # no exception
        assert estado.evaluation() is not None
        assert jugar(estado, "e2", "e4") is Click.MOVED

    def test_readings_are_not_taken_twice_for_the_same_position(
        self, partida: PlayState
    ) -> None:
        """Selecting a piece redraws the panel without changing the position.

        Without the cache that would put a Stockfish search on every click of a
        piece, including the ones that only pick it up and put it back down.
        """
        partida.evaluation()
        llamadas = partida.evaluator.network_calls
        partida.click(chess.E2)
        partida.evaluation()
        assert partida.evaluator.network_calls == llamadas

    def test_changing_the_model_drops_the_cached_readings(
        self, partida: PlayState
    ) -> None:
        """Otherwise one architecture's number is shown under the other's name."""
        partida.evaluation()
        llamadas = partida.evaluator.network_calls
        partida.invalidate_readings()
        partida.evaluation()
        assert partida.evaluator.network_calls > llamadas


class TestReinicio:
    def test_reset_clears_everything(self, partida: PlayState) -> None:
        jugar(partida, "e2", "e4")
        partida.engine_move()
        partida.reset()
        assert partida.history == []
        assert partida.last_move is None
        assert partida.last_search is None
        assert partida.board.fen() == chess.STARTING_FEN

    def test_reset_can_change_sides(self, partida: PlayState) -> None:
        partida.reset(human_color=chess.BLACK)
        assert partida.flipped
        assert not partida.human_turn
