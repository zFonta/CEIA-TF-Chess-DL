"""The state machine behind a click-to-play board, with no widgets in sight.

The interactive board is two halves. This is the half that decides *what a click
means*: whether it selects a piece, completes a move, asks which piece a pawn
promotes to, or does nothing at all. The other half -- :mod:`chessdl.ui.board` --
draws the result and knows about ipywidgets.

Splitting them is not ceremony. Click handling is where an interactive board
actually goes wrong: a second click that silently deselects instead of capturing,
a promotion that defaults to a queen without asking, an undo that leaves the
human on the wrong side of the board. None of those raise, all of them are
visible only by playing, and none of them can be tested through a widget. Here
they are ordinary function calls with ordinary return values.

The rules come from python-chess and nothing else. This module never decides
whether a move is legal, only which of the legal moves a pair of clicks names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess

from ..engine.evaluator import Evaluator
from ..engine.search import GameOverError, SearchResult, search, value_white
from ..normalize import value_to_cp


class Click(str, Enum):
    """What a click on a square turned out to mean.

    The view redraws on anything but :data:`IGNORED`, and only asks for a piece
    when it gets :data:`PROMOTION`.
    """

    #: A piece of the side to move was picked up.
    SELECTED = "selected"
    #: The selection moved to another of the player's own pieces.
    RESELECTED = "reselected"
    #: The move was played.
    MOVED = "moved"
    #: The move is a pawn promotion; :meth:`PlayState.promote` finishes it.
    PROMOTION = "promotion"
    #: The selection was dropped without moving.
    CLEARED = "cleared"
    #: Nothing to do: an empty square, an enemy piece, the engine's turn.
    IGNORED = "ignored"


@dataclass
class PlayState:
    """A game in progress between a person and the engine.

    The person plays :attr:`human_color` and the engine answers. Everything the
    view needs to draw a frame is readable from here -- the position, what is
    selected, where that selection can go, the engine's last ranking -- so the
    view holds no state of its own and cannot drift out of step with the game.
    """

    evaluator: Evaluator
    human_color: chess.Color = chess.WHITE
    depth: int = 2
    board: chess.Board = field(default_factory=chess.Board)

    #: The square the person picked up, while they hold it.
    selected: int | None = None
    #: A promotion waiting on a choice of piece, as ``(from_square, to_square)``.
    pending_promotion: tuple[int, int] | None = None
    #: The move just played by either side, for the view to highlight.
    last_move: chess.Move | None = None
    #: What the engine saw when it last moved, including its ranked candidates.
    last_search: SearchResult | None = None
    #: Moves in algebraic notation, in order.
    history: list[str] = field(default_factory=list)
    #: Whether the board is drawn from Black's side.
    flipped: bool = False

    def __post_init__(self) -> None:
        self.flipped = self.human_color == chess.BLACK

    # ------------------------------------------------------------------ state

    @property
    def human_turn(self) -> bool:
        return self.board.turn == self.human_color and not self.finished

    @property
    def finished(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def destinations(self) -> dict[int, chess.Move]:
        """Where the selected piece can go, keyed by target square.

        A promotion contributes several legal moves to the same square; the one
        kept here is only a placeholder, because which piece it becomes is a
        question for the person and not for this mapping.
        """
        if self.selected is None:
            return {}
        salidas: dict[int, chess.Move] = {}
        for move in self.board.legal_moves:
            if move.from_square == self.selected:
                salidas.setdefault(move.to_square, move)
        return salidas

    # ----------------------------------------------------------------- clicks

    def click(self, square: int) -> Click:
        """Interpret a click on ``square`` and, when it completes one, move.

        The ordinary rules: an own piece selects or reselects, clicking the
        selected piece again drops it, and a piece with no legal moves is not a
        selection -- picking one up would leave the board looking live with
        nothing to click.

        The one addition is castling by clicking the rook. python-chess writes
        castling as the king's move to g1 or c1, so clicking the rook is not a
        legal destination and would read as picking the rook up. Plenty of
        people castle that way, and the intent is unambiguous when the king is
        selected and the rook on that side can still castle, so it is resolved
        as the castling move.
        """
        if not self.human_turn or self.pending_promotion is not None:
            return Click.IGNORED

        destinos = self.destinations()
        if square in destinos:
            return self._play_from_selection(square)

        enroque = self._castling_via_rook(square)
        if enroque is not None:
            self._push(enroque)
            return Click.MOVED

        pieza = self.board.piece_at(square)
        if pieza is not None and pieza.color == self.human_color:
            if square == self.selected:
                self.selected = None
                return Click.CLEARED
            anterior = self.selected
            self.selected = square
            if not self.destinations():      # a piece that cannot move is not a selection
                self.selected = anterior
                return Click.IGNORED
            return Click.RESELECTED if anterior is not None else Click.SELECTED

        if self.selected is not None:
            self.selected = None
            return Click.CLEARED
        return Click.IGNORED

    def _castling_via_rook(self, square: int) -> chess.Move | None:
        """The castling move a click on one's own rook names, if it is legal."""
        if self.selected is None:
            return None
        rey = self.board.piece_at(self.selected)
        torre = self.board.piece_at(square)
        if rey is None or rey.piece_type != chess.KING:
            return None
        if torre is None or torre.piece_type != chess.ROOK or torre.color != rey.color:
            return None
        del_lado_del_rey = chess.square_file(square) > chess.square_file(self.selected)
        for move in self.board.legal_moves:
            if move.from_square == self.selected and self.board.is_castling(move):
                if self.board.is_kingside_castling(move) == del_lado_del_rey:
                    return move
        return None

    def _play_from_selection(self, square: int) -> Click:
        assert self.selected is not None
        candidatas = [
            move for move in self.board.legal_moves
            if move.from_square == self.selected and move.to_square == square
        ]
        if len(candidatas) > 1:              # only promotions produce several
            self.pending_promotion = (self.selected, square)
            return Click.PROMOTION
        self._push(candidatas[0])
        return Click.MOVED

    def promote(self, piece_type: int) -> Click:
        """Finish the promotion the last click left pending."""
        if self.pending_promotion is None:
            return Click.IGNORED
        origen, destino = self.pending_promotion
        self.pending_promotion = None
        self._push(chess.Move(origen, destino, promotion=piece_type))
        return Click.MOVED

    def cancel_promotion(self) -> Click:
        self.pending_promotion = None
        self.selected = None
        return Click.CLEARED

    def _push(self, move: chess.Move) -> None:
        self.history.append(self.board.san(move))   # SAN needs the position *before*
        self.board.push(move)
        self.last_move = move
        self.selected = None

    # ----------------------------------------------------------------- engine

    def engine_move(self) -> SearchResult | None:
        """Let the engine answer. Returns ``None`` if the game is already over."""
        if self.finished:
            return None
        try:
            resultado = search(self.board, self.evaluator, depth=self.depth)
        except GameOverError:
            return None
        self.last_search = resultado
        self._push(resultado.move)
        return resultado

    def hint(self) -> SearchResult | None:
        """What the engine would play *for the person*, without playing it.

        The same call the engine makes on its own turn, which is the point: the
        suggestion is not a weaker advisory mode, it is the engine's actual
        opinion about the position in front of the person.
        """
        if self.finished:
            return None
        try:
            return search(self.board, self.evaluator, depth=self.depth)
        except GameOverError:
            return None

    # --------------------------------------------------------------- readings

    def evaluation(self) -> tuple[float, float]:
        """The current position as ``(value from White's side, centipawns)``.

        White's point of view rather than the side to move's, because a bar that
        flips meaning every ply is unreadable -- and because requirement 1.4
        defines the reported value that way.
        """
        valor_stm = self.evaluator.evaluate_board(self.board)
        blancas = value_white(self.board, valor_stm)
        return blancas, value_to_cp(max(-1.0, min(1.0, blancas)))

    def status(self) -> str:
        """One line on how the game stands, in the language of the notebooks."""
        if self.board.is_checkmate():
            ganador = "negras" if self.board.turn == chess.WHITE else "blancas"
            return f"Jaque mate: ganan las {ganador}."
        if self.board.is_stalemate():
            return "Tablas por ahogado."
        if self.board.is_insufficient_material():
            return "Tablas por material insuficiente."
        if self.board.is_fifty_moves():
            return "Tablas por la regla de las 50 jugadas."
        if self.board.is_repetition(3):
            return "Tablas por repeticion."
        if self.pending_promotion is not None:
            return "Coronacion: elegi la pieza."
        turno = "blancas" if self.board.turn == chess.WHITE else "negras"
        jaque = " (jaque)" if self.board.is_check() else ""
        de_quien = "vos" if self.human_turn else "el motor"
        return f"Juegan las {turno}{jaque} - le toca a {de_quien}."

    def move_pairs(self) -> list[tuple[int, str, str]]:
        """The history as ``(number, white, black)`` rows, for a move list."""
        raiz = self.board.root()          # the position the move stack started from
        primero = raiz.fullmove_number
        empiezan_negras = raiz.turn == chess.BLACK

        jugadas = list(self.history)
        if empiezan_negras:
            jugadas.insert(0, "...")
        filas = []
        for i in range(0, len(jugadas), 2):
            par = jugadas[i:i + 2]
            filas.append((primero + i // 2, par[0], par[1] if len(par) > 1 else ""))
        return filas

    # ---------------------------------------------------------------- control

    def undo(self) -> None:
        """Take back one full move: the engine's answer and the person's move."""
        for _ in range(2):
            if not self.board.move_stack:
                break
            self.board.pop()
            self.history.pop()
            if self.board.turn == self.human_color:
                break
        self.selected = None
        self.pending_promotion = None
        self.last_move = self.board.move_stack[-1] if self.board.move_stack else None
        self.last_search = None

    def reset(self, fen: str | None = None, human_color: chess.Color | None = None) -> None:
        """Start again, optionally from a position or on the other side."""
        self.board = chess.Board(fen) if fen else chess.Board()
        if human_color is not None:
            self.human_color = human_color
            self.flipped = human_color == chess.BLACK
        self.selected = None
        self.pending_promotion = None
        self.last_move = None
        self.last_search = None
        self.history = []
