"""One-ply search: pick a move by evaluating what it leads to (WBS 5, req. 1.6).

The plan defines move selection as a one-level search over the evaluation
function, not as a classifier over moves. So: play each legal move, score the
resulting position, keep the best one.

**The sign is the whole thing.** The network always reports from the point of
view of the side to move, and after *my* move it is the *opponent's* turn.
So what comes back is how good the position is **for the opponent**, and the
move to play is the one that leaves that number as low as possible.

That is worth stating twice, because an engine with this backwards does not
fail. It returns legal moves, raises nothing, prints a confident evaluation --
and plays the worst move available, every single time. No test of shapes or
types catches it; only a test that knows what a good move looks like does.

Every legal move is scored in **one batch**. A position has around thirty of
them, and thirty separate forward passes would pay the launch cost thirty times
over for the same arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import numpy as np

from ..normalize import value_to_cp
from .evaluator import Evaluator


class GameOverError(ValueError):
    """Raised when a move is requested in a position that is already decided."""


@dataclass(frozen=True)
class SearchResult:
    """What the search chose, and what it saw.

    ``value`` is from the point of view of **the side that moved** -- the
    negation of what the network reported for the resulting position. Positive
    means the engine believes it came out ahead.
    """

    move: chess.Move
    value: float
    ranked: tuple[tuple[chess.Move, float], ...]
    seconds: float

    @property
    def centipawns(self) -> float:
        """The chosen move's value in centipawns (requirement 4.2)."""
        return value_to_cp(self.value)

    def table(self, board: chess.Board, top: int = 5) -> str:
        """The best few moves with their evaluations, in algebraic notation."""
        lines = [f"{'jugada':<10}{'valor':>9}{'cp':>9}"]
        lines.append("-" * 28)
        for move, value in self.ranked[:top]:
            marca = " *" if move == self.move else ""
            lines.append(
                f"{board.san(move):<10}{value:>9.4f}{value_to_cp(value):>9.0f}{marca}"
            )
        return "\n".join(lines)


def search(
    board: chess.Board,
    evaluator: Evaluator,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    """Choose a move for the side to move, by one-ply search.

    ``rng`` only breaks ties. Left as ``None`` the choice is deterministic --
    the first move in python-chess's own generation order wins -- which is what
    makes a game reproducible. Pass a generator when several games from the same
    position should not be identical.
    """
    started = time.perf_counter()

    moves = list(board.legal_moves)
    if not moves:
        raise GameOverError(
            "no hay jugadas legales: la posicion ya termino "
            f"({board.result(claim_draw=True)})"
        )

    # The resulting positions, each with the opponent to move.
    posiciones: list[chess.Board] = []
    for move in moves:
        board.push(move)
        # A copy is needed because the evaluator holds the whole list at once.
        # The move stack travels with it: the repetition and fifty-move rules
        # are about history, and a copy without it would silently stop
        # detecting the very draws `terminal_value` exists to catch.
        posiciones.append(board.copy())
        board.pop()

    # Values for the OPPONENT, who is to move in each resulting position.
    del_rival = evaluator.evaluate(posiciones)

    # ...so the engine's own value is the negation, and it wants the maximum.
    propios = -del_rival
    mejor = float(propios.max())

    empatados = [move for move, value in zip(moves, propios) if value == mejor]
    elegida = (
        empatados[0] if rng is None or len(empatados) == 1
        else empatados[int(rng.integers(len(empatados)))]
    )

    orden = np.argsort(-propios, kind="stable")
    ranked = tuple((moves[i], float(propios[i])) for i in orden)

    return SearchResult(
        move=elegida,
        value=mejor,
        ranked=ranked,
        seconds=time.perf_counter() - started,
    )


def best_move(
    board: chess.Board,
    evaluator: Evaluator,
    rng: np.random.Generator | None = None,
) -> chess.Move:
    """The move alone, for callers that do not need the rest."""
    return search(board, evaluator, rng).move


def value_white(board: chess.Board, value_stm: float) -> float:
    """Convert a side-to-move value to white's point of view (requirement 1.4).

    ``board`` is the position the value refers to, so the flip follows its turn.
    """
    return value_stm if board.turn == chess.WHITE else -value_stm
