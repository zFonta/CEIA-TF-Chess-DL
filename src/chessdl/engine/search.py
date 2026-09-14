"""Negamax search over the value network (WBS 5, requirement 1.6).

The plan defines move selection as a one-level search over the evaluation
function: play each legal move, score the resulting position, keep the best. That
is ``depth=1`` here and it is the deliverable. Greater depths are an extension,
measured rather than assumed, because each extra ply multiplies the positions to
evaluate by the branching factor -- around 33 in a middlegame.

**The sign is the whole thing.** The network always reports from the point of
view of the side to move, and after *my* move it is the *opponent's* turn. So
what comes back is how good the position is **for the opponent**, and the move to
play is the one that leaves that number as low as possible. That is what negamax
formalises: a node's value is ``max(-value(child))``.

Worth stating twice, because an engine with this backwards does not fail. It
returns legal moves, raises nothing, prints a confident evaluation -- and plays
the worst move available, every single time. No test of shapes or types catches
it; only a test that knows what a good move looks like does.

**Depth two is the one that fixes something specific.** At depth one the leaf is
the position after the engine's own move, so the opponent's reply is not in the
tree at all and a recapture is invisible: the engine wins a rook with its queen
and never sees the king take back. Depth two ends *after* the reply, which is
exactly that blind spot closed. Odd depths end on the engine's own move and are
structurally optimistic again -- depth three pushes the same problem two plies
further out rather than removing it.

**All the leaves go out in one batch.** The evaluation is the expensive part and
it parallelises; the tree walk does not. So the tree is expanded first, collecting
leaf tensors, then the whole set is evaluated at once, then the values propagate
back up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import chess
import numpy as np

from ..encoding import board_to_tensor
from ..normalize import value_to_cp
from .evaluator import DRAW, Evaluator, terminal_value

#: Score of a checkmate, outside the range the network can ever produce.
#:
#: ``tanh`` keeps the network strictly inside (-1, 1), so anything at or beyond 1
#: is unreachable for it and a real mate always outranks an evaluated position.
#: That separation has to be maintained by construction: a mate scored at 1.0
#: would compete with a saturated network output of 0.9999, and lose ties.
MATE = 2.0

#: How much a mate loses per ply of delay, so the search prefers the fast one.
#:
#: Without it every mate scores the same and the engine, having found one, has no
#: reason to play the move that delivers it now rather than one that keeps it
#: available -- which is how an engine "sees" a mate for fifty moves and never
#: gets there. The step is small enough that four plies of mate still sit well
#: above any evaluated position.
MATE_STEP = 0.01


class GameOverError(ValueError):
    """Raised when a move is requested in a position that is already decided."""


def mate_in(value: float) -> int | None:
    """Moves to mate encoded in a search value, or ``None`` if it is not a mate.

    A mate scores in the :data:`MATE` band with the ply distance subtracted, so
    the distance comes back out by inverting that. The sign follows the value:
    positive means the side to move delivers the mate, negative that it receives
    one.

    This is the inverse of a convention written in exactly one place -- the
    scoring in :func:`_exact` -- and it belongs next to it. Decoding the band by
    hand wherever a value is displayed is how the two drift apart, and a
    displayed mate distance that disagrees with the search is worse than no
    distance at all.
    """
    if abs(value) < MATE - MATE_STEP * 64:
        return None
    plies = round((MATE - abs(value)) / MATE_STEP)
    moves = max(1, (plies + 1) // 2)
    return moves if value > 0 else -moves


@dataclass(frozen=True)
class SearchResult:
    """What the search chose, and what it saw.

    ``value`` is from the point of view of **the side that moved**: positive
    means the engine believes it came out ahead. It lives in [-1, 1] except when
    a mate is in sight, where it takes a score in the :data:`MATE` band.
    """

    move: chess.Move
    value: float
    ranked: tuple[tuple[chess.Move, float], ...]
    seconds: float
    depth: int
    leaves: int

    @property
    def is_mate(self) -> bool:
        return mate_in(self.value) is not None

    @property
    def centipawns(self) -> float:
        """The chosen move's value in centipawns (requirement 4.2)."""
        return value_to_cp(max(-1.0, min(1.0, self.value)))

    def table(self, board: chess.Board, top: int = 5) -> str:
        """The best few moves with their evaluations, in algebraic notation."""
        lines = [f"{'jugada':<10}{'valor':>9}{'cp':>9}"]
        lines.append("-" * 28)
        for move, value in self.ranked[:top]:
            marca = " *" if move == self.move else ""
            mate = mate_in(value)
            if mate is not None:
                # A mate scores outside the network's range on purpose, so
                # printing 1.9900 next to a centipawn figure invites the reader
                # to compare it with an evaluation. It is not one.
                lines.append(f"{board.san(move):<10}{f'#{mate}':>9}{'':>9}{marca}")
                continue
            acotado = max(-1.0, min(1.0, value))
            lines.append(
                f"{board.san(move):<10}{value:>9.4f}{value_to_cp(acotado):>9.0f}{marca}"
            )
        return "\n".join(lines)


def _exact(board: chess.Board, ply: int, claim_draws: bool) -> float | None:
    """The value of a decided position, or ``None`` if play continues.

    Checkmate is scored in the :data:`MATE` band with the distance subtracted, so
    the search prefers to deliver it sooner and to suffer it later. Everything
    else a decided position can be is a draw, and a draw is a draw whenever it
    arrives.
    """
    if board.is_checkmate():
        return -(MATE - ply * MATE_STEP)
    decided = terminal_value(board, claim_draws)
    return DRAW if decided is not None else None


def _expand(
    board: chess.Board,
    depth: int,
    ply: int,
    tensors: list[np.ndarray],
    claim_draws: bool,
):
    """Walk the tree, collecting leaf tensors; returns the shape of the subtree.

    Nodes travel as ``(kind, payload)``: ``exact`` for a decided position,
    ``leaf`` for an index into ``tensors``, ``node`` for a list of children.

    The board is mutated with push/pop rather than copied. At depth three a copy
    per node would mean tens of thousands of them, and a ``chess.Board`` copy is
    not cheap -- the tree walk is already the part of this that does not
    parallelise.
    """
    decided = _exact(board, ply, claim_draws)
    if decided is not None:
        return ("exact", decided)

    if depth == 0:
        tensors.append(board_to_tensor(board))
        return ("leaf", len(tensors) - 1)

    children = []
    for move in board.legal_moves:
        board.push(move)
        children.append(_expand(board, depth - 1, ply + 1, tensors, claim_draws))
        board.pop()
    return ("node", children)


def _value(node, values: np.ndarray) -> float:
    """Fold the evaluated leaves back up the tree, negating at every level."""
    kind, payload = node
    if kind == "exact":
        return payload
    if kind == "leaf":
        return float(values[payload])
    return max(-_value(child, values) for child in payload)


def search(
    board: chess.Board,
    evaluator: Evaluator,
    *,
    depth: int = 1,
    beam: int | None = None,
    rng: np.random.Generator | None = None,
) -> SearchResult:
    """Choose a move for the side to move.

    ``depth`` counts plies: 1 is the engine's own move, 2 adds the opponent's
    reply, 3 the engine's answer to that.

    ``beam`` keeps only the best ``beam`` moves after a depth-one look and
    searches those to full depth. It trades exactness for cost -- a good move
    ranked poorly at depth one is discarded before it can prove itself -- and
    exists because the cost of an exact search grows by a factor of the branching
    factor per ply. Left as ``None`` the search is exact.

    ``rng`` only breaks ties. Left as ``None`` the choice is deterministic, which
    is what makes a game reproducible; pass a generator when several games from
    the same position should not be identical.
    """
    if depth < 1:
        raise ValueError(f"la profundidad tiene que ser al menos 1, no {depth}")

    started = time.perf_counter()

    moves = list(board.legal_moves)
    if not moves:
        raise GameOverError(
            "no hay jugadas legales: la posicion ya termino "
            f"({board.result(claim_draw=True)})"
        )

    if beam is not None and depth > 1 and beam < len(moves):
        preliminar = search(board, evaluator, depth=1)
        moves = [move for move, _ in preliminar.ranked[:beam]]

    tensors: list[np.ndarray] = []
    branches = []
    for move in moves:
        board.push(move)
        branches.append(_expand(board, depth - 1, 1, tensors, evaluator.claim_draws))
        board.pop()

    values = evaluator.evaluate_tensors(tensors)

    # The branch value belongs to the opponent, who moves in the resulting
    # position; the engine's own value is its negation.
    own = np.array([-_value(branch, values) for branch in branches], dtype=np.float64)
    best = float(own.max())

    tied = [move for move, value in zip(moves, own) if value == best]
    chosen = (
        tied[0] if rng is None or len(tied) == 1
        else tied[int(rng.integers(len(tied)))]
    )

    order = np.argsort(-own, kind="stable")
    ranked = tuple((moves[i], float(own[i])) for i in order)

    return SearchResult(
        move=chosen,
        value=best,
        ranked=ranked,
        seconds=time.perf_counter() - started,
        depth=depth,
        leaves=len(tensors),
    )


def best_move(
    board: chess.Board,
    evaluator: Evaluator,
    *,
    depth: int = 1,
    beam: int | None = None,
    rng: np.random.Generator | None = None,
) -> chess.Move:
    """The move alone, for callers that do not need the rest."""
    return search(board, evaluator, depth=depth, beam=beam, rng=rng).move


def value_white(board: chess.Board, value_stm: float) -> float:
    """Convert a side-to-move value to white's point of view (requirement 1.4).

    ``board`` is the position the value refers to, so the flip follows its turn.
    """
    return value_stm if board.turn == chess.WHITE else -value_stm
