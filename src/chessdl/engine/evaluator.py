"""Scoring positions with a trained network, for the engine (WBS 5).

The contract is one sentence: :meth:`Evaluator.evaluate` returns, for each
position, how good it is **for the side to move**, on the same [-1, 1] scale the
network was trained on.

Two things make this more than a thin wrapper around ``model(x)``.

**Decided positions never reach the network.** The dataset excluded terminal
positions on purpose -- a checkmate or a stalemate is not something Stockfish
evaluates, it is a result -- so the network has never seen one and has no reason
whatsoever to score one correctly. Asking it anyway is not a small inaccuracy: a
search that evaluates a mating position with the network will happily rank the
mate below some quiet move and **miss mate in one**. Those positions are resolved
here, from the rules, before any tensor is built.

**Everything else goes out in batches.** A one-ply search asks about every legal
move at once -- some thirty positions -- and thirty separate forward passes cost
thirty times the launch overhead for the same arithmetic. Feeding them as one
batch is what keeps a move inside the budget of requirement 1.7.
"""

from __future__ import annotations

from collections.abc import Sequence

import chess
import numpy as np
import torch

from ..encoding import board_to_tensor

#: Value of a position whose side to move has been checkmated. The worst there
#: is, and reachable: the network's ``tanh`` can approach it but never reach it,
#: so a real mate always outranks anything the network reports.
LOSS = -1.0

#: Value of any drawn position, from either point of view.
DRAW = 0.0


def terminal_value(board: chess.Board, claim_draws: bool = True) -> float | None:
    """The exact value of a decided position, or ``None`` if play continues.

    From the side to move's point of view, so a checkmated side scores
    :data:`LOSS` and every draw scores :data:`DRAW`.

    ``claim_draws`` covers the two endings that are *claimable* rather than
    automatic -- threefold repetition and the fifty-move rule. Treating them as
    draws is what stops an engine with a winning position from shuffling pieces
    forever: without it, a repetition looks like just another position and the
    engine has no reason to avoid it, while the opponent who is losing will
    certainly claim. The automatic endings (fivefold, seventy-five moves) are
    always honoured, claim or no claim.
    """
    if board.is_checkmate():
        return LOSS
    if board.is_stalemate() or board.is_insufficient_material():
        return DRAW
    if board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return DRAW
    if claim_draws and (board.is_repetition(3) or board.is_fifty_moves()):
        return DRAW
    return None


class Evaluator:
    """Scores positions with a trained value network.

    The model can be any of this project's architectures: they share an
    interface -- input ``(batch, 18, 8, 8)``, output ``(batch,)`` in [-1, 1] --
    precisely so the engine does not have to know which one it is holding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | None = None,
        batch_size: int = 512,
        claim_draws: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.claim_draws = claim_draws
        #: Positions sent to the network, and positions answered from the rules.
        #: Kept because "the search never asked the network about a mate" is a
        #: property worth being able to check rather than assume.
        self.network_calls = 0
        self.terminal_calls = 0

    @torch.no_grad()
    def evaluate_tensors(self, planes: Sequence[np.ndarray]) -> np.ndarray:
        """Run the network over already-encoded positions, in batches.

        The raw path, with no rules applied: whatever is passed here reaches the
        network. Callers that start from boards should use :meth:`evaluate`,
        which resolves decided positions first; the search uses this one because
        it resolves them itself, and with the distance to mate that a search
        needs and a position on its own has no way to know.
        """
        if not planes:
            return np.empty(0, dtype=np.float32)

        values = np.empty(len(planes), dtype=np.float32)
        for start in range(0, len(planes), self.batch_size):
            corte = slice(start, start + self.batch_size)
            batch = torch.from_numpy(np.stack(planes[corte]))
            values[corte] = self.model(batch.to(self.device)).float().cpu().numpy()
            self.network_calls += len(values[corte])
        return values

    def evaluate(self, boards: Sequence[chess.Board]) -> np.ndarray:
        """Score every position, from each one's own side-to-move perspective.

        Decided positions are answered from the rules and never reach the
        network.
        """
        values = np.empty(len(boards), dtype=np.float32)

        rows: list[int] = []
        planes: list[np.ndarray] = []
        for index, board in enumerate(boards):
            decided = terminal_value(board, self.claim_draws)
            if decided is None:
                rows.append(index)
                planes.append(board_to_tensor(board))
            else:
                values[index] = decided
                self.terminal_calls += 1

        if rows:
            values[rows] = self.evaluate_tensors(planes)
        return values

    def evaluate_board(self, board: chess.Board) -> float:
        """Score a single position. Convenience over :meth:`evaluate`."""
        return float(self.evaluate([board])[0])

    def describe(self) -> str:
        parametros = sum(p.numel() for p in self.model.parameters())
        return (
            f"{type(self.model).__name__} en {self.device} "
            f"({parametros:,} parametros, lotes de {self.batch_size})"
        )
