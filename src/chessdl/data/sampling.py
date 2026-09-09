"""Position sampling within an accepted game (requirement 1.3).

Four positions are drawn at random from each game, stratified by side to move:
two where white is to move and two where black is. Sampling within each stratum
is still random, but the split makes the colour balance exact by construction
rather than something that merely tends to hold on average.

Openings are deliberately *not* skipped -- the whole game is sampled.

The random seed for a game is derived from the global seed and the game id, so
the same game always yields the same positions no matter which worker processes
it or in what order.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import chess
import chess.pgn

from chessdl.config import SamplingConfig


@dataclass(frozen=True)
class SampledPosition:
    """One position drawn from a game, before it has been labelled."""

    ply: int
    fen: str
    turn_white: bool


def game_rng(game_id: str, seed: int) -> random.Random:
    """A per-game RNG that does not depend on processing order.

    ``hash()`` is salted per interpreter, so a stable digest is used instead:
    reproducibility has to survive both multiprocessing and a rerun weeks later.
    """
    digest = hashlib.blake2b(
        f"{seed}:{game_id}".encode("utf-8"), digest_size=8
    ).digest()
    return random.Random(int.from_bytes(digest, "big"))


def sample_plies(n_plies: int, cfg: SamplingConfig, rng: random.Random) -> list[int]:
    """Choose which plies of a game to sample.

    Candidate plies are ``0 .. n_plies - 1``: the position *before* each played
    move. Every one of them is non-terminal by construction, since a legal move
    was played from it.
    """
    if n_plies <= 0:
        return []

    wanted = min(cfg.positions_per_game, n_plies)

    if not cfg.balance_by_turn:
        return sorted(rng.sample(range(n_plies), wanted))

    # White moves on even plies, black on odd ones.
    white_plies = list(range(0, n_plies, 2))
    black_plies = list(range(1, n_plies, 2))

    n_white = wanted // 2
    n_black = wanted - n_white
    if wanted % 2 and rng.random() < 0.5:
        # Split the odd position randomly so no colour is systematically favoured.
        n_white, n_black = n_black, n_white

    # If one stratum is short (a very short game), top up from the other rather
    # than silently returning fewer positions than asked for.
    n_white = min(n_white, len(white_plies))
    n_black = min(n_black, len(black_plies))
    shortfall = wanted - n_white - n_black
    if shortfall > 0:
        n_white = min(n_white + shortfall, len(white_plies))
        shortfall = wanted - n_white - n_black
        if shortfall > 0:
            n_black = min(n_black + shortfall, len(black_plies))

    chosen = rng.sample(white_plies, n_white) + rng.sample(black_plies, n_black)
    return sorted(chosen)


def sample_positions(
    game: chess.pgn.Game,
    cfg: SamplingConfig,
    game_id: str,
) -> list[SampledPosition]:
    """Replay a game once and return the sampled positions as FENs."""
    moves = list(game.mainline_moves())
    targets = set(sample_plies(len(moves), cfg, game_rng(game_id, cfg.seed)))
    if not targets:
        return []

    positions: list[SampledPosition] = []
    board = game.board()  # honours a FEN header, if the game has one
    for ply, move in enumerate(moves):
        if ply in targets:
            positions.append(
                SampledPosition(
                    ply=ply,
                    fen=board.fen(),
                    turn_white=board.turn == chess.WHITE,
                )
            )
        board.push(move)
    return positions
