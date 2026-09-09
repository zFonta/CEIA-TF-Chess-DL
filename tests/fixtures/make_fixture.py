"""Regenerate the PGN fixture used by the tests.

    python tests/fixtures/make_fixture.py

The fixture stands in for a Lichess dump: it carries the same headers and mixes
games that should pass the filter with games that should be rejected for every
reason the filter knows about (low Elo, excluded time control, unrated,
abandoned, too short). The moves are random but legal, and the seed is fixed, so
the file is reproducible.
"""

from __future__ import annotations

import random
from pathlib import Path

import chess
import chess.pgn

SEED = 20260909
OUTPUT = Path(__file__).parent / "sample_games.pgn"

# (event, white_elo, black_elo, time_control, termination, target_plies)
# The first block is what the filter should accept; the rest is what it should
# reject, one row per reason.
GAME_SPECS = [
    *[("Rated Blitz game", 2300, 2410, "300+0", "Normal", 44) for _ in range(8)],
    *[("Rated Rapid game", 2250, 2200, "600+5", "Normal", 52) for _ in range(6)],
    *[("Rated Classical game", 2500, 2480, "1800+20", "Normal", 60) for _ in range(4)],
    *[("Rated Blitz tournament https://lichess.org/tournament/abc", 2260, 2340, "180+2", "Time forfeit", 38) for _ in range(2)],
    # --- rejected below ---
    *[("Rated Blitz game", 2100, 2400, "300+0", "Normal", 40) for _ in range(4)],  # white under threshold
    *[("Rated Blitz game", 2400, 1900, "300+0", "Normal", 40) for _ in range(3)],  # black under threshold
    *[("Rated Bullet game", 2600, 2650, "60+0", "Normal", 40) for _ in range(4)],  # excluded control
    *[("Rated UltraBullet game", 2400, 2450, "15+0", "Normal", 40) for _ in range(2)],
    *[("Rated Correspondence game", 2400, 2450, "-", "Normal", 40) for _ in range(2)],
    *[("Casual Blitz game", 2400, 2450, "300+0", "Normal", 40) for _ in range(2)],  # unrated
    *[("Rated Blitz game", 2400, 2450, "300+0", "Abandoned", 40) for _ in range(2)],
    # Passes the header filter but is too short to sample: dropped later, by
    # `min_plies`, which is a different code path worth exercising.
    *[("Rated Blitz game", 2400, 2450, "300+0", "Normal", 12) for _ in range(2)],
]


def random_game(rng: random.Random, target_plies: int) -> tuple[chess.Board, list[chess.Move]]:
    """Play random legal moves, stopping at the target length or at game over."""
    board = chess.Board()
    moves: list[chess.Move] = []
    while len(moves) < target_plies and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        moves.append(move)
    return board, moves


def build_game(index: int, spec: tuple, rng: random.Random) -> chess.pgn.Game:
    event, white_elo, black_elo, time_control, termination, target_plies = spec
    board, moves = random_game(rng, target_plies)

    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Site"] = f"https://lichess.org/fixture{index:04d}"
    game.headers["Date"] = "2026.01.15"
    game.headers["White"] = f"player_w_{index}"
    game.headers["Black"] = f"player_b_{index}"
    # Random play rarely reaches a real finish, so an unfinished game gets a
    # plausible result rather than the "*" that no Lichess game ever carries.
    result = board.result(claim_draw=True)
    game.headers["Result"] = (
        result if result != "*" else rng.choice(["1-0", "0-1", "1/2-1/2"])
    )
    game.headers["WhiteElo"] = str(white_elo)
    game.headers["BlackElo"] = str(black_elo)
    game.headers["TimeControl"] = time_control
    game.headers["Termination"] = termination

    node: chess.pgn.GameNode = game
    for move in moves:
        node = node.add_variation(move)
    return game


def export(game: chess.pgn.Game) -> str:
    """Render one game. A StringExporter accumulates, so each game needs its own."""
    return game.accept(
        chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    )


def main() -> None:
    rng = random.Random(SEED)
    chunks = [export(build_game(i, spec, rng)) for i, spec in enumerate(GAME_SPECS)]
    OUTPUT.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunks)} games to {OUTPUT}")


if __name__ == "__main__":
    main()
