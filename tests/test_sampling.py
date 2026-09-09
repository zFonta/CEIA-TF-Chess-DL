"""Tests for position sampling (requirement 1.3)."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import chess
import pytest

from chessdl.config import SamplingConfig
from chessdl.data.pgn import iter_raw_games, parse_game
from chessdl.data.sampling import game_rng, sample_plies, sample_positions


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


def accepted_games(sample_pgn: Path, cfg, min_plies: int | None = None):
    """Parsed fixture games that are long enough to sample."""
    threshold = cfg.filter.min_plies if min_plies is None else min_plies
    with sample_pgn.open(encoding="utf-8") as handle:
        raws = list(iter_raw_games(handle, cfg.filter))

    out = []
    for raw in raws:
        game = parse_game(raw)
        if game is None:
            continue
        if sum(1 for _ in game.mainline_moves()) >= threshold:
            out.append((raw, game))
    return out


# --- sample_plies ----------------------------------------------------------


def test_draws_the_requested_number_of_plies(cfg, rng):
    plies = sample_plies(40, cfg.sampling, rng)
    assert len(plies) == cfg.sampling.positions_per_game == 4


def test_plies_are_distinct_and_in_range(cfg, rng):
    plies = sample_plies(40, cfg.sampling, rng)
    assert len(set(plies)) == len(plies)
    assert all(0 <= ply < 40 for ply in plies)


def test_plies_come_back_sorted(cfg, rng):
    assert sample_plies(60, cfg.sampling, rng) == sorted(sample_plies(60, cfg.sampling, random.Random(0)))


def test_balanced_sampling_splits_the_colours_evenly(cfg, rng):
    """Requirement 1.3: two white-to-move plies and two black-to-move ones."""
    plies = sample_plies(40, cfg.sampling, rng)
    white = [ply for ply in plies if ply % 2 == 0]
    black = [ply for ply in plies if ply % 2 == 1]
    assert len(white) == len(black) == 2


def test_balance_holds_for_every_seed(cfg):
    for seed in range(200):
        plies = sample_plies(40, cfg.sampling, random.Random(seed))
        assert sum(1 for ply in plies if ply % 2 == 0) == 2


def test_unbalanced_sampling_does_not_guarantee_the_split(cfg):
    """Without the stratification the balance is only true on average.

    This is why `balance_by_turn` is on: it turns a statistical tendency into a
    property of every single game.
    """
    unbalanced = replace(cfg.sampling, balance_by_turn=False)
    splits = {
        sum(1 for ply in sample_plies(40, unbalanced, random.Random(seed)) if ply % 2 == 0)
        for seed in range(200)
    }
    assert splits != {2}


def test_openings_are_not_skipped(cfg):
    """Early plies must be reachable: the plan explicitly keeps openings in."""
    seen = set()
    for seed in range(300):
        seen.update(sample_plies(40, cfg.sampling, random.Random(seed)))
    assert 0 in seen or 1 in seen
    assert min(seen) <= 3


def test_sampling_covers_the_whole_game(cfg):
    seen = set()
    for seed in range(300):
        seen.update(sample_plies(40, cfg.sampling, random.Random(seed)))
    assert max(seen) >= 36


def test_short_games_return_what_they_can(cfg, rng):
    plies = sample_plies(3, cfg.sampling, rng)
    assert len(plies) == 3
    assert sorted(plies) == [0, 1, 2]


def test_a_game_with_a_single_ply(cfg, rng):
    assert sample_plies(1, cfg.sampling, rng) == [0]


def test_a_game_with_no_moves(cfg, rng):
    assert sample_plies(0, cfg.sampling, rng) == []


def test_odd_position_counts_do_not_favour_one_colour(cfg):
    """With an odd request the extra position goes to a random colour."""
    odd = replace(cfg.sampling, positions_per_game=3)
    white_counts = Counter(
        sum(1 for ply in sample_plies(40, odd, random.Random(seed)) if ply % 2 == 0)
        for seed in range(400)
    )
    assert set(white_counts) == {1, 2}
    # Neither split should dominate; a fair coin over 400 draws stays well inside this.
    assert 0.35 < white_counts[2] / 400 < 0.65


# --- determinism -----------------------------------------------------------


def test_the_same_game_always_yields_the_same_plies(cfg):
    first = sample_plies(40, cfg.sampling, game_rng("abc123", cfg.sampling.seed))
    second = sample_plies(40, cfg.sampling, game_rng("abc123", cfg.sampling.seed))
    assert first == second


def test_different_games_yield_different_plies(cfg):
    draws = {
        tuple(sample_plies(40, cfg.sampling, game_rng(f"game{i}", cfg.sampling.seed)))
        for i in range(50)
    }
    assert len(draws) > 1


def test_changing_the_seed_changes_the_sample(cfg):
    base = sample_plies(60, cfg.sampling, game_rng("abc123", cfg.sampling.seed))
    other = sample_plies(60, cfg.sampling, game_rng("abc123", cfg.sampling.seed + 1))
    assert base != other


def test_game_rng_does_not_depend_on_interpreter_hash_seed(cfg):
    """`hash()` is salted per process; the per-game RNG must not be.

    Reproducibility has to survive multiprocessing and a rerun weeks later, so
    the seed is derived from a stable digest instead.
    """
    expected = game_rng("stable-id", 1234).random()
    assert game_rng("stable-id", 1234).random() == expected


# --- sample_positions over real games --------------------------------------


def test_positions_from_a_real_game(sample_pgn: Path, cfg):
    raw, game = accepted_games(sample_pgn, cfg)[0]
    positions = sample_positions(game, cfg.sampling, raw.game_id)

    assert len(positions) == cfg.sampling.positions_per_game
    assert len({position.ply for position in positions}) == len(positions)


def test_sampled_positions_are_balanced_by_colour(sample_pgn: Path, cfg):
    for raw, game in accepted_games(sample_pgn, cfg):
        positions = sample_positions(game, cfg.sampling, raw.game_id)
        white = sum(1 for position in positions if position.turn_white)
        assert white == len(positions) - white == 2


def test_the_fen_matches_the_declared_turn(sample_pgn: Path, cfg):
    for raw, game in accepted_games(sample_pgn, cfg)[:5]:
        for position in sample_positions(game, cfg.sampling, raw.game_id):
            board = chess.Board(position.fen)
            assert (board.turn == chess.WHITE) == position.turn_white


def test_sampled_positions_are_never_terminal(sample_pgn: Path, cfg):
    """A finished position has no evaluation and no move to search."""
    for raw, game in accepted_games(sample_pgn, cfg):
        for position in sample_positions(game, cfg.sampling, raw.game_id):
            board = chess.Board(position.fen)
            assert not board.is_game_over()
            assert board.legal_moves.count() > 0


def test_the_fen_is_the_position_at_that_ply(sample_pgn: Path, cfg):
    raw, game = accepted_games(sample_pgn, cfg)[0]
    moves = list(game.mainline_moves())

    for position in sample_positions(game, cfg.sampling, raw.game_id):
        replayed = game.board()
        for move in moves[: position.ply]:
            replayed.push(move)
        assert replayed.fen() == position.fen


def test_sampling_a_game_twice_gives_the_same_positions(sample_pgn: Path, cfg):
    raw, game = accepted_games(sample_pgn, cfg)[0]
    first = sample_positions(game, cfg.sampling, raw.game_id)
    second = sample_positions(game, cfg.sampling, raw.game_id)
    assert [p.fen for p in first] == [p.fen for p in second]
