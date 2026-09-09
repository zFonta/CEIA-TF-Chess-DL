"""Tests for PGN header parsing and the game-level filter (requirement 1.1)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chessdl.config import FilterConfig
from chessdl.data.pgn import (
    ScanStats,
    classify_time_control,
    count_plies,
    headers_pass_filter,
    iter_raw_games,
    parse_game,
    parse_header_line,
)
from tests.conftest import EXPECTED_ACCEPTED_GAMES


def headers(**overrides: str) -> dict[str, str]:
    base = {
        "Event": "Rated Blitz game",
        "Site": "https://lichess.org/abcd1234",
        "Result": "1-0",
        "WhiteElo": "2400",
        "BlackElo": "2350",
        "TimeControl": "300+0",
        "Termination": "Normal",
    }
    base.update(overrides)
    return base


# --- header parsing --------------------------------------------------------


def test_parse_header_line():
    assert parse_header_line('[WhiteElo "2400"]\n') == ("WhiteElo", "2400")


def test_parse_header_line_keeps_values_with_spaces_and_urls():
    line = '[Event "Rated Blitz tournament https://lichess.org/tournament/xyz"]\n'
    key, value = parse_header_line(line)
    assert key == "Event"
    assert value.endswith("tournament/xyz")


def test_parse_header_line_rejects_movetext():
    assert parse_header_line("1. e4 e5 2. Nf3 *\n") is None


# --- time control classification -------------------------------------------


@pytest.mark.parametrize(
    "event, expected",
    [
        ("Rated Blitz game", "Blitz"),
        ("Rated Rapid game", "Rapid"),
        ("Rated Classical game", "Classical"),
        ("Rated Bullet game", "Bullet"),
        ("Rated UltraBullet game", "UltraBullet"),
        ("Rated Correspondence game", "Correspondence"),
        ("Rated Blitz tournament https://lichess.org/tournament/x", "Blitz"),
    ],
)
def test_classify_from_the_event_header(event, expected):
    assert classify_time_control(headers(Event=event)) == expected


def test_ultrabullet_is_not_mistaken_for_bullet():
    """"Bullet" is a substring of "UltraBullet"; the order of checks matters."""
    assert classify_time_control(headers(Event="Rated UltraBullet game")) == "UltraBullet"


@pytest.mark.parametrize(
    "time_control, expected",
    [
        ("15+0", "UltraBullet"),
        ("60+0", "Bullet"),
        ("300+0", "Blitz"),
        ("600+0", "Rapid"),
        ("1800+20", "Classical"),
        ("-", "Correspondence"),
    ],
)
def test_classify_falls_back_to_the_clock(time_control, expected):
    """With an Event header that names no category, the clock decides."""
    raw = headers(Event="Rated game", TimeControl=time_control)
    assert classify_time_control(raw) == expected


def test_classify_returns_none_when_nothing_is_usable():
    assert classify_time_control({"Event": "Rated game", "TimeControl": "?"}) is None


# --- the filter itself -----------------------------------------------------


def test_accepts_a_qualifying_game(cfg):
    assert headers_pass_filter(headers(), cfg.filter)


@pytest.mark.parametrize("colour", ["WhiteElo", "BlackElo"])
def test_rejects_when_either_player_is_below_the_threshold(colour, cfg):
    """Requirement 1.1: the threshold applies to both players, not the average."""
    assert not headers_pass_filter(headers(**{colour: "2100"}), cfg.filter)


def test_accepts_exactly_at_the_threshold(cfg):
    raw = headers(WhiteElo=str(cfg.filter.min_elo), BlackElo=str(cfg.filter.min_elo))
    assert headers_pass_filter(raw, cfg.filter)


def test_rejects_unrated_players(cfg):
    assert not headers_pass_filter(headers(WhiteElo="?"), cfg.filter)


def test_rejects_missing_elo_headers(cfg):
    raw = headers()
    del raw["BlackElo"]
    assert not headers_pass_filter(raw, cfg.filter)


@pytest.mark.parametrize("category", ["Bullet", "UltraBullet", "Correspondence"])
def test_rejects_the_excluded_time_controls(category, cfg):
    raw = headers(Event=f"Rated {category} game", TimeControl="60+0")
    assert not headers_pass_filter(raw, cfg.filter)


@pytest.mark.parametrize("category", ["Blitz", "Rapid", "Classical"])
def test_accepts_the_configured_time_controls(category, cfg):
    assert headers_pass_filter(headers(Event=f"Rated {category} game"), cfg.filter)


def test_rejects_casual_games(cfg):
    assert not headers_pass_filter(headers(Event="Casual Blitz game"), cfg.filter)


def test_accepts_casual_games_when_rated_only_is_off(cfg):
    relaxed = replace(cfg.filter, rated_only=False)
    assert headers_pass_filter(headers(Event="Casual Blitz game"), relaxed)


@pytest.mark.parametrize("termination", ["Abandoned", "Rules infraction", "Unterminated"])
def test_rejects_excluded_terminations(termination, cfg):
    assert not headers_pass_filter(headers(Termination=termination), cfg.filter)


def test_accepts_a_time_forfeit(cfg):
    """Losing on time is a normal way for a real game to end."""
    assert headers_pass_filter(headers(Termination="Time forfeit"), cfg.filter)


# --- streaming over the fixture --------------------------------------------


def test_iterating_the_fixture_yields_only_accepted_games(sample_pgn: Path, cfg):
    stats = ScanStats()
    with sample_pgn.open(encoding="utf-8") as handle:
        games = list(iter_raw_games(handle, cfg.filter, stats=stats))

    assert stats.games_seen == 41
    assert stats.games_accepted == EXPECTED_ACCEPTED_GAMES
    assert len(games) == EXPECTED_ACCEPTED_GAMES
    assert 0 < stats.acceptance_rate < 1


def test_accepted_games_really_satisfy_the_filter(sample_pgn: Path, cfg):
    with sample_pgn.open(encoding="utf-8") as handle:
        for raw in iter_raw_games(handle, cfg.filter):
            assert int(raw.headers["WhiteElo"]) >= cfg.filter.min_elo
            assert int(raw.headers["BlackElo"]) >= cfg.filter.min_elo
            assert classify_time_control(raw.headers) in cfg.filter.time_controls


def test_game_id_comes_from_the_site_header(sample_pgn: Path, cfg):
    with sample_pgn.open(encoding="utf-8") as handle:
        first = next(iter_raw_games(handle, cfg.filter))
    assert first.game_id.startswith("fixture")
    assert "/" not in first.game_id


def test_accepted_games_parse_back_into_real_games(sample_pgn: Path, cfg):
    with sample_pgn.open(encoding="utf-8") as handle:
        raws = list(iter_raw_games(handle, cfg.filter))

    for raw in raws:
        game = parse_game(raw)
        assert game is not None
        assert count_plies(game) > 0


def test_short_games_are_caught_by_the_ply_filter(sample_pgn: Path, cfg):
    """Two fixture games pass the headers but are too short to sample."""
    with sample_pgn.open(encoding="utf-8") as handle:
        raws = list(iter_raw_games(handle, cfg.filter))

    plies = [count_plies(parse_game(raw)) for raw in raws]
    too_short = [n for n in plies if n < cfg.filter.min_plies]
    assert len(too_short) == 2


def test_a_stricter_filter_accepts_fewer_games(sample_pgn: Path, cfg):
    strict = replace(cfg.filter, min_elo=2450)
    with sample_pgn.open(encoding="utf-8") as handle:
        strict_count = sum(1 for _ in iter_raw_games(handle, strict))
    assert strict_count < EXPECTED_ACCEPTED_GAMES


def test_parse_game_returns_none_when_no_move_is_readable(cfg):
    from chessdl.data.pgn import RawGame

    broken = RawGame(headers=headers(), movetext="1. Qxz9 nonsense *\n")
    assert parse_game(broken) is None


def test_corrupt_movetext_is_truncated_rather_than_rejected(cfg):
    """python-chess skips tokens it cannot read instead of raising.

    The surviving prefix is still a sequence of legal moves, so the positions it
    yields are perfectly good; `min_plies` discards anything truncated badly
    enough to matter.
    """
    from chessdl.data.pgn import RawGame

    partial = RawGame(headers=headers(), movetext="1. e4 e5 2. Qxz9 nonsense *\n")
    game = parse_game(partial)
    assert game is not None
    assert count_plies(game) == 2
    assert count_plies(game) < cfg.filter.min_plies  # so the pipeline drops it


def test_filtering_is_streaming_not_buffering(sample_pgn: Path, cfg):
    """The scanner must consume a plain iterator of lines, not a seekable file.

    The real source is a zstd decompression stream that cannot be rewound, so
    this guards the property the whole ingestion design depends on.
    """
    lines = iter(sample_pgn.read_text(encoding="utf-8").splitlines(keepends=True))
    assert len(list(iter_raw_games(lines, cfg.filter))) == EXPECTED_ACCEPTED_GAMES
