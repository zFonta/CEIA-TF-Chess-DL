"""Streaming PGN parsing and game-level filtering (requirement 1.1).

A monthly Lichess dump holds tens of millions of games, and only a small
fraction pass the ELO and time-control filter. Parsing every game's movetext to
throw almost all of them away would dominate the runtime, so this module works
in two stages:

1. A cheap line scanner reads the header block and decides from it alone whether
   the game is wanted. Rejected games have their movetext skipped without ever
   being accumulated or parsed.
2. Only accepted games are handed to ``python-chess`` for real parsing.

The stream is consumed strictly forward, because the source is a zstd
decompression stream and cannot be seeked backwards.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

import chess.pgn

from chessdl.config import FilterConfig

#: Header keys the filter needs. Everything else is ignored while scanning.
WANTED_HEADERS = frozenset(
    {
        "Event",
        "Site",
        "Result",
        "WhiteElo",
        "BlackElo",
        "TimeControl",
        "Termination",
    }
)

#: Lichess time-control categories, longest name first so that "UltraBullet" is
#: matched before "Bullet" when scanning the Event header.
TIME_CONTROL_CATEGORIES = (
    "UltraBullet",
    "Bullet",
    "Blitz",
    "Rapid",
    "Classical",
    "Correspondence",
)

#: Upper bound in seconds of the estimated game duration for each category, as
#: Lichess buckets them. Used only as a fallback when the Event header is
#: unusable. Classical is everything above the Rapid bound.
_CATEGORY_UPPER_BOUNDS = (
    (29, "UltraBullet"),
    (179, "Bullet"),
    (479, "Blitz"),
    (1499, "Rapid"),
)

_HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
_TIME_CONTROL_RE = re.compile(r"^(\d+)\+(\d+)$")


@dataclass
class ScanStats:
    """Counters for one pass over a PGN stream.

    ``games_seen`` counts every game in the stream and ``games_accepted`` only
    those that pass the header filter; their ratio is the yield the survey step
    reports before any Stockfish time is committed.
    """

    games_seen: int = 0
    games_accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.games_accepted / self.games_seen if self.games_seen else 0.0


@dataclass(frozen=True)
class RawGame:
    """A game's headers plus its unparsed movetext."""

    headers: dict[str, str]
    movetext: str

    @property
    def game_id(self) -> str:
        """The Lichess game id, taken from the Site header.

        Falls back to the raw Site value so a non-Lichess PGN (a test fixture,
        say) still produces something usable for traceability.
        """
        site = self.headers.get("Site", "")
        return site.rsplit("/", 1)[-1] if site else ""


def parse_header_line(line: str) -> tuple[str, str] | None:
    """Parse one ``[Key "Value"]`` line, or return None if it is not a header."""
    match = _HEADER_RE.match(line)
    return (match.group(1), match.group(2)) if match else None


def classify_time_control(headers: dict[str, str]) -> str | None:
    """Determine the Lichess time-control category for a game.

    Uses the Event header ("Rated Blitz game", "Rated Blitz tournament ..."),
    which is Lichess's own bucketing, and falls back to deriving the category
    from the TimeControl clock when the Event header does not name one.
    """
    event = headers.get("Event", "")
    for category in TIME_CONTROL_CATEGORIES:
        if category in event:
            return category

    time_control = headers.get("TimeControl", "")
    if time_control == "-":
        return "Correspondence"

    match = _TIME_CONTROL_RE.match(time_control)
    if not match:
        return None

    # Lichess estimates a game's duration as base + 40 * increment.
    estimated = int(match.group(1)) + 40 * int(match.group(2))
    for upper_bound, category in _CATEGORY_UPPER_BOUNDS:
        if estimated <= upper_bound:
            return category
    return "Classical"


def _parse_elo(raw: str) -> int | None:
    """Parse an Elo header, tolerating the '?' Lichess uses for unrated players."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def headers_pass_filter(headers: dict[str, str], cfg: FilterConfig) -> bool:
    """Decide from the headers alone whether a game is worth parsing.

    This runs on every game in the dump, so it stays deliberately cheap: no
    movetext, no board, no object construction.
    """
    if cfg.rated_only and not headers.get("Event", "").startswith("Rated"):
        return False

    category = classify_time_control(headers)
    if category not in cfg.time_controls:
        return False

    # Requirement 1.1: the threshold applies to *both* players.
    for key in ("WhiteElo", "BlackElo"):
        elo = _parse_elo(headers.get(key, ""))
        if elo is None or elo < cfg.min_elo:
            return False

    if headers.get("Termination", "") in cfg.excluded_terminations:
        return False

    return True


def iter_raw_games(
    lines: Iterable[str],
    cfg: FilterConfig,
    stats: ScanStats | None = None,
) -> Iterator[RawGame]:
    """Yield the games in a PGN stream that pass the header filter.

    Games are separated by their header blocks: a ``[`` line arriving while
    movetext is being read starts the next game. Rejected games never have their
    movetext collected.
    """
    headers: dict[str, str] = {}
    movetext: list[str] = []
    in_movetext = False
    accepted = False

    for line in lines:
        if line.startswith("["):
            if in_movetext:  # the previous game ended; this is a new header block
                if accepted and movetext:
                    yield RawGame(headers=headers, movetext="".join(movetext))
                headers, movetext, in_movetext, accepted = {}, [], False, False
            parsed = parse_header_line(line)
            if parsed and parsed[0] in WANTED_HEADERS:
                headers[parsed[0]] = parsed[1]
        elif line.strip():
            if not in_movetext:
                # First movetext line: decide once, then either collect or skip.
                in_movetext = True
                accepted = headers_pass_filter(headers, cfg)
                if stats is not None:
                    stats.games_seen += 1
                    stats.games_accepted += accepted
            if accepted:
                movetext.append(line)

    if accepted and movetext:
        yield RawGame(headers=headers, movetext="".join(movetext))


def parse_game(raw: RawGame) -> chess.pgn.Game | None:
    """Turn an accepted :class:`RawGame` into a parsed ``python-chess`` game.

    Returns None for games that yield no moves at all. A handful of malformed
    entries in a dump of tens of millions should not abort a multi-hour run.

    Note that ``python-chess`` does not fail on a corrupt movetext: it skips the
    tokens it cannot read, so a damaged game comes back *truncated* rather than
    rejected. That is harmless here -- every position reached is still legal and
    still evaluable -- and the ``min_plies`` filter downstream discards anything
    truncated badly enough to matter.
    """
    header_block = "".join(f'[{k} "{v}"]\n' for k, v in raw.headers.items())
    text = f"{header_block}\n{raw.movetext}\n"
    try:
        game = chess.pgn.read_game(io.StringIO(text))
    except (ValueError, RuntimeError):
        return None
    if game is None or not game.variations:
        return None
    return game


def count_plies(game: chess.pgn.Game) -> int:
    """Number of moves in the game's mainline."""
    return sum(1 for _ in game.mainline_moves())
