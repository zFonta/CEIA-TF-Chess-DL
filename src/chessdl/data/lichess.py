"""Streaming access to the Lichess monthly PGN dumps (requirement 5.1).

The dumps are public standard-chess games published under CC0 at
https://database.lichess.org. Each month is a single zstd-compressed PGN file of
tens of gigabytes, so it is never downloaded whole: the HTTP response is
decompressed incrementally and consumed line by line.

The expensive network pass runs *once* and writes the games that survive the
filter to a small local PGN. Everything downstream reads that extract instead of
the dump, which matters on Colab: a zstd stream cannot be seeked, so without the
extract every resume would mean re-streaming tens of gigabytes.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import requests
import zstandard as zstd

from chessdl.config import FilterConfig, SourceConfig
from chessdl.data.pgn import RawGame, ScanStats, iter_raw_games

#: Read size for the HTTP response and the decompressor. Large enough that the
#: per-chunk overhead is irrelevant against a multi-gigabyte stream.
CHUNK_BYTES = 1 << 20

#: Lichess asks for a descriptive user agent on database downloads.
USER_AGENT = "chessdl/0.1 (CEIA-FIUBA final project; academic use)"


@dataclass
class ExtractStats:
    """Result of one pass over a dump."""

    dump: str
    games_seen: int = 0
    games_accepted: int = 0
    games_written: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.games_accepted / self.games_seen if self.games_seen else 0.0

    def as_dict(self) -> dict:
        return {**asdict(self), "acceptance_rate": self.acceptance_rate}

    def summary(self) -> str:
        return (
            f"{self.dump}: {self.games_seen:,} games seen, "
            f"{self.games_accepted:,} accepted "
            f"({self.acceptance_rate:.3%}), {self.games_written:,} written"
        )


def _open_local(path: Path) -> io.TextIOBase:
    if path.suffix == ".zst":
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw, read_size=CHUNK_BYTES)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _open_remote(url: str) -> io.TextIOBase:
    response = requests.get(
        url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    reader = zstd.ZstdDecompressor().stream_reader(
        response.raw, read_size=CHUNK_BYTES
    )
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def open_pgn_stream(source: str | Path) -> io.TextIOBase:
    """Open a PGN source as a text stream.

    Accepts an ``http(s)`` URL or a local path, zstd-compressed or plain, so the
    same pipeline runs against a real dump and against a small test fixture.
    """
    text = str(source)
    if text.startswith(("http://", "https://")):
        return _open_remote(text)
    return _open_local(Path(text))


def iter_dump_games(
    source: str | Path,
    filter_cfg: FilterConfig,
    stats: ScanStats | None = None,
) -> Iterator[RawGame]:
    """Stream a dump and yield only the games that pass the header filter."""
    with open_pgn_stream(source) as stream:
        yield from iter_raw_games(stream, filter_cfg, stats=stats)


def _format_game(raw: RawGame) -> str:
    headers = "".join(f'[{k} "{v}"]\n' for k, v in raw.headers.items())
    return f"{headers}\n{raw.movetext.rstrip()}\n\n"


def extract_games(
    source: str | Path,
    filter_cfg: FilterConfig,
    out_path: str | Path | None = None,
    max_games: int | None = None,
    max_scanned: int | None = None,
    progress: bool = True,
) -> ExtractStats:
    """Scan a dump, keeping the games that pass the filter.

    With ``out_path`` the accepted games are written to a zstd-compressed PGN;
    without it nothing is written and the call is a pure survey of the yield,
    which is the cheap way to size the run before committing Stockfish time.

    ``max_scanned`` bounds how many games are *looked at* (useful for surveying
    a slice of a dump); ``max_games`` bounds how many are kept.
    """
    dump_name = Path(str(source)).name
    stats = ExtractStats(dump=dump_name)
    scan = ScanStats()

    writer = _ShardWriter(out_path) if out_path else None
    bar = _progress_bar(progress, dump_name)

    try:
        for raw in iter_dump_games(source, filter_cfg, stats=scan):
            if writer is not None:
                writer.write(_format_game(raw))
                stats.games_written += 1
            if bar is not None:
                bar.update(1)
            if max_games is not None and scan.games_accepted >= max_games:
                break
            if max_scanned is not None and scan.games_seen >= max_scanned:
                break
    finally:
        if writer is not None:
            writer.close()
        if bar is not None:
            bar.close()

    stats.games_seen = scan.games_seen
    stats.games_accepted = scan.games_accepted
    return stats


class _ShardWriter:
    """Buffered zstd text writer that creates its parent directory."""

    def __init__(self, path: str | Path, level: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = self.path.open("wb")
        self._compressor = zstd.ZstdCompressor(level=level).stream_writer(self._raw)
        self._text = io.TextIOWrapper(self._compressor, encoding="utf-8")

    def write(self, text: str) -> None:
        self._text.write(text)

    def close(self) -> None:
        self._text.flush()
        self._text.close()
        self._compressor.close()
        self._raw.close()


def _progress_bar(enabled: bool, description: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        return None
    return tqdm(desc=f"accepted from {description}", unit="game")


def dump_sources(source_cfg: SourceConfig) -> list[tuple[str, str]]:
    """Return ``(dump, url)`` pairs for every dump listed in the config.

    ``CHESSDL_DUMP_DIR`` overrides the URL with a local file when a dump has
    already been downloaded, which keeps repeated Colab runs off the network.
    """
    local_dir = os.environ.get("CHESSDL_DUMP_DIR")
    pairs: list[tuple[str, str]] = []
    for dump in source_cfg.dumps:
        if local_dir:
            candidate = Path(local_dir) / f"{source_cfg.dump_name(dump)}.pgn.zst"
            if candidate.exists():
                pairs.append((dump, str(candidate)))
                continue
        pairs.append((dump, source_cfg.dump_url(dump)))
    return pairs
