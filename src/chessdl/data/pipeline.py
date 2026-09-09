"""End-to-end orchestration of the dataset build (WBS 3.6).

The build runs in two passes per dump:

1. **Extract** -- stream the multi-gigabyte Lichess dump once and write the games
   that pass the filter to a small local PGN. A zstd stream cannot be seeked, so
   doing this once is what makes everything afterwards resumable without
   re-downloading anything.
2. **Label** -- read the extract, sample positions, evaluate them with Stockfish
   and write Parquet shards, pushing each shard to the Hub as it is finished.

Progress is checkpointed after every shard, so a Colab disconnect costs at most
one shard of work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from chessdl.config import DatasetConfig
from chessdl.data import lichess, pgn, sampling, schema
from chessdl.data.labeling import label_fens
from chessdl.data.pgn import RawGame
from chessdl.data.state import PipelineState, SeenKeys, seen_keys_path


@dataclass
class ShardResult:
    """Outcome of building one shard."""

    index: int
    path: Path
    n_games: int
    n_positions: int
    n_duplicates: int
    n_dropped: int


@dataclass
class BuildSummary:
    """Everything one invocation of the build produced."""

    shards: list[ShardResult] = field(default_factory=list)
    extract_stats: list[lichess.ExtractStats] = field(default_factory=list)

    @property
    def n_positions(self) -> int:
        return sum(shard.n_positions for shard in self.shards)

    @property
    def n_duplicates(self) -> int:
        return sum(shard.n_duplicates for shard in self.shards)

    def summary(self) -> str:
        lines = [stats.summary() for stats in self.extract_stats]
        lines.append(
            f"{len(self.shards)} shard(s) written, {self.n_positions:,} positions, "
            f"{self.n_duplicates:,} duplicates skipped"
        )
        return "\n".join(lines)


def extract_path_for(cfg: DatasetConfig, dump: str) -> Path:
    """Where a dump's filtered extract is stored."""
    return Path(cfg.output.extract_dir) / f"{cfg.source.dump_name(dump)}_filtered.pgn.zst"


def resolve_sources(
    cfg: DatasetConfig,
    sources: Sequence[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """The ``(dump, location)`` pairs to process, honouring an explicit override."""
    return list(sources) if sources is not None else lichess.dump_sources(cfg.source)


def survey(
    cfg: DatasetConfig,
    max_scanned: int | None = None,
    progress: bool = True,
    sources: Sequence[tuple[str, str]] | None = None,
) -> list[lichess.ExtractStats]:
    """Count how many games pass the filter, without labelling anything.

    This is the cheap measurement that sizes the run: it says how many usable
    games a dump actually yields before any Stockfish time is spent.
    """
    results = []
    for dump, source in resolve_sources(cfg, sources):
        stats = lichess.extract_games(
            source,
            cfg.filter,
            out_path=None,
            max_scanned=max_scanned,
            progress=progress,
        )
        stats.dump = dump
        results.append(stats)
    return results


def ensure_extract(
    cfg: DatasetConfig,
    dump: str,
    source: str,
    state: PipelineState,
    max_games: int | None = None,
    progress: bool = True,
) -> Path:
    """Produce the filtered extract for a dump, or reuse the existing one."""
    path = extract_path_for(cfg, dump)
    dump_state = state.for_dump(dump)

    if dump_state.extracted and path.exists():
        return path

    stats = lichess.extract_games(
        source, cfg.filter, out_path=path, max_games=max_games, progress=progress
    )
    dump_state.extracted = True
    dump_state.games_seen = stats.games_seen
    dump_state.games_accepted = stats.games_accepted
    state.save()
    return path


def iter_game_batches(
    extract: Path,
    cfg: DatasetConfig,
    skip_games: int = 0,
) -> Iterator[list[RawGame]]:
    """Read the extract and yield batches of ``games_per_shard`` games.

    ``skip_games`` fast-forwards past the games already turned into shards. The
    extract is small and local, so skipping is cheap compared with re-streaming
    the original dump.
    """
    batch: list[RawGame] = []
    for index, raw in enumerate(lichess.iter_dump_games(extract, cfg.filter)):
        if index < skip_games:
            continue
        batch.append(raw)
        if len(batch) >= cfg.output.games_per_shard:
            yield batch
            batch = []
    if batch:
        yield batch


def positions_from_games(
    games: Sequence[RawGame],
    cfg: DatasetConfig,
    seen: SeenKeys,
) -> tuple[list[dict], int, int]:
    """Sample positions from a batch of games, dropping duplicates.

    Returns the pending rows (everything but the labels), how many duplicates
    were skipped, and how many games were dropped for being unparseable or too
    short.
    """
    pending: list[dict] = []
    duplicates = 0
    dropped_games = 0

    for raw in games:
        game = pgn.parse_game(raw)
        if game is None:
            dropped_games += 1
            continue
        if pgn.count_plies(game) < cfg.filter.min_plies:
            dropped_games += 1
            continue

        game_id = raw.game_id
        for position in sampling.sample_positions(game, cfg.sampling, game_id):
            key = schema.position_key(position.fen)
            if not seen.add_new(key):
                duplicates += 1
                continue
            pending.append(
                {
                    "game_id": game_id,
                    "ply": position.ply,
                    "fen": position.fen,
                    "pos_key": key,
                    "turn_white": position.turn_white,
                    "white_elo": int(raw.headers.get("WhiteElo", 0) or 0),
                    "black_elo": int(raw.headers.get("BlackElo", 0) or 0),
                    "result": raw.headers.get("Result", "*"),
                    "time_control": pgn.classify_time_control(raw.headers) or "Unknown",
                }
            )

    return pending, duplicates, dropped_games


def label_pending_rows(
    pending: Sequence[dict],
    cfg: DatasetConfig,
    sf_version: str,
    dump: str,
    progress: bool = True,
) -> list[dict]:
    """Evaluate the sampled positions and complete the rows.

    Positions the engine could not evaluate are dropped, so the labels are
    matched back to their rows by FEN rather than by position in the list.
    """
    labeled = label_fens(
        [row["fen"] for row in pending], cfg.labeling, cfg.normalization, progress
    )
    labels_by_fen = {item.fen: item.label for item in labeled}

    rows: list[dict] = []
    for row in pending:
        label = labels_by_fen.get(row["fen"])
        if label is None:
            continue
        rows.append(
            {
                **row,
                "cp_white": label.cp_white,
                "cp_stm": label.cp_stm,
                "value_white": label.value_white,
                "value_stm": label.value_stm,
                "is_mate": label.is_mate,
                "mate_in": schema.clamp_mate_in(label.mate_in),
                "sf_depth": cfg.labeling.depth,
                "sf_version": sf_version,
                "src_dump": dump,
            }
        )
    return rows


def build_shard(
    games: Sequence[RawGame],
    cfg: DatasetConfig,
    seen: SeenKeys,
    sf_version: str,
    dump: str,
    index: int,
    progress: bool = True,
) -> ShardResult | None:
    """Sample, label and persist one shard. Returns None if nothing survived."""
    pending, duplicates, dropped = positions_from_games(games, cfg, seen)
    rows = label_pending_rows(pending, cfg, sf_version, dump, progress=progress)
    if not rows:
        return None

    path = Path(cfg.output.local_dir) / schema.shard_filename(dump, index)
    schema.write_shard(rows, path)
    return ShardResult(
        index=index,
        path=path,
        n_games=len(games),
        n_positions=len(rows),
        n_duplicates=duplicates,
        n_dropped=dropped,
    )


def push_shard(path: Path, cfg: DatasetConfig) -> None:
    """Push a finished shard to the Hub, where it is actually durable."""
    from chessdl import hf  # imported lazily so offline runs need no token

    token = hf.get_token(required=True)
    hf.ensure_repo(cfg.output.hf_repo_id, token=token)
    hf.upload_shard(path, cfg.output.hf_repo_id, token=token)


def run_build(
    cfg: DatasetConfig,
    sf_version: str,
    max_shards: int | None = None,
    max_games: int | None = None,
    push: bool | None = None,
    progress: bool = True,
    sources: Sequence[tuple[str, str]] | None = None,
) -> BuildSummary:
    """Run the full build for every configured dump, resuming where it stopped."""
    state = PipelineState.load(cfg.output.state_path)
    seen = SeenKeys(seen_keys_path(cfg.output.state_path))
    should_push = cfg.output.push_to_hub if push is None else push

    summary = BuildSummary()
    shards_built = 0

    for dump, source in resolve_sources(cfg, sources):
        extract = ensure_extract(
            cfg, dump, source, state, max_games=max_games, progress=progress
        )
        dump_state = state.for_dump(dump)
        skip = dump_state.games_to_skip(cfg.output.games_per_shard)

        for offset, games in enumerate(iter_game_batches(extract, cfg, skip_games=skip)):
            if max_shards is not None and shards_built >= max_shards:
                return summary

            index = dump_state.shards_done
            result = build_shard(
                games, cfg, seen, sf_version, dump, index, progress=progress
            )

            # The shard index advances even when a batch produced nothing, so a
            # resume does not replay a batch that is known to be empty.
            dump_state.shards_done += 1
            if result is not None:
                dump_state.positions_written += result.n_positions
                summary.shards.append(result)
                if should_push:
                    push_shard(result.path, cfg)

            state.save()
            seen.save()
            shards_built += 1

    return summary
