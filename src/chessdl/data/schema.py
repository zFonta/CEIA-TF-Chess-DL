"""Parquet schema and shard I/O (requirement 1.2).

The dataset stores FENs and labels, not tensors. Labelling with Stockfish is the
expensive, hard-to-repeat part of the pipeline; the tensor encoding is cheap and
will keep changing while the network is designed. Keeping the encoding out of
the stored data means the representation can be revised without re-labelling a
single position.

Every row also carries its own provenance -- Stockfish version, search depth and
source dump -- so the dataset documents itself as requirement 2.3 asks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        # --- provenance of the position
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("ply", pa.int16(), nullable=False),
        pa.field("fen", pa.string(), nullable=False),
        pa.field("pos_key", pa.uint64(), nullable=False),
        pa.field("turn_white", pa.bool_(), nullable=False),
        # --- labels, in both points of view
        pa.field("cp_white", pa.int32(), nullable=False),
        pa.field("cp_stm", pa.int32(), nullable=False),
        pa.field("value_white", pa.float32(), nullable=False),
        pa.field("value_stm", pa.float32(), nullable=False),
        pa.field("is_mate", pa.bool_(), nullable=False),
        pa.field("mate_in", pa.int16(), nullable=False),
        # --- game metadata, for auditing the filter
        pa.field("white_elo", pa.int16(), nullable=False),
        pa.field("black_elo", pa.int16(), nullable=False),
        pa.field("result", pa.string(), nullable=False),
        pa.field("time_control", pa.string(), nullable=False),
        # --- labelling provenance (requirement 2.3)
        pa.field("sf_depth", pa.int16(), nullable=False),
        pa.field("sf_version", pa.string(), nullable=False),
        pa.field("src_dump", pa.string(), nullable=False),
    ]
)

COLUMNS: tuple[str, ...] = tuple(SCHEMA.names)

#: A mate distance is tiny at any practical search depth, but the column is
#: still clamped so a pathological engine reply cannot overflow it.
MATE_IN_LIMIT = 32_000


def position_key(fen: str) -> int:
    """A stable 64-bit key for deduplication (requirement 3.2).

    Built from the first four FEN fields -- placement, side to move, castling
    rights and en passant square -- so the same position reached with different
    move counters keys the same. ``hash()`` is salted per interpreter, so a
    digest is used instead: the key has to agree across worker processes and
    across runs.
    """
    canonical = " ".join(fen.split(" ")[:4])
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def clamp_mate_in(mate_in: int) -> int:
    return max(-MATE_IN_LIMIT, min(MATE_IN_LIMIT, int(mate_in)))


class SchemaMismatchError(ValueError):
    """Raised when a row does not carry exactly the schema's columns."""


def rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """Build a table, failing loudly if a row does not match the schema.

    Arrow quietly fills a missing key with null rather than raising, even for a
    field declared non-nullable. That would turn a dropped column into nulls
    discovered only at validation time, so the columns are checked here instead
    -- the error then points at the code that built the row.
    """
    expected = set(COLUMNS)
    for index, row in enumerate(rows):
        keys = set(row)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise SchemaMismatchError(
                f"Row {index} does not match the schema. "
                f"Missing: {missing or 'none'}. Unexpected: {extra or 'none'}."
            )
    return pa.Table.from_pylist(list(rows), schema=SCHEMA)


def write_shard(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Write one shard of rows to Parquet."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rows_to_table(rows), out, compression="zstd")
    return out


def read_shard(path: str | Path) -> pa.Table:
    """Read one shard back."""
    return pq.read_table(Path(path))


def shard_paths(directory: str | Path) -> list[Path]:
    """Every shard in a directory, in stable order."""
    return sorted(Path(directory).glob("*.parquet"))


def read_dataset(paths: Iterable[str | Path]) -> pa.Table:
    """Read and concatenate shards into a single table."""
    tables = [read_shard(path) for path in paths]
    if not tables:
        return SCHEMA.empty_table()
    return pa.concat_tables(tables)


def shard_filename(dump: str, index: int) -> str:
    return f"{dump}_shard_{index:05d}.parquet"
