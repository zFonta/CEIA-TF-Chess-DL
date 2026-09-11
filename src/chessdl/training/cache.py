"""A precomputed tensor cache, so the GPU is not left waiting on the CPU.

Encoding one position through the direct path (FEN -> ``Board`` -> tensor) costs
about 79 microseconds, of which 44 are spent parsing the FEN. That is roughly
12,600 positions per second per core; on the two vCPUs Colab hands out, an epoch
over 2.5 million positions spends around 100 seconds just encoding -- the same
order as the GPU step itself. Without a cache, epoch time nearly doubles and
half the compute budget is spent waiting.

So the encoding is done once and stored as ``uint8``: 2.9 GB for the full
dataset, against 11.8 GB if it were kept as ``float32``. Every plane is binary
except the halfmove clock, which is stored as its raw 0-100 count and divided on
load. That round trip is bit-exact -- verified exhaustively over all 101 possible
clock values -- so a cached tensor is byte-identical to one built directly by
``board_to_tensor``. Anything less would mean training on one representation and
running the engine on another.

This does not change the dataset. The Parquet files still store FENs, which is
what lets the representation change without re-labelling a single position. The
cache is a derived, throwaway artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import chess
import numpy as np

from ..encoding import (
    HALFMOVE_CLOCK_MAX,
    PLANE_HALFMOVE_CLOCK,
    TENSOR_SHAPE,
    board_to_tensor,
)

#: Bytes one cached position occupies: 18 * 8 * 8.
BYTES_PER_POSITION = int(np.prod(TENSOR_SHAPE))

CACHE_DTYPE = np.uint8


class CacheMismatchError(RuntimeError):
    """An existing cache file does not match the dataset it is being used with."""


def fen_to_cached(fen: str) -> np.ndarray:
    """Encode one position into the cache's ``uint8`` representation."""
    board = chess.Board(fen)
    planes = board_to_tensor(board, dtype=np.float32)

    cached = planes.astype(CACHE_DTYPE)
    # Every other plane is 0/1 and survives the cast. The clock plane holds a
    # fraction, so the raw count is stored instead and scaled back on load.
    cached[PLANE_HALFMOVE_CLOCK] = min(
        board.halfmove_clock, HALFMOVE_CLOCK_MAX
    )
    return cached


def cached_to_tensor(cached: np.ndarray) -> np.ndarray:
    """Undo :func:`fen_to_cached`, reproducing ``board_to_tensor`` exactly.

    Works on a single ``(18, 8, 8)`` entry or on a batch of them.
    """
    planes = cached.astype(np.float32)
    planes[..., PLANE_HALFMOVE_CLOCK, :, :] /= np.float32(HALFMOVE_CLOCK_MAX)
    return planes


def cache_path_for(directory: str | Path, name: str = "tensors") -> Path:
    """Where the cache for a given working directory lives.

    ``.npy`` because that is genuinely what it is: the file carries a NumPy
    header, which is what lets :func:`load_cache` check its shape instead of
    trusting the caller to remember it.
    """
    return Path(directory) / f"{name}.npy"


def build_cache(
    fens: Sequence[str],
    path: str | Path,
    progress: bool = False,
) -> Path:
    """Encode every position once and write it to a flat ``uint8`` file.

    Written through a memory map so peak memory stays at one position rather than
    the whole dataset, which matters on a 12.7 GB Colab runtime.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".partial")
    array = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=CACHE_DTYPE, shape=(len(fens), *TENSOR_SHAPE)
    )

    iterator: Iterable[tuple[int, str]] = enumerate(fens)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(fens), desc="encoding", unit="pos")
        except ImportError:  # pragma: no cover - tqdm is a hard dependency
            pass

    for index, fen in iterator:
        array[index] = fen_to_cached(fen)

    array.flush()
    del array
    # Renamed only once complete: an interrupted build must not leave behind a
    # short file that later looks like a valid cache.
    tmp.replace(path)
    return path


def load_cache(path: str | Path, expected_rows: int | None = None) -> np.ndarray:
    """Memory-map an existing cache.

    Mapping rather than reading keeps the 2.9 GB off the heap; the operating
    system pages in what the batch actually touches.
    """
    path = Path(path)
    array = np.load(path, mmap_mode="r")

    if array.shape[1:] != TENSOR_SHAPE:
        raise CacheMismatchError(
            f"cache holds {array.shape[1:]} tensors, expected {TENSOR_SHAPE}"
        )
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise CacheMismatchError(
            f"cache has {array.shape[0]:,} rows but the dataset has "
            f"{expected_rows:,}. It was built from different shards; rebuild it."
        )
    return array


def cache_size_bytes(n_positions: int) -> int:
    return n_positions * BYTES_PER_POSITION
