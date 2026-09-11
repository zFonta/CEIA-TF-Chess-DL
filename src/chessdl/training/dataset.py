"""The PyTorch ``Dataset`` the training loop reads from.

Two ways in, deliberately:

* From the **cache** (:mod:`chessdl.training.cache`), which is what a real
  training run uses -- indexing a memory-mapped ``uint8`` array is essentially
  free, so the GPU is never left waiting.
* From **FENs**, encoding on demand. Slower, but it needs no precomputation and
  keeps the tests honest: a test that only ever exercises the cache would not
  notice the cache drifting away from :func:`~chessdl.encoding.board_to_tensor`.

Both paths must yield identical tensors, and a test asserts exactly that.

``torch`` is imported at module scope, so this module is only importable with
the ``train`` extra installed. Everything else under
:mod:`chessdl.training` stays NumPy-only and works without it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..encoding import TENSOR_SHAPE
from .cache import cached_to_tensor, fen_to_cached


class PositionDataset(Dataset):
    """Positions and their labels, as ``(tensor, target)`` pairs.

    The target is ``value_stm`` -- the label from the point of view of the side
    to move, which is the one that matches the mirrored encoding. Converting to
    white's point of view is a sign flip and belongs in the engine, not here.
    """

    def __init__(
        self,
        targets: Sequence[float] | np.ndarray,
        cache: np.ndarray | None = None,
        fens: Sequence[str] | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        if (cache is None) == (fens is None):
            raise ValueError("pass exactly one of `cache` or `fens`")

        self._cache = cache
        self._fens = fens
        self._targets = np.asarray(targets, dtype=np.float32)

        source_len = len(cache) if cache is not None else len(fens)  # type: ignore[arg-type]
        if source_len != len(self._targets):
            raise ValueError(
                f"{source_len:,} positions but {len(self._targets):,} targets"
            )

        # `indices` selects a split without copying the underlying tensors, which
        # for the cache would mean duplicating gigabytes.
        self._indices = (
            np.arange(source_len, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self._indices[position])

        if self._cache is not None:
            planes = cached_to_tensor(np.asarray(self._cache[row]))
        else:
            planes = cached_to_tensor(fen_to_cached(self._fens[row]))  # type: ignore[index]

        return (
            torch.from_numpy(planes),
            torch.tensor(self._targets[row], dtype=torch.float32),
        )

    @property
    def targets(self) -> np.ndarray:
        """The labels of this split, in order -- for the baselines."""
        return self._targets[self._indices]

    def tensor_shape(self) -> tuple[int, int, int]:
        return TENSOR_SHAPE


def split_datasets(
    targets: np.ndarray,
    masks: dict[str, np.ndarray],
    cache: np.ndarray | None = None,
    fens: Sequence[str] | None = None,
) -> dict[str, PositionDataset]:
    """Build one dataset per split from boolean masks over the rows."""
    return {
        name: PositionDataset(
            targets=targets, cache=cache, fens=fens, indices=np.flatnonzero(mask)
        )
        for name, mask in masks.items()
    }
