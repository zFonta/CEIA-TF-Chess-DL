"""The PyTorch dataset, and the equivalence of its two paths.

The cache exists so the GPU is not starved, but a cache that disagrees with the
encoder is worse than no cache: the model would train on one representation and
the engine would run on another, with nothing failing anywhere.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.encoding import TENSOR_SHAPE, board_to_tensor  # noqa: E402
from chessdl.training.cache import build_cache, cache_path_for, load_cache  # noqa: E402
from chessdl.training.dataset import PositionDataset, split_datasets  # noqa: E402
from chessdl.training.split import SPLITS, split_masks  # noqa: E402

FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/p4ppp/1pnbpn2/8/3PB3/5NB1/PP1N1PPP/R2QK2R b KQ - 0 11",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/5ppp/8/4k3/8/8/4KPPP/8 w - - 87 60",
]
TARGETS = np.array([0.05, -0.53, 0.12, -0.99], dtype=np.float32)


@pytest.fixture
def cache(tmp_path):
    return load_cache(build_cache(FENS, cache_path_for(tmp_path)))


class TestConstruction:
    def test_requires_exactly_one_source(self, cache):
        with pytest.raises(ValueError, match="exactly one"):
            PositionDataset(TARGETS, cache=cache, fens=FENS)
        with pytest.raises(ValueError, match="exactly one"):
            PositionDataset(TARGETS)

    def test_rejects_a_target_count_mismatch(self):
        """Silently truncating here would misalign every label in the dataset."""
        with pytest.raises(ValueError, match="targets"):
            PositionDataset(TARGETS[:2], fens=FENS)


class TestItems:
    def test_returns_a_tensor_and_its_target(self):
        dataset = PositionDataset(TARGETS, fens=FENS)
        x, y = dataset[1]
        assert x.shape == TENSOR_SHAPE
        assert x.dtype == torch.float32
        assert y.item() == pytest.approx(TARGETS[1])

    def test_the_two_paths_agree_exactly(self, cache):
        """From the cache or from the FEN, the tensor must be the same one."""
        from_cache = PositionDataset(TARGETS, cache=cache)
        from_fens = PositionDataset(TARGETS, fens=FENS)

        for index in range(len(FENS)):
            assert torch.equal(from_cache[index][0], from_fens[index][0]), FENS[index]

    def test_matches_the_encoder_directly(self):
        dataset = PositionDataset(TARGETS, fens=FENS)
        for index, fen in enumerate(FENS):
            expected = torch.from_numpy(board_to_tensor(chess.Board(fen)))
            assert torch.equal(dataset[index][0], expected)


class TestIndexing:
    def test_indices_select_a_subset_without_copying(self, cache):
        subset = PositionDataset(TARGETS, cache=cache, indices=np.array([0, 3]))
        assert len(subset) == 2
        assert subset[1][1].item() == pytest.approx(TARGETS[3])

    def test_targets_property_follows_the_indices(self, cache):
        subset = PositionDataset(TARGETS, cache=cache, indices=np.array([2, 0]))
        assert np.allclose(subset.targets, [TARGETS[2], TARGETS[0]])


class TestSplitDatasets:
    def test_builds_one_dataset_per_split_covering_every_row(self):
        game_ids = [f"game{i // 4:04d}" for i in range(400)]
        targets = np.zeros(len(game_ids), dtype=np.float32)
        fens = [chess.STARTING_FEN] * len(game_ids)

        masks = split_masks(game_ids)
        datasets = split_datasets(targets, masks, fens=fens)

        assert set(datasets) == set(SPLITS)
        assert sum(len(d) for d in datasets.values()) == len(game_ids)


def test_works_with_a_dataloader(cache):
    """The whole point is to be consumed in batches by the training loop."""
    from torch.utils.data import DataLoader

    loader = DataLoader(PositionDataset(TARGETS, cache=cache), batch_size=2)
    batches = list(loader)

    assert len(batches) == 2
    x, y = batches[0]
    assert x.shape == (2, *TENSOR_SHAPE)
    assert y.shape == (2,)
