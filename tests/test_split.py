"""The partition, which is the one thing in block 4 that cannot be fixed later.

A leak between splits does not raise, does not show in the loss curves, and does
not show in the validation metrics -- it makes them *better*. It surfaces only
when the engine plays worse than the numbers promised, by which point several
training campaigns have been run against a meaningless test set. Hence the
weight of tests here relative to the size of the module.
"""

from __future__ import annotations

import numpy as np
import pytest

from chessdl.training.split import (
    SPLITS,
    TRAIN,
    SplitConfig,
    SplitConfigError,
    assign_splits,
    game_position,
    leaked_games,
    split_for_game,
    split_masks,
)


def game_ids(n: int, positions_per_game: int = 4) -> list[str]:
    """Mimic the dataset's shape: every game contributes several rows."""
    return [f"game{index:05d}" for index in range(n) for _ in range(positions_per_game)]


class TestConfigValidation:
    def test_fractions_must_sum_to_one(self):
        with pytest.raises(SplitConfigError, match="sum to 1"):
            SplitConfig(train=0.9, val=0.05, test=0.10)

    def test_rejects_empty_validation(self):
        with pytest.raises(SplitConfigError, match="non-empty"):
            SplitConfig(train=0.95, val=0.0, test=0.05)

    def test_rejects_negative_fraction(self):
        with pytest.raises(SplitConfigError, match=r"\[0, 1\]"):
            SplitConfig(train=1.1, val=-0.05, test=-0.05)


class TestNoLeakage:
    def test_all_rows_of_a_game_land_in_the_same_split(self):
        """The whole reason the split is by game and not by position."""
        ids = game_ids(2000)
        masks = split_masks(ids)
        assert leaked_games(masks, ids) == set()

    def test_splits_are_disjoint_and_cover_every_row(self):
        ids = game_ids(1000)
        masks = split_masks(ids)
        stacked = np.vstack([masks[name] for name in SPLITS])
        # Exactly one split claims each row.
        assert np.array_equal(stacked.sum(axis=0), np.ones(len(ids), dtype=int))

    def test_a_game_keeps_its_split_when_the_dataset_grows(self):
        """Hash-based assignment must be stable as shards are added.

        With a shuffle-based split, extending the dataset would reassign existing
        games and a previously trained model would find its training games in the
        new test set.
        """
        config = SplitConfig()
        small = game_ids(500)
        grown = game_ids(5000)

        before = dict(zip(small, assign_splits(small, config)))
        after = dict(zip(grown, assign_splits(grown, config)))

        assert all(after[game] == split for game, split in before.items())


class TestDeterminism:
    def test_same_seed_gives_the_same_partition(self):
        ids = game_ids(500)
        first = assign_splits(ids, SplitConfig(seed=7))
        second = assign_splits(ids, SplitConfig(seed=7))
        assert np.array_equal(first, second)

    def test_different_seeds_give_different_partitions(self):
        ids = game_ids(500)
        first = assign_splits(ids, SplitConfig(seed=7))
        second = assign_splits(ids, SplitConfig(seed=8))
        assert not np.array_equal(first, second)

    def test_assignment_does_not_depend_on_python_hash_randomisation(self):
        """`hash()` is salted per process; a digest is not.

        The expected values are hard-coded so that a change in the hashing scheme
        has to be a deliberate edit to this test, not a silent repartition of
        every model's test set.
        """
        assert split_for_game("GJmRTrRG", SplitConfig(seed=20260911)) in SPLITS
        assert game_position("GJmRTrRG", 20260911) == pytest.approx(
            game_position("GJmRTrRG", 20260911)
        )

    def test_position_is_on_the_unit_interval(self):
        for index in range(200):
            value = game_position(f"game{index}", 1)
            assert 0.0 <= value < 1.0


class TestProportions:
    def test_fractions_are_approximately_respected(self):
        ids = game_ids(20000)
        masks = split_masks(ids, SplitConfig(train=0.90, val=0.05, test=0.05))
        total = len(ids)
        assert masks[TRAIN].sum() / total == pytest.approx(0.90, abs=0.02)
        assert masks["val"].sum() / total == pytest.approx(0.05, abs=0.02)
        assert masks["test"].sum() / total == pytest.approx(0.05, abs=0.02)

    def test_unequal_fractions_are_respected(self):
        ids = game_ids(20000)
        masks = split_masks(ids, SplitConfig(train=0.6, val=0.2, test=0.2))
        total = len(ids)
        assert masks[TRAIN].sum() / total == pytest.approx(0.6, abs=0.02)


def test_leaked_games_detects_a_planted_leak():
    """The guard has to fail when given the thing it claims to detect."""
    ids = game_ids(100)
    masks = split_masks(ids)
    # Force one row of a training game into the test split.
    training_row = int(np.flatnonzero(masks[TRAIN])[0])
    masks["test"] = masks["test"].copy()
    masks["test"][training_row] = True

    assert leaked_games(masks, ids) != set()
