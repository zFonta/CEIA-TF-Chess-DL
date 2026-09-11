"""The tensor cache, whose only job is to be indistinguishable from the encoder.

If a cached tensor differs from what :func:`board_to_tensor` produces, the model
trains on one representation and the engine runs on another. Nothing crashes;
the engine is just quietly worse than the metrics said. Every test here exists to
make that impossible.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessdl.encoding import (
    HALFMOVE_CLOCK_MAX,
    PLANE_EN_PASSANT,
    PLANE_HALFMOVE_CLOCK,
    TENSOR_SHAPE,
    board_to_tensor,
)
from chessdl.training.cache import (
    BYTES_PER_POSITION,
    CacheMismatchError,
    build_cache,
    cache_path_for,
    cache_size_bytes,
    cached_to_tensor,
    fen_to_cached,
    load_cache,
)

FENS = [
    chess.STARTING_FEN,
    # Black to move: exercises the mirroring path.
    "r1bq1rk1/p4ppp/1pnbpn2/8/3PB3/5NB1/PP1N1PPP/R2QK2R b KQ - 0 11",
    # A legal en passant capture.
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    # No castling rights left, and a high halfmove clock.
    "8/5ppp/8/4k3/8/8/4KPPP/8 w - - 87 60",
    # An endgame with very little material.
    "8/8/8/4k3/8/8/4Q3/4K3 b - - 3 40",
    # Mid-clock value, to catch a scaling mistake that vanishes at 0 and 100.
    "r1bqk2r/5ppp/p1np1b2/1p1Np3/4P3/N7/PPP2PPP/R2Q1RK1 w kq - 37 20",
]


class TestRoundTripIsExact:
    @pytest.mark.parametrize("fen", FENS)
    def test_cached_tensor_matches_the_encoder_bit_for_bit(self, fen: str):
        direct = board_to_tensor(chess.Board(fen))
        through_cache = cached_to_tensor(fen_to_cached(fen))

        assert through_cache.dtype == direct.dtype == np.float32
        # Exact equality, not approximate: a cache that is merely close means the
        # engine and the trained model disagree about the same position.
        assert np.array_equal(through_cache, direct), fen

    def test_every_halfmove_clock_value_survives_the_round_trip(self):
        """The clock is the only non-binary plane, so it is the only one that can drift."""
        for clock in range(0, HALFMOVE_CLOCK_MAX + 1):
            board = chess.Board()
            board.halfmove_clock = clock
            fen = board.fen()

            direct = board_to_tensor(chess.Board(fen))
            through_cache = cached_to_tensor(fen_to_cached(fen))
            assert np.array_equal(through_cache, direct), f"clock={clock}"

    def test_clock_above_the_cap_is_clamped_like_the_encoder(self):
        board = chess.Board()
        board.halfmove_clock = 150
        fen = board.fen()

        assert np.array_equal(
            cached_to_tensor(fen_to_cached(fen)), board_to_tensor(chess.Board(fen))
        )
        assert cached_to_tensor(fen_to_cached(fen))[PLANE_HALFMOVE_CLOCK][0][0] == 1.0

    def test_en_passant_plane_survives(self):
        fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
        assert cached_to_tensor(fen_to_cached(fen))[PLANE_EN_PASSANT].sum() == 1.0


class TestCacheStorage:
    def test_cached_entry_is_uint8_and_the_expected_size(self):
        cached = fen_to_cached(chess.STARTING_FEN)
        assert cached.dtype == np.uint8
        assert cached.shape == TENSOR_SHAPE
        assert cached.nbytes == BYTES_PER_POSITION

    def test_build_and_load_round_trips_every_position(self, tmp_path):
        path = build_cache(FENS, cache_path_for(tmp_path))
        loaded = load_cache(path, expected_rows=len(FENS))

        assert loaded.shape == (len(FENS), *TENSOR_SHAPE)
        for index, fen in enumerate(FENS):
            assert np.array_equal(
                cached_to_tensor(np.asarray(loaded[index])),
                board_to_tensor(chess.Board(fen)),
            )

    def test_batch_decoding_matches_one_at_a_time(self, tmp_path):
        """`cached_to_tensor` is used on whole batches in the training loop."""
        path = build_cache(FENS, cache_path_for(tmp_path))
        loaded = np.asarray(load_cache(path))

        batch = cached_to_tensor(loaded)
        for index in range(len(FENS)):
            assert np.array_equal(batch[index], cached_to_tensor(loaded[index]))

    def test_row_count_mismatch_is_refused(self, tmp_path):
        """A cache built from different shards must not be used silently."""
        path = build_cache(FENS, cache_path_for(tmp_path))
        with pytest.raises(CacheMismatchError, match="rebuild it"):
            load_cache(path, expected_rows=len(FENS) + 1)

    def test_an_interrupted_build_leaves_no_usable_cache(self, tmp_path):
        """The file is renamed into place only once complete."""
        path = cache_path_for(tmp_path)

        with pytest.raises(RuntimeError, match="runtime died"):
            build_cache(ExplodingSequence(), path)

        assert not path.exists()

    def test_size_estimate_matches_reality(self, tmp_path):
        path = build_cache(FENS, cache_path_for(tmp_path))
        loaded = load_cache(path)
        assert loaded.nbytes == cache_size_bytes(len(FENS))


class ExplodingSequence:
    """A sequence that fails partway through, to simulate a lost runtime."""

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index):  # pragma: no cover - iteration path is what matters
        raise RuntimeError("runtime died")

    def __iter__(self):
        yield chess.STARTING_FEN
        raise RuntimeError("runtime died")


def test_full_dataset_estimate_is_the_documented_figure():
    """2.9 GB in uint8 is the number the scope document commits to."""
    gigabytes = cache_size_bytes(2_552_804) / 1e9
    assert gigabytes == pytest.approx(2.9, abs=0.1)
