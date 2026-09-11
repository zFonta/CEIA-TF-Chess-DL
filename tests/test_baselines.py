"""The reference points that make the network's RMSE readable.

These are also the first sanity check of the whole training block: if the
material fit does not recover something close to the textbook piece values, the
encoding is wrong, and it is much cheaper to learn that here than after a
training campaign.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from chessdl.training.baselines import (
    describe_weights,
    material_baseline,
    material_features,
    mean_baseline,
    rmse,
    score,
)
from chessdl.training.cache import fen_to_cached

# Rough piece values in pawns, to check the fit lands somewhere sane.
PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
                chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}


def synthetic_dataset(n: int = 600, seed: int = 0):
    """Positions whose label really is a function of material.

    Built by removing random pieces from the starting position and labelling with
    the material balance, so the material baseline *should* fit it almost
    perfectly. That makes the test about the implementation rather than about
    chess.
    """
    rng = np.random.default_rng(seed)
    tensors, targets = [], []

    for _ in range(n):
        board = chess.Board()
        squares = list(board.piece_map())
        rng.shuffle(squares)
        for square in squares[: rng.integers(0, 8)]:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:
                board.remove_piece_at(square)

        balance = 0.0
        for piece in board.piece_map().values():
            sign = 1.0 if piece.color == board.turn else -1.0
            balance += sign * PIECE_VALUES[piece.piece_type]

        tensors.append(fen_to_cached(board.fen()))
        targets.append(np.tanh(balance / 9.0))

    return np.stack(tensors), np.array(targets, dtype=np.float32)


class TestMetrics:
    def test_rmse_of_a_perfect_prediction_is_zero(self):
        values = np.array([0.1, -0.4, 0.9], dtype=np.float32)
        assert rmse(values, values) == 0.0

    def test_score_reports_both_metrics(self):
        predicted = np.array([0.0, 0.0], dtype=np.float32)
        actual = np.array([1.0, -1.0], dtype=np.float32)
        result = score("x", predicted, actual)
        assert result.rmse == pytest.approx(1.0)
        assert result.mae == pytest.approx(1.0)


class TestMeanBaseline:
    def test_rmse_equals_the_standard_deviation(self):
        """Predicting the mean gives an RMSE of exactly the target's std."""
        rng = np.random.default_rng(1)
        targets = rng.normal(0.05, 0.49, size=5000).astype(np.float32)
        result = mean_baseline(targets, targets)
        assert result.rmse == pytest.approx(float(np.std(targets)), rel=1e-5)

    def test_uses_the_training_mean_not_the_evaluation_mean(self):
        """Fitting on the evaluation split would be leakage, even for a baseline."""
        train = np.full(100, 0.5, dtype=np.float32)
        evaluation = np.full(100, -0.5, dtype=np.float32)
        # Predicting the train mean (0.5) against -0.5 gives an error of 1.0.
        assert mean_baseline(train, evaluation).rmse == pytest.approx(1.0)


class TestMaterialFeatures:
    def test_starting_position_has_no_material_difference(self):
        features = material_features(fen_to_cached(chess.STARTING_FEN)[None])
        assert np.allclose(features[0, :6], 0.0)

    def test_bias_column_is_present(self):
        features = material_features(fen_to_cached(chess.STARTING_FEN)[None])
        assert features[0, -1] == 1.0

    def test_missing_opponent_rook_shows_as_a_positive_difference(self):
        # Black is missing the a8 rook, white to move -> white is a rook up.
        fen = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"
        features = material_features(fen_to_cached(fen)[None])
        assert features[0, chess.ROOK - 1] == pytest.approx(1.0)

    def test_difference_is_from_the_movers_point_of_view(self):
        """The same material edge reads positive for whoever owns it."""
        white_up = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1"
        black_up = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR b Kkq - 0 1"
        assert material_features(fen_to_cached(white_up)[None])[0, chess.ROOK - 1] == 1.0
        assert material_features(fen_to_cached(black_up)[None])[0, chess.ROOK - 1] == 1.0


class TestMaterialBaseline:
    def test_beats_the_mean_on_material_driven_labels(self):
        tensors, targets = synthetic_dataset()
        split = len(targets) // 2

        constant = mean_baseline(targets[:split], targets[split:])
        material, _ = material_baseline(
            tensors[:split], targets[:split], tensors[split:], targets[split:]
        )
        assert material.rmse < constant.rmse

    def test_recovers_sensible_relative_piece_values(self):
        """A queen must weigh more than a rook, and a rook more than a pawn."""
        tensors, targets = synthetic_dataset(n=1500, seed=3)
        _, coefficients = material_baseline(tensors, targets, tensors, targets)

        pawn, knight, _bishop, rook, queen = (coefficients[i] for i in range(5))
        assert queen > rook > knight > pawn > 0

    def test_predictions_stay_inside_the_label_range(self):
        """A linear fit does not know the targets live in [-1, 1]."""
        tensors, targets = synthetic_dataset(n=400, seed=5)
        # Extreme material imbalance, far outside the fitted range.
        lopsided = np.stack([fen_to_cached("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")] * 4)
        _, coefficients = material_baseline(tensors, targets, tensors, targets)

        raw = material_features(lopsided) @ coefficients
        clipped = np.clip(raw, -1.0, 1.0)
        assert np.all(np.abs(clipped) <= 1.0)


class TestDescribeWeights:
    def test_renders_ratios_when_the_pawn_weight_is_positive(self):
        text = describe_weights([0.075, 0.231, 0.225, 0.374, 0.660, 0.0, 0.04])
        assert "3.08" in text  # caballo / peon
        assert "8.80" in text  # dama / peon
        assert "degenerado" not in text

    def test_withholds_ratios_when_the_fit_is_degenerate(self):
        """A negative pawn weight flips every ratio; printing them would mislead.

        This is not hypothetical: it happens whenever the fitting positions do
        not vary material independently, which a biased sample easily produces.
        """
        text = describe_weights([-0.087, 0.100, 0.131, 0.207, 0.027, 0.0, 0.29])
        assert "degenerado" in text
        assert "--" in text
        # The raw weights are still shown; only the ratios are suppressed.
        assert "-0.0870" in text
