"""Evaluation metrics.

Two of these encode decisions that are easy to get wrong silently: reporting
centipawns as MAE rather than RMSE, and averaging tied ranks. Both would produce
plausible-looking numbers if implemented carelessly.
"""

from __future__ import annotations

import numpy as np
import pytest

from chessdl.normalize import value_to_cp, values_to_cp
from chessdl.training.metrics import (
    _ranks,
    evaluate,
    sign_agreement,
    spearman,
)


class TestVectorisedInverse:
    def test_matches_the_scalar_transform_elementwise(self):
        """Two implementations of one transform drift apart unless pinned together."""
        values = np.linspace(-1.0, 1.0, 501)
        scalar = np.array([value_to_cp(v) for v in values])
        assert np.allclose(values_to_cp(values), scalar, atol=1e-9)

    def test_saturated_predictions_do_not_diverge(self):
        """A network that outputs exactly +/-1 must not produce infinities."""
        assert np.isfinite(values_to_cp(np.array([-1.0, 1.0]))).all()
        assert values_to_cp(np.array([1.0]))[0] == pytest.approx(2000.0)


class TestSpearman:
    def test_perfect_ordering_is_one(self):
        actual = np.array([-0.9, -0.2, 0.1, 0.5, 0.8])
        assert spearman(actual * 0.3 - 0.2, actual) == pytest.approx(1.0)

    def test_reversed_ordering_is_minus_one(self):
        actual = np.array([-0.9, -0.2, 0.1, 0.5, 0.8])
        assert spearman(-actual, actual) == pytest.approx(-1.0)

    def test_tied_labels_get_averaged_ranks(self):
        """The labels are heavily tied: clipping pins many at exactly +/-0.9999.

        Ranking ties by their position in the array would invent an ordering that
        is not in the data. Averaging is the standard treatment and the one the
        correlation below assumes.
        """
        assert _ranks(np.array([1.0, 1.0, 2.0, 3.0])).tolist() == [0.5, 0.5, 2.0, 3.0]
        assert _ranks(np.array([7.0, 7.0, 7.0])).tolist() == [1.0, 1.0, 1.0]

    def test_a_tie_in_the_labels_costs_a_prediction_that_orders_it(self):
        """Not a bug: Spearman correlates the ranks of *both* series.

        The labels tie the first two positions, the prediction does not, and that
        imposed ordering is a genuine disagreement. Hand-computed:
        ranks [0.5, 0.5, 2, 3] against [1, 0, 2, 3] gives 4.5 / sqrt(22.5).
        """
        actual = np.array([1.0, 1.0, 2.0, 3.0])
        expected = 4.5 / np.sqrt(22.5)
        assert spearman(np.array([5.0, 4.0, 6.0, 7.0]), actual) == pytest.approx(expected)

    def test_a_constant_prediction_is_not_a_number(self):
        assert np.isnan(spearman(np.zeros(10), np.arange(10.0)))


class TestSignAgreement:
    def test_perfect_agreement(self):
        actual = np.array([-0.5, 0.4, 0.9, -0.8])
        assert sign_agreement(actual, actual) == pytest.approx(1.0)

    def test_total_disagreement(self):
        actual = np.array([-0.5, 0.4, 0.9, -0.8])
        assert sign_agreement(-actual, actual) == pytest.approx(0.0)

    def test_near_equal_positions_are_excluded(self):
        """Whether a dead-drawn position reads +0.001 or -0.001 is noise."""
        actual = np.array([0.001, -0.002, 0.7, 0.8])
        predicted = np.array([-1.0, 1.0, 0.6, 0.9])  # wrong on the two tiny ones
        assert sign_agreement(predicted, actual) == pytest.approx(1.0)

    def test_all_equal_positions_gives_not_a_number(self):
        assert np.isnan(sign_agreement(np.zeros(4), np.full(4, 0.001)))


class TestEvaluate:
    def test_a_perfect_model_scores_perfectly(self):
        actual = np.tanh(np.linspace(-3, 3, 200))
        metrics = evaluate(actual, actual)
        assert metrics.rmse == pytest.approx(0.0)
        assert metrics.mae_cp == pytest.approx(0.0)
        assert metrics.sign_agreement == pytest.approx(1.0)
        assert metrics.spearman == pytest.approx(1.0)

    def test_centipawn_error_uses_mae_not_rmse(self):
        """The reason: the inverse transform diverges near +/-1.

        One saturated position with a large centipawn error must not dominate the
        reported figure the way a squared error would.
        """
        actual = np.concatenate([np.zeros(999), [0.9999]])
        predicted = np.concatenate([np.zeros(999), [0.0]])

        metrics = evaluate(predicted, actual)
        worst_cp = abs(values_to_cp(np.array([0.9999]))[0])
        # The mean absolute error is that one error spread over 1000 positions.
        assert metrics.mae_cp == pytest.approx(worst_cp / 1000, rel=1e-6)

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="predictions against"):
            evaluate(np.zeros(5), np.zeros(4))

    def test_summary_and_row_render(self):
        actual = np.tanh(np.linspace(-2, 2, 50))
        metrics = evaluate(actual * 0.9, actual)
        assert "RMSE" in metrics.summary()
        assert "centipeones" in metrics.summary()
        assert "rho" in metrics.row()

    def test_predicting_the_mean_reproduces_the_standard_deviation(self):
        """Ties the metric back to the baseline the notebook reports."""
        rng = np.random.default_rng(0)
        actual = rng.normal(0.05, 0.49, size=10_000)
        metrics = evaluate(np.full_like(actual, actual.mean()), actual)
        assert metrics.rmse == pytest.approx(float(np.std(actual)), rel=1e-6)
