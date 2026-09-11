"""Evaluation metrics for the position evaluator.

RMSE in ``value`` space is the primary number, because that is the space the
model is trained in and the space the baselines are quoted in.

Centipawns are reported as **MAE, not RMSE**, and the distinction matters. The
inverse transform ``cp = 400 * atanh(value)`` diverges as ``value`` approaches
+/-1, so a squared error in centipawns is dominated by the handful of saturated
positions -- a dataset where 1.43 % of positions are forced mates would produce a
centipawn RMSE that says more about mates than about ordinary play. MAE is far
less sensitive to that tail.

The two rank-based metrics exist because the engine never uses the evaluation as
a number: it compares the evaluations of every legal move and picks the best one.
A model with a large but *consistent* bias would look mediocre by RMSE and play
perfectly well. Sign agreement and Spearman measure what the search actually
depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..normalize import DEFAULT_CP_CLIP, DEFAULT_SCALE, values_to_cp


@dataclass(frozen=True)
class EvaluationMetrics:
    """Everything measured on one split, in one place."""

    n: int
    rmse: float
    mae: float
    mae_cp: float
    sign_agreement: float
    spearman: float

    def summary(self) -> str:
        return "\n".join(
            [
                f"{'posiciones':<26}{self.n:>12,}",
                f"{'RMSE (value)':<26}{self.rmse:>12.4f}",
                f"{'MAE (value)':<26}{self.mae:>12.4f}",
                f"{'MAE (centipeones)':<26}{self.mae_cp:>12.1f}",
                f"{'acuerdo de signo':<26}{self.sign_agreement:>11.2%}",
                f"{'Spearman':<26}{self.spearman:>12.4f}",
            ]
        )

    def row(self) -> str:
        """One line, for printing an epoch inside the training loop."""
        return (
            f"RMSE {self.rmse:.4f}  MAE {self.mae:.4f}  "
            f"MAEcp {self.mae_cp:6.1f}  signo {self.sign_agreement:.1%}  "
            f"rho {self.spearman:.4f}"
        )


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, without pulling in SciPy for one function."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    # Average the ranks of tied runs. The labels are heavily tied -- clipping
    # puts thousands of positions at exactly +/-0.9999 -- so ignoring ties would
    # bias the correlation.
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return ranks


def spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation: does the model order positions the way Stockfish does?"""
    if len(predicted) < 2:
        return float("nan")
    pred_ranks = _ranks(np.asarray(predicted, dtype=np.float64))
    true_ranks = _ranks(np.asarray(actual, dtype=np.float64))

    pred_centred = pred_ranks - pred_ranks.mean()
    true_centred = true_ranks - true_ranks.mean()
    denominator = np.sqrt((pred_centred**2).sum() * (true_centred**2).sum())
    if denominator == 0:
        return float("nan")
    return float((pred_centred * true_centred).sum() / denominator)


def sign_agreement(
    predicted: np.ndarray, actual: np.ndarray, deadband: float = 0.05
) -> float:
    """Fraction of positions where model and engine agree on who stands better.

    Positions the engine calls near-equal are excluded: whether a dead-drawn
    position reads +0.001 or -0.001 is noise, and counting it would drown the
    signal from positions that actually have a better side.
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)

    decided = np.abs(actual) > deadband
    if not decided.any():
        return float("nan")
    return float((np.sign(predicted[decided]) == np.sign(actual[decided])).mean())


def evaluate(
    predicted: np.ndarray,
    actual: np.ndarray,
    scale: float = DEFAULT_SCALE,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> EvaluationMetrics:
    """Compute every metric on one split."""
    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError(
            f"{predicted.shape} predictions against {actual.shape} labels"
        )

    error = predicted - actual
    cp_error = np.abs(
        values_to_cp(predicted, scale, cp_clip) - values_to_cp(actual, scale, cp_clip)
    )

    return EvaluationMetrics(
        n=len(predicted),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        mae_cp=float(np.mean(cp_error)),
        sign_agreement=sign_agreement(predicted, actual),
        spearman=spearman(predicted, actual),
    )
