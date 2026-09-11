"""Non-neural reference points, so the network's RMSE means something.

An RMSE reported on its own is not interpretable: 0.35 could be excellent or
embarrassing. These two baselines cost minutes to compute and bracket the range.

* **Mean.** Always predict the training mean. Its RMSE is the standard deviation
  of the targets -- about 0.488 on this dataset. A model that fails to beat it
  has a bug, not an architecture problem, and that is worth knowing on day one
  rather than after a tuning campaign.

* **Material.** A least-squares fit on piece counts alone: the "knows some
  chess, understands no chess" line. It captures everything a beginner's
  material count captures and nothing else, so the gap between it and the
  network is the part the network actually contributes.

Both are fitted on train and reported on validation, exactly like the models, so
the numbers sit in the same table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..encoding import N_PIECE_TYPES, PLANE_OPPONENT_PIECES, PLANE_OWN_PIECES


@dataclass(frozen=True)
class BaselineScore:
    """What a baseline achieved, in the same units as the models."""

    name: str
    rmse: float
    mae: float

    def __str__(self) -> str:
        return f"{self.name:<28}RMSE {self.rmse:.4f}   MAE {self.mae:.4f}"


def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def score(name: str, predicted: np.ndarray, actual: np.ndarray) -> BaselineScore:
    return BaselineScore(name=name, rmse=rmse(predicted, actual), mae=mae(predicted, actual))


def mean_baseline(train_targets: np.ndarray, eval_targets: np.ndarray) -> BaselineScore:
    """Predict the training mean for every position."""
    prediction = np.full_like(eval_targets, float(np.mean(train_targets)))
    return score("media constante", prediction, eval_targets)


def material_features(tensors: np.ndarray) -> np.ndarray:
    """Count each piece type, own minus opponent, from cached or raw tensors.

    Takes the difference rather than the twelve raw counts because that is what
    material balance means, and because the encoding is already from the mover's
    point of view: "my knights minus your knights" is the same feature whichever
    colour is to move.
    """
    tensors = np.asarray(tensors)
    own = tensors[:, PLANE_OWN_PIECES : PLANE_OWN_PIECES + N_PIECE_TYPES]
    opponent = tensors[
        :, PLANE_OPPONENT_PIECES : PLANE_OPPONENT_PIECES + N_PIECE_TYPES
    ]
    difference = own.sum(axis=(2, 3)).astype(np.float64) - opponent.sum(
        axis=(2, 3)
    ).astype(np.float64)

    # A bias column, so the fit can learn the tempo advantage of being to move.
    return np.column_stack([difference, np.ones(len(difference))])


def material_baseline(
    train_tensors: np.ndarray,
    train_targets: np.ndarray,
    eval_tensors: np.ndarray,
    eval_targets: np.ndarray,
) -> tuple[BaselineScore, np.ndarray]:
    """Least-squares fit on material difference. Returns the score and the weights.

    The weights are worth looking at, but **not against the textbook 1/3/3/5/9**.
    Those values live in centipawns; this fit is in ``value`` space, which
    ``tanh`` compresses. A queen is +900 cp, and ``tanh(900/400) = 0.978`` against
    ``tanh(100/400) = 0.245`` for a pawn -- a ratio of 4, not 9. Real positions
    compress further still, because material explains only part of the label and
    a linear fit shrinks toward the mean.

    What *should* hold is the ordering (pawn < knight <= bishop < rook < queen),
    a king weight of exactly zero -- both sides always have one, so the feature is
    identically zero -- and a bias term close to the value of having the move.
    """
    features = material_features(train_tensors)
    coefficients, *_ = np.linalg.lstsq(features, train_targets.astype(np.float64), rcond=None)

    prediction = material_features(eval_tensors) @ coefficients
    # The targets live in [-1, 1]; a linear fit does not know that.
    prediction = np.clip(prediction, -1.0, 1.0)

    return score("material (lineal)", prediction.astype(eval_targets.dtype), eval_targets), coefficients


def describe_weights(coefficients: Sequence[float]) -> str:
    """Render the material weights relative to a pawn, for a sanity read.

    Expressing the weights in pawns only means anything when the pawn weight is
    positive. It may not be: if the training positions never vary some piece
    count independently -- say every sampled position removes material from the
    same side -- the design matrix is degenerate and the fit comes back with
    arbitrary, even negative, coefficients. Dividing by a negative pawn weight
    flips every ratio and prints a confident-looking table of nonsense, so the
    ratios are withheld and the reason stated instead.
    """
    names = ["peon", "caballo", "alfil", "torre", "dama", "rey"]
    pawn = coefficients[0]
    usable = pawn > 1e-9

    lines = [f"{'pieza':<10}{'peso':>10}{'en peones':>12}"]
    lines.append("-" * 32)
    for name, weight in zip(names, coefficients[: len(names)]):
        relative = f"{weight / pawn:>12.2f}" if usable else f"{'--':>12}"
        lines.append(f"{name:<10}{weight:>10.4f}{relative}")
    lines.append(f"{'sesgo':<10}{coefficients[len(names)]:>10.4f}")

    if not usable:
        lines.append("")
        lines.append(
            f"El peso del peon es {pawn:+.4f}, no positivo: el ajuste es degenerado y"
        )
        lines.append(
            "las razones en peones no significan nada. Suele indicar que las"
        )
        lines.append(
            "posiciones de ajuste no varian el material de forma independiente."
        )
        return "\n".join(lines)

    # Without this note the natural reflex is to compare against 1/3/3/5/9 and
    # conclude something is broken. Those values are centipawns; this fit is in
    # value space, where tanh has already compressed the top end.
    lines.append("")
    lines.append("Las razones NO deben compararse con 1/3/3/5/9: esa es la escala en")
    lines.append("centipeones. El ajuste es en espacio value, comprimido por tanh, donde")
    lines.append("una dama (+900 cp -> 0,978) vale ~4 peones (+100 cp -> 0,245) y no 9.")
    lines.append("Lo que tiene que cumplirse es el orden, rey = 0, y un sesgo cercano al")
    lines.append("valor de tener la jugada.")
    return "\n".join(lines)
