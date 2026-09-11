"""Centipawn <-> normalized value transforms.

Requirement 1.1 asks for Stockfish centipawn evaluations to be mapped into
``[-1, 1]`` before being used as training labels; requirement 4.2 asks for the
inverse transform so the engine can report evaluations on Stockfish's own scale.

    value = tanh(cp / scale)          cp = scale * atanh(value)

``cp`` is clipped to ``+/- cp_clip`` before the transform. That keeps ``|value|``
strictly below 1, so ``atanh`` never diverges -- including when the network
saturates and outputs exactly +/-1.

Mate scores collapse to the clip bound, so ``mate_in`` is carried alongside the
label to preserve the information the clipping throws away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import chess.engine

DEFAULT_SCALE = 400.0
DEFAULT_CP_CLIP = 2000


def clip_cp(cp: float, cp_clip: int = DEFAULT_CP_CLIP) -> int:
    """Clip a centipawn score to the representable range."""
    return int(max(-cp_clip, min(cp_clip, cp)))


def cp_to_value(
    cp: float,
    scale: float = DEFAULT_SCALE,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> float:
    """Map centipawns to ``[-1, 1]`` (requirement 1.1)."""
    return math.tanh(clip_cp(cp, cp_clip) / scale)


def value_to_cp(
    value: float,
    scale: float = DEFAULT_SCALE,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> float:
    """Map a normalized value back to centipawns (requirement 4.2).

    The value is first clamped to the range actually reachable by
    :func:`cp_to_value`, which both keeps ``atanh`` finite and stops a saturated
    network prediction from being reported as an absurd centipawn score.
    """
    limit = math.tanh(cp_clip / scale)
    clamped = max(-limit, min(limit, float(value)))
    return scale * math.atanh(clamped)


def values_to_cp(
    values: "np.ndarray",
    scale: float = DEFAULT_SCALE,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> "np.ndarray":
    """Vectorised :func:`value_to_cp`, for scoring a whole split at once.

    Same semantics, element for element -- a test pins the two together, because
    two implementations of one transform is exactly the sort of pair that drifts
    apart without anything failing.
    """
    import numpy as np

    limit = math.tanh(cp_clip / scale)
    clamped = np.clip(np.asarray(values, dtype=np.float64), -limit, limit)
    return scale * np.arctanh(clamped)


def score_to_cp(
    score: chess.engine.Score,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> tuple[int, int | None]:
    """Convert a one-sided engine score into clipped centipawns.

    Returns ``(cp, mate_in)``, where ``mate_in`` is ``None`` for ordinary scores
    and the signed distance to mate otherwise (positive: the side this score
    belongs to delivers mate).
    """
    mate_in = score.mate()
    if mate_in is not None:
        # A forced mate is beyond any centipawn evaluation, so it pins to the
        # clip bound; `mate_in` keeps the detail that the clip discards.
        return (cp_clip if mate_in > 0 else -cp_clip), mate_in
    cp = score.score()
    if cp is None:  # pragma: no cover - guarded by python-chess's own invariants
        raise ValueError(f"Score carries neither a centipawn nor a mate value: {score!r}")
    return clip_cp(cp, cp_clip), None


@dataclass(frozen=True)
class Label:
    """One labelled position, in both points of view.

    The network trains on ``value_stm`` (side-to-move perspective, matching the
    mirrored board encoding); the engine and CLI report ``value_white``, which is
    the white-relative scale requirements 1.4 and 4.2 are written against. The
    two are the same number up to sign, and :mod:`chessdl.data.validate` checks
    that invariant across the whole dataset.
    """

    cp_white: int
    cp_stm: int
    value_white: float
    value_stm: float
    is_mate: bool
    mate_in: int  # signed, from white's point of view; 0 when there is no mate


def label_from_povscore(
    pov_score: chess.engine.PovScore,
    turn: chess.Color,
    scale: float = DEFAULT_SCALE,
    cp_clip: int = DEFAULT_CP_CLIP,
) -> Label:
    """Build a :class:`Label` from an engine score and the side to move.

    ``cp_stm`` is *derived* from ``cp_white`` rather than converted independently,
    which makes the sign invariant exact by construction instead of something
    that merely ought to hold.
    """
    cp_white, mate_white = score_to_cp(pov_score.white(), cp_clip)
    cp_stm = cp_white if turn == chess.WHITE else -cp_white

    return Label(
        cp_white=cp_white,
        cp_stm=cp_stm,
        value_white=cp_to_value(cp_white, scale, cp_clip),
        value_stm=cp_to_value(cp_stm, scale, cp_clip),
        is_mate=mate_white is not None,
        mate_in=int(mate_white) if mate_white is not None else 0,
    )
