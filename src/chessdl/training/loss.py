"""Loss functions for the value head (WBS 4.3).

**MSE** is the starting point: the task is regression onto a bounded scalar and
the metric that decides the comparison is RMSE, so training on the squared error
optimises the thing being reported.

**Huber** is the alternative worth measuring, and the reason is specific to this
dataset rather than general. The labels are not smoothly spread over [-1, 1]:
1.43 % of positions are forced mates pinned to +/-0.9999, and the +/-2000
centipawn clip stacks more mass at the ends. MSE weights an error by its square,
so those saturated positions dominate the gradient -- and they are exactly the
positions where the precise value carries least information, since "winning" and
"winning more" are the same thing when choosing a move. Huber is linear beyond
``delta`` and so lets the ordinary positions, which is where the engine spends
its time, count for more.

Which one wins is an empirical question, and settling it is what task 4.5 is for.
Both are exposed through one factory so switching is a config change rather than
an edit to the training loop.
"""

from __future__ import annotations

import torch
from torch import nn

MSE = "mse"
HUBER = "huber"
LOSSES = (MSE, HUBER)

#: Where Huber switches from squared to linear. In value units, 0.1 is about 40
#: centipawns near the centre of the scale -- small enough that ordinary
#: positions stay in the quadratic regime, large enough that the saturated tail
#: does not.
DEFAULT_HUBER_DELTA = 0.1


class UnknownLossError(ValueError):
    """The configured loss name is not one this project implements."""


def build_loss(name: str = MSE, huber_delta: float = DEFAULT_HUBER_DELTA) -> nn.Module:
    """Return the loss module named in the configuration."""
    if name == MSE:
        return nn.MSELoss()
    if name == HUBER:
        return nn.HuberLoss(delta=huber_delta)
    raise UnknownLossError(
        f"Unknown loss {name!r}. Available: {', '.join(LOSSES)}."
    )


def describe_loss(name: str, huber_delta: float = DEFAULT_HUBER_DELTA) -> str:
    if name == MSE:
        return "MSE"
    if name == HUBER:
        return f"Huber (delta={huber_delta})"
    raise UnknownLossError(f"Unknown loss {name!r}.")


@torch.no_grad()
def saturated_share(targets: torch.Tensor, threshold: float = 0.99) -> float:
    """Fraction of labels sitting in the saturated tail.

    The number that motivates trying Huber at all, so it is worth printing next
    to the loss choice rather than leaving in a docstring.
    """
    return float((targets.abs() >= threshold).float().mean())
