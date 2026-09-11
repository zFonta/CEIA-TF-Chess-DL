"""Residual network for position evaluation -- the baseline architecture (WBS 4.2).

Written from scratch rather than adapted from ``torchvision``, and the reason is
structural, not stylistic: an ImageNet ResNet opens with a 7x7 convolution of
stride 2 followed by max-pooling, which would take an 8x8 board down to 2x2 in
two steps. Those stems exist to make 224x224 images tractable. A chess board is
already small, and every square matters.

So the design is the AlphaZero one: 3x3 convolutions with padding 1, which hold
the board at 8x8 from input to head, a stack of residual blocks, and a value
head that collapses to one number.

    (18, 8, 8) -> stem -> N residual blocks -> value head -> scalar in [-1, 1]

The output passes through ``tanh``, so the range required by requirement 1.4 is
guaranteed by construction rather than by hoping the training data taught it.
That also matches the label transform exactly: the targets were produced by
``tanh(cp / 400)``, so the network's last operation is the same one that made
the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..encoding import N_PLANES


@dataclass(frozen=True)
class ResNetConfig:
    """Shape of the network. Recorded with the weights, since it is needed to load them."""

    channels: int = 128
    blocks: int = 8
    value_channels: int = 32
    value_hidden: int = 256
    input_planes: int = N_PLANES

    def describe(self) -> str:
        return (
            f"ResNet {self.channels} canales x {self.blocks} bloques "
            f"(cabeza {self.value_channels}/{self.value_hidden})"
        )


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection around them.

    The second batch-norm is applied *before* the skip is added and the final
    ReLU *after*, which is the ordering the original residual paper uses.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # No bias on the convolutions: the batch-norm that follows has its own
        # shift, so a bias term here would be redundant parameters.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class ValueHead(nn.Module):
    """Collapse the board representation to a single evaluation in [-1, 1]."""

    def __init__(self, channels: int, value_channels: int, hidden: int) -> None:
        super().__init__()
        # A 1x1 convolution first: it cuts the width before the flatten, which is
        # where the parameter count would otherwise explode (8*8*channels).
        self.conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(value_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(value_channels * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm(self.conv(x)))
        out = out.flatten(start_dim=1)
        out = self.relu(self.fc1(out))
        return torch.tanh(self.fc2(out)).squeeze(-1)


class ChessResNet(nn.Module):
    """Evaluate a position from the side-to-move's point of view.

    Input is ``(batch, 18, 8, 8)``; output is ``(batch,)`` in [-1, 1], on the
    same scale as the ``value_stm`` column of the dataset.
    """

    def __init__(self, config: ResNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResNetConfig()

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.config.input_planes,
                self.config.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.config.channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock(self.config.channels) for _ in range(self.config.blocks))
        )
        self.value_head = ValueHead(
            self.config.channels, self.config.value_channels, self.config.value_hidden
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.blocks(self.stem(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return f"{self.config.describe()} -- {self.count_parameters():,} parametros"
