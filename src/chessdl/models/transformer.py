"""Transformer encoder for position evaluation -- the second architecture (WBS 4.8).

The residual network reads the board the way it reads an image: through 3x3
windows, so a relationship between two distant squares only exists after enough
layers have stacked to span the distance between them. A bishop on c1 bearing on
h6 is five squares away, which is several blocks of indirection.

A transformer removes that distance entirely. The board is cut into **64 tokens,
one per square**, and every layer of self-attention lets any square look at any
other in a single step. Long-range relations -- pins, batteries, an unprotected
piece on the far side of the board -- are one attention hop, not five convolutions.

    (18, 8, 8) -> 64 tokens of 18 features -> projection + position -> N encoder
    layers -> pooled -> value head -> scalar in [-1, 1]

Three design points are worth stating, because each one is a decision that could
reasonably have gone the other way:

**Learned positional embeddings, not sinusoidal.** A chess board is not a
sequence: the "distance" between a1 and a2 is not comparable to the distance
between a1 and b1 in any way a sine wave captures. There are only 64 positions
and they never change, so each one simply gets its own learned vector and the
network works out the geometry from the data.

**Pre-norm (``norm_first=True``).** Post-norm transformers are notoriously hard
to start from scratch without a long warm-up, and the training budget here is a
Colab session, not a research cluster. Pre-norm puts the normalisation inside
the residual branch, which keeps the gradient path clean from the first step.

**The same 18-plane input as the ResNet.** The tensor cache, the split and the
training loop are all reused unchanged, so the comparison between the two
architectures is about the architecture and nothing else. The cost is that the
five global planes (castling rights and the halfmove clock) are constant across
squares and therefore repeated in all 64 tokens; the input projection is free to
collapse that redundancy, and paying for it is cheaper than maintaining a second
encoding.

The output passes through ``tanh``, exactly as in the ResNet, so requirement
1.4's range holds by construction and the last operation of the network is the
same one that produced the labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..encoding import BOARD_SIZE, N_PLANES

#: One token per square of the board.
N_SQUARES = BOARD_SIZE * BOARD_SIZE


@dataclass(frozen=True)
class TransformerConfig:
    """Shape of the network. Recorded with the weights, since it is needed to load them.

    The defaults land at roughly 2.7 M parameters, deliberately close to the
    ResNet's 2.91 M: the two architectures are compared at the same budget, so a
    difference in the results is a difference in how the board is read and not in
    how much capacity was spent reading it.
    """

    d_model: int = 192
    layers: int = 6
    heads: int = 8
    #: Width of the feed-forward block inside each layer. Four times ``d_model``
    #: is the usual ratio and holds most of the layer's parameters.
    feedforward: int = 768
    value_hidden: int = 256
    input_planes: int = N_PLANES
    #: Dropout inside the encoder layers (attention and feed-forward).
    dropout: float = 0.1
    #: Dropout before the last linear layer of the value head, mirroring the
    #: ResNet's. Kept separate because head dropout and body dropout regularise
    #: different things and the sweep may want to move only one.
    head_dropout: float = 0.0
    #: How the 64 square tokens are collapsed into one vector: ``"cls"`` adds a
    #: 65th learned token that attends to the board, ``"mean"`` averages the
    #: squares. Mean weights every square equally, empty ones included; the CLS
    #: token lets attention decide. Which wins is an empirical question, so it is
    #: a configuration field rather than a hard-coded choice.
    pooling: str = "cls"

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError(
                f"d_model={self.d_model} no es divisible por heads={self.heads}"
            )
        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"pooling desconocido: {self.pooling!r}")

    def describe(self) -> str:
        extra = f", dropout cabeza {self.head_dropout}" if self.head_dropout else ""
        return (
            f"Transformer d_model {self.d_model} x {self.layers} capas x "
            f"{self.heads} cabezas (ffn {self.feedforward}, pooling {self.pooling}, "
            f"dropout {self.dropout}{extra})"
        )


class ValueHead(nn.Module):
    """Collapse the pooled board representation to a single evaluation in [-1, 1].

    Deliberately the same two-layer shape as the ResNet's head, so any difference
    between the architectures comes from the body rather than from one of them
    having been given a larger head.
    """

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.activation(self.fc1(x)))
        return torch.tanh(self.fc2(out)).squeeze(-1)


class ChessTransformer(nn.Module):
    """Evaluate a position from the side-to-move's point of view.

    Input is ``(batch, 18, 8, 8)`` and output is ``(batch,)`` in [-1, 1] -- the
    same contract as :class:`~chessdl.models.resnet.ChessResNet`, so the two are
    interchangeable in the training loop, the checkpoints and the engine.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()

        self.input_projection = nn.Linear(self.config.input_planes, self.config.d_model)
        # One vector per square. 64 positions that never change, so there is
        # nothing to extrapolate to and nothing a fixed encoding would buy.
        self.positions = nn.Parameter(
            torch.zeros(1, N_SQUARES, self.config.d_model)
        )
        if self.config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        else:
            self.register_parameter("cls_token", None)

        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.heads,
            dim_feedforward=self.config.feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # `enable_nested_tensor` is a fast path for padded batches, and it does
        # not apply here twice over: every board has exactly 64 squares, so there
        # is no padding to skip, and PyTorch disables it under `norm_first`
        # anyway. Saying so explicitly keeps it from warning about it on every
        # model built in a notebook.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=self.config.layers, enable_nested_tensor=False
        )
        # Pre-norm layers leave the residual stream un-normalised on the way out,
        # so the stack needs a final normalisation before the head reads it.
        self.norm = nn.LayerNorm(self.config.d_model)
        self.value_head = ValueHead(
            self.config.d_model, self.config.value_hidden, self.config.head_dropout
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        """Small random embeddings, rather than the zeros they were allocated as.

        The positional embeddings start at zero-mean noise of the scale ViT uses.
        Leaving them at exactly zero is not fatal -- the gradient would move them
        -- but every square would be indistinguishable on the first forward pass,
        and the network would spend its first epochs undoing that symmetry.
        """
        nn.init.trunc_normal_(self.positions, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Turn ``(batch, 18, 8, 8)`` planes into ``(batch, 64, 18)`` square tokens.

        ``flatten(2)`` gives ``(batch, 18, 64)`` -- plane-major -- and the
        transpose turns it into one row per square, each carrying that square's
        18 features. Token ``i`` is square ``rank * 8 + file``, which is
        python-chess's own square numbering from the mover's perspective.
        """
        return x.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(self.tokens(x)) + self.positions

        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.norm(self.encoder(tokens))
        pooled = encoded[:, 0] if self.cls_token is not None else encoded.mean(dim=1)
        return self.value_head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return f"{self.config.describe()} -- {self.count_parameters():,} parametros"
