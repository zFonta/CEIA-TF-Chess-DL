"""The residual baseline (WBS 4.2).

The tests that matter here are the ones that catch an architecture which *runs*
but does not learn: a broken skip connection, a head that throws away the board,
a detached graph. Those produce no error and no warning -- just a model that
converges to the mean and a training campaign spent finding that out.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.encoding import N_PLANES  # noqa: E402
from chessdl.models.resnet import ChessResNet, ResNetConfig  # noqa: E402
from chessdl.training.cache import cached_to_tensor, fen_to_cached  # noqa: E402

SMALL = ResNetConfig(channels=32, blocks=2)


def batch_of(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(cached_to_tensor(np.stack([fen_to_cached(f) for f in fens])))


def material_batch(n: int, seed: int = 0):
    """Positions whose label is a function of material, so learning is possible."""
    rng = np.random.default_rng(seed)
    values = [0, 1, 3, 3, 5, 9, 0]
    fens, targets = [], []
    for _ in range(n):
        board = chess.Board()
        for square in list(board.piece_map())[: rng.integers(0, 10)]:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:
                board.remove_piece_at(square)
        balance = sum(
            (1 if piece.color == board.turn else -1) * values[piece.piece_type]
            for piece in board.piece_map().values()
        )
        fens.append(board.fen())
        targets.append(np.tanh(balance / 9))
    return batch_of(fens), torch.tensor(targets, dtype=torch.float32)


class TestShapes:
    def test_maps_a_batch_of_boards_to_one_number_each(self):
        model = ChessResNet(SMALL).eval()
        x = torch.zeros(5, N_PLANES, 8, 8)
        with torch.no_grad():
            assert model(x).shape == (5,)

    def test_board_stays_8x8_through_the_body(self):
        """An ImageNet stem would have downsampled it; this one must not."""
        model = ChessResNet(SMALL).eval()
        x = torch.zeros(2, N_PLANES, 8, 8)
        with torch.no_grad():
            features = model.blocks(model.stem(x))
        assert features.shape == (2, SMALL.channels, 8, 8)

    def test_parameter_count_matches_the_documented_budget(self):
        """The scope document commits to ~2.9 M for the default configuration."""
        model = ChessResNet()
        assert model.count_parameters() == pytest.approx(2.9e6, rel=0.05)


class TestOutputRange:
    def test_output_is_inside_the_required_range(self):
        """Requirement 1.4: the evaluation must live in [-1, 1].

        Guaranteed by the final tanh rather than learned, so it holds from the
        first random initialisation and cannot drift.
        """
        model = ChessResNet(SMALL).eval()
        with torch.no_grad():
            out = model(torch.randn(64, N_PLANES, 8, 8) * 100)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_untrained_model_is_not_already_saturated(self):
        model = ChessResNet(SMALL).eval()
        x, _ = material_batch(64)
        with torch.no_grad():
            out = model(x)
        assert out.abs().mean() < 0.95, "saturates before training; head is mis-scaled"


class TestItActuallyLearns:
    def test_overfits_a_small_batch(self):
        """The standard check that gradients reach every layer and the head sees the board.

        A model that cannot drive the loss to zero on 64 fixed examples has a
        structural problem, and no amount of hyperparameter tuning will fix it.
        """
        torch.manual_seed(0)
        x, y = material_batch(64)
        model = ChessResNet(SMALL).train()
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)

        for _ in range(400):
            optimiser.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimiser.step()

        assert loss.item() < 0.01, f"did not overfit: final loss {loss.item():.4f}"

    def test_every_parameter_receives_a_finite_gradient(self):
        torch.manual_seed(0)
        x, y = material_batch(16)
        model = ChessResNet(SMALL).train()

        torch.nn.functional.mse_loss(model(x), y).backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
            assert parameter.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_the_prediction_depends_on_the_pieces(self):
        """Guards against a head that ignores the board and predicts a constant."""
        model = ChessResNet(SMALL).eval()
        x, _ = material_batch(32, seed=4)
        with torch.no_grad():
            out = model(x)
        assert out.std() > 1e-4, "same output for every position; board is ignored"


class TestConfiguration:
    def test_depth_and_width_change_the_parameter_count(self):
        small = ChessResNet(ResNetConfig(channels=32, blocks=2)).count_parameters()
        deeper = ChessResNet(ResNetConfig(channels=32, blocks=4)).count_parameters()
        wider = ChessResNet(ResNetConfig(channels=64, blocks=2)).count_parameters()
        assert small < deeper < wider

    def test_weights_round_trip_through_a_state_dict(self):
        """Checkpoints are only useful if they reload into an identical model."""
        torch.manual_seed(0)
        original = ChessResNet(SMALL).eval()
        restored = ChessResNet(SMALL).eval()
        restored.load_state_dict(original.state_dict())

        x, _ = material_batch(8, seed=2)
        with torch.no_grad():
            assert torch.equal(original(x), restored(x))
