"""The transformer encoder (WBS 4.8).

The same bar as the ResNet's tests: catch an architecture that *runs* but does
not learn. A transformer adds two failure modes a convolutional network does not
have, and both are silent:

* **Tokens that lose their square.** If the flatten/transpose is wrong, each
  token carries one plane across all 64 squares instead of one square across all
  18 planes. The shapes are identical, nothing errors, and the network is being
  fed nonsense.
* **Positional embeddings that do nothing.** Self-attention is permutation
  invariant, so without a working positional signal the model sees a *bag* of
  pieces with no geometry -- and still trains, just to a much worse ceiling.

Both are tested directly rather than inferred from the loss going down.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.encoding import N_PLANES, board_to_tensor  # noqa: E402
from chessdl.models.resnet import ChessResNet  # noqa: E402
from chessdl.models.transformer import (  # noqa: E402
    N_SQUARES,
    ChessTransformer,
    TransformerConfig,
)
from chessdl.training.cache import cached_to_tensor, fen_to_cached  # noqa: E402

SMALL = TransformerConfig(d_model=32, layers=2, heads=4, feedforward=64, dropout=0.0)


def batch_of(fens: list[str]) -> torch.Tensor:
    return torch.from_numpy(cached_to_tensor(np.stack([fen_to_cached(f) for f in fens])))


def material_batch(n: int, seed: int = 0):
    """Positions whose label is a function of material, so learning is possible."""
    rng = np.random.default_rng(seed)
    values = [0, 1, 3, 3, 5, 9, 0]
    fens, targets = [], []
    for _ in range(n):
        board = chess.Board()
        squares = list(board.piece_map())
        rng.shuffle(squares)
        for square in squares[: rng.integers(0, 10)]:
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


class TestTokenisation:
    def test_there_is_one_token_per_square(self):
        model = ChessTransformer(SMALL)
        tokens = model.tokens(torch.zeros(3, N_PLANES, 8, 8))
        assert tokens.shape == (3, N_SQUARES, N_PLANES)

    def test_each_token_carries_one_square_across_all_planes(self):
        """The failure this catches -- a missing transpose -- has the right shape.

        Only 64 == 8*8 and 18 != 64 keep it from being undetectable by shape
        alone, so the content is checked against the plane layout directly.
        """
        board = chess.Board()
        planes = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        tokens = ChessTransformer(SMALL).tokens(planes)[0]

        # e1 holds the side-to-move's king: plane 5 (KING - 1), square 4.
        king_square = chess.E1
        assert tokens[king_square, 5] == 1.0
        # ...and no other piece. Only the piece planes are checked: the castling
        # planes are global, so every token on the starting position carries all
        # four of them, which is the documented redundancy and not a bug.
        assert tokens[king_square, :12].sum() == 1.0

        # a8 holds the opponent's rook: plane 6 + ROOK - 1 = 9.
        assert tokens[chess.A8, 9] == 1.0
        assert tokens[chess.A8, :12].sum() == 1.0

        # e4 is empty on the starting position.
        assert tokens[chess.E4, :12].sum() == 0.0

    def test_token_index_is_the_python_chess_square_number(self):
        """Position embeddings are learned, so only consistency matters -- but the
        mapping still has to be the documented one, or the engine and the trained
        weights would disagree about which square is which."""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.C3, chess.Piece(chess.QUEEN, chess.WHITE))
        planes = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
        tokens = ChessTransformer(SMALL).tokens(planes)[0]

        occupied = (tokens[:, :12].sum(dim=1) > 0).nonzero().flatten()
        assert occupied.tolist() == [chess.C3]


class TestShapes:
    def test_maps_a_batch_of_boards_to_one_number_each(self):
        model = ChessTransformer(SMALL).eval()
        with torch.no_grad():
            assert model(torch.zeros(5, N_PLANES, 8, 8)).shape == (5,)

    def test_mean_pooling_is_also_a_valid_configuration(self):
        model = ChessTransformer(
            TransformerConfig(
                d_model=32, layers=1, heads=4, feedforward=64, pooling="mean"
            )
        ).eval()
        with torch.no_grad():
            assert model(torch.zeros(4, N_PLANES, 8, 8)).shape == (4,)

    def test_parameter_count_sits_in_the_resnet_budget(self):
        """Both architectures are compared at the same budget, or the comparison
        measures capacity instead of architecture."""
        transformer = ChessTransformer().count_parameters()
        resnet = ChessResNet().count_parameters()
        assert transformer == pytest.approx(resnet, rel=0.15)


class TestConfiguration:
    def test_a_head_count_that_does_not_divide_the_width_is_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(d_model=100, heads=8)

    def test_an_unknown_pooling_is_rejected(self):
        with pytest.raises(ValueError, match="pooling"):
            TransformerConfig(pooling="max")

    def test_depth_and_width_change_the_parameter_count(self):
        base = dict(heads=4, feedforward=64)
        small = ChessTransformer(TransformerConfig(d_model=32, layers=2, **base))
        deeper = ChessTransformer(TransformerConfig(d_model=32, layers=4, **base))
        wider = ChessTransformer(TransformerConfig(d_model=64, layers=2, **base))
        assert small.count_parameters() < deeper.count_parameters()
        assert small.count_parameters() < wider.count_parameters()

    def test_weights_round_trip_through_a_state_dict(self):
        """Checkpoints are only useful if they reload into an identical model."""
        torch.manual_seed(0)
        original = ChessTransformer(SMALL).eval()
        restored = ChessTransformer(SMALL).eval()
        restored.load_state_dict(original.state_dict())

        x, _ = material_batch(8, seed=2)
        with torch.no_grad():
            assert torch.equal(original(x), restored(x))


class TestOutputRange:
    def test_output_is_inside_the_required_range(self):
        """Requirement 1.4: guaranteed by the final tanh, not learned."""
        model = ChessTransformer(SMALL).eval()
        with torch.no_grad():
            out = model(torch.randn(64, N_PLANES, 8, 8) * 100)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)

    def test_untrained_model_is_not_already_saturated(self):
        model = ChessTransformer(SMALL).eval()
        x, _ = material_batch(64)
        with torch.no_grad():
            out = model(x)
        assert out.abs().mean() < 0.95, "saturates before training; head is mis-scaled"


class TestPositionMatters:
    """Self-attention is permutation invariant; the geometry has to be supplied."""

    def test_positional_embeddings_are_not_left_at_zero(self):
        model = ChessTransformer(SMALL)
        assert model.positions.abs().sum() > 0, "todas las casillas son indistinguibles"

    def test_moving_a_piece_changes_the_evaluation(self):
        """Without a working positional signal the model sees a bag of pieces:
        the same material on different squares would evaluate identically."""
        torch.manual_seed(0)
        model = ChessTransformer(SMALL).eval()

        def value(square: chess.Square) -> float:
            board = chess.Board()
            board.clear()
            board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            planes = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
            with torch.no_grad():
                return float(model(planes))

        assert value(chess.D4) != value(chess.H1)

    def test_the_positions_parameter_receives_a_gradient(self):
        torch.manual_seed(0)
        x, y = material_batch(16)
        model = ChessTransformer(SMALL).train()
        torch.nn.functional.mse_loss(model(x), y).backward()

        assert model.positions.grad is not None
        assert model.positions.grad.abs().sum() > 0


class TestItActuallyLearns:
    def test_overfits_a_small_batch(self):
        """A model that cannot drive the loss to zero on 64 fixed examples has a
        structural problem, and no hyperparameter will fix it."""
        torch.manual_seed(0)
        x, y = material_batch(64)
        model = ChessTransformer(SMALL).train()
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(400):
            optimiser.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimiser.step()

        assert loss.item() < 0.01, f"did not overfit: final loss {loss.item():.4f}"

    def test_every_parameter_receives_a_finite_gradient(self):
        torch.manual_seed(0)
        x, y = material_batch(16)
        model = ChessTransformer(SMALL).train()

        torch.nn.functional.mse_loss(model(x), y).backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} gradient is not finite"
            assert parameter.grad.abs().sum() > 0, f"{name} gradient is all zeros"

    def test_the_prediction_depends_on_the_pieces(self):
        """Guards against a head that ignores the board and predicts a constant."""
        model = ChessTransformer(SMALL).eval()
        x, _ = material_batch(32, seed=4)
        with torch.no_grad():
            out = model(x)
        assert out.std() > 1e-4, "same output for every position; board is ignored"


class TestDropIn:
    """The transformer has to be usable everywhere the ResNet is, unchanged."""

    def test_the_training_loop_accepts_it(self, tmp_path):
        from chessdl.training.loop import train

        rng = np.random.default_rng(0)
        cache = np.stack([fen_to_cached(chess.Board().fen()) for _ in range(64)])
        targets = rng.normal(0, 0.3, size=64).astype(np.float32)

        run = train(
            ChessTransformer(SMALL),
            cache,
            targets,
            np.arange(0, 48),
            np.arange(48, 64),
            epochs=1,
            batch_size=16,
            device="cpu",
            progress=False,
            amp=False,
        )
        assert run.final_epoch == 1
        assert run.best is not None

    def test_it_evaluates_a_split_like_the_resnet(self):
        from chessdl.training.loop import evaluate_split

        cache = np.stack([fen_to_cached(chess.Board().fen()) for _ in range(32)])
        targets = np.zeros(32, dtype=np.float32)
        metrics = evaluate_split(
            ChessTransformer(SMALL).eval(), cache, targets, np.arange(32), "cpu"
        )
        assert np.isfinite(metrics.rmse)
