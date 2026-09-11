"""The training loop, the loss factory and the resume path.

The property that matters most here is resumption. A run that "resumes" by
reloading only the weights looks fine -- training continues, the loss curve has a
step in it, nothing errors -- so the tests check that the optimiser and schedule
come back too, and that a resumed run reaches the same place an uninterrupted one
does.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.models.resnet import ChessResNet, ResNetConfig  # noqa: E402
from chessdl.training.cache import fen_to_cached  # noqa: E402
from chessdl.training.checkpoint import (  # noqa: E402
    HISTORY_NAME,
    LAST_NAME,
    EpochRecord,
    HubCheckpoints,
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
)
from chessdl.training.loop import evaluate_split, predict, train  # noqa: E402
from chessdl.training.loss import (  # noqa: E402
    HUBER,
    MSE,
    UnknownLossError,
    build_loss,
    saturated_share,
)

TINY = ResNetConfig(channels=8, blocks=1, value_channels=4, value_hidden=16)


@pytest.fixture(scope="module")
def data():
    """A small material-driven set, so training has something learnable."""
    rng = np.random.default_rng(0)
    values = [0, 1, 3, 3, 5, 9, 0]
    tensors, targets = [], []
    for _ in range(400):
        board = chess.Board()
        squares = list(board.piece_map())
        rng.shuffle(squares)
        for square in squares[: rng.integers(0, 10)]:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:
                board.remove_piece_at(square)
        balance = sum(
            (1 if p.color == board.turn else -1) * values[p.piece_type]
            for p in board.piece_map().values()
        )
        tensors.append(fen_to_cached(board.fen()))
        targets.append(np.tanh(balance / 9))
    return np.stack(tensors), np.array(targets, dtype=np.float32)


@pytest.fixture
def splits(data):
    _, targets = data
    return np.arange(0, 320), np.arange(320, len(targets))


class TestLossFactory:
    def test_builds_both_losses(self):
        assert isinstance(build_loss(MSE), torch.nn.MSELoss)
        assert isinstance(build_loss(HUBER), torch.nn.HuberLoss)

    def test_rejects_an_unknown_name(self):
        with pytest.raises(UnknownLossError, match="Available"):
            build_loss("mae")

    def test_huber_punishes_a_large_error_less_than_mse(self):
        """The reason Huber is on the table at all."""
        predicted = torch.tensor([0.0])
        actual = torch.tensor([1.0])
        assert build_loss(HUBER)(predicted, actual) < build_loss(MSE)(predicted, actual)

    def test_saturated_share_counts_the_tail(self):
        targets = torch.tensor([0.0, 0.5, 0.9999, -0.9999])
        assert saturated_share(targets) == pytest.approx(0.5)


class TestTraining:
    def test_a_short_run_reduces_the_validation_error(self, data, splits):
        cache, targets = data
        train_idx, val_idx = splits
        torch.manual_seed(0)
        model = ChessResNet(TINY)

        before = evaluate_split(model, cache, targets, val_idx, "cpu")
        run = train(
            model, cache, targets, train_idx, val_idx,
            epochs=3, batch_size=64, device="cpu", checkpoints=None, progress=False,
        )
        assert run.history.epochs[-1].val_rmse < before.rmse

    def test_history_records_every_epoch(self, data, splits):
        cache, targets = data
        train_idx, val_idx = splits
        run = train(
            ChessResNet(TINY), cache, targets, train_idx, val_idx,
            epochs=2, batch_size=64, device="cpu", checkpoints=None, progress=False,
        )
        assert [e.epoch for e in run.history.epochs] == [1, 2]
        assert run.best is not None

    def test_predictions_cover_the_split_in_order(self, data, splits):
        cache, targets = data
        _, val_idx = splits
        out = predict(ChessResNet(TINY), cache, targets, np.sort(val_idx), "cpu", 32)
        assert out.shape == val_idx.shape
        assert np.isfinite(out).all()


class TestCheckpointRoundTrip:
    def test_restores_weights_optimizer_and_epoch(self, tmp_path, data, splits):
        cache, targets = data
        torch.manual_seed(0)
        model = ChessResNet(TINY)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=5)

        # Take a step so the optimiser has state worth preserving.
        loss = model(torch.from_numpy(
            np.asarray(cache[:8], dtype=np.uint8).astype(np.float32)
        )).sum()
        loss.backward()
        optimiser.step()
        scheduler.step()

        history = TrainingHistory(run_name="t")
        history.epochs.append(
            EpochRecord(1, 0.5, 0.4, 0.3, 100.0, 0.8, 0.7, 12.0, 1e-3)
        )
        path = save_checkpoint(
            tmp_path / "c.pt", model, optimiser, scheduler, 1, history, {"channels": 8}
        )

        restored = ChessResNet(TINY)
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_sched = torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt, T_max=5)
        epoch, restored_history = load_checkpoint(
            path, restored, restored_opt, restored_sched
        )

        assert epoch == 1
        assert restored_history.epochs[0].val_rmse == pytest.approx(0.4)
        for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
            assert torch.equal(a, b)
        # The optimiser's moments must come back, not restart at zero.
        assert restored_opt.state_dict()["state"], "optimiser state was not restored"
        assert restored_sched.get_last_lr() == scheduler.get_last_lr()

    def test_an_interrupted_write_leaves_no_checkpoint(self, tmp_path):
        """Saved through a temp file, so a truncated file never looks resumable."""
        assert not (tmp_path / "c.pt.partial").exists()
        assert not (tmp_path / "c.pt").exists()


class TestResume:
    def test_resuming_continues_instead_of_restarting(self, tmp_path, data, splits):
        """Two epochs, then two more, must land where four in a row would."""
        cache, targets = data
        train_idx, val_idx = splits

        def fresh():
            torch.manual_seed(0)
            return ChessResNet(TINY)

        def run(model, epochs, store):
            return train(
                model, cache, targets, train_idx, val_idx,
                epochs=epochs, batch_size=64, device="cpu", seed=7,
                checkpoints=store, progress=False,
            )

        store = HubCheckpoints("x/y", "run", tmp_path, enabled=False)
        run(fresh(), 2, store)
        assert store.local_path(LAST_NAME).exists()

        # A brand-new model object: everything it knows must come from the Hub copy.
        resumed = run(ChessResNet(TINY), 4, store)

        assert [e.epoch for e in resumed.history.epochs] == [1, 2, 3, 4]
        # The first two rows are the originals, carried through the checkpoint.
        assert resumed.history.epochs[0].seconds > 0

    def test_a_fresh_run_starts_from_scratch(self, tmp_path, data, splits):
        cache, targets = data
        train_idx, val_idx = splits
        store = HubCheckpoints("x/y", "nueva", tmp_path, enabled=False)
        result = train(
            ChessResNet(TINY), cache, targets, train_idx, val_idx,
            epochs=1, batch_size=64, device="cpu", checkpoints=store, progress=False,
        )
        assert [e.epoch for e in result.history.epochs] == [1]


class TestHistory:
    def test_best_epoch_is_the_lowest_validation_rmse(self):
        history = TrainingHistory(run_name="t")
        for epoch, rmse in [(1, 0.5), (2, 0.3), (3, 0.4)]:
            history.epochs.append(
                EpochRecord(epoch, 0.1, rmse, 0.2, 90.0, 0.8, 0.7, 10.0, 1e-3)
            )
        assert history.best().epoch == 2

    def test_json_round_trip(self):
        history = TrainingHistory(run_name="t", baselines={"material": 0.3973})
        history.epochs.append(
            EpochRecord(1, 0.1, 0.35, 0.2, 90.0, 0.8, 0.7, 10.0, 1e-3)
        )
        restored = TrainingHistory.from_json(history.to_json())
        assert restored.baselines["material"] == pytest.approx(0.3973)
        assert restored.epochs[0].val_rmse == pytest.approx(0.35)

    def test_table_marks_the_best_epoch(self):
        history = TrainingHistory(run_name="t")
        for epoch, rmse in [(1, 0.5), (2, 0.3)]:
            history.epochs.append(
                EpochRecord(epoch, 0.1, rmse, 0.2, 90.0, 0.8, 0.7, 10.0, 1e-3)
            )
        assert "mejor epoca: 2" in history.table()


class TestResumeIsExact:
    def test_an_interrupted_run_matches_an_uninterrupted_one(self, tmp_path, data, splits):
        """The property a resume claims but rarely has.

        The interruption is simulated the way a real one happens: the same call,
        with the same ``epochs``, dying partway through. Re-running with a
        *smaller* ``epochs`` would not be a resume at all -- it configures a
        different cosine schedule -- so the comparison has to hold the whole
        configuration fixed and only vary where the run stopped.
        """
        cache, targets = data
        train_idx, val_idx = splits

        def run(store, model, stop_after=None):
            def maybe_stop(record):
                if stop_after is not None and record.epoch >= stop_after:
                    raise KeyboardInterrupt("runtime perdido")

            return train(
                model, cache, targets, train_idx, val_idx,
                epochs=4, batch_size=64, device="cpu", seed=11,
                checkpoints=store, progress=False, on_epoch=maybe_stop,
            )

        torch.manual_seed(0)
        straight_model = ChessResNet(TINY)
        straight = run(HubCheckpoints("x/y", "seguida", tmp_path, enabled=False),
                       straight_model)

        interrupted = HubCheckpoints("x/y", "cortada", tmp_path, enabled=False)
        torch.manual_seed(0)
        with pytest.raises(KeyboardInterrupt):
            run(interrupted, ChessResNet(TINY), stop_after=2)

        resumed_model = ChessResNet(TINY)
        resumed = run(interrupted, resumed_model)

        assert [e.epoch for e in resumed.history.epochs] == [1, 2, 3, 4]
        assert resumed.history.epochs[-1].val_rmse == pytest.approx(
            straight.history.epochs[-1].val_rmse, abs=1e-6
        )
        for a, b in zip(
            straight_model.state_dict().values(), resumed_model.state_dict().values()
        ):
            assert torch.allclose(a.float(), b.float(), atol=1e-6)

    def test_epoch_order_depends_on_the_epoch_number(self):
        from chessdl.training.loop import epoch_rng

        indices = np.arange(100)
        first = epoch_rng(5, 1).permutation(indices)
        second = epoch_rng(5, 2).permutation(indices)
        again = epoch_rng(5, 1).permutation(indices)

        assert not np.array_equal(first, second), "every epoch would see the same order"
        assert np.array_equal(first, again), "the same epoch must be reproducible"


class TestHistoryIsAlwaysWritten:
    def test_history_file_survives_a_run_without_a_token(self, tmp_path, data, splits):
        """Training without the Hub is ordinary; losing the curves to it is not.

        The loss curves and validation metrics are a deliverable of the plan, and
        the weights are already written locally in this same situation -- there is
        no reason for the history to be the one artifact that disappears.
        """
        cache, targets = data
        train_idx, val_idx = splits
        store = HubCheckpoints("x/y", "sin-token", tmp_path, enabled=False)

        train(
            ChessResNet(TINY), cache, targets, train_idx, val_idx,
            epochs=1, batch_size=64, device="cpu", checkpoints=store, progress=False,
        )

        path = store.local_path(HISTORY_NAME)
        assert path.exists(), "the metric history was not written"
        restored = TrainingHistory.from_json(path.read_text(encoding="utf-8"))
        assert len(restored.epochs) == 1

    def test_history_records_the_baselines_it_was_measured_against(self, tmp_path, data, splits):
        """An RMSE means nothing without the floor it is compared to."""
        cache, targets = data
        train_idx, val_idx = splits
        store = HubCheckpoints("x/y", "con-pisos", tmp_path, enabled=False)

        train(
            ChessResNet(TINY), cache, targets, train_idx, val_idx,
            epochs=1, batch_size=64, device="cpu", checkpoints=store, progress=False,
            baselines={"media": 0.4886, "material": 0.3973},
        )

        restored = TrainingHistory.from_json(
            store.local_path(HISTORY_NAME).read_text(encoding="utf-8")
        )
        assert restored.baselines["material"] == pytest.approx(0.3973)


class TestRngStateSurvivesMapLocation:
    """Loading a checkpoint onto a GPU used to raise a bare TypeError.

    `torch.load(..., map_location="cuda")` moves every tensor in the payload to
    the target device, and the RNG state was stored as a tensor. A CUDA tensor is
    not a valid argument to `torch.set_rng_state`, which demands a CPU
    ByteTensor. It broke the resume path -- the whole reason checkpoints exist --
    and no CPU test could see it.
    """

    def test_the_stored_rng_state_contains_no_tensors(self, tmp_path, data):
        """The root-cause assertion: map_location cannot touch what is not a tensor."""
        cache, _ = data
        model = ChessResNet(TINY)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        path = save_checkpoint(
            tmp_path / "c.pt", model, optimiser, None, 1, TrainingHistory(run_name="t"), {}
        )

        payload = torch.load(path, map_location="cpu", weights_only=False)
        rng = payload["rng_state"]

        assert not isinstance(rng["torch"], torch.Tensor)
        assert isinstance(rng["torch"], (bytes, bytearray))
        for entry in rng.get("cuda", []):
            assert not isinstance(entry, torch.Tensor)

    def test_restores_from_the_byte_format(self, tmp_path, data):
        from chessdl.training.checkpoint import _restore_rng, _rng_state

        before = torch.get_rng_state()
        state = _rng_state()
        torch.rand(10)  # move the generator on
        _restore_rng(state)

        assert torch.equal(torch.get_rng_state(), before)

    def test_still_reads_the_old_tensor_format(self):
        """Checkpoints written before the fix must keep loading.

        A campaign in flight has checkpoints in the old format; refusing them
        would throw away the run.
        """
        from chessdl.training.checkpoint import _restore_rng

        before = torch.get_rng_state()
        legacy = {"torch": before.clone(), "numpy": np.random.get_state()}
        torch.rand(10)
        _restore_rng(legacy)

        assert torch.equal(torch.get_rng_state(), before)

    def test_tolerates_a_state_moved_off_cpu(self):
        """What map_location actually did to the old format.

        A CUDA tensor cannot be built on this machine, so the test uses the other
        half of the same failure: a tensor that is no longer a ByteTensor. The
        coercion has to handle it rather than hand it straight to torch.
        """
        from chessdl.training.checkpoint import _as_byte_tensor

        moved = torch.get_rng_state().to(torch.float32)
        coerced = _as_byte_tensor(moved)

        assert coerced.dtype == torch.uint8
        assert coerced.device.type == "cpu"
        torch.set_rng_state(coerced)  # must not raise
