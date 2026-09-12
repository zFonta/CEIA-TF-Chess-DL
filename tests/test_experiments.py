"""The hyperparameter sweep (WBS 4.5).

The sweep runs for hours on a runtime that gets recycled, so the property that
matters is that an interrupted sweep does not redo finished work -- and that each
experiment's overrides actually reach the training loop, which is the sort of
plumbing that fails silently and leaves you comparing five identical runs.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.models.resnet import ChessResNet, ResNetConfig  # noqa: E402
from chessdl.training.cache import fen_to_cached  # noqa: E402
from chessdl.training.experiments import (  # noqa: E402
    DEFAULT_SWEEP,
    DEFAULT_TRANSFORMER_SWEEP,
    Experiment,
    SweepResult,
    run_sweep,
)
from chessdl.training.loop import _build_scheduler  # noqa: E402

TINY = ResNetConfig(channels=8, blocks=1, value_channels=4, value_hidden=16)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    values = [0, 1, 3, 3, 5, 9, 0]
    tensors, targets = [], []
    for _ in range(240):
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


class TestExperimentOverrides:
    def test_only_the_changed_fields_are_passed_through(self):
        """Everything else must come from the campaign settings, not be reset."""
        assert Experiment("x", "y").overrides() == {}
        assert Experiment("x", "y", weight_decay=1e-3).overrides() == {"weight_decay": 1e-3}

    def test_warmup_is_only_passed_when_requested(self):
        assert "warmup_epochs" not in Experiment("x", "y").overrides()
        assert Experiment("x", "y", warmup_epochs=2).overrides()["warmup_epochs"] == 2

    def test_dropout_lands_on_the_model_and_not_the_loop(self):
        """Dropout is architecture, so it must not be passed to `train`."""
        experiment = Experiment("x", "y", dropout=0.3)
        assert "dropout" not in experiment.overrides()
        assert experiment.model_config(TINY).dropout == 0.3

    def test_an_arm_that_does_not_mention_dropout_leaves_it_alone(self):
        """Not the same as setting it to zero.

        The ResNet defaults to no dropout so the two used to coincide, but the
        transformer defaults to 0.1 inside its encoder. An arm about weight decay
        that silently turned that off would be measuring two changes at once.
        """
        from chessdl.models.transformer import TransformerConfig

        base = TransformerConfig(dropout=0.1)
        assert Experiment("x", "y", weight_decay=1e-3).model_config(base).dropout == 0.1
        assert Experiment("x", "y", dropout=0.0).model_config(base).dropout == 0.0

    def test_architecture_fields_other_than_dropout_can_be_swept(self):
        from chessdl.models.transformer import TransformerConfig

        experiment = Experiment("x", "y", architecture=(("pooling", "mean"),))
        assert experiment.model_config(TransformerConfig()).pooling == "mean"
        assert "pooling" not in experiment.overrides()

    def test_the_default_sweep_changes_something_in_every_arm(self):
        """A sweep arm that changes nothing is a wasted hour of GPU."""
        for experiment in DEFAULT_SWEEP:
            if experiment.name == "base":
                continue
            changes = experiment.overrides() or experiment.dropout
            assert changes, f"{experiment.name} no cambia nada"

    def test_every_arm_states_why_it_exists(self):
        for experiment in DEFAULT_SWEEP:
            assert experiment.rationale


class TestTransformerSweep:
    """The transformer's arms pull the opposite way to the ResNet's, on purpose.

    Its first campaign underfitted -- validation error squared over training loss
    was 1.19, against the ResNet's 3.75 -- so an arm that adds regularisation
    would be pushing the wrong way, however well it worked for the ResNet.
    """

    def test_no_arm_adds_regularisation(self):
        for experiment in DEFAULT_TRANSFORMER_SWEEP:
            assert experiment.dropout == 0.0, f"{experiment.name} agrega dropout"
            assert experiment.weight_decay is not None
            assert experiment.weight_decay <= 1e-4, (
                f"{experiment.name}: weight decay {experiment.weight_decay} contra un "
                "modelo que subajusta"
            )

    def test_every_arm_raises_the_learning_rate_of_the_first_campaign(self):
        """The first campaign ran at 3e-4, a third of the ResNet's."""
        for experiment in DEFAULT_TRANSFORMER_SWEEP:
            assert experiment.learning_rate > 3e-4

    def test_every_arm_clips_the_gradient(self):
        """What makes the higher rate safe on a network without batch norm."""
        for experiment in DEFAULT_TRANSFORMER_SWEEP:
            assert experiment.overrides()["grad_clip"] == 1.0

    def test_the_arms_differ_in_technique_and_not_only_in_learning_rate(self):
        """The point of three arms is three levers, not three values of one."""
        rates = {e.learning_rate for e in DEFAULT_TRANSFORMER_SWEEP}
        assert len(rates) == 1, "los brazos solo varian el learning rate"

        changes = set()
        for experiment in DEFAULT_TRANSFORMER_SWEEP:
            if experiment.batch_size is not None:
                changes.add("batch")
            if experiment.architecture:
                changes.add("arquitectura")
        assert changes == {"batch", "arquitectura"}

    def test_the_architecture_arm_reaches_the_model(self):
        from chessdl.models.transformer import ChessTransformer, TransformerConfig

        arm = next(e for e in DEFAULT_TRANSFORMER_SWEEP if e.architecture)
        config = arm.model_config(TransformerConfig(d_model=32, layers=1, heads=4))
        assert config.pooling == "mean"
        assert ChessTransformer(config).cls_token is None

    def test_grad_clip_is_not_mistaken_for_an_architecture_field(self):
        assert "grad_clip" not in DEFAULT_TRANSFORMER_SWEEP[0].architecture


class TestWarmupScheduler:
    def test_without_warmup_the_rate_starts_at_the_maximum(self):
        model = torch.nn.Linear(2, 1)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _build_scheduler(optimiser, 10, 0)
        assert optimiser.param_groups[0]["lr"] == pytest.approx(1e-3)

    def test_with_warmup_the_rate_ramps_up_then_anneals(self):
        model = torch.nn.Linear(2, 1)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = _build_scheduler(optimiser, 10, 2)

        rates = []
        for _ in range(10):
            rates.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()

        assert rates[0] < rates[1] < rates[2], "no sube durante el calentamiento"
        assert rates[2] == pytest.approx(1e-3), "no llega al maximo tras el calentamiento"
        assert rates[-1] < rates[2], "no anela despues"

    def test_the_schedule_still_reaches_near_zero_at_the_end(self):
        """Warm-up must not eat the annealing: the cosine gets the remaining epochs."""
        model = torch.nn.Linear(2, 1)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = _build_scheduler(optimiser, 10, 2)
        for _ in range(10):
            optimiser.step()
            scheduler.step()
        assert optimiser.param_groups[0]["lr"] < 1e-4


class TestSweep:
    def test_runs_every_experiment_and_ranks_them(self, tmp_path, data):
        cache, targets = data
        arms = (
            Experiment("a", "control"),
            Experiment("b", "mas decaimiento", weight_decay=1e-2),
        )
        result = run_sweep(
            arms, cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=2, batch_size=64, device="cpu", progress=False,
            baselines={"material": 0.3973}, reference=0.2584,
        )

        assert set(result.runs) == {"a", "b"}
        assert result.best_name() in {"a", "b"}
        table = result.table()
        assert "mejor:" in table
        assert "campana 1" in table

    def test_an_interrupted_sweep_does_not_redo_finished_experiments(self, tmp_path, data):
        """A lost runtime mid-sweep must not cost the hours already spent."""
        cache, targets = data
        arms = (Experiment("a", "control"),)
        kwargs = dict(
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=2, batch_size=64, device="cpu", progress=False,
        )

        first = run_sweep(arms, cache, targets, np.arange(0, 192), np.arange(192, 240), **kwargs)
        again = run_sweep(arms, cache, targets, np.arange(0, 192), np.arange(192, 240), **kwargs)

        # The second call resumes a finished run: same epochs, nothing appended.
        assert len(again.runs["a"].history.epochs) == len(first.runs["a"].history.epochs) == 2
        assert again.runs["a"].best.val_rmse == pytest.approx(first.runs["a"].best.val_rmse)

    def test_each_experiment_gets_its_own_checkpoint_prefix(self, tmp_path, data):
        """Otherwise the arms overwrite each other's weights on the Hub."""
        cache, targets = data
        arms = (Experiment("a", "uno"), Experiment("b", "dos"))
        run_sweep(
            arms, cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=1, batch_size=64, device="cpu", progress=False,
            prefix="s",
        )
        assert (tmp_path / "s-a").is_dir()
        assert (tmp_path / "s-b").is_dir()

    def test_dropout_reaches_the_trained_model(self, tmp_path, data):
        cache, targets = data
        result = run_sweep(
            (Experiment("d", "con dropout", dropout=0.25),),
            cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=1, batch_size=64, device="cpu", progress=False,
        )
        assert result.runs["d"].history.model["dropout"] == 0.25


class TestSweepsAnyArchitecture:
    """The transformer reuses the sweep rather than getting a second one.

    Two implementations would be two places for the arms to drift apart, and the
    point of the sweep is that the arms differ only in the hyperparameter.
    """

    def test_the_factory_decides_which_network_is_trained(self, tmp_path, data):
        from chessdl.models.transformer import ChessTransformer, TransformerConfig
        from chessdl.training.experiments import transformer_summary

        cache, targets = data
        config = TransformerConfig(d_model=32, layers=1, heads=4, feedforward=64)
        result = run_sweep(
            (Experiment("t", "transformer"),),
            cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=config, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=1, batch_size=64, device="cpu", progress=False,
            model_factory=ChessTransformer, model_summary=transformer_summary,
            amp=False,
        )
        assert result.runs["t"].best is not None
        assert result.runs["t"].history.model["d_model"] == 32

    def test_the_resnet_history_is_recorded_exactly_as_before(self, tmp_path, data):
        """Campaigns 1 and 2 are already published; their fields must not move."""
        cache, targets = data
        result = run_sweep(
            (Experiment("a", "control"),),
            cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=1, batch_size=64, device="cpu", progress=False,
        )
        assert set(result.runs["a"].history.model) == {"channels", "blocks", "dropout"}


class TestEmptySweepResult:
    def test_a_sweep_with_no_runs_has_no_winner(self):
        assert SweepResult().best_name() is None
        assert "experimento" in SweepResult().table()


class TestTableReadsHonestly:
    """A sweep table is read at a glance; a wrong sign there is a wrong decision."""

    def _result(self, rmse: float, piso: float, referencia: float) -> SweepResult:
        from chessdl.training.checkpoint import EpochRecord, TrainingHistory
        from chessdl.training.loop import TrainingRun

        record = EpochRecord(3, 0.1, rmse, 0.1, 90.0, 0.9, 0.9, 10.0, 1e-3)
        history = TrainingHistory(run_name="r")
        history.epochs.append(record)
        return SweepResult(
            runs={"a": TrainingRun(history=history, best=record, final_epoch=3)},
            experiments={"a": Experiment("a", "control")},
            baselines={"material": piso},
            reference=referencia,
        )

    def test_a_result_below_the_floor_is_not_called_an_improvement(self):
        table = self._result(rmse=0.45, piso=0.3973, referencia=0.2584).table()
        assert "EMPEORA" in table
        assert "mejora -" not in table

    def test_a_result_above_the_floor_is_called_an_improvement(self):
        table = self._result(rmse=0.25, piso=0.3973, referencia=0.2584).table()
        assert "mejora" in table
        assert "EMPEORA" not in table

    def test_a_regression_against_the_first_campaign_is_flagged(self):
        table = self._result(rmse=0.30, piso=0.3973, referencia=0.2584).table()
        # Better than the material floor, worse than campaign 1: both must show.
        assert "mejora 24.5%" in table
        assert "EMPEORA 16.10%" in table   # 2 decimales: contra la campana 1 las diferencias son chicas


class TestArmsAreComparable:
    """Every arm must start from the same weights, or the sweep measures noise.

    `train` seeds, but only after the caller has built the model, so each arm was
    initialised from whatever state the interpreter happened to be in. With the
    differences between arms expected to be a few percent, initialisation noise
    would be the same size as the effect.
    """

    def test_two_identical_arms_give_identical_results(self, tmp_path, data):
        cache, targets = data
        result = run_sweep(
            (Experiment("a", "x"), Experiment("b", "x")),
            cache, targets, np.arange(0, 192), np.arange(192, 240),
            base_model=TINY, repo_id="x/y", local_dir=str(tmp_path), token=None,
            push_to_hub=False, epochs=2, batch_size=64, device="cpu",
            progress=False, seed=7,
        )
        assert result.runs["a"].best.val_rmse == result.runs["b"].best.val_rmse

    def test_seeding_makes_initial_weights_reproducible(self):
        from chessdl.training.loop import seed_everything

        seed_everything(11)
        first = ChessResNet(TINY).stem[0].weight.detach().clone()
        seed_everything(11)
        second = ChessResNet(TINY).stem[0].weight.detach().clone()

        assert torch.equal(first, second)

    def test_different_seeds_give_different_initial_weights(self):
        from chessdl.training.loop import seed_everything

        seed_everything(1)
        first = ChessResNet(TINY).stem[0].weight.detach().clone()
        seed_everything(2)
        second = ChessResNet(TINY).stem[0].weight.detach().clone()

        assert not torch.equal(first, second)
