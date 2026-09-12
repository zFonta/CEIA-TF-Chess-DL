"""Hyperparameter experiments (WBS 4.5), driven by the first campaign's diagnosis.

This is a short list of deliberate comparisons, not a grid search. The first
campaign already narrowed the space, and every configuration costs real GPU time
on a fixed Colab budget, so guessing broadly would be expensive and uninformative.

What the diagnosis established, and what it rules out:

* **The network overfits from epoch 14 on.** Training loss fell 42 % between
  epochs 14 and 30 while validation RMSE did not move at all -- an hour of GPU
  for 0.58 %. So more capacity and more epochs are not the lever; regularisation
  is.
* **The learning rate is too high at the start.** Validation spiked at epochs 2,
  4 and 10 while the rate was near its maximum; epoch 2 came back worse than
  predicting the mean.
* **The labels are not the limit.** Stockfish at depth 12 and depth 20 disagree
  by RMSE 0.057, against the model's 0.261 -- label noise accounts for under 5 %
  of the error, so there is real headroom to chase.

Every experiment gets its own run name on the Hub, so a sweep interrupted by a
lost runtime resumes exactly where it stopped: finished experiments are detected
and skipped, and a half-finished one continues from its last epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from ..models.resnet import ChessResNet, ResNetConfig
from ..models.transformer import TransformerConfig
from .baselines import rmse_map
from .checkpoint import HubCheckpoints
from .loop import TrainingRun, seed_everything, train


@dataclass(frozen=True)
class Experiment:
    """One configuration to try, as a delta from the campaign's settings."""

    name: str
    rationale: str
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    loss_name: str | None = None
    warmup_epochs: int = 0
    grad_clip: float | None = None
    #: ``None`` means "leave the architecture's own default alone", which is not
    #: the same as 0.0. The ResNet defaults to no dropout, so the two coincided
    #: there; the transformer defaults to 0.1 inside its encoder layers, and a
    #: sweep arm that was not about dropout would otherwise turn it off without
    #: saying so -- a confound that looks like a result.
    dropout: float | None = None
    #: Architecture fields other than dropout, for sweeps over the shape of the
    #: network rather than the optimiser -- the transformer's pooling, say, or
    #: its depth. Applied with ``dataclasses.replace``, so the keys have to be
    #: fields of the base configuration.
    architecture: tuple[tuple[str, Any], ...] = ()

    def model_config(self, base: Any) -> Any:
        """The architecture this experiment trains, as a delta from ``base``.

        Typed loosely on purpose: it works for any frozen configuration
        dataclass with a ``dropout`` field, which is what both
        :class:`~chessdl.models.resnet.ResNetConfig` and
        :class:`~chessdl.models.transformer.TransformerConfig` are.
        """
        changes: dict[str, Any] = dict(self.architecture)
        if self.dropout is not None:
            changes["dropout"] = self.dropout
        return replace(base, **changes)

    def overrides(self) -> dict[str, Any]:
        """Only the fields this experiment actually changes."""
        out: dict[str, Any] = {}
        for name in ("epochs", "batch_size", "learning_rate", "weight_decay", "loss_name"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.warmup_epochs:
            out["warmup_epochs"] = self.warmup_epochs
        if self.grad_clip:
            out["grad_clip"] = self.grad_clip
        return out


#: The sweep the first campaign's diagnosis points at.
#:
#: `base` is the control: the same settings as the first campaign but over a
#: shorter cosine schedule. It is not a wasted run -- annealing to zero in 12
#: epochs is a different, better-finished trajectory than stopping a 30-epoch
#: schedule at 12, and without it the other four have nothing to be compared to
#: at the same budget.
DEFAULT_SWEEP: tuple[Experiment, ...] = (
    Experiment(
        name="base",
        rationale="control: la configuracion de la campana 1, con coseno corto",
    ),
    Experiment(
        name="wd-alto",
        rationale="weight decay x10, contra el sobreajuste desde la epoca 14",
        weight_decay=1e-3,
    ),
    Experiment(
        name="dropout",
        rationale="dropout en la cabeza, donde se concentran los parametros densos",
        dropout=0.3,
    ),
    Experiment(
        name="warmup",
        rationale="calentamiento lineal, contra los picos de las epocas 2, 4 y 10",
        warmup_epochs=2,
    ),
    Experiment(
        name="huber",
        rationale="perdida robusta: el 1,43 % de mates saturados domina el gradiente con MSE",
        loss_name="huber",
    ),
)


#: The transformer's sweep (WBS 4.8), and it pulls the opposite way to the ResNet's.
#:
#: The first transformer campaign landed at 0.2978 on test against the ResNet's
#: 0.2511, and the curves say why. At epoch 30 its validation error squared over
#: its training loss was **1.19**; the ResNet's was **3.75**. A ratio near one
#: means validation and training are the same number: nothing is being memorised,
#: so nothing is being over-learned. The transformer is *underfitting*.
#:
#: The clincher is blunter still. The transformer's final training loss (0.0759)
#: is worse than the ResNet's validation error at its best epoch (0.0622): it
#: cannot fit the training set as well as the ResNet generalises.
#:
#: So every lever that won the ResNet's sweep -- weight decay, dropout -- is
#: exactly wrong here, and the settings the first campaign used were the cause:
#: a learning rate three times lower than the ResNet's and a weight decay a
#: hundred times higher, both chosen as insurance against an instability that the
#: loss curve shows never existed. Thirty epochs, monotone, not one spike.
#:
#: Three arms, three different techniques rather than three learning rates. Each
#: builds on the one before, so the sweep is staged rather than orthogonal: that
#: is deliberate, because the diagnosis points every lever the same way and
#: there is no budget to cross them.
DEFAULT_TRANSFORMER_SWEEP: tuple[Experiment, ...] = (
    # The control, and also the fairest comparison the project can make: the
    # ResNet's exact training recipe. If the transformer trains well under it,
    # the two architectures differ only in architecture -- same parameter
    # budget, same optimiser, same schedule, same epochs, same data.
    #
    # Gradient clipping is the one addition, and it is what makes the tenfold
    # jump in learning rate a reasonable thing to try rather than a gamble: the
    # transformer has no batch norm to bound its activations.
    Experiment(
        name="receta-resnet",
        rationale="la receta de la ResNet: lr 1e-3, wd 1e-4, sin dropout, con recorte",
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        warmup_epochs=2,
        grad_clip=1.0,
    ),
    # Twice the optimisation steps for the same pass over the data: 4,488 updates
    # per epoch instead of 2,244. An underfitting model is one that has not moved
    # far enough, and steps are how it moves.
    #
    # Worth stating plainly: batch size and learning rate are coupled, and
    # halving the batch while holding the rate raises the effective step size and
    # the gradient noise together. So this is not a clean second axis -- it is a
    # different way of spending the same budget, and if it wins, the honest
    # reading is "more, noisier steps helped", not "512 is the right batch".
    Experiment(
        name="lotes-chicos",
        rationale="lote 512: el doble de pasos de optimizacion por epoca",
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        warmup_epochs=2,
        grad_clip=1.0,
        batch_size=512,
    ),
    # The only arm that changes the network rather than how it is trained, and
    # the only question the architecture left genuinely open.
    #
    # With a CLS token the value head reads one vector that has to gather the
    # whole board through attention; the 64 squares only reach the loss through
    # it. Averaging instead gives every square a direct path to the output and to
    # the gradient. That matters more when a model is underfitting than when it
    # is overfitting, which is exactly the situation. It costs 192 parameters.
    Experiment(
        name="pooling-medio",
        rationale="promedio de las 64 casillas en vez del token CLS",
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.0,
        warmup_epochs=2,
        grad_clip=1.0,
        architecture=(("pooling", "mean"),),
    ),
)


@dataclass
class SweepResult:
    """Every experiment's outcome, ready to compare."""

    runs: dict[str, TrainingRun] = field(default_factory=dict)
    experiments: dict[str, Experiment] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    reference: float | None = None

    def __post_init__(self) -> None:
        self.baselines = rmse_map(self.baselines)

    def best_name(self) -> str | None:
        scored = {
            name: run.best.val_rmse
            for name, run in self.runs.items()
            if run.best is not None
        }
        return min(scored, key=scored.get) if scored else None

    def table(self) -> str:
        lines = [
            f"{'experimento':<12}{'val RMSE':>10}{'MAE cp':>9}{'signo':>8}"
            f"{'rho':>8}{'epoca':>7}   que cambia"
        ]
        lines.append("-" * 92)

        ganador = self.best_name()
        ordenados = sorted(
            self.runs.items(),
            key=lambda kv: kv[1].best.val_rmse if kv[1].best else float("inf"),
        )
        for name, run in ordenados:
            if run.best is None:
                lines.append(f"{name:<12}{'sin datos':>10}")
                continue
            marca = " *" if name == ganador else "  "
            lines.append(
                f"{name:<12}{run.best.val_rmse:>10.4f}{run.best.val_mae_cp:>9.1f}"
                f"{run.best.val_sign_agreement:>8.1%}{run.best.val_spearman:>8.4f}"
                f"{run.best.epoch:>7}{marca} {self.experiments[name].rationale}"
            )

        if ganador is not None:
            mejor = self.runs[ganador].best.val_rmse
            lines.append("")
            lines.append(f"* mejor: {ganador} (val RMSE {mejor:.4f})")
            if self.reference is not None:
                delta = (self.reference - mejor) / self.reference
                verbo = "mejora" if delta > 0 else "EMPEORA"
                lines.append(
                    f"  contra la campana 1 ({self.reference:.4f}): {verbo} {abs(delta):.2%}"
                )
            piso = self.baselines.get("material")
            if piso is not None:
                # Signed, like TrainingRun.summary: printing "mejora -4.7 %" for a
                # result that is worse than the floor reads as a win at a glance,
                # which is exactly the reading a sweep table must not invite.
                delta = (piso - mejor) / piso
                verbo = "mejora" if delta > 0 else "EMPEORA"
                lines.append(
                    f"  contra el piso de material ({piso:.4f}): {verbo} {abs(delta):.1%}"
                )
        return "\n".join(lines)


def resnet_summary(config: ResNetConfig) -> dict[str, Any]:
    """What gets recorded in ``history.json`` for a ResNet run.

    Kept as an explicit short list rather than ``dataclasses.asdict`` so the
    histories already published for campaigns 1 and 2 stay comparable with the
    ones this sweep writes.
    """
    return {
        "channels": config.channels,
        "blocks": config.blocks,
        "dropout": config.dropout,
    }


def transformer_summary(config: TransformerConfig) -> dict[str, Any]:
    """What gets recorded in ``history.json`` for a transformer run."""
    return {
        "d_model": config.d_model,
        "layers": config.layers,
        "heads": config.heads,
        "feedforward": config.feedforward,
        "pooling": config.pooling,
        "dropout": config.dropout,
        "head_dropout": config.head_dropout,
    }


def run_sweep(
    experiments: tuple[Experiment, ...],
    cache: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    base_model: Any,
    repo_id: str,
    local_dir: str,
    token: str | None,
    push_to_hub: bool = True,
    epochs: int = 12,
    prefix: str = "sweep",
    baselines: dict[str, float] | None = None,
    reference: float | None = None,
    model_factory: Callable[[Any], Any] = ChessResNet,
    model_summary: Callable[[Any], dict[str, Any]] = resnet_summary,
    **campaign,
) -> SweepResult:
    """Run each experiment in turn and collect the results.

    ``epochs`` is the screening budget, deliberately shorter than a full
    campaign. Twelve is enough because the first campaign had already flattened
    by epoch 14, and because the cosine schedule anneals over whatever budget it
    is given -- each run is a complete short campaign, not a truncated long one.

    One caveat worth keeping in mind when reading the results: **a reduced
    budget is never a neutral referee**, and which way it leans depends on what
    is being screened.

    * Screening *regularisation*, as the ResNet's sweep did: less training
      favours less regularisation, so a weight-decay or dropout setting that
      wins here may be conservative at full length.
    * Screening *optimisation speed*, as the transformer's does: a short budget
      favours whatever moves fastest early. An arm with a smaller batch takes
      more steps per epoch, and an architecture with a shorter gradient path
      gets going sooner -- neither advantage need survive to the end.

    Both biases point at the arm being tested rather than at the control, which
    is precisely the direction that flatters a false positive. So the winner is
    confirmed at full length rather than adopted from this table alone -- and if
    the winner is not the control, confirming it measures how good the best
    configuration is, not which change earned it.

    ``model_factory`` and ``model_summary`` default to the ResNet, so the calls
    from task 4.5 are unchanged. Pass ``ChessTransformer`` and
    ``transformer_summary`` to sweep the second architecture: the loop, the
    resume logic and the comparison table are the same, and writing them twice
    would be two places for the arms to stop being comparable.
    """
    result = SweepResult(baselines=baselines or {}, reference=reference)

    for experiment in experiments:
        run_name = f"{prefix}-{experiment.name}"
        print(f"\n{'='*72}\n{run_name}: {experiment.rationale}\n{'='*72}")

        model_config = experiment.model_config(base_model)
        checkpoints = HubCheckpoints(
            repo_id=repo_id,
            run_name=run_name,
            local_dir=local_dir,
            token=token,
            enabled=push_to_hub and token is not None,
        )

        settings = {"epochs": epochs, **campaign, **experiment.overrides()}

        # Every arm starts from the same initial weights, so the difference
        # between arms is the hyperparameter and not where each one happened to
        # land at initialisation.
        seed_everything(int(settings.get("seed", 20260911)))

        run = train(
            model_factory(model_config),
            cache,
            targets,
            train_indices,
            val_indices,
            checkpoints=checkpoints,
            model_config=model_summary(model_config),
            baselines=baselines,
            **settings,
        )
        result.runs[experiment.name] = run
        result.experiments[experiment.name] = experiment

    return result
