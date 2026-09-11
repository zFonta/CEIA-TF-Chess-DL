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
from typing import Any

import numpy as np

from ..models.resnet import ChessResNet, ResNetConfig
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
    dropout: float = 0.0

    def model_config(self, base: ResNetConfig) -> ResNetConfig:
        return replace(base, dropout=self.dropout)

    def overrides(self) -> dict[str, Any]:
        """Only the fields this experiment actually changes."""
        out: dict[str, Any] = {}
        for name in ("epochs", "batch_size", "learning_rate", "weight_decay", "loss_name"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        if self.warmup_epochs:
            out["warmup_epochs"] = self.warmup_epochs
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


@dataclass
class SweepResult:
    """Every experiment's outcome, ready to compare."""

    runs: dict[str, TrainingRun] = field(default_factory=dict)
    experiments: dict[str, Experiment] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    reference: float | None = None

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


def run_sweep(
    experiments: tuple[Experiment, ...],
    cache: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    base_model: ResNetConfig,
    repo_id: str,
    local_dir: str,
    token: str | None,
    push_to_hub: bool = True,
    epochs: int = 12,
    prefix: str = "sweep",
    baselines: dict[str, float] | None = None,
    reference: float | None = None,
    **campaign,
) -> SweepResult:
    """Run each experiment in turn and collect the results.

    ``epochs`` is the screening budget, deliberately shorter than a full
    campaign. Twelve is enough because the first campaign had already flattened
    by epoch 14, and because the cosine schedule anneals over whatever budget it
    is given -- each run is a complete short campaign, not a truncated long one.

    One caveat worth keeping in mind when reading the results: screening
    regularisation at a reduced budget is not neutral. Less training favours less
    regularisation, so a weight-decay or dropout setting that wins here may be
    conservative for the full-length campaign. The winner is confirmed at full
    length in task 4.6, not adopted from this table alone.
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
            ChessResNet(model_config),
            cache,
            targets,
            train_indices,
            val_indices,
            checkpoints=checkpoints,
            model_config={
                "channels": model_config.channels,
                "blocks": model_config.blocks,
                "dropout": model_config.dropout,
            },
            baselines=baselines,
            **settings,
        )
        result.runs[experiment.name] = run
        result.experiments[experiment.name] = experiment

    return result
