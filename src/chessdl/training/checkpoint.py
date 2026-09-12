"""Checkpoints on the Hugging Face Hub, so a lost runtime costs one epoch.

Colab recycles runtimes, and a training campaign runs for hours. The data
pipeline solved the same problem by publishing each shard as it closed; this is
the same idea for weights: after every epoch the full training state goes to a
*model* repository on the Hub, and resuming reads it back.

"Full state" means more than the weights. Restarting from weights alone silently
restarts the optimiser too, throwing away Adam's moment estimates and whatever
the schedule had reached -- the run continues, the loss jumps, and nothing says
why. So the optimiser, the scheduler, the epoch counter, the RNG states and the
metric history all travel together.

Two checkpoints are kept: ``last``, to resume from, and ``best``, by validation
RMSE, because the final epoch is not usually the best one. The history is also
written on its own, so the loss curves the plan asks for as a deliverable survive
even if the weights are replaced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import hf
from .baselines import rmse_map

LAST_NAME = "checkpoint_last.pt"
BEST_NAME = "checkpoint_best.pt"
HISTORY_NAME = "history.json"


@dataclass
class EpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_rmse: float
    val_mae: float
    val_mae_cp: float
    val_sign_agreement: float
    val_spearman: float
    seconds: float
    learning_rate: float


@dataclass
class TrainingHistory:
    """Every epoch of a run, plus what produced it."""

    run_name: str
    model: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, float] = field(default_factory=dict)
    epochs: list[EpochRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Every path that builds a history goes through here -- `train`, the
        # sweep, and `from_json` reading a history back off the Hub -- so this is
        # the one place where the baselines can be pinned to floats. It also
        # repairs histories written before this existed, whose baselines were
        # serialised as nested dicts.
        self.baselines = rmse_map(self.baselines)

    def best(self) -> EpochRecord | None:
        return min(self.epochs, key=lambda e: e.val_rmse, default=None)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "TrainingHistory":
        raw = json.loads(text)
        epochs = [EpochRecord(**row) for row in raw.pop("epochs", [])]
        return cls(epochs=epochs, **raw)

    def table(self) -> str:
        """The history as a table, for the notebook and the report."""
        lines = [
            f"{'epoca':>6}{'train':>10}{'val RMSE':>11}{'val MAE':>10}"
            f"{'MAE cp':>9}{'signo':>9}{'rho':>8}{'seg':>8}"
        ]
        lines.append("-" * 71)
        best = self.best()
        for row in self.epochs:
            marca = " *" if best is not None and row.epoch == best.epoch else ""
            lines.append(
                f"{row.epoch:>6}{row.train_loss:>10.5f}{row.val_rmse:>11.4f}"
                f"{row.val_mae:>10.4f}{row.val_mae_cp:>9.1f}"
                f"{row.val_sign_agreement:>8.1%}{row.val_spearman:>8.4f}"
                f"{row.seconds:>8.0f}{marca}"
            )
        if best is not None:
            lines.append("")
            lines.append(f"* mejor epoca: {best.epoch} (val RMSE {best.val_rmse:.4f})")
        return "\n".join(lines)


def _rng_state() -> dict[str, Any]:
    """Capture the random state as plain bytes, not as tensors.

    ``torch.load(..., map_location="cuda")`` moves *every* tensor in the payload
    to the target device, and the RNG state is a tensor. A CUDA tensor is not a
    valid argument to ``torch.set_rng_state``, which demands a CPU ByteTensor, so
    loading a checkpoint on a GPU used to fail with a bare TypeError -- and it
    failed on the resume path, which is the entire point of checkpointing.

    Bytes are not tensors, so ``map_location`` leaves them alone. The RNG state
    is not model data and has no business being moved to a device.
    """
    state: dict[str, Any] = {
        "torch": torch.get_rng_state().numpy().tobytes(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [s.numpy().tobytes() for s in torch.cuda.get_rng_state_all()]
    return state


def _as_byte_tensor(value: Any) -> torch.Tensor:
    """Coerce a stored RNG state back into the CPU ByteTensor torch expects.

    Handles both formats: raw bytes (written by :func:`_rng_state`) and the
    tensors that earlier checkpoints stored, which may arrive on the wrong device
    after ``map_location``.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.uint8)
    return torch.frombuffer(bytearray(value), dtype=torch.uint8)


def _restore_rng(state: dict[str, Any] | None) -> None:
    if not state:
        return
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch: int,
    history: TrainingHistory,
    model_config: dict[str, Any],
    scaler: Any | None = None,
) -> Path:
    """Write the complete training state to disk, atomically.

    The mixed-precision scaler is part of that state. Its loss scale adapts
    during training, so dropping it means the first steps after a resume run at
    a different scale than the ones before it -- harmless in the end, but enough
    to make a resumed run stop matching the one it continues.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_config": model_config,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "history": json.loads(history.to_json()),
    }

    # Through a temporary file: an interruption mid-write must not leave a
    # truncated checkpoint that looks resumable and then fails to load.
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
    scaler: Any | None = None,
) -> tuple[int, TrainingHistory]:
    """Restore a checkpoint, returning the epoch it left off at and the history."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    _restore_rng(payload.get("rng_state"))

    history = TrainingHistory.from_json(json.dumps(payload["history"]))
    return int(payload["epoch"]), history


class HubCheckpoints:
    """Reads and writes this run's checkpoints in a Hub model repository.

    Every run gets its own prefix, so training a second architecture -- the
    transformer of task 4.8 -- does not overwrite the ResNet's weights.
    """

    def __init__(
        self,
        repo_id: str,
        run_name: str,
        local_dir: str | Path,
        token: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.run_name = run_name
        self.local_dir = Path(local_dir)
        self.token = token
        self.enabled = enabled
        self._repo_ready = False

    def _remote(self, name: str) -> str:
        return f"{self.run_name}/{name}"

    def _ensure_repo(self) -> None:
        if self._repo_ready or not self.enabled:
            return
        hf.ensure_repo(self.repo_id, token=self.token, repo_type=hf.MODEL)
        self._repo_ready = True

    def local_path(self, name: str) -> Path:
        return self.local_dir / self.run_name / name

    def push(self, local: str | Path, name: str, message: str | None = None) -> None:
        if not self.enabled:
            return
        self._ensure_repo()
        hf.upload_file(
            local,
            self.repo_id,
            self._remote(name),
            token=self.token,
            commit_message=message or f"{self.run_name}: update {name}",
            repo_type=hf.MODEL,
        )

    def push_history(self, history: TrainingHistory) -> Path:
        """Write the metric history, and publish it when the Hub is enabled.

        Separate from the weights because it is a deliverable in its own right --
        the plan asks for the loss curves and validation metrics -- and because
        it stays readable on the Hub without downloading a checkpoint.

        The local file is written **even with the Hub disabled**. Running without
        a token is an ordinary case (a local experiment, a session where the
        secret is not granted), and losing the curves of a completed campaign to
        that would be a poor trade: the weights are already saved locally in the
        same situation.
        """
        path = self.local_path(HISTORY_NAME)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(history.to_json(), encoding="utf-8")

        epoch = history.epochs[-1].epoch if history.epochs else 0
        self.push(path, HISTORY_NAME, f"{self.run_name}: metrics through epoch {epoch}")
        return path

    def fetch(self, name: str) -> Path | None:
        """Download one checkpoint, or return None when the run is new."""
        if not self.enabled:
            local = self.local_path(name)
            return local if local.exists() else None
        return hf.download_file(
            self.repo_id,
            self._remote(name),
            self.local_path(name),
            token=self.token,
            repo_type=hf.MODEL,
        )
