"""The training loop (WBS 4.4), built to survive a lost Colab runtime.

Batching is done by hand rather than through ``DataLoader``. With the tensor
cache the per-sample work is a memory copy, so worker processes would add
inter-process overhead to hide latency that is not there -- and on the two vCPUs
Colab assigns there is nothing to gain. Indexing the cache directly is simpler,
deterministic, and faster here.

One trick is worth explaining. The row order is shuffled globally each epoch and
then cut into batches, but the indices *within* a batch are sorted before
touching the cache. Batch membership stays random, which is all the optimiser
cares about; the sort just makes each batch read the 2.9 GB memory map in
increasing order instead of jumping around it, which matters while the file is
still being paged in.

After every epoch the model is scored on validation and the whole training state
is pushed to the Hub. That cadence is the contract: a disconnect costs at most
one epoch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .cache import cached_to_tensor
from .checkpoint import (
    BEST_NAME,
    LAST_NAME,
    EpochRecord,
    HubCheckpoints,
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
)
from .loss import build_loss, describe_loss
from .metrics import EvaluationMetrics, evaluate


@dataclass(frozen=True)
class TrainingRun:
    """What a campaign produced."""

    history: TrainingHistory
    best: EpochRecord | None
    final_epoch: int

    def summary(self) -> str:
        lines = [self.history.table()]
        if self.best is not None:
            floor = self.history.baselines.get("material")
            if floor is not None:
                delta = (floor - self.best.val_rmse) / floor
                lines.append("")
                lines.append(
                    f"Contra el piso de material ({floor:.4f}): "
                    f"{'mejora' if delta > 0 else 'EMPEORA'} {abs(delta):.1%}"
                )
        return "\n".join(lines)


def seed_everything(seed: int) -> None:
    """Seed the generators that decide a run's initial weights.

    Must be called **before** the model is constructed. ``train`` also seeds, but
    by then the caller has already built the network, so the initial weights come
    from whatever state the interpreter happened to be in -- which is not
    reproducible across sessions, and across the arms of a sweep means every arm
    starts somewhere different. With differences between arms expected to be a
    few percent, that noise would be the same size as the effect being measured.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_rng(seed: int, epoch: int) -> np.random.Generator:
    """The generator that shuffles one epoch, derived from ``(seed, epoch)``.

    Deriving it per epoch rather than carrying one generator across the run is
    what makes resuming exact. A single generator advances with every epoch, so
    after reloading a checkpoint it would restart from the seed and epoch 3 would
    replay the order epoch 1 used -- training continues and nothing fails, but a
    resumed run no longer matches the uninterrupted one it claims to continue.
    """
    return np.random.default_rng([seed, epoch])


def _build_scheduler(optimizer, epochs: int, warmup_epochs: int):
    """Cosine annealing, optionally preceded by a linear warm-up.

    The first campaign spiked on validation at epochs 2, 4 and 10 while the
    learning rate was still near its maximum -- epoch 2 came back worse than
    predicting the mean. Ramping the rate up over the first epochs is the usual
    remedy, and each of those spikes cost roughly four minutes of GPU to undo.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs)
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def _batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    """Yield index batches, sorted inside each batch for cache locality.

    ``rng`` of None means evaluation order: no shuffle.
    """
    order = indices if rng is None else rng.permutation(indices)
    for start in range(0, len(order), batch_size):
        yield np.sort(order[start : start + batch_size])


def _to_device(
    cache: np.ndarray, rows: np.ndarray, targets: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    planes = cached_to_tensor(np.asarray(cache[rows]))
    x = torch.from_numpy(planes).to(device, non_blocking=True)
    y = torch.from_numpy(targets[rows]).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    cache: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Run the model over a split and return its predictions, in order."""
    model.eval()
    out = np.empty(len(indices), dtype=np.float32)

    position = 0
    for rows in _batches(indices, batch_size):
        x, _ = _to_device(cache, rows, targets, device)
        out[position : position + len(rows)] = model(x).float().cpu().numpy()
        position += len(rows)
    return out


def evaluate_split(
    model: torch.nn.Module,
    cache: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    device: str,
    batch_size: int = 2048,
) -> EvaluationMetrics:
    """Score the model on one split."""
    # `indices` is sorted batch by batch, so the predictions come back aligned
    # with the sorted order -- the labels are gathered the same way.
    ordered = np.sort(indices)
    predictions = predict(model, cache, targets, ordered, device, batch_size)
    return evaluate(predictions, targets[ordered])


def train(
    model: torch.nn.Module,
    cache: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_name: str = "mse",
    warmup_epochs: int = 0,
    device: str | None = None,
    seed: int = 20260911,
    checkpoints: HubCheckpoints | None = None,
    model_config: dict[str, Any] | None = None,
    baselines: dict[str, float] | None = None,
    amp: bool | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
    resume: bool = True,
    progress: bool = True,
) -> TrainingRun:
    """Train the value network, resuming from the Hub if a run is already there."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available() if amp is None else amp

    torch.manual_seed(seed)

    model = model.to(device)
    criterion = build_loss(loss_name)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = TrainingHistory(
        run_name=checkpoints.run_name if checkpoints else "local",
        model=model_config or {},
        hyperparameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "loss": describe_loss(loss_name),
            "optimizer": "AdamW",
            "scheduler": ("LinearWarmup+CosineAnnealingLR" if warmup_epochs
                          else "CosineAnnealingLR"),
            "warmup_epochs": warmup_epochs,
            "amp": amp,
            "seed": seed,
            "train_positions": int(len(train_indices)),
            "val_positions": int(len(val_indices)),
        },
        baselines=baselines or {},
    )

    start_epoch = 0
    if resume and checkpoints is not None:
        previous = checkpoints.fetch(LAST_NAME)
        if previous is not None:
            start_epoch, history = load_checkpoint(
                previous, model, optimizer, scheduler, map_location=device,
                scaler=scaler,
            )
            print(f"Reanudando desde la epoca {start_epoch}.")

    best_rmse = min((e.val_rmse for e in history.epochs), default=float("inf"))

    for epoch in range(start_epoch + 1, epochs + 1):
        started = time.time()
        model.train()
        running, seen = 0.0, 0

        # Derived from (seed, epoch), so the order of an epoch is the same
        # whether the run reached it straight through or after a resume.
        steps = _batches(train_indices, batch_size, epoch_rng(seed, epoch))
        if progress:
            try:
                from tqdm.auto import tqdm

                total = int(np.ceil(len(train_indices) / batch_size))
                steps = tqdm(steps, total=total, desc=f"epoca {epoch}", unit="lote")
            except ImportError:  # pragma: no cover - tqdm is a dependency
                pass

        for rows in steps:
            x, y = _to_device(cache, rows, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * len(rows)
            seen += len(rows)

        scheduler.step()
        metrics = evaluate_split(model, cache, targets, val_indices, device)

        record = EpochRecord(
            epoch=epoch,
            train_loss=running / max(seen, 1),
            val_rmse=metrics.rmse,
            val_mae=metrics.mae,
            val_mae_cp=metrics.mae_cp,
            val_sign_agreement=metrics.sign_agreement,
            val_spearman=metrics.spearman,
            seconds=time.time() - started,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
        )
        history.epochs.append(record)
        print(f"epoca {epoch:>3}  perdida {record.train_loss:.5f}  {metrics.row()}")

        if checkpoints is not None:
            path = checkpoints.local_path(LAST_NAME)
            save_checkpoint(
                path, model, optimizer, scheduler, epoch, history,
                model_config or {}, scaler=scaler,
            )
            checkpoints.push(path, LAST_NAME, f"epoca {epoch}")
            if metrics.rmse < best_rmse:
                best_path = checkpoints.local_path(BEST_NAME)
                save_checkpoint(
                    best_path, model, optimizer, scheduler, epoch, history,
                    model_config or {}, scaler=scaler,
                )
                checkpoints.push(
                    best_path, BEST_NAME, f"mejor: epoca {epoch}, RMSE {metrics.rmse:.4f}"
                )
            checkpoints.push_history(history)

        best_rmse = min(best_rmse, metrics.rmse)
        if on_epoch is not None:
            on_epoch(record)

    return TrainingRun(
        history=history,
        best=history.best(),
        final_epoch=history.epochs[-1].epoch if history.epochs else start_epoch,
    )
