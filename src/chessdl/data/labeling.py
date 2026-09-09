"""Position labelling with Stockfish (requirements 1.1 and 2.3).

Every sampled position is analysed at a fixed, configured search depth and the
resulting score is normalized into ``[-1, 1]``. The exact Stockfish version is
read from the engine's own UCI handshake and stored on every row, so the dataset
never depends on remembering which binary produced it.

Labelling is the pipeline's bottleneck and is pure CPU work, so it runs in a
process pool with one long-lived engine per worker. Starting an engine per
position would cost far more than the search itself.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import chess
import chess.engine

from chessdl.config import LabelingConfig, NormalizationConfig
from chessdl.normalize import Label, label_from_povscore


class EngineNotAvailableError(RuntimeError):
    """Raised when the Stockfish binary cannot be started."""


@dataclass(frozen=True)
class LabeledPosition:
    """A FEN together with the label Stockfish produced for it."""

    fen: str
    label: Label


def _engine_options(cfg: LabelingConfig) -> dict[str, int]:
    return {"Threads": cfg.threads_per_engine, "Hash": cfg.hash_mb}


def open_engine(cfg: LabelingConfig) -> chess.engine.SimpleEngine:
    """Start a Stockfish process configured from ``cfg``."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(cfg.engine_path)
    except (OSError, chess.engine.EngineError) as exc:
        raise EngineNotAvailableError(
            f"Could not start the engine at {cfg.engine_path!r}. "
            "Run scripts/setup_stockfish.sh or set labeling.engine_path."
        ) from exc
    engine.configure(_engine_options(cfg))
    return engine


def engine_version(cfg: LabelingConfig) -> str:
    """The engine's self-reported name and version, for the provenance columns."""
    engine = open_engine(cfg)
    try:
        return engine.id.get("name", "unknown")
    finally:
        engine.quit()


def analyse_fen(
    engine: chess.engine.SimpleEngine,
    fen: str,
    labeling_cfg: LabelingConfig,
    norm_cfg: NormalizationConfig,
) -> Label:
    """Analyse one position and normalize the score."""
    board = chess.Board(fen)
    info = engine.analyse(board, chess.engine.Limit(depth=labeling_cfg.depth))
    return label_from_povscore(
        info["score"], board.turn, norm_cfg.scale, norm_cfg.cp_clip
    )


# --- worker-process plumbing ----------------------------------------------
#
# One engine per worker, created on first use and shut down at process exit.

_WORKER_ENGINE: chess.engine.SimpleEngine | None = None
_WORKER_LABELING: LabelingConfig | None = None
_WORKER_NORM: NormalizationConfig | None = None


def _shutdown_worker_engine() -> None:
    global _WORKER_ENGINE
    if _WORKER_ENGINE is not None:
        try:
            _WORKER_ENGINE.quit()
        except chess.engine.EngineError:  # pragma: no cover - best-effort cleanup
            pass
        _WORKER_ENGINE = None


def _init_worker(labeling_cfg: LabelingConfig, norm_cfg: NormalizationConfig) -> None:
    global _WORKER_ENGINE, _WORKER_LABELING, _WORKER_NORM
    _WORKER_LABELING = labeling_cfg
    _WORKER_NORM = norm_cfg
    _WORKER_ENGINE = open_engine(labeling_cfg)
    atexit.register(_shutdown_worker_engine)


def _label_one(fen: str) -> tuple[str, Label] | None:
    """Label one position, restarting the engine once if it dies.

    A dead engine is worth one retry: a multi-hour run should not be lost to a
    single crashed subprocess. A position that fails twice is dropped, and the
    caller reports how many were lost.
    """
    global _WORKER_ENGINE
    assert _WORKER_LABELING is not None and _WORKER_NORM is not None

    for attempt in (1, 2):
        try:
            if _WORKER_ENGINE is None:
                _WORKER_ENGINE = open_engine(_WORKER_LABELING)
            label = analyse_fen(_WORKER_ENGINE, fen, _WORKER_LABELING, _WORKER_NORM)
            return fen, label
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            _shutdown_worker_engine()
            if attempt == 2:
                return None
        except ValueError:
            # An unparseable FEN is a bug upstream, not a transient failure.
            return None
    return None  # pragma: no cover - unreachable


def label_fens(
    fens: Sequence[str],
    labeling_cfg: LabelingConfig,
    norm_cfg: NormalizationConfig,
    progress: bool = True,
) -> list[LabeledPosition]:
    """Label a batch of positions, in parallel when more than one worker is set.

    Results come back in input order. Positions the engine could not evaluate
    are omitted, so the result may be shorter than the input.
    """
    if not fens:
        return []

    workers = min(labeling_cfg.resolved_workers(), len(fens))
    if workers <= 1:
        results = _label_serially(fens, labeling_cfg, norm_cfg)
    else:
        results = _label_in_parallel(fens, labeling_cfg, norm_cfg, workers)

    return [
        LabeledPosition(fen=fen, label=label)
        for fen, label in _with_progress(results, len(fens), progress)
        if label is not None
    ]


def _label_serially(
    fens: Sequence[str],
    labeling_cfg: LabelingConfig,
    norm_cfg: NormalizationConfig,
) -> Iterator[tuple[str, Label | None]]:
    engine = open_engine(labeling_cfg)
    try:
        for fen in fens:
            try:
                yield fen, analyse_fen(engine, fen, labeling_cfg, norm_cfg)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
                yield fen, None
    finally:
        engine.quit()


def _label_in_parallel(
    fens: Sequence[str],
    labeling_cfg: LabelingConfig,
    norm_cfg: NormalizationConfig,
    workers: int,
) -> Iterator[tuple[str, Label | None]]:
    # A modest chunk size keeps the workers busy without letting one slow chunk
    # stall the batch.
    chunksize = max(1, min(32, len(fens) // (workers * 4) or 1))
    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(labeling_cfg, norm_cfg),
    ) as pool:
        for result in pool.imap(_label_one, fens, chunksize=chunksize):
            yield result if result is not None else ("", None)


def _with_progress(
    results: Iterable[tuple[str, Label | None]],
    total: int,
    enabled: bool,
) -> Iterator[tuple[str, Label | None]]:
    if not enabled:
        yield from results
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - tqdm is a declared dependency
        yield from results
        return
    yield from tqdm(results, total=total, desc="labelling", unit="pos")
