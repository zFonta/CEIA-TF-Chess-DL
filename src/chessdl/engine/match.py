"""Measuring how well the engine actually plays (WBS block 6).

RMSE says how closely the network reproduces Stockfish's numbers. It does not
say how well the engine plays, and those are not the same question: two models a
fraction of a percent apart on RMSE can be far apart over a board, or identical.
This module answers the second question, three ways.

**Games against Stockfish**, with its strength capped via ``UCI_Elo``, to put a
number on the engine -- fitted against the whole ladder at once by
:func:`pooled_elo`, never read off a single rung. And against Stockfish *limited
to one ply*, which is the
comparison that isolates what this project built: same search, so the only
difference left is the evaluation function -- one learned from two million
positions, one written by hand over twenty years.

**Average centipawn loss**, which does most of the work. A hundred games give a
score rate with an uncertainty of several percent, enough to hide the difference
between two similar engines. Asking Stockfish, over a few thousand positions,
how much each chosen move throws away against the best one gives a continuous
measurement where every position counts, and a far tighter interval for the same
wall-clock cost.

Two details keep the games from producing a number that means nothing:

* **Openings come from the test split.** A deterministic engine plays the same
  game every time, so a hundred games from the start position would be a hundred
  copies of one game. Real positions the model never saw are better than random
  legal moves, which mostly produce garbage neither side understands.
* **Every opening is played twice, with colours swapped.** An opening set that
  happens to favour White would otherwise land entirely in one engine's column.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import chess
import chess.engine
import numpy as np

from ..normalize import score_to_cp
from .evaluator import Evaluator
from .search import GameOverError, search

#: Plies before a game is called a draw. Two weak engines can shuffle forever,
#: and an unfinished game still has to become a row in the table.
MAX_PLIES = 200

#: How a game ended, in the words the tables use. The keys are the
#: ``chess.Termination`` names in lower case, plus the two ways a game here can
#: stop without the rules ending it.
#:
#: Worth recording because a draw is not one thing. "Held Stockfish to a draw"
#: and "the two engines repeated moves until the rules stopped them" both score
#: half a point, and only one of them says anything about the evaluation.
ENDINGS = {
    "checkmate": "mate",
    "stalemate": "ahogado",
    "insufficient_material": "material insuficiente",
    "seventyfive_moves": "75 jugadas",
    "fivefold_repetition": "repeticion quintuple",
    "fifty_moves": "50 jugadas",
    "threefold_repetition": "repeticion triple",
    "max_plies": "limite de jugadas",
    "interrupted": "interrumpida",
}


@dataclass(frozen=True)
class GameResult:
    """One game, from the network engine's point of view."""

    score: float            # 1 win, 0.5 draw, 0 loss
    plies: int
    outcome: str
    engine_white: bool
    seconds_per_move: float
    termination: str = ""   # a key of ENDINGS; empty for results built by hand

    @property
    def won(self) -> bool:
        return self.score == 1.0

    @property
    def drawn(self) -> bool:
        return self.score == 0.5


@dataclass
class MatchResult:
    """A set of games, and what they say."""

    games: list[GameResult] = field(default_factory=list)
    label: str = ""

    @property
    def score_rate(self) -> float:
        """Points per game, in [0, 1]: the quantity Elo is derived from."""
        if not self.games:
            return float("nan")
        return sum(g.score for g in self.games) / len(self.games)

    @property
    def record(self) -> tuple[int, int, int]:
        wins = sum(1 for g in self.games if g.won)
        draws = sum(1 for g in self.games if g.drawn)
        return wins, draws, len(self.games) - wins - draws

    def elo_difference(self) -> float:
        """Elo gap implied by the score rate, positive when the engine is ahead.

        Undefined at a clean sweep in either direction -- a rate of 0 or 1 says
        only "further than these games can measure" -- so those come back as an
        infinity rather than as a number that looks precise.
        """
        rate = self.score_rate
        if not 0.0 < rate < 1.0:
            return math.copysign(math.inf, rate - 0.5)
        return -400.0 * math.log10(1.0 / rate - 1.0)

    def margin(self) -> float:
        """Rough +- on the score rate: one standard error of the mean.

        Worth printing next to the rate, because the number of games needed for
        a confident answer is usually larger than the number anyone runs.
        """
        if len(self.games) < 2:
            return float("nan")
        scores = np.array([g.score for g in self.games])
        return float(scores.std(ddof=1) / math.sqrt(len(scores)))

    def summary(self) -> str:
        wins, draws, losses = self.record
        lines = [f"{self.label or 'torneo'}: {len(self.games)} partidas"]
        lines.append(f"  {wins}W {draws}T {losses}D")
        lines.append(f"  puntos por partida  {self.score_rate:.3f} +- {self.margin():.3f}")
        elo = self.elo_difference()
        lines.append(
            f"  diferencia de Elo   {'sin acotar (barrida)' if math.isinf(elo) else f'{elo:+.0f}'}"
        )
        tablas = Counter(g.termination for g in self.games if g.drawn)
        if tablas:
            lines.append("  tablas por          " + ", ".join(
                f"{n} {ENDINGS.get(motivo, motivo or '?')}"
                for motivo, n in tablas.most_common()))
        if self.games:
            plies = np.array([g.plies for g in self.games])
            lines.append(f"  largo medio         {plies.mean():.0f} plies")
        return "\n".join(lines)


@dataclass(frozen=True)
class PooledElo:
    """One rating fitted to the results against several rated opponents."""

    rating: float   # +inf / -inf when every opponent was beaten / lost to outright
    sigma: float    # half-width of the one-sigma interval; nan when unbounded
    games: int

    @property
    def bounded(self) -> bool:
        return math.isfinite(self.rating)

    def describe(self) -> str:
        if self.rating == math.inf:
            return "por encima de todos los rivales (les gano todo)"
        if self.rating == -math.inf:
            return "por debajo de todos los rivales (perdio todo)"
        return f"{self.rating:.0f} +- {self.sigma:.0f}"


def pooled_elo(
    results: Mapping[float, MatchResult] | Iterable[tuple[float, MatchResult]],
    margin: float = 2000.0,
    step: float = 1.0,
) -> PooledElo:
    """Fit one Elo rating to the scores against opponents of known rating.

    ``results`` maps each opponent's rating -- the ``UCI_Elo`` of a ladder rung
    -- to the match played against it. The rating returned is the one under
    which those scores are most likely, with the usual logistic expectation and
    draws counted as half a point.

    The alternative, reading the rating off a single rung, is what a ladder
    invites and it goes wrong in two ways. It throws away the games against the
    other rungs. And it is worst exactly where it is most tempting: against a
    rung the engine dominates, the score rate sits near 1, where a few games
    move the implied rating by hundreds of points -- and a clean sweep gives no
    number at all. Fitted against every rung at once, a sweep at one of them is
    still pinned down by the others; only sweeping all of them in the same
    direction leaves the rating unbounded, and that comes back as an infinity
    rather than as a number that looks precise.

    ``sigma`` is half the width of the interval where the log-likelihood stays
    within 0.5 of its peak: one standard error, under the same model.
    """
    pares = results.items() if isinstance(results, Mapping) else results
    filas = [(float(rival), r) for rival, r in pares if r.games]
    if not filas:
        raise ValueError("ningun rival tiene partidas")

    rivales = np.array([rival for rival, _ in filas])
    partidas = np.array([len(r.games) for _, r in filas], dtype=np.float64)
    puntos = np.array([r.score_rate for _, r in filas])
    total = int(partidas.sum())

    if np.all(puntos >= 1.0):
        return PooledElo(math.inf, float("nan"), total)
    if np.all(puntos <= 0.0):
        return PooledElo(-math.inf, float("nan"), total)

    grilla = np.arange(rivales.min() - margin, rivales.max() + margin + step, step)
    esperado = 1.0 / (1.0 + 10.0 ** ((rivales[None, :] - grilla[:, None]) / 400.0))
    esperado = np.clip(esperado, 1e-12, 1.0 - 1e-12)
    verosimilitud = (partidas * (puntos * np.log(esperado)
                                 + (1.0 - puntos) * np.log(1.0 - esperado))).sum(axis=1)

    mejor = int(np.argmax(verosimilitud))
    if mejor in (0, len(grilla) - 1):
        # Further out than the grid: as good as unbounded in that direction.
        return PooledElo(math.copysign(math.inf, mejor - 0.5), float("nan"), total)
    dentro = grilla[verosimilitud >= verosimilitud[mejor] - 0.5]
    return PooledElo(float(grilla[mejor]), float((dentro[-1] - dentro[0]) / 2), total)


def opening_positions(
    fens: Sequence[str], plies: Sequence[int], count: int, max_ply: int = 16, seed: int = 0
) -> list[str]:
    """Pick opening positions from a split, so games do not all coincide.

    ``max_ply`` keeps them recognisably openings: far enough in that the sides
    have committed to something, early enough that the game is still open.
    """
    early = [fen for fen, ply in zip(fens, plies) if ply <= max_ply]
    if not early:
        raise ValueError(f"ninguna posicion con ply <= {max_ply}")
    rng = np.random.default_rng(seed)
    elegidas = rng.choice(len(early), size=min(count, len(early)), replace=False)
    return [early[i] for i in elegidas]


def play_game(
    evaluator: Evaluator,
    opponent: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    start_fen: str = chess.STARTING_FEN,
    engine_white: bool = True,
    depth: int = 1,
    beam: int | None = None,
    rng: np.random.Generator | None = None,
    max_plies: int = MAX_PLIES,
) -> GameResult:
    """Play one game, the network engine against a configured Stockfish."""
    board = chess.Board(start_fen)
    nuestro_turno = chess.WHITE if engine_white else chess.BLACK
    tiempos: list[float] = []

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        if board.turn == nuestro_turno:
            try:
                resultado = search(board, evaluator, depth=depth, beam=beam, rng=rng)
            except GameOverError:
                break
            tiempos.append(resultado.seconds)
            board.push(resultado.move)
        else:
            jugada = opponent.play(board, limit).move
            if jugada is None:
                break
            board.push(jugada)

    outcome = board.result(claim_draw=True)
    if outcome == "1/2-1/2" or outcome == "*":
        score = 0.5
    else:
        gano_blancas = outcome == "1-0"
        score = 1.0 if gano_blancas == engine_white else 0.0

    final = board.outcome(claim_draw=True)
    if final is not None:
        termination = final.termination.name.lower()
    elif len(board.move_stack) >= max_plies:
        termination = "max_plies"           # adjudicated as a draw, see MAX_PLIES
    else:
        termination = "interrupted"

    return GameResult(
        score=score,
        plies=len(board.move_stack),
        outcome=outcome,
        engine_white=engine_white,
        seconds_per_move=float(np.mean(tiempos)) if tiempos else float("nan"),
        termination=termination,
    )


def play_match(
    evaluator: Evaluator,
    opponent: chess.engine.SimpleEngine,
    openings: Iterable[str],
    limit: chess.engine.Limit,
    depth: int = 1,
    beam: int | None = None,
    label: str = "",
    seed: int = 0,
    progress: bool = False,
    max_plies: int = MAX_PLIES,
) -> MatchResult:
    """Play every opening twice, once with each colour.

    The colour swap is not a detail: an opening set that leans towards White
    would otherwise put the whole lean in one engine's column, and the table
    would report as playing strength what is actually a property of the
    positions chosen.
    """
    resultado = MatchResult(label=label)
    rng = np.random.default_rng(seed)

    aperturas = list(openings)
    partidas = [(fen, blancas) for fen in aperturas for blancas in (True, False)]
    if progress:
        try:
            from tqdm.auto import tqdm

            partidas = tqdm(partidas, desc=label or "partidas", unit="partida")
        except ImportError:  # pragma: no cover - tqdm is a dependency
            pass

    for fen, blancas in partidas:
        resultado.games.append(
            play_game(
                evaluator, opponent, limit, fen, blancas, depth, beam, rng, max_plies
            )
        )
    return resultado


def estimated_seconds(
    openings: int, seconds_per_move: float, plies: int = 80
) -> float:
    """Rough wall-clock for a match, to size one before launching it.

    Games cost the branching factor squared per extra ply of search, so a
    tournament that is minutes at depth one is hours at depth three. Worth
    knowing before, not after.
    """
    return openings * 2 * (plies / 2) * seconds_per_move


@dataclass
class MoveQuality:
    """How much each chosen move gives away, judged by Stockfish."""

    losses_cp: np.ndarray
    agreements: np.ndarray
    label: str = ""

    @property
    def acpl(self) -> float:
        """Average centipawn loss: the standard measure of how much is thrown away."""
        return float(np.mean(self.losses_cp))

    @property
    def agreement(self) -> float:
        """Share of positions where the engine picked Stockfish's move."""
        return float(np.mean(self.agreements))

    def blunder_rate(self, threshold: float = 300.0) -> float:
        return float(np.mean(self.losses_cp > threshold))

    def summary(self) -> str:
        error = self.losses_cp.std(ddof=1) / math.sqrt(len(self.losses_cp))
        return "\n".join([
            f"{self.label or 'calidad de jugada'}: {len(self.losses_cp):,} posiciones",
            f"  perdida media       {self.acpl:.1f} cp +- {error:.1f}",
            f"  mediana             {np.median(self.losses_cp):.1f} cp",
            f"  acuerdo con SF      {self.agreement:.1%}",
            f"  errores > 300 cp    {self.blunder_rate():.1%}",
        ])


def move_quality(
    evaluator: Evaluator,
    reference: chess.engine.SimpleEngine,
    fens: Sequence[str],
    limit: chess.engine.Limit,
    depth: int = 1,
    beam: int | None = None,
    label: str = "",
    progress: bool = False,
) -> MoveQuality:
    """Score each chosen move against the best one, in centipawns.

    The reference engine evaluates the position twice: once as it stands, which
    is the value of playing its own best move, and once after the move under
    test. The difference is what that choice costs.

    Both readings are taken **from the same side's point of view** -- the side
    that had to move -- which is where this goes wrong if it goes wrong. After
    the move it is the opponent's turn, so the second reading arrives with the
    opposite sign and has to be flipped before the subtraction. Skip that and
    every good move looks like a catastrophe and every blunder like a triumph.
    """
    perdidas: list[float] = []
    aciertos: list[float] = []

    posiciones = list(fens)
    if progress:
        try:
            from tqdm.auto import tqdm

            posiciones = tqdm(posiciones, desc=label or "posiciones", unit="pos")
        except ImportError:  # pragma: no cover
            pass

    for fen in posiciones:
        board = chess.Board(fen)
        if board.is_game_over():
            continue

        mejor = reference.analyse(board, limit)
        mejor_cp, _ = score_to_cp(mejor["score"].pov(board.turn))

        elegida = search(board, evaluator, depth=depth, beam=beam).move
        aciertos.append(float(elegida == mejor["pv"][0]) if mejor.get("pv") else 0.0)

        board.push(elegida)
        if board.is_game_over():
            # A decided position has no analysis to ask for; a mate delivered
            # loses nothing, and a move into a draw loses what the position was
            # worth.
            tras = 0.0 if not board.is_checkmate() else float(mejor_cp)
        else:
            despues = reference.analyse(board, limit)
            # The reading comes back from the opponent's side; flip it.
            despues_cp, _ = score_to_cp(despues["score"].pov(board.turn))
            tras = -despues_cp

        perdidas.append(max(0.0, float(mejor_cp) - tras))

    return MoveQuality(
        losses_cp=np.array(perdidas, dtype=np.float64),
        agreements=np.array(aciertos, dtype=np.float64),
        label=label,
    )


def stockfish_at_elo(engine: chess.engine.SimpleEngine, elo: int) -> None:
    """Cap Stockfish's strength, so the result lands on a scale people read."""
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})


def stockfish_full_strength(engine: chess.engine.SimpleEngine) -> None:
    engine.configure({"UCI_LimitStrength": False})


def timed(fn, *args, **kwargs):
    """Run something and report how long it took, for the notebook's tables."""
    started = time.perf_counter()
    salida = fn(*args, **kwargs)
    return salida, time.perf_counter() - started
