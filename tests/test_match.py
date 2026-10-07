"""Playing and measuring against Stockfish (WBS block 6).

Two parts, with different risks.

The **arithmetic** -- score rates, implied Elo, averaging centipawn losses -- is
cheap to get subtly wrong and produces a number that looks fine either way, so it
is tested directly against cases whose answer is known by hand.

The **measurement of move quality** has one specific way of being wrong, and it
is silent: the two readings that get subtracted come from opposite sides of the
board. Stockfish reports relative to whoever is to move, and after the move under
test that is the opponent, so the second reading has to be flipped before the
subtraction. Miss it and every good move reads as a catastrophe and every blunder
as a triumph -- with no error, no warning, and a table full of plausible numbers.
That one is tested against the real engine.
"""

from __future__ import annotations

import math
import shutil

import chess
import chess.engine
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.encoding import board_to_tensor  # noqa: E402
from chessdl.engine.evaluator import Evaluator  # noqa: E402
from chessdl.engine.match import (  # noqa: E402
    ENDINGS,
    GameResult,
    MatchResult,
    move_quality,
    opening_positions,
    play_game,
    play_match,
    pooled_elo,
    stockfish_at_elo,
)
from test_engine import MaterialNet  # noqa: E402

STOCKFISH = shutil.which("stockfish") or "./bin/stockfish"
requiere_stockfish = pytest.mark.skipif(
    not shutil.which(STOCKFISH), reason="requires a Stockfish binary"
)


@pytest.fixture
def evaluador() -> Evaluator:
    return Evaluator(MaterialNet(), device="cpu")


class PreferirJugada(Evaluator):
    """An evaluator rigged so a one-ply search picks one chosen move.

    Scores the position that move leads to at -1 and every other at 0. The
    search negates, so -1 becomes the best available and the move is chosen --
    which lets a test fix what the engine plays and check what the measurement
    says about it.
    """

    def __init__(self, board: chess.Board, move: chess.Move) -> None:
        super().__init__(MaterialNet(), device="cpu")
        despues = board.copy()
        despues.push(move)
        self.objetivo = board_to_tensor(despues)

    def evaluate_tensors(self, planes):
        return np.array(
            [-1.0 if np.array_equal(p, self.objetivo) else 0.0 for p in planes],
            dtype=np.float32,
        )


@pytest.fixture
def stockfish():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    try:
        yield engine
    finally:
        engine.quit()


def partida(score: float, plies: int = 40, blancas: bool = True) -> GameResult:
    return GameResult(score, plies, "?", blancas, 0.01)


class TestScoreArithmetic:
    def test_an_even_match_implies_no_elo_difference(self):
        resultado = MatchResult([partida(1.0), partida(0.0)])
        assert resultado.score_rate == pytest.approx(0.5)
        assert resultado.elo_difference() == pytest.approx(0.0)

    def test_the_textbook_conversion_holds(self):
        """A 0.75 score rate is the classic +191 Elo."""
        resultado = MatchResult([partida(1.0)] * 3 + [partida(0.0)])
        assert resultado.elo_difference() == pytest.approx(190.8, abs=1.0)

    def test_losing_gives_a_negative_difference(self):
        assert MatchResult([partida(0.0)] * 3 + [partida(1.0)]).elo_difference() < 0

    def test_a_clean_sweep_is_reported_as_unbounded_and_not_as_a_number(self):
        """A rate of 1 says "further than these games can measure", not "+inf Elo"."""
        assert math.isinf(MatchResult([partida(1.0)] * 8).elo_difference())
        assert math.isinf(MatchResult([partida(0.0)] * 8).elo_difference())

    def test_the_record_adds_up(self):
        resultado = MatchResult([partida(1.0), partida(0.5), partida(0.0), partida(1.0)])
        assert resultado.record == (2, 1, 1)

    def test_the_margin_shrinks_as_games_accumulate(self):
        """The reason the tables print it: ten games do not settle much."""
        pocas = MatchResult([partida(1.0), partida(0.0)] * 5)
        muchas = MatchResult([partida(1.0), partida(0.0)] * 50)
        assert muchas.margin() < pocas.margin()

    def test_the_summary_reports_what_it_measured(self):
        texto = MatchResult([partida(1.0), partida(0.0)], label="prueba").summary()
        assert "prueba" in texto and "Elo" in texto

    def test_the_summary_says_how_the_draws_came_about(self):
        """Half a point by repetition and half a point held are not the same result."""
        tablas = [GameResult(0.5, 60, "1/2-1/2", True, 0.01, "threefold_repetition")] * 3
        tablas += [GameResult(0.5, 200, "*", False, 0.01, "max_plies")]
        texto = MatchResult(tablas + [partida(0.0)]).summary()
        assert "3 repeticion triple" in texto
        assert "1 limite de jugadas" in texto

    def test_a_match_without_draws_does_not_mention_them(self):
        assert "tablas por" not in MatchResult([partida(1.0), partida(0.0)]).summary()

    def test_the_percentages_add_up_to_the_record(self):
        """Requirement 2.4 asks for percentages, not only counts."""
        resultado = MatchResult([partida(1.0), partida(0.5), partida(0.0), partida(1.0)])
        assert resultado.percentages == pytest.approx((0.5, 0.25, 0.25))
        assert "victorias 50%, tablas 25%, derrotas 25%" in resultado.summary()

    def test_a_game_cut_off_is_counted_as_unfinished(self):
        """2.4 wants no interrupted games, so the table has to be able to say so."""
        cortada = GameResult(0.5, 400, "*", True, 0.01, "max_plies")
        reglas = GameResult(0.5, 80, "1/2-1/2", True, 0.01, "threefold_repetition")
        resultado = MatchResult([cortada, reglas, partida(1.0)])
        assert resultado.unfinished == 1
        assert "terminadas por las reglas  2 de 3" in resultado.summary()

    def test_hand_built_results_count_as_finished(self):
        assert MatchResult([partida(1.0), partida(0.0)]).unfinished == 0


def match_simulado(elo_motor: float, rival: float, n: int, rng) -> MatchResult:
    """Games drawn from the logistic model itself, so the right answer is known."""
    esperado = 1.0 / (1.0 + 10 ** ((rival - elo_motor) / 400.0))
    return MatchResult([partida(1.0 if rng.random() < esperado else 0.0) for _ in range(n)])


class TestPooledElo:
    """One rating from a whole ladder, instead of reading it off one rung."""

    ESCALONES = (1320, 1500, 1700)

    def test_it_recovers_the_rating_the_games_were_drawn_from(self):
        rng = np.random.default_rng(0)
        for real in (1400, 1550, 1650):
            ajuste = pooled_elo({e: match_simulado(real, e, 300, rng) for e in self.ESCALONES})
            assert ajuste.rating == pytest.approx(real, abs=40)
            assert ajuste.games == 900

    def test_one_rung_gives_what_that_rung_alone_implies(self):
        """With nothing to pool, it has to agree with `elo_difference`."""
        match = MatchResult([partida(1.0)] * 3 + [partida(0.0)])
        ajuste = pooled_elo({1500: match})
        assert ajuste.rating == pytest.approx(1500 + match.elo_difference(), abs=1.0)

    def test_an_even_score_puts_the_engine_on_the_rung(self):
        ajuste = pooled_elo({1500: MatchResult([partida(1.0), partida(0.0)] * 10)})
        assert ajuste.rating == pytest.approx(1500, abs=1.0)

    def test_a_sweep_at_one_rung_is_pinned_down_by_the_others(self):
        """The case that sank the single-rung reading: no number at all."""
        barrida = MatchResult([partida(1.0)] * 30)
        assert math.isinf(barrida.elo_difference())
        ajuste = pooled_elo({
            1320: barrida,
            1500: MatchResult([partida(1.0)] * 18 + [partida(0.0)] * 12),
            1700: MatchResult([partida(1.0)] * 9 + [partida(0.0)] * 21),
        })
        assert ajuste.bounded
        assert 1500 < ajuste.rating < 1700

    def test_sweeping_every_rung_is_unbounded_and_says_so(self):
        todo = {e: MatchResult([partida(1.0)] * 10) for e in self.ESCALONES}
        nada = {e: MatchResult([partida(0.0)] * 10) for e in self.ESCALONES}
        assert pooled_elo(todo).rating == math.inf
        assert pooled_elo(nada).rating == -math.inf
        assert "encima" in pooled_elo(todo).describe()
        assert "debajo" in pooled_elo(nada).describe()

    def test_the_interval_narrows_with_more_games(self):
        def con(n):
            return pooled_elo({e: MatchResult([partida(1.0), partida(0.0)] * n)
                               for e in self.ESCALONES})
        assert con(60).sigma == pytest.approx(con(15).sigma / 2, rel=0.1)

    def test_it_takes_pairs_as_well_as_a_mapping(self):
        match = MatchResult([partida(1.0)] * 3 + [partida(0.0)])
        assert pooled_elo([(1500, match)]).rating == pooled_elo({1500: match}).rating

    def test_rungs_without_games_are_ignored_and_none_at_all_is_an_error(self):
        match = MatchResult([partida(1.0), partida(0.0)])
        assert pooled_elo({1500: match, 1700: MatchResult()}).games == 2
        with pytest.raises(ValueError):
            pooled_elo({1500: MatchResult()})


class TestHowGamesEnd:
    """Each game records why it stopped, without needing an engine to find out."""

    def test_a_dead_position_is_insufficient_material(self, evaluador):
        resultado = play_game(evaluador, None, chess.engine.Limit(time=0.01),
                              start_fen="8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        assert resultado.termination == "insufficient_material"
        assert resultado.score == 0.5

    def test_a_game_cut_by_the_ply_limit_says_so(self, evaluador):
        resultado = play_game(evaluador, None, chess.engine.Limit(time=0.01), max_plies=0)
        assert resultado.termination == "max_plies"
        assert resultado.score == 0.5

    def test_a_mated_engine_lost_by_checkmate(self, evaluador):
        tonto = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        resultado = play_game(evaluador, None, chess.engine.Limit(time=0.01),
                              start_fen=tonto, engine_white=True)
        assert resultado.termination == "checkmate"
        assert resultado.score == 0.0

    def test_an_illegal_move_stops_the_game_instead_of_corrupting_it(self, evaluador):
        """`Board.push` takes anything; the match must not."""

        class Tramposo:
            def play(self, board, limit):
                return chess.engine.PlayResult(chess.Move.from_uci("e7e4"), None)

        with pytest.raises(chess.IllegalMoveError):
            play_game(evaluador, Tramposo(), chess.engine.Limit(time=0.01),
                      engine_white=True, max_plies=4)

    def test_every_ending_has_a_label(self):
        for motivo in chess.Termination:
            if motivo is not chess.Termination.VARIANT_WIN and \
                    motivo is not chess.Termination.VARIANT_LOSS and \
                    motivo is not chess.Termination.VARIANT_DRAW:
                assert motivo.name.lower() in ENDINGS


class TestOpenings:
    def test_only_early_positions_are_offered(self):
        fens = [chess.STARTING_FEN] * 5
        elegidas = opening_positions(fens, [2, 40, 8, 60, 12], count=5, max_ply=16)
        assert len(elegidas) == 3

    def test_it_says_so_when_there_is_nothing_early_enough(self):
        with pytest.raises(ValueError, match="ply"):
            opening_positions([chess.STARTING_FEN], [90], count=1, max_ply=16)

    def test_the_choice_is_reproducible(self):
        fens = [f"8/8/8/8/8/8/8/K{c}k4 w - - 0 1".replace("K", "K") for c in "QRBN"]
        plies = [2, 4, 6, 8]
        assert (opening_positions(fens, plies, 2, seed=7)
                == opening_positions(fens, plies, 2, seed=7))


@requiere_stockfish
class TestPlayingRealGames:
    def test_a_game_finishes_and_scores_from_our_point_of_view(self, evaluador, stockfish):
        stockfish_at_elo(stockfish, 1320)
        resultado = play_game(
            evaluador, stockfish, chess.engine.Limit(depth=1),
            engine_white=True, max_plies=30,
        )
        assert resultado.score in (0.0, 0.5, 1.0)
        assert 0 < resultado.plies <= 30
        assert resultado.seconds_per_move > 0

    def test_playing_black_is_scored_the_other_way_round(self, evaluador, stockfish):
        """The colour flip in the scoring is exactly the kind of thing to get backwards."""
        stockfish_at_elo(stockfish, 1320)
        resultado = play_game(
            evaluador, stockfish, chess.engine.Limit(depth=1),
            engine_white=False, max_plies=20,
        )
        assert not resultado.engine_white
        if resultado.outcome == "1-0":
            assert resultado.score == 0.0
        elif resultado.outcome == "0-1":
            assert resultado.score == 1.0

    def test_a_match_plays_each_opening_with_both_colours(self, evaluador, stockfish):
        stockfish_at_elo(stockfish, 1320)
        aperturas = [chess.STARTING_FEN,
                     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"]
        resultado = play_match(
            evaluador, stockfish, aperturas, chess.engine.Limit(depth=1), label="prueba",
        )
        assert len(resultado.games) == 4
        assert sum(g.engine_white for g in resultado.games) == 2


@requiere_stockfish
class TestMoveQuality:
    def test_playing_stockfishs_own_move_loses_nothing(self, stockfish):
        """The sign test, through the real measurement and the real engine.

        The evaluator is rigged so the search picks exactly the move Stockfish
        would, which makes the expected loss zero by definition. With the flip
        missing the same call returns roughly twice the evaluation instead -- a
        large number, in the wrong direction, that no type or shape objects to.
        """
        # The position is chosen so the best move is not in doubt: White takes a
        # free queen, and there is nothing else to consider. That matters more
        # than it looks -- **Stockfish keeps its hash between calls**, so asking
        # twice for "the best move" in a busy session can give two answers, and
        # a version of this test that picked the move from a separate analysis
        # failed for that reason rather than for a real one.
        #
        # Lopsided on purpose too: in a balanced position a flipped sign adds up
        # to twice a small evaluation, close enough to the ordinary wobble
        # between two searches to be arguable. Here it would be thousands.
        fen = "q3k3/8/8/8/8/8/8/R3K3 w - - 0 1"
        limite = chess.engine.Limit(depth=10)
        captura = chess.Move.from_uci("a1a8")

        calidad = move_quality(
            PreferirJugada(chess.Board(fen), captura), stockfish, [fen], limite,
        )

        assert calidad.agreement == 1.0, "Txa8 tendria que ser tambien la de Stockfish"
        # The bound is loose, and deliberately so. Two searches of the same
        # position rarely agree to the centipawn -- here the evaluation tracks
        # how far off a mate looks, which moves with the hash -- and readings a
        # hundred or so apart are ordinary. What this test discriminates is a
        # sign inversion, which in a position this lopsided lands in the
        # thousands. Tightening the bound would only buy flakiness.
        assert calidad.acpl < 400, (
            f"llevarse la dama gratis perdio {calidad.acpl:.0f} cp; "
            "con el signo invertido esto daria miles"
        )

    def test_a_deliberately_bad_move_shows_up_as_a_loss(self, stockfish):
        """The other half: the measurement has to be able to say "malo".

        The blunder has to be a plain one and the position free of mate. A first
        version forced the engine to decline a mate in one, which it declined to
        decline: a mate is scored outside the network's range precisely so that
        nothing can outrank it, rigged evaluator included.
        """
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        limite = chess.engine.Limit(depth=10)
        regala_caballo = chess.Move.from_uci("f3e5")   # Cxe5, y Cxe5 responde

        calidad = move_quality(
            PreferirJugada(chess.Board(fen), regala_caballo), stockfish, [fen], limite,
        )
        assert calidad.acpl > 250
        assert calidad.agreement == 0.0

    def test_it_measures_a_real_engine(self, evaluador, stockfish):
        fens = [
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "rnbqkb1r/pp2pppp/3p1n2/2pP4/4P3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 8",
        ]
        calidad = move_quality(
            evaluador, stockfish, fens, chess.engine.Limit(depth=8), label="material",
        )

        assert len(calidad.losses_cp) == 3
        assert np.all(calidad.losses_cp >= 0), "una perdida nunca puede ser negativa"
        assert 0.0 <= calidad.agreement <= 1.0
        assert "perdida media" in calidad.summary()

    def test_a_material_only_engine_loses_more_than_zero(self, evaluador, stockfish):
        """Sanity: counting material is not the same as playing well."""
        fens = [
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P3/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 8",
            "rnbqkb1r/pp2pppp/3p1n2/2pP4/4P3/8/PPP2PPP/RNBQKBNR w KQkq - 0 5",
            "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7",
        ]
        calidad = move_quality(evaluador, stockfish, fens, chess.engine.Limit(depth=8))
        assert calidad.acpl > 0
