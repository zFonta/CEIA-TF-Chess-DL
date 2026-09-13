"""Rebuilding a trained model from a checkpoint, and the engine's CLI.

The property under test is that a checkpoint is enough to get the *same* network
back. Weights on their own are not: ``load_state_dict`` refuses an incompatible
shape, but it accepts a compatible wrong one without a word, and what comes out
is a model that is quietly not the one that was trained.
"""

from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch", reason="requires the `train` extra")

from chessdl.engine.loader import (  # noqa: E402
    UnknownArchitectureError,
    build_model,
    load_local,
)
from chessdl.models.resnet import ChessResNet, ResNetConfig  # noqa: E402
from chessdl.models.transformer import ChessTransformer, TransformerConfig  # noqa: E402
from chessdl.scripts.play import main  # noqa: E402
from chessdl.training.checkpoint import TrainingHistory, save_checkpoint  # noqa: E402

RESNET = {"channels": 16, "blocks": 1, "dropout": 0.0}
TRANSFORMER = {"d_model": 32, "layers": 1, "heads": 4, "feedforward": 64,
               "pooling": "cls", "dropout": 0.0, "head_dropout": 0.0}


def escribir(tmp_path, modelo, model_config):
    """A checkpoint exactly as `train` writes one."""
    optimizador = torch.optim.AdamW(modelo.parameters())
    return save_checkpoint(
        tmp_path / "checkpoint_best.pt", modelo, optimizador, None, 1,
        TrainingHistory(run_name="t"), model_config,
    )


class TestBuildModel:
    def test_a_resnet_config_builds_a_resnet(self):
        assert isinstance(build_model(RESNET), ChessResNet)

    def test_a_transformer_config_builds_a_transformer(self):
        assert isinstance(build_model(TRANSFORMER), ChessTransformer)

    def test_the_architecture_is_recognised_by_its_fields(self):
        """Not by a name the older checkpoints never recorded."""
        assert isinstance(build_model({"channels": 128, "blocks": 8}), ChessResNet)

    def test_extra_keys_in_the_record_are_ignored(self):
        """Histories gained fields over time; an old checkpoint must still load."""
        modelo = build_model({**RESNET, "algo_que_no_existe": 7})
        assert modelo.config.channels == 16

    def test_an_unrecognisable_record_says_so(self):
        with pytest.raises(UnknownArchitectureError, match="channels"):
            build_model({"capas": 4})


class TestRoundTrip:
    @pytest.mark.parametrize(
        "config, fabrica",
        [(RESNET, lambda: ChessResNet(ResNetConfig(channels=16, blocks=1))),
         (TRANSFORMER, lambda: ChessTransformer(
             TransformerConfig(d_model=32, layers=1, heads=4, feedforward=64)))],
        ids=["resnet", "transformer"],
    )
    def test_the_reloaded_model_predicts_exactly_the_same(self, tmp_path, config, fabrica):
        torch.manual_seed(0)
        original = fabrica().eval()
        ruta = escribir(tmp_path, original, config)

        recuperado = load_local(ruta).eval()

        x = torch.randn(4, 18, 8, 8)
        with torch.no_grad():
            assert torch.equal(original(x), recuperado(x))

    def test_the_shape_comes_from_the_checkpoint_and_not_from_a_default(self, tmp_path):
        """A model rebuilt with default sizes would not even load these weights."""
        torch.manual_seed(0)
        original = ChessResNet(ResNetConfig(channels=16, blocks=1))
        recuperado = load_local(escribir(tmp_path, original, RESNET))
        assert recuperado.config.channels == 16
        assert recuperado.config.blocks == 1


class TestCli:
    def _checkpoint(self, tmp_path):
        torch.manual_seed(0)
        return escribir(tmp_path, ChessResNet(ResNetConfig(channels=16, blocks=1)), RESNET)

    def test_it_analyses_a_position(self, tmp_path, capsys):
        ruta = self._checkpoint(tmp_path)
        assert main(["--checkpoint", str(ruta), "--top", "3"]) == 0

        salida = capsys.readouterr().out
        assert "Jugada elegida" in salida
        assert "centipeones" in salida
        assert chess.STARTING_FEN in salida

    def test_it_analyses_a_given_fen(self, tmp_path, capsys):
        ruta = self._checkpoint(tmp_path)
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        main(["--checkpoint", str(ruta), "--fen", fen])
        assert "Juegan         blancas" in capsys.readouterr().out

    def test_self_play_reports_the_time_budget(self, tmp_path, capsys):
        """Requirement 1.7 is answered with a measurement, not a promise."""
        ruta = self._checkpoint(tmp_path)
        main(["--checkpoint", str(ruta), "--self-play", "6"])

        salida = capsys.readouterr().out
        assert "requerimiento 1.7" in salida
        assert "Resultado:" in salida

    def test_without_weights_it_refuses_instead_of_playing_at_random(self, capsys):
        """An untrained network returns moves, so silence here would look fine."""
        with pytest.raises(SystemExit, match="pesos entrenados"):
            main([])
