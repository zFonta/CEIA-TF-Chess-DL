"""Play or analyse with the trained engine (requirements 1.6, 1.7 and 4.2).

    python -m chessdl.scripts.play --checkpoint ./checkpoints/campana2-warmup/checkpoint_best.pt
    python -m chessdl.scripts.play --run-name campana2-warmup --fen "<FEN>"
    python -m chessdl.scripts.play --run-name campana2-warmup --self-play 40

Evaluations are printed from **white's point of view** and in centipawns, which
is the scale requirements 1.4 and 4.2 speak, and the one a chess player reads
without translating. Internally everything is the side-to-move value the network
was trained on; the flip happens once, on the way out.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

from chessdl import hf
from chessdl.config import load_config
from chessdl.engine.evaluator import Evaluator
from chessdl.engine.loader import load_from_hub, load_local
from chessdl.engine.search import GameOverError, search, value_white
from chessdl.normalize import value_to_cp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    origen = parser.add_argument_group("de donde sale el modelo")
    origen.add_argument("--checkpoint", type=Path, default=None,
                        help="Archivo .pt local.")
    origen.add_argument("--run-name", default=None,
                        help="Nombre de la corrida en el repositorio de modelos del Hub.")
    origen.add_argument("--repo-id", default=None,
                        help="Repositorio de modelos (por defecto, el de la config).")
    origen.add_argument("--device", default=None, help="cpu o cuda.")

    que = parser.add_argument_group("que hacer")
    que.add_argument("--fen", default=chess.STARTING_FEN,
                     help="Posicion a analizar (por defecto, la inicial).")
    que.add_argument("--self-play", type=int, default=0, metavar="N",
                     help="Jugar N jugadas contra si mismo y mostrar la partida.")
    que.add_argument("--top", type=int, default=5,
                     help="Cuantas jugadas mostrar en el analisis.")
    que.add_argument("--seed", type=int, default=None,
                     help="Semilla para desempatar; sin ella la eleccion es determinista.")
    return parser


def cargar_evaluador(args) -> Evaluator:
    if args.checkpoint is not None:
        modelo = load_local(args.checkpoint, device=args.device or "cpu")
    elif args.run_name is not None:
        cfg = load_config()
        repo = args.repo_id or f"{cfg.output.hf_namespace}/{cfg.training.hf_models_repo}"
        modelo = load_from_hub(
            repo, args.run_name, token=hf.get_token(), device=args.device or "cpu"
        )
    else:
        raise SystemExit(
            "Hace falta --checkpoint o --run-name: sin pesos entrenados el motor\n"
            "jugaria con una red al azar, que es peor que no jugar."
        )
    return Evaluator(modelo, device=args.device)


def analizar(board: chess.Board, evaluador: Evaluator, top: int, rng) -> None:
    resultado = search(board, evaluador, rng=rng)
    blancas = value_white(board, resultado.value)

    print(board)
    print()
    print(f"FEN            {board.fen()}")
    print(f"Juegan         {'blancas' if board.turn == chess.WHITE else 'negras'}")
    print(f"Jugada elegida {board.san(resultado.move)}")
    print(f"Evaluacion     {blancas:+.4f} en escala value  "
          f"({value_to_cp(blancas):+.0f} centipeones, vista de las blancas)")
    print(f"Tiempo         {resultado.seconds * 1000:.0f} ms "
          f"sobre {len(resultado.ranked)} jugadas legales")
    print()
    print(resultado.table(board, top=top))


def auto_partida(board: chess.Board, evaluador: Evaluator, jugadas: int, rng) -> None:
    empezo = time.perf_counter()
    tiempos: list[float] = []
    san: list[str] = []

    for _ in range(jugadas):
        try:
            resultado = search(board, evaluador, rng=rng)
        except GameOverError:
            break
        tiempos.append(resultado.seconds)
        san.append(board.san(resultado.move))
        board.push(resultado.move)

    for numero in range(0, len(san), 2):
        blancas = san[numero]
        negras = san[numero + 1] if numero + 1 < len(san) else ""
        print(f"{numero // 2 + 1:>3}. {blancas:<10}{negras}")

    t = np.array(tiempos) * 1000
    print()
    print(f"{len(san)} jugadas en {time.perf_counter() - empezo:.1f} s")
    if len(t):
        print(f"por jugada: mediana {np.median(t):.0f} ms, maximo {t.max():.0f} ms")
        print(f"requerimiento 1.7 (5 s): {5000 / t.max():.0f}x de margen en el peor caso")
    print(f"Resultado: {board.result(claim_draw=True)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluador = cargar_evaluador(args)
    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    print(evaluador.describe())
    print()

    board = chess.Board(args.fen)
    if args.self_play:
        auto_partida(board, evaluador, args.self_play, rng)
    else:
        analizar(board, evaluador, args.top, rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
