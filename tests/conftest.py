"""Shared fixtures for the data pipeline tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from chessdl.config import DatasetConfig, load_config

FIXTURE_PGN = Path(__file__).parent / "fixtures" / "sample_games.pgn"


@dataclass(frozen=True)
class FixtureExpectations:
    """What the PGN fixture should yield at each stage of the pipeline."""

    #: Games whose headers pass the filter.
    accepted_games: int = 22
    #: Of those, the ones also long enough to sample; two are deliberately short.
    sampleable_games: int = 20

    @property
    def too_short_games(self) -> int:
        return self.accepted_games - self.sampleable_games


@pytest.fixture
def expected_counts() -> FixtureExpectations:
    """Expected counts for the PGN fixture, injected rather than imported.

    These used to be module-level constants that other test modules imported with
    ``from tests.conftest import ...``. That only works when the repository root
    is on ``sys.path`` *and* nothing else on it claims the name ``tests`` -- and
    a environment with a large pre-installed set of packages (Colab, for one)
    can easily ship its own top-level ``tests`` package that shadows this one.
    A fixture is resolved by pytest itself, so it depends on neither.
    """
    return FixtureExpectations()


def find_stockfish() -> str | None:
    """Locate a Stockfish binary, including the Debian /usr/games location."""
    found = shutil.which("stockfish")
    if found:
        return found
    debian_path = Path("/usr/games/stockfish")
    return str(debian_path) if debian_path.is_file() else None


@pytest.fixture(scope="session")
def stockfish_path() -> str:
    path = find_stockfish()
    if path is None:
        pytest.skip("Stockfish is not installed in this environment")
    return path


@pytest.fixture
def sample_pgn() -> Path:
    return FIXTURE_PGN


@pytest.fixture
def shard_with_keys():
    """Write a shard containing exactly the given deduplication keys.

    A fixture rather than an importable helper: importing between test modules
    needs the repository root on ``sys.path`` and nothing else on it claiming
    the name ``tests``, which is not something to rely on.
    """
    import chess

    from chessdl.data import schema
    from chessdl.normalize import cp_to_value

    def build(path, keys):
        filas = [
            {
                "game_id": f"g{clave}", "ply": 10, "fen": chess.STARTING_FEN,
                "pos_key": clave, "turn_white": True,
                "cp_white": 0, "cp_stm": 0,
                "value_white": cp_to_value(0), "value_stm": cp_to_value(0),
                "is_mate": False, "mate_in": 0,
                "white_elo": 2400, "black_elo": 2350,
                "result": "1-0", "time_control": "Blitz",
                "sf_depth": 12, "sf_version": "Stockfish 17.1", "src_dump": "2025-06",
            }
            for clave in keys
        ]
        return schema.write_shard(filas, path)

    return build


@pytest.fixture
def cfg() -> DatasetConfig:
    """The repository's own config, so the tests check the shipped defaults."""
    return load_config()


@pytest.fixture
def local_cfg(cfg: DatasetConfig, tmp_path: Path, stockfish_path: str) -> DatasetConfig:
    """A config wired to temporary directories and a shallow search depth.

    Depth 6 keeps the end-to-end test to a few seconds while exercising exactly
    the same code path as a real depth-12 run.
    """
    return replace(
        cfg,
        output=replace(
            cfg.output,
            local_dir=str(tmp_path / "shards"),
            extract_dir=str(tmp_path / "extracts"),
            state_path=str(tmp_path / "state.json"),
            push_to_hub=False,
            games_per_shard=10,
        ),
        labeling=replace(cfg.labeling, engine_path=stockfish_path, depth=6, workers=2),
    )
