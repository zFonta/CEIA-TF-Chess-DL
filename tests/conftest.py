"""Shared fixtures for the data pipeline tests."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from chessdl.config import DatasetConfig, load_config

FIXTURE_PGN = Path(__file__).parent / "fixtures" / "sample_games.pgn"

#: Games in the fixture whose headers should pass the filter. Twenty of them are
#: also long enough to sample; two are deliberately too short.
EXPECTED_ACCEPTED_GAMES = 22
EXPECTED_SAMPLEABLE_GAMES = 20


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
