"""End-to-end tests for the dataset build.

The build is exercised against the PGN fixture with a real Stockfish at a
shallow depth: same code path as a depth-12 production run, a few seconds long.
Tests that need the engine skip cleanly when it is not installed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import chess
import pytest

from chessdl.data import lichess, pipeline, schema
from chessdl.data.labeling import analyse_fen, engine_version, label_fens, open_engine
from chessdl.data.pgn import iter_raw_games
from chessdl.data.state import PipelineState, SeenKeys, seen_keys_path
from chessdl.data.validate import describe_table, validate_table
from chessdl.normalize import value_to_cp
from tests.conftest import EXPECTED_ACCEPTED_GAMES, EXPECTED_SAMPLEABLE_GAMES


def fixture_sources(sample_pgn: Path) -> list[tuple[str, str]]:
    return [("fixture", str(sample_pgn))]


# --- resume state ----------------------------------------------------------


def test_state_round_trips(tmp_path):
    state = PipelineState.load(tmp_path / "state.json")
    state.for_dump("2025-06").shards_done = 3
    state.for_dump("2025-06").positions_written = 1200
    state.save()

    reloaded = PipelineState.load(tmp_path / "state.json")
    assert reloaded.for_dump("2025-06").shards_done == 3
    assert reloaded.total_positions == 1200


def test_missing_state_starts_empty(tmp_path):
    state = PipelineState.load(tmp_path / "nothing.json")
    assert state.dumps == {}
    assert state.for_dump("2025-06").shards_done == 0


def test_games_to_skip_follows_the_shard_size(tmp_path):
    state = PipelineState.load(tmp_path / "state.json")
    dump_state = state.for_dump("2025-06")
    dump_state.shards_done = 4
    assert dump_state.games_to_skip(5000) == 20_000


def test_seen_keys_persist_across_runs(tmp_path):
    """Deduplication has to survive a Colab disconnect, not just a process."""
    path = tmp_path / "seen.npy"
    keys = SeenKeys(path)
    assert keys.add_new(11) is True
    assert keys.add_new(11) is False
    keys.add_new(22)
    keys.save()

    reloaded = SeenKeys(path)
    assert len(reloaded) == 2
    assert 11 in reloaded
    assert reloaded.add_new(11) is False
    assert reloaded.add_new(33) is True


def test_seen_keys_handle_large_values(tmp_path):
    """Position keys are full 64-bit values and must not overflow on the way out."""
    path = tmp_path / "seen.npy"
    big = 2**64 - 1
    keys = SeenKeys(path)
    keys.add_new(big)
    keys.save()
    assert big in SeenKeys(path)


def test_seen_keys_path_sits_next_to_the_state():
    path = seen_keys_path("/content/drive/MyDrive/ceia-chess/state.json")
    assert path.name == "state_seen_keys.npy"
    assert path.parent.name == "ceia-chess"


# --- extraction ------------------------------------------------------------


def test_extract_writes_only_the_accepted_games(sample_pgn: Path, cfg, tmp_path):
    out = tmp_path / "extract.pgn.zst"
    stats = lichess.extract_games(sample_pgn, cfg.filter, out_path=out, progress=False)

    assert stats.games_seen == 41
    assert stats.games_accepted == stats.games_written == EXPECTED_ACCEPTED_GAMES
    assert out.exists()
    assert out.stat().st_size < sample_pgn.stat().st_size


def test_the_extract_can_be_read_back(sample_pgn: Path, cfg, tmp_path):
    """The extract replaces the dump downstream, so it must reparse identically."""
    out = tmp_path / "extract.pgn.zst"
    lichess.extract_games(sample_pgn, cfg.filter, out_path=out, progress=False)

    with sample_pgn.open(encoding="utf-8") as handle:
        original = [raw.game_id for raw in iter_raw_games(handle, cfg.filter)]
    reread = [raw.game_id for raw in lichess.iter_dump_games(out, cfg.filter)]
    assert reread == original


def test_survey_counts_without_writing(sample_pgn: Path, cfg, tmp_path):
    stats = lichess.extract_games(sample_pgn, cfg.filter, out_path=None, progress=False)
    assert stats.games_accepted == EXPECTED_ACCEPTED_GAMES
    assert stats.games_written == 0
    assert list(tmp_path.iterdir()) == []


def test_survey_reports_the_yield(sample_pgn: Path, cfg):
    results = pipeline.survey(cfg, progress=False, sources=fixture_sources(sample_pgn))
    assert len(results) == 1
    assert 0 < results[0].acceptance_rate < 1
    assert "accepted" in results[0].summary()


def test_max_games_bounds_the_extract(sample_pgn: Path, cfg, tmp_path):
    stats = lichess.extract_games(
        sample_pgn, cfg.filter, out_path=tmp_path / "e.pgn.zst", max_games=5, progress=False
    )
    assert stats.games_written == 5


# --- sampling and deduplication --------------------------------------------


def test_positions_from_games_samples_every_usable_game(sample_pgn: Path, cfg, tmp_path):
    with sample_pgn.open(encoding="utf-8") as handle:
        games = list(iter_raw_games(handle, cfg.filter))

    seen = SeenKeys(tmp_path / "seen.npy")
    pending, duplicates, dropped = pipeline.positions_from_games(games, cfg, seen)

    assert dropped == EXPECTED_ACCEPTED_GAMES - EXPECTED_SAMPLEABLE_GAMES
    assert len(pending) == EXPECTED_SAMPLEABLE_GAMES * cfg.sampling.positions_per_game
    assert duplicates >= 0


def test_duplicate_positions_are_skipped(sample_pgn: Path, cfg, tmp_path):
    """Running the same games twice must add nothing the second time."""
    with sample_pgn.open(encoding="utf-8") as handle:
        games = list(iter_raw_games(handle, cfg.filter))

    seen = SeenKeys(tmp_path / "seen.npy")
    first, _, _ = pipeline.positions_from_games(games, cfg, seen)
    second, duplicates, _ = pipeline.positions_from_games(games, cfg, seen)

    assert second == []
    assert duplicates == len(first)


def test_sampled_positions_are_colour_balanced(sample_pgn: Path, cfg, tmp_path):
    with sample_pgn.open(encoding="utf-8") as handle:
        games = list(iter_raw_games(handle, cfg.filter))

    pending, _, _ = pipeline.positions_from_games(games, cfg, SeenKeys(tmp_path / "s.npy"))
    white = sum(1 for row in pending if row["turn_white"])
    assert white == len(pending) - white


# --- labelling (needs Stockfish) -------------------------------------------


def test_engine_version_is_reported(local_cfg):
    version = engine_version(local_cfg.labeling)
    assert "Stockfish" in version


def test_the_starting_position_is_roughly_balanced(local_cfg):
    """Sanity check on the labels themselves, per the plan's risk-4 mitigation."""
    engine = open_engine(local_cfg.labeling)
    try:
        label = analyse_fen(
            engine, chess.STARTING_FEN, local_cfg.labeling, local_cfg.normalization
        )
    finally:
        engine.quit()
    assert abs(label.value_white) < 0.2
    assert not label.is_mate


def test_a_large_material_advantage_is_labelled_decisively(local_cfg):
    """White a queen up, black to move: clearly winning for white."""
    engine = open_engine(local_cfg.labeling)
    try:
        label = analyse_fen(
            engine, "4k3/8/8/8/8/8/8/3QK3 b - - 0 1", local_cfg.labeling, local_cfg.normalization
        )
    finally:
        engine.quit()

    assert label.value_white > 0.5
    assert label.value_stm < 0  # bad for black, who is to move
    assert label.cp_white > 0


def test_mate_in_one_is_labelled_as_mate(local_cfg):
    engine = open_engine(local_cfg.labeling)
    try:
        label = analyse_fen(
            engine, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", local_cfg.labeling, local_cfg.normalization
        )
    finally:
        engine.quit()
    assert label.value_white > 0.9


def test_labels_invert_with_the_point_of_view(local_cfg):
    """The same position, mirrored, must get the opposite white-relative label."""
    engine = open_engine(local_cfg.labeling)
    try:
        white_up = analyse_fen(
            engine, "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", local_cfg.labeling, local_cfg.normalization
        )
        black_up = analyse_fen(
            engine,
            chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").mirror().fen(),
            local_cfg.labeling,
            local_cfg.normalization,
        )
    finally:
        engine.quit()

    assert white_up.value_white > 0
    assert black_up.value_white < 0


def test_parallel_labelling_returns_every_position(local_cfg):
    fens = [chess.STARTING_FEN, "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "8/8/8/4k3/8/8/4K3/7R b - - 0 1"]
    labeled = label_fens(fens, local_cfg.labeling, local_cfg.normalization, progress=False)
    assert [item.fen for item in labeled] == fens


def test_labelling_an_empty_batch(local_cfg):
    assert label_fens([], local_cfg.labeling, local_cfg.normalization, progress=False) == []


# --- the full build --------------------------------------------------------


@pytest.fixture
def built(sample_pgn: Path, local_cfg):
    """Run the whole pipeline once and reuse the result across tests."""
    summary = pipeline.run_build(
        local_cfg,
        sf_version=engine_version(local_cfg.labeling),
        progress=False,
        sources=fixture_sources(sample_pgn),
    )
    return summary, local_cfg


def test_the_build_produces_shards(built):
    summary, cfg = built
    assert summary.shards
    assert summary.n_positions > 0
    for shard in summary.shards:
        assert shard.path.exists()


def test_the_built_dataset_passes_every_integrity_check(built):
    """The end-to-end guarantee requirement 3.2 asks for."""
    summary, cfg = built
    table = schema.read_dataset(schema.shard_paths(cfg.output.local_dir))
    report = validate_table(table, cfg)
    assert report.passed, report.summary()


def test_the_built_dataset_has_the_expected_size(built):
    summary, cfg = built
    table = schema.read_dataset(schema.shard_paths(cfg.output.local_dir))
    assert table.num_rows == EXPECTED_SAMPLEABLE_GAMES * cfg.sampling.positions_per_game


def test_provenance_is_recorded_on_every_row(built):
    """Requirement 2.3: the dataset documents how it was made."""
    summary, cfg = built
    table = schema.read_dataset(schema.shard_paths(cfg.output.local_dir))

    assert set(table["sf_depth"].to_pylist()) == {cfg.labeling.depth}
    assert all("Stockfish" in v for v in set(table["sf_version"].to_pylist()))
    assert set(table["src_dump"].to_pylist()) == {"fixture"}


def test_labels_can_be_read_back_as_centipawns(built):
    """Requirement 4.2: the inverse transform recovers Stockfish's own scale."""
    summary, cfg = built
    table = schema.read_dataset(schema.shard_paths(cfg.output.local_dir))

    for value, cp in zip(table["value_white"].to_pylist(), table["cp_white"].to_pylist()):
        recovered = value_to_cp(value, cfg.normalization.scale, cfg.normalization.cp_clip)
        assert recovered == pytest.approx(cp, abs=1.0)


def test_descriptive_statistics_are_available(built):
    summary, cfg = built
    table = schema.read_dataset(schema.shard_paths(cfg.output.local_dir))
    stats = describe_table(table)

    assert stats["n_positions"] == table.num_rows
    assert stats["n_games"] == EXPECTED_SAMPLEABLE_GAMES
    assert stats["white_to_move_share"] == pytest.approx(0.5)


def test_the_build_resumes_instead_of_redoing_work(sample_pgn: Path, local_cfg):
    """A disconnect must cost at most one shard, and must not duplicate rows."""
    sources = fixture_sources(sample_pgn)
    version = engine_version(local_cfg.labeling)

    first = pipeline.run_build(
        local_cfg, sf_version=version, max_shards=1, progress=False, sources=sources
    )
    assert len(first.shards) == 1

    second = pipeline.run_build(
        local_cfg, sf_version=version, progress=False, sources=sources
    )
    assert second.shards
    assert first.shards[0].path not in [shard.path for shard in second.shards]

    table = schema.read_dataset(schema.shard_paths(local_cfg.output.local_dir))
    assert table.num_rows == EXPECTED_SAMPLEABLE_GAMES * local_cfg.sampling.positions_per_game
    assert validate_table(table, local_cfg).passed


def test_rerunning_a_finished_build_adds_nothing(sample_pgn: Path, local_cfg):
    sources = fixture_sources(sample_pgn)
    version = engine_version(local_cfg.labeling)

    pipeline.run_build(local_cfg, sf_version=version, progress=False, sources=sources)
    before = schema.read_dataset(schema.shard_paths(local_cfg.output.local_dir)).num_rows

    pipeline.run_build(local_cfg, sf_version=version, progress=False, sources=sources)
    after = schema.read_dataset(schema.shard_paths(local_cfg.output.local_dir)).num_rows
    assert after == before


def test_the_extract_is_not_rebuilt_on_resume(sample_pgn: Path, local_cfg):
    """The extract costs a full pass over the dump; it must be built once."""
    sources = fixture_sources(sample_pgn)
    version = engine_version(local_cfg.labeling)

    pipeline.run_build(
        local_cfg, sf_version=version, max_shards=1, progress=False, sources=sources
    )
    extract = pipeline.extract_path_for(local_cfg, "fixture")
    stamp = extract.stat().st_mtime_ns

    pipeline.run_build(
        local_cfg, sf_version=version, max_shards=1, progress=False, sources=sources
    )
    assert extract.stat().st_mtime_ns == stamp
