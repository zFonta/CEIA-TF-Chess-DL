"""Tests for the Parquet schema and the integrity checks (requirements 1.2, 3.2).

Two things are verified here: that the storage round-trips without losing
anything, and that each integrity check actually fires when the problem it
describes is present. A check that always passes would be worse than no check.
"""

from __future__ import annotations

import math
from dataclasses import replace

import chess
import pyarrow as pa
import pytest

from chessdl.data import schema
from chessdl.data.validate import (
    check_elo_threshold,
    check_fens_are_legal,
    check_label_range,
    check_mate_consistency,
    check_no_duplicate_positions,
    check_sign_invariant,
    check_turn_balance,
    validate_table,
)
from chessdl.normalize import cp_to_value

START_FEN = chess.STARTING_FEN


def make_row(**overrides) -> dict:
    """A valid row, with the sign invariant satisfied."""
    cp_white = overrides.pop("cp_white", 50)
    turn_white = overrides.pop("turn_white", True)
    cp_stm = cp_white if turn_white else -cp_white

    row = {
        "game_id": "abc123",
        "ply": 10,
        "fen": START_FEN,
        "pos_key": schema.position_key(START_FEN),
        "turn_white": turn_white,
        "cp_white": cp_white,
        "cp_stm": cp_stm,
        "value_white": cp_to_value(cp_white),
        "value_stm": cp_to_value(cp_stm),
        "is_mate": False,
        "mate_in": 0,
        "white_elo": 2400,
        "black_elo": 2350,
        "result": "1-0",
        "time_control": "Blitz",
        "sf_depth": 12,
        "sf_version": "Stockfish 17.1",
        "src_dump": "2025-06",
    }
    row.update(overrides)
    return row


def make_rows(n: int) -> list[dict]:
    """`n` distinct, balanced, valid rows."""
    rows = []
    board = chess.Board()
    for index in range(n):
        board.push(list(board.legal_moves)[index % board.legal_moves.count()])
        fen = board.fen()
        rows.append(
            make_row(
                game_id=f"game{index}",
                ply=index,
                fen=fen,
                pos_key=schema.position_key(fen),
                turn_white=board.turn == chess.WHITE,
                cp_white=(index % 7) * 30 - 90,
            )
        )
    return rows


def table_of(rows) -> pa.Table:
    return schema.rows_to_table(rows)


# --- schema and storage ----------------------------------------------------


def test_position_key_is_stable_across_calls():
    assert schema.position_key(START_FEN) == schema.position_key(START_FEN)


def test_position_key_ignores_the_move_counters():
    """The same position with different clocks is the same position."""
    base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    assert schema.position_key(f"{base} 0 1") == schema.position_key(f"{base} 12 34")


def test_position_key_separates_different_castling_rights():
    a = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    b = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"
    assert schema.position_key(a) != schema.position_key(b)


def test_position_key_separates_the_side_to_move():
    a = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    b = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    assert schema.position_key(a) != schema.position_key(b)


def test_position_key_fits_in_the_column_type():
    assert 0 <= schema.position_key(START_FEN) < 2**64


def test_shard_round_trip_preserves_everything(tmp_path):
    """Requirement 1.2: reloading a shard must not lose or change anything."""
    rows = make_rows(25)
    path = schema.write_shard(rows, tmp_path / "shard.parquet")
    table = schema.read_shard(path)

    assert table.num_rows == len(rows)
    assert table.schema.equals(schema.SCHEMA)
    assert table.to_pylist() == table_of(rows).to_pylist()


def test_reading_several_shards_concatenates_them(tmp_path):
    schema.write_shard(make_rows(10), tmp_path / "a.parquet")
    schema.write_shard(make_rows(10)[:6], tmp_path / "b.parquet")
    table = schema.read_dataset(schema.shard_paths(tmp_path))
    assert table.num_rows == 16


def test_reading_an_empty_directory_gives_an_empty_table(tmp_path):
    table = schema.read_dataset(schema.shard_paths(tmp_path))
    assert table.num_rows == 0
    assert table.schema.equals(schema.SCHEMA)


def test_a_row_missing_a_column_is_rejected():
    """Arrow would fill the gap with nulls; the writer must not let it."""
    row = make_row()
    del row["sf_version"]
    with pytest.raises(schema.SchemaMismatchError, match="sf_version"):
        table_of([row])


def test_a_row_with_an_unexpected_column_is_rejected():
    with pytest.raises(schema.SchemaMismatchError, match="typo_column"):
        table_of([make_row(typo_column=1)])


def test_mate_distance_is_clamped_into_the_column():
    assert schema.clamp_mate_in(10) == 10
    assert abs(schema.clamp_mate_in(10**9)) <= schema.MATE_IN_LIMIT


# --- the integrity checks ---------------------------------------------------


def test_a_clean_dataset_passes_every_check(cfg):
    report = validate_table(table_of(make_rows(60)), cfg)
    assert report.passed, report.summary()
    assert len(report.checks) == 8


def test_duplicate_positions_are_caught(cfg):
    rows = make_rows(20)
    rows.append(dict(rows[0]))  # the same position twice
    result = check_no_duplicate_positions(table_of(rows), cfg)
    assert not result.passed
    assert "1 duplicates" in result.detail


def test_colour_imbalance_is_caught(cfg):
    """Requirement 1.3: no colour above the configured share."""
    rows = [make_row(turn_white=True, game_id=f"g{i}", ply=i) for i in range(20)]
    for index, row in enumerate(rows):  # keep the keys distinct
        row["pos_key"] = index
    result = check_turn_balance(table_of(rows), cfg)
    assert not result.passed
    assert "100.00%" in result.detail


def test_a_balanced_dataset_passes_the_colour_check(cfg):
    assert check_turn_balance(table_of(make_rows(60)), cfg).passed


def test_the_balance_limit_is_configurable(cfg):
    rows = make_rows(60)
    strict = replace(cfg, validation=replace(cfg.validation, max_turn_share=0.50001))
    assert check_turn_balance(table_of(rows), strict).passed


def test_labels_outside_the_range_are_caught(cfg):
    rows = make_rows(10)
    rows[0]["value_white"] = 1.5
    result = check_label_range(table_of(rows), cfg)
    assert not result.passed
    assert "escapes" in result.detail


def test_nan_labels_are_caught(cfg):
    rows = make_rows(10)
    rows[0]["value_stm"] = math.nan
    result = check_label_range(table_of(rows), cfg)
    assert not result.passed
    assert "NaN" in result.detail


def test_a_game_below_the_elo_threshold_is_caught(cfg):
    rows = make_rows(10)
    rows[3]["black_elo"] = 1500
    result = check_elo_threshold(table_of(rows), cfg)
    assert not result.passed


def test_a_broken_sign_invariant_is_caught(cfg):
    """The failure mode that would silently mislabel half the dataset."""
    rows = make_rows(10)
    rows[2]["cp_stm"] = -rows[2]["cp_stm"]
    rows[2]["value_stm"] = -rows[2]["value_stm"]
    result = check_sign_invariant(table_of(rows), cfg)
    assert not result.passed
    assert "mismatch" in result.detail


def test_the_sign_invariant_holds_for_a_clean_dataset(cfg):
    assert check_sign_invariant(table_of(make_rows(60)), cfg).passed


def test_a_terminal_position_is_caught(cfg):
    """Fool's mate has no evaluation and no legal move to search."""
    checkmate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    rows = make_rows(10)
    rows[0]["fen"] = checkmate
    rows[0]["pos_key"] = schema.position_key(checkmate)
    result = check_fens_are_legal(table_of(rows), cfg)
    assert not result.passed
    assert "terminal" in result.detail


def test_an_unparseable_fen_is_caught(cfg):
    rows = make_rows(10)
    rows[0]["fen"] = "not a fen at all"
    result = check_fens_are_legal(table_of(rows), cfg)
    assert not result.passed
    assert "unparseable" in result.detail


def test_inconsistent_mate_flags_are_caught(cfg):
    rows = make_rows(10)
    rows[1]["is_mate"] = True  # but mate_in stays 0
    result = check_mate_consistency(table_of(rows), cfg)
    assert not result.passed


def test_a_well_formed_mate_row_passes(cfg):
    rows = make_rows(10)
    clip = cfg.normalization.cp_clip
    rows[1].update(
        cp_white=clip,
        cp_stm=clip if rows[1]["turn_white"] else -clip,
        value_white=cp_to_value(clip),
        value_stm=cp_to_value(clip if rows[1]["turn_white"] else -clip),
        is_mate=True,
        mate_in=3,
    )
    assert check_mate_consistency(table_of(rows), cfg).passed


def test_an_empty_dataset_does_not_pass(cfg):
    report = validate_table(schema.SCHEMA.empty_table(), cfg)
    assert not report.passed


def test_the_report_reads_clearly(cfg):
    report = validate_table(table_of(make_rows(20)), cfg)
    text = report.summary()
    assert "RESULT: PASS" in text
    assert "colour balance" in text
    assert not report.failures
