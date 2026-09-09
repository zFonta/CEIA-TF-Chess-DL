"""Dataset integrity checks (requirement 3.2).

These are the checks the plan commits to: no duplicate positions, a balanced
split between white-to-move and black-to-move positions, labels inside the valid
range with no nulls, every game above the configured Elo threshold, parseable
non-terminal FENs, and the sign relation between the two points of view.

They run over the Parquet table with Arrow compute kernels rather than row by
row, so validating millions of positions stays cheap enough to do on every run.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import chess
import pyarrow as pa
import pyarrow.compute as pc

from chessdl.config import DatasetConfig

#: How many FENs to parse when checking legality. Parsing is the one check that
#: cannot be vectorised, so on a large dataset it runs on a random sample.
DEFAULT_FEN_SAMPLE = 5_000


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class ValidationReport:
    n_rows: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        lines = [f"Dataset validation over {self.n_rows:,} positions", ""]
        lines += [str(check) for check in self.checks]
        lines += ["", "RESULT: " + ("PASS" if self.passed else "FAIL")]
        return "\n".join(lines)


def _sum_bool(column: pa.ChunkedArray) -> int:
    return pc.sum(pc.cast(column, pa.int64())).as_py() or 0


def check_no_duplicate_positions(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """Requirement 3.2: the dataset must not repeat positions."""
    n_rows = table.num_rows
    n_unique = pc.count_distinct(table["pos_key"]).as_py()
    duplicates = n_rows - n_unique
    return CheckResult(
        name="no duplicate positions",
        passed=duplicates == 0,
        detail=f"{n_unique:,} distinct keys over {n_rows:,} rows ({duplicates:,} duplicates)",
    )


def check_turn_balance(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """Requirement 1.3: neither colour may exceed the configured share."""
    n_rows = table.num_rows
    if n_rows == 0:
        return CheckResult("colour balance", False, "empty dataset")

    n_white = _sum_bool(table["turn_white"])
    white_share = n_white / n_rows
    limit = cfg.validation.max_turn_share
    passed = max(white_share, 1 - white_share) <= limit
    return CheckResult(
        name="colour balance",
        passed=passed,
        detail=(
            f"white to move {white_share:.2%}, black to move {1 - white_share:.2%} "
            f"(limit {limit:.0%})"
        ),
    )


def check_label_range(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """Requirements 1.1 and 1.4: labels live in [-1, 1], with no nulls or NaN."""
    problems: list[str] = []
    for column in ("value_white", "value_stm"):
        array = table[column]
        if array.null_count:
            problems.append(f"{column}: {array.null_count:,} nulls")
        if pc.any(pc.is_nan(array)).as_py():
            problems.append(f"{column}: contains NaN")
            continue
        bounds = pc.min_max(array).as_py()
        low, high = bounds["min"], bounds["max"]
        if low is None or high is None:
            problems.append(f"{column}: no values")
        elif low < -1.0 or high > 1.0:
            problems.append(f"{column}: range [{low:.4f}, {high:.4f}] escapes [-1, 1]")

    if problems:
        return CheckResult("label range", False, "; ".join(problems))

    bounds = pc.min_max(table["value_white"]).as_py()
    return CheckResult(
        name="label range",
        passed=True,
        detail=f"value_white in [{bounds['min']:.4f}, {bounds['max']:.4f}], no nulls or NaN",
    )


def check_no_nulls(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """Every column is declared non-nullable; verify the data agrees."""
    offenders = {
        name: table[name].null_count
        for name in table.column_names
        if table[name].null_count
    }
    return CheckResult(
        name="no null values",
        passed=not offenders,
        detail="no nulls in any column" if not offenders else f"nulls found: {offenders}",
    )


def check_elo_threshold(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """Requirement 1.1: both players must be above the threshold."""
    threshold = cfg.filter.min_elo
    minimums = {
        column: pc.min(table[column]).as_py() for column in ("white_elo", "black_elo")
    }
    below = {k: v for k, v in minimums.items() if v is None or v < threshold}
    return CheckResult(
        name="Elo threshold",
        passed=not below,
        detail=(
            f"min white {minimums['white_elo']}, min black {minimums['black_elo']} "
            f"(threshold {threshold})"
        ),
    )


def check_sign_invariant(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """The relation the two-point-of-view design rests on.

    ``cp_white == cp_stm`` when white is to move and ``-cp_stm`` otherwise. If
    this ever broke, half the dataset would be labelled with the wrong sign and
    nothing else in the pipeline would notice.
    """
    expected_cp = pc.if_else(table["turn_white"], table["cp_stm"], pc.negate(table["cp_stm"]))
    cp_mismatches = table.num_rows - _sum_bool(pc.equal(table["cp_white"], expected_cp))

    expected_value = pc.if_else(
        table["turn_white"], table["value_stm"], pc.negate(table["value_stm"])
    )
    value_close = pc.less(
        pc.abs(pc.subtract(table["value_white"], expected_value)), 1e-6
    )
    value_mismatches = table.num_rows - _sum_bool(value_close)

    passed = cp_mismatches == 0 and value_mismatches == 0
    return CheckResult(
        name="sign invariant between points of view",
        passed=passed,
        detail=(
            "cp and value agree on every row"
            if passed
            else f"{cp_mismatches:,} cp mismatches, {value_mismatches:,} value mismatches"
        ),
    )


def check_fens_are_legal(
    table: pa.Table,
    cfg: DatasetConfig,
    sample_size: int = DEFAULT_FEN_SAMPLE,
    seed: int = 0,
) -> CheckResult:
    """FENs must parse and must not already be finished positions.

    A checkmated or stalemated position has no meaningful evaluation and no legal
    move to search, so it has no business in the dataset.
    """
    n_rows = table.num_rows
    if n_rows == 0:
        return CheckResult("FEN legality", False, "empty dataset")

    indices = range(n_rows)
    if n_rows > sample_size:
        indices = random.Random(seed).sample(range(n_rows), sample_size)

    fens = table["fen"]
    unparseable = 0
    terminal = 0
    for index in indices:
        fen = fens[index].as_py()
        try:
            board = chess.Board(fen)
        except ValueError:
            unparseable += 1
            continue
        if board.is_game_over():
            terminal += 1

    checked = len(list(indices)) if not isinstance(indices, range) else n_rows
    passed = unparseable == 0 and terminal == 0
    return CheckResult(
        name="FEN legality",
        passed=passed,
        detail=(
            f"{checked:,} FENs checked, all parseable and non-terminal"
            if passed
            else f"{unparseable:,} unparseable, {terminal:,} terminal (of {checked:,} checked)"
        ),
    )


def check_mate_consistency(table: pa.Table, cfg: DatasetConfig) -> CheckResult:
    """``is_mate`` and ``mate_in`` must agree, and mates must sit at the clip."""
    is_mate = table["is_mate"]
    has_distance = pc.not_equal(table["mate_in"], 0)
    mismatches = table.num_rows - _sum_bool(pc.equal(is_mate, has_distance))

    clip = cfg.normalization.cp_clip
    at_clip = pc.equal(pc.abs(table["cp_white"]), clip)
    # Every mate must sit at the clip bound; a non-mate may also reach it.
    mates_off_clip = _sum_bool(pc.and_(is_mate, pc.invert(at_clip)))

    passed = mismatches == 0 and mates_off_clip == 0
    n_mates = _sum_bool(is_mate)
    return CheckResult(
        name="mate flags",
        passed=passed,
        detail=(
            f"{n_mates:,} mate positions, all consistent"
            if passed
            else f"{mismatches:,} flag mismatches, {mates_off_clip:,} mates away from the clip"
        ),
    )


CHECKS: tuple[Callable[[pa.Table, DatasetConfig], CheckResult], ...] = (
    check_no_duplicate_positions,
    check_turn_balance,
    check_label_range,
    check_no_nulls,
    check_elo_threshold,
    check_sign_invariant,
    check_fens_are_legal,
    check_mate_consistency,
)


def validate_table(table: pa.Table, cfg: DatasetConfig) -> ValidationReport:
    """Run every integrity check and collect the results."""
    report = ValidationReport(n_rows=table.num_rows)
    for check in CHECKS:
        report.checks.append(check(table, cfg))
    return report


def describe_table(table: pa.Table) -> dict[str, float | int]:
    """Descriptive statistics for the exploratory analysis and the dataset card."""
    if table.num_rows == 0:
        return {"n_positions": 0}

    values = table["value_white"]
    bounds = pc.min_max(values).as_py()
    mean = pc.mean(values).as_py()
    stddev = pc.stddev(values).as_py()

    return {
        "n_positions": table.num_rows,
        "n_games": pc.count_distinct(table["game_id"]).as_py(),
        "white_to_move_share": _sum_bool(table["turn_white"]) / table.num_rows,
        "mate_share": _sum_bool(table["is_mate"]) / table.num_rows,
        "value_mean": mean,
        "value_std": stddev if stddev is not None and not math.isnan(stddev) else 0.0,
        "value_min": bounds["min"],
        "value_max": bounds["max"],
        "mean_white_elo": pc.mean(table["white_elo"]).as_py(),
        "mean_black_elo": pc.mean(table["black_elo"]).as_py(),
        "mean_ply": pc.mean(table["ply"]).as_py(),
    }
