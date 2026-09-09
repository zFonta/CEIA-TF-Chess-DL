"""Run the dataset integrity checks and report the result (requirement 3.2).

    python -m chessdl.scripts.validate_dataset --shards-dir /content/ceia-chess/shards
    python -m chessdl.scripts.validate_dataset --from-hub

Exits non-zero when any check fails, so it can gate a build.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chessdl.config import load_config
from chessdl.data import schema
from chessdl.data.validate import describe_table, validate_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="Directory holding the Parquet shards (defaults to output.local_dir).",
    )
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="Download the dataset from the Hub before validating it.",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Also print descriptive statistics."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    shards_dir = args.shards_dir or Path(cfg.output.local_dir)
    if args.from_hub:
        from chessdl import hf

        shards_dir = hf.download_dataset(cfg.output.hf_repo_id, shards_dir)

    paths = schema.shard_paths(shards_dir)
    if not paths:
        print(f"ERROR: no Parquet shards found in {shards_dir}", file=sys.stderr)
        return 2

    table = schema.read_dataset(paths)
    report = validate_table(table, cfg)
    stats = describe_table(table) if args.stats else None

    if args.json:
        payload = {
            "n_rows": report.n_rows,
            "passed": report.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in report.checks
            ],
        }
        if stats is not None:
            payload["stats"] = stats
        print(json.dumps(payload, indent=2))
    else:
        print(f"{len(paths)} shard(s) from {shards_dir}\n")
        print(report.summary())
        if stats is not None:
            print("\nDescriptive statistics")
            print("-" * 40)
            for key, value in stats.items():
                formatted = f"{value:,.4f}" if isinstance(value, float) else f"{value:,}"
                print(f"{key:>22}: {formatted}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
