"""Build the labelled position dataset.

    python -m chessdl.scripts.build_dataset --survey
    python -m chessdl.scripts.build_dataset --max-shards 1 --no-push
    python -m chessdl.scripts.build_dataset

The first form measures how many games pass the filter without spending any
Stockfish time; the second runs a single pilot shard locally; the third is the
full, resumable build.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from chessdl.colab import describe_runtime
from chessdl.config import DatasetConfig, load_config
from chessdl.data import pipeline
from chessdl.data.labeling import EngineNotAvailableError, engine_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to the dataset YAML config."
    )
    parser.add_argument(
        "--survey",
        action="store_true",
        help="Only count how many games pass the filter; label nothing.",
    )
    parser.add_argument(
        "--max-scanned",
        type=int,
        default=None,
        help="With --survey, stop after looking at this many games.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop extracting after this many accepted games (pilot runs).",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Build at most this many shards in this invocation.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep shards local instead of pushing them to the Hub.",
    )
    parser.add_argument(
        "--shards-dir", type=Path, default=None, help="Override output.local_dir."
    )
    parser.add_argument(
        "--extract-dir", type=Path, default=None, help="Override output.extract_dir."
    )
    parser.add_argument(
        "--state-path", type=Path, default=None, help="Override output.state_path."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Use this PGN file or URL instead of the configured dumps.",
    )
    parser.add_argument(
        "--engine", type=str, default=None, help="Override labeling.engine_path."
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override labeling.workers."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars.")
    parser.add_argument(
        "--json", action="store_true", help="Emit the summary as JSON."
    )
    return parser


def apply_overrides(cfg: DatasetConfig, args: argparse.Namespace) -> DatasetConfig:
    """Fold the command line overrides into the loaded configuration."""
    output = cfg.output
    if args.shards_dir is not None:
        output = replace(output, local_dir=str(args.shards_dir))
    if args.extract_dir is not None:
        output = replace(output, extract_dir=str(args.extract_dir))
    if args.state_path is not None:
        output = replace(output, state_path=str(args.state_path))

    labeling = cfg.labeling
    if args.engine is not None:
        labeling = replace(labeling, engine_path=args.engine)
    if args.workers is not None:
        labeling = replace(labeling, workers=args.workers)

    source = cfg.source
    if args.source is not None:
        # A single explicit source replaces the configured dump list. The dump
        # label keeps the file's stem so the provenance column stays meaningful.
        source = replace(source, dumps=(Path(args.source).stem,))

    return replace(cfg, output=output, labeling=labeling, source=source)


def explicit_sources(args: argparse.Namespace) -> list[tuple[str, str]] | None:
    """An explicit ``--source`` overrides the configured dump list.

    The dump label keeps the file's stem, so the ``src_dump`` provenance column
    still says where the positions came from.
    """
    if args.source is None:
        return None
    return [(Path(args.source).stem, args.source)]


def _run_survey(cfg: DatasetConfig, args: argparse.Namespace) -> int:
    results = pipeline.survey(
        cfg,
        max_scanned=args.max_scanned,
        progress=not args.quiet,
        sources=explicit_sources(args),
    )
    if args.json:
        print(json.dumps([stats.as_dict() for stats in results], indent=2))
    else:
        print("\nSurvey of the configured dumps")
        print("-" * 60)
        for stats in results:
            print(stats.summary())
        total = sum(stats.games_accepted for stats in results)
        print("-" * 60)
        print(
            f"{total:,} usable games -> about "
            f"{total * cfg.sampling.positions_per_game:,} positions"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)

    if args.survey:
        return _run_survey(cfg, args)

    runtime = describe_runtime(cfg.labeling.engine_path)
    for warning in runtime.warnings():
        print(f"WARNING: {warning}", file=sys.stderr)

    try:
        sf_version = engine_version(cfg.labeling)
    except EngineNotAvailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Labelling with {sf_version} at depth {cfg.labeling.depth} "
          f"({cfg.labeling.resolved_workers()} workers)")

    summary = pipeline.run_build(
        cfg,
        sf_version=sf_version,
        max_shards=args.max_shards,
        max_games=args.max_games,
        push=False if args.no_push else None,
        progress=not args.quiet,
        sources=explicit_sources(args),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "n_positions": summary.n_positions,
                    "n_duplicates": summary.n_duplicates,
                    "shards": [str(shard.path) for shard in summary.shards],
                },
                indent=2,
            )
        )
    else:
        print("\n" + summary.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
