"""Typed configuration for the data pipeline, loaded from YAML.

The YAML file (``configs/dataset_v1.yaml``) is the single source of truth for every
pipeline parameter. Keeping it in one versioned place is what makes the dataset
reproducible and what feeds the dataset card required by requirement 2.3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HF_NAMESPACE = "zFonta"


@dataclass(frozen=True)
class SourceConfig:
    """Where the raw PGN dumps come from."""

    base_url: str = "https://database.lichess.org/standard"
    dumps: tuple[str, ...] = ("2025-06",)

    def dump_url(self, dump: str) -> str:
        return f"{self.base_url}/lichess_db_standard_rated_{dump}.pgn.zst"

    def dump_name(self, dump: str) -> str:
        return f"lichess_db_standard_rated_{dump}"


@dataclass(frozen=True)
class FilterConfig:
    """Game-level filter (requirement 1.1)."""

    min_elo: int = 2200
    time_controls: tuple[str, ...] = ("Blitz", "Rapid", "Classical")
    rated_only: bool = True
    min_plies: int = 20
    excluded_terminations: tuple[str, ...] = (
        "Abandoned",
        "Rules infraction",
        "Unterminated",
    )


@dataclass(frozen=True)
class SamplingConfig:
    """Position sampling within an accepted game (requirement 1.3)."""

    positions_per_game: int = 4
    balance_by_turn: bool = True
    seed: int = 20260909


@dataclass(frozen=True)
class LabelingConfig:
    """Stockfish labelling parameters (requirement 2.3)."""

    engine_path: str = "stockfish"
    depth: int = 12
    threads_per_engine: int = 1
    hash_mb: int = 64
    workers: int | None = None

    def resolved_workers(self) -> int:
        """Worker count, defaulting to the runtime's CPU count.

        Colab hands out different machine shapes from session to session, so the
        parallelism is resolved at run time rather than pinned in the config.
        """
        if self.workers is not None:
            return max(1, self.workers)
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class NormalizationConfig:
    """Centipawn <-> value transform (requirements 1.1 and 4.2)."""

    scale: float = 400.0
    cp_clip: int = 2000


@dataclass(frozen=True)
class OutputConfig:
    """Shard layout, resume state and Hugging Face destination."""

    games_per_shard: int = 5000
    local_dir: str = "/content/ceia-chess/shards"
    # The filtered extracts and the resume state are expensive to regenerate
    # (the extract costs a full pass over a multi-gigabyte dump), so on Colab
    # they belong on Drive rather than the recycled local disk.
    extract_dir: str = "/content/drive/MyDrive/ceia-chess/extracts"
    state_path: str = "/content/drive/MyDrive/ceia-chess/state.json"
    push_to_hub: bool = True
    hf_dataset: str = "ceia-chess-eval"
    hf_namespace: str = field(
        default_factory=lambda: os.environ.get("HF_NAMESPACE", DEFAULT_HF_NAMESPACE)
    )

    @property
    def hf_repo_id(self) -> str:
        return f"{self.hf_namespace}/{self.hf_dataset}"


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds for the dataset integrity checks (requirement 3.2)."""

    max_turn_share: float = 0.55


@dataclass(frozen=True)
class DatasetConfig:
    """Root configuration object."""

    version: str = "v1"
    source: SourceConfig = field(default_factory=SourceConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetConfig":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            if is_dataclass(f.type) or isinstance(value, dict):
                kwargs[f.name] = _build_section(f.name, value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)


_SECTIONS: dict[str, type] = {
    "source": SourceConfig,
    "filter": FilterConfig,
    "sampling": SamplingConfig,
    "labeling": LabelingConfig,
    "normalization": NormalizationConfig,
    "output": OutputConfig,
    "validation": ValidationConfig,
}


def _build_section(name: str, value: Any) -> Any:
    """Instantiate one config section, converting YAML lists into tuples.

    The dataclasses are frozen and use tuples so a config object can be shared
    with worker processes without any risk of one of them mutating it.
    """
    section_cls = _SECTIONS.get(name)
    if section_cls is None or not isinstance(value, dict):
        return value

    known = {f.name for f in fields(section_cls)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) in config section '{name}': {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    coerced = {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    return section_cls(**coerced)


def default_config_path() -> Path:
    """Path to the bundled ``configs/dataset_v1.yaml`` when running from a clone."""
    return Path(__file__).resolve().parents[2] / "configs" / "dataset_v1.yaml"


def load_config(path: str | Path | None = None) -> DatasetConfig:
    """Load the pipeline config, falling back to the repository default.

    The default lives in the repository rather than inside the package, since the
    documented workflow is to clone and install in editable mode. A non-editable
    install has no ``configs/`` directory, so say that plainly instead of raising
    a bare FileNotFoundError.
    """
    resolved = Path(path) if path is not None else default_config_path()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. The default config ships with the "
            "repository, so either clone it and install with `pip install -e .`, "
            "or pass an explicit config path."
        )
    return DatasetConfig.from_yaml(resolved)
