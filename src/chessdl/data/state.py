"""Resume state for the dataset build.

Colab recycles the local disk and disconnects long sessions, so a multi-hour
labelling run has to be able to pick up exactly where it stopped. Two things are
persisted between runs:

* which dumps have been extracted and how many shards of each are finished;
* the set of position keys already written, so deduplication survives a restart
  instead of only holding within a single process.

Both live wherever ``output.state_path`` points -- on Colab that should be a
Drive path, since anything under ``/content`` is gone after a disconnect.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DumpState:
    """Progress through a single monthly dump."""

    extracted: bool = False
    games_seen: int = 0
    games_accepted: int = 0
    shards_done: int = 0
    positions_written: int = 0

    def games_to_skip(self, games_per_shard: int) -> int:
        """Games already turned into shards, to be skipped when resuming."""
        return self.shards_done * games_per_shard


@dataclass
class PipelineState:
    """The build's progress across every dump."""

    path: Path
    dumps: dict[str, DumpState] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineState":
        state_path = Path(path)
        if not state_path.exists():
            return cls(path=state_path)
        with state_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        dumps = {name: DumpState(**values) for name, values in raw.get("dumps", {}).items()}
        return cls(path=state_path, dumps=dumps)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dumps": {name: asdict(state) for name, state in self.dumps.items()}}
        # Write through a temporary file so an interrupted save cannot leave a
        # truncated state behind -- losing the progress record would mean
        # re-labelling everything.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp.replace(self.path)

    def for_dump(self, dump: str) -> DumpState:
        return self.dumps.setdefault(dump, DumpState())

    @property
    def total_positions(self) -> int:
        return sum(state.positions_written for state in self.dumps.values())


class SeenKeys:
    """The set of position keys already written, persisted across runs.

    Deduplication (requirement 3.2) has to hold across shards and across
    restarts, so the keys are kept in memory for the run and flushed to a small
    binary file next to the state.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._keys: set[int] = set()
        if self.path.exists():
            self._keys = set(np.load(self.path).tolist())

    def __contains__(self, key: int) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: int) -> None:
        self._keys.add(key)

    def add_new(self, key: int) -> bool:
        """Add a key, returning True only the first time it is seen."""
        if key in self._keys:
            return False
        self._keys.add(key)
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        array = np.fromiter(self._keys, dtype=np.uint64, count=len(self._keys))
        # The temporary name already ends in .npy, so numpy will not append a
        # second suffix and the replace below hits the file it just wrote.
        tmp = self.path.with_name(self.path.name + ".tmp.npy")
        np.save(tmp, array)
        tmp.replace(self.path)


def seen_keys_path(state_path: str | Path) -> Path:
    """Location of the dedup key file that goes with a state file."""
    return Path(state_path).with_name(Path(state_path).stem + "_seen_keys.npy")
