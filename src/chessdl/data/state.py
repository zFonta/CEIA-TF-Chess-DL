"""Resume state for the dataset build.

A run has to survive losing the machine it started on: Colab recycles runtimes
and disconnects long sessions. Nothing durable is kept on local disk, so the
state has to come back from somewhere else.

Two different things are needed to resume, and they are handled differently:

* **How far through each dump the build got.** A few counters, kept in a small
  JSON on the Hub. It is a few hundred bytes, so pushing it after every shard
  costs nothing.

* **Which positions have already been written**, so deduplication (requirement
  3.2) holds across shards and across sessions. This is *not* stored: it is
  rebuilt by reading the ``pos_key`` column of the shards already published.
  The dataset is its own record of what it contains, which means there is no
  second file to keep in step with it, nothing to clobber, and no drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

#: Where the state file lives inside the working repository.
STATE_PATH_IN_REPO = "state.json"


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
    def load(
        cls,
        path: str | Path,
        repo_id: str | None = None,
        token: str | None = None,
    ) -> "PipelineState":
        """Load the state, falling back to the copy on the Hub.

        The local file is only a cache. On a fresh machine it does not exist, so
        the Hub copy is fetched; if there is none either, the build starts from
        the beginning, which is the right answer for a first run.
        """
        state_path = Path(path)

        if not state_path.exists() and repo_id is not None:
            from chessdl import hf

            hf.download_file(repo_id, STATE_PATH_IN_REPO, state_path, token=token)

        if not state_path.exists():
            return cls(path=state_path)

        with state_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        dumps = {
            name: DumpState(**values) for name, values in raw.get("dumps", {}).items()
        }
        return cls(path=state_path, dumps=dumps)

    def save(self, repo_id: str | None = None, token: str | None = None) -> None:
        """Write the state locally and, when given a repository, push it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dumps": {name: asdict(state) for name, state in self.dumps.items()}}

        # Write through a temporary file so an interrupted save cannot leave a
        # truncated state behind -- losing the progress record would mean
        # re-labelling everything.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp.replace(self.path)

        if repo_id is not None:
            from chessdl import hf

            hf.upload_file(
                self.path,
                repo_id,
                STATE_PATH_IN_REPO,
                token=token,
                commit_message="Update build state",
            )

    def for_dump(self, dump: str) -> DumpState:
        return self.dumps.setdefault(dump, DumpState())

    @property
    def total_positions(self) -> int:
        return sum(state.positions_written for state in self.dumps.values())


class SeenKeys:
    """The set of position keys already written to the dataset.

    Built by reading the shards rather than from a file of its own. Storing it
    separately would mean a second artefact to upload after every shard -- it
    grows into the megabytes -- and a chance for the two to disagree. The shards
    already say exactly which positions exist.
    """

    def __init__(self, keys: Iterable[int] = ()) -> None:
        self._keys: set[int] = set(keys)

    @classmethod
    def from_shards(cls, paths: Iterable[str | Path]) -> "SeenKeys":
        """Rebuild the set from the ``pos_key`` column of published shards.

        Only that one column is read, so the cost stays close to the size of the
        keys themselves rather than the whole dataset.
        """
        keys: set[int] = set()
        for path in paths:
            table = pq.read_table(Path(path), columns=["pos_key"])
            keys.update(table["pos_key"].to_pylist())
        return cls(keys)

    def __contains__(self, key: int) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def add_new(self, key: int) -> bool:
        """Add a key, returning True only the first time it is seen."""
        if key in self._keys:
            return False
        self._keys.add(key)
        return True
