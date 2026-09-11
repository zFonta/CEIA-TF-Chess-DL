"""Train/validation/test partitioning, by game rather than by position.

This is the one decision in the training block that cannot be fixed after the
fact. The dataset holds four positions per game, and positions from the same
game share an opening, both players and most of the board. Splitting by
*position* puts siblings on both sides of the wall: the model sees a position
from game X in training and is then scored on another position from game X,
which inflates validation and is only discovered later, when the engine plays
worse than the metrics promised.

Global deduplication by ``pos_key`` removes exact repeats but not this
correlation -- the positions really are different, they are just not
independent. So the split is by ``game_id``.

Assignment is a hash of the game id, not a shuffle of the game list. A hash is
*stable as the dataset grows*: adding shards later leaves every existing game in
the split it already had, so a model trained today can still be scored against a
test set that never contained its training games. A shuffle would reassign
everything and silently destroy that guarantee.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import blake2b

import numpy as np

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLITS: tuple[str, str, str] = (TRAIN, VAL, TEST)

#: Width of the digest used to place a game on the unit interval.
_DIGEST_BYTES = 8
_DIGEST_SCALE = float(1 << (8 * _DIGEST_BYTES))


class SplitConfigError(ValueError):
    """The requested split fractions do not describe a partition."""


@dataclass(frozen=True)
class SplitConfig:
    """How to divide the games into train, validation and test.

    ``seed`` is part of the dataset's identity: the same seed always produces the
    same partition, on any machine and in any process. Record it alongside the
    trained weights, because a model's test set is only meaningful together with
    the seed that defined it.
    """

    train: float = 0.90
    val: float = 0.05
    test: float = 0.05
    seed: int = 20260911

    def __post_init__(self) -> None:
        for name in SPLITS:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise SplitConfigError(f"{name} fraction must be in [0, 1], got {value}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise SplitConfigError(f"fractions must sum to 1, got {total}")
        if self.val <= 0.0 or self.test <= 0.0:
            raise SplitConfigError("validation and test must be non-empty")

    @property
    def bounds(self) -> tuple[float, float]:
        """Cut points on the unit interval: below the first is train, and so on."""
        return self.train, self.train + self.val


def game_position(game_id: str, seed: int) -> float:
    """Map a game id onto [0, 1) deterministically.

    Uses a keyed digest rather than :func:`hash`, which Python salts per process:
    with ``hash`` the partition would silently differ between runs, and a model
    would end up evaluated on games it had trained on.
    """
    digest = blake2b(
        f"{seed}:{game_id}".encode(), digest_size=_DIGEST_BYTES
    ).digest()
    return int.from_bytes(digest, "big") / _DIGEST_SCALE


def split_for_game(game_id: str, config: SplitConfig | None = None) -> str:
    """Return which split a single game belongs to."""
    config = config or SplitConfig()
    train_end, val_end = config.bounds
    position = game_position(game_id, config.seed)
    if position < train_end:
        return TRAIN
    if position < val_end:
        return VAL
    return TEST


def assign_splits(
    game_ids: Iterable[str], config: SplitConfig | None = None
) -> np.ndarray:
    """Label every row with its split, given that row's ``game_id``.

    Rows of the same game always get the same label, which is the entire point.
    """
    config = config or SplitConfig()
    train_end, val_end = config.bounds

    # Games repeat across rows -- four times each, by construction -- so the
    # digest is computed once per distinct game and reused.
    cache: dict[str, str] = {}
    labels = []
    for game_id in game_ids:
        label = cache.get(game_id)
        if label is None:
            position = game_position(game_id, config.seed)
            label = TRAIN if position < train_end else VAL if position < val_end else TEST
            cache[game_id] = label
        labels.append(label)
    return np.array(labels, dtype=object)


def split_masks(
    game_ids: Sequence[str], config: SplitConfig | None = None
) -> dict[str, np.ndarray]:
    """Boolean masks over the rows, one per split, ready to index a table."""
    labels = assign_splits(game_ids, config)
    return {name: labels == name for name in SPLITS}


def describe_split(masks: dict[str, np.ndarray], game_ids: Sequence[str]) -> str:
    """A short report of the partition, for the notebook to print."""
    total = len(game_ids)
    games = np.asarray(game_ids, dtype=object)
    lines = [f"{'split':<8}{'posiciones':>14}{'%':>8}{'partidas':>12}"]
    lines.append("-" * 42)
    for name in SPLITS:
        mask = masks[name]
        n = int(mask.sum())
        n_games = len(set(games[mask].tolist()))
        lines.append(f"{name:<8}{n:>14,}{n / total:>8.2%}{n_games:>12,}")
    return "\n".join(lines)


def leaked_games(masks: dict[str, np.ndarray], game_ids: Sequence[str]) -> set[str]:
    """Games appearing in more than one split -- must always be empty.

    Cheap enough to run on every training start, and the failure it guards
    against is invisible in the loss curves.
    """
    games = np.asarray(game_ids, dtype=object)
    seen: dict[str, str] = {}
    leaked: set[str] = set()
    for name in SPLITS:
        for game_id in set(games[masks[name]].tolist()):
            previous = seen.setdefault(game_id, name)
            if previous != name:
                leaked.add(game_id)
    return leaked
