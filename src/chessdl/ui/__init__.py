"""Interactive board for playing against the trained engine (WBS block 5).

:mod:`~chessdl.ui.game` decides what a click means and holds the game; it
imports no widget library and is the half under test. :mod:`~chessdl.ui.board`
draws it with ipywidgets, which is an optional dependency -- so importing this
package costs nothing in an environment without it, and the import error only
appears if someone actually asks for a board.
"""

from .game import Click, PlayState

__all__ = ["Click", "PlayState", "PlayUI"]


def __getattr__(name: str):
    """Defer the ipywidgets import until a board is really requested."""
    if name == "PlayUI":
        from .board import PlayUI

        return PlayUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
