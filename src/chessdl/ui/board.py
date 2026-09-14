"""A clickable chess board for the notebook, built out of sixty-four buttons.

The point of this file is to let someone *play* against the trained network
instead of reading a table about it, and to show, while they do, what the
network thinks: the evaluation of the position on the board and the moves the
engine ranked highest with their scores. A number in a report is an assertion;
a move you can disagree with is evidence.

**Why buttons and not a drawn board.** A real board would be HTML with drag and
drop, which needs a channel from JavaScript back to Python -- and that channel
is different in Colab (``google.colab.output``) than in Jupyter (``comm``), so a
board built on one does not run on the other. A grid of ``ipywidgets.Button``
needs no channel at all: the widget protocol already carries the click. The
board ends up being eight rows of eight buttons whose labels are the pieces and
whose background colours are the squares, and it runs anywhere ipywidgets does.

Clicking, not dragging: click the piece, then click where it goes. What each
click *means* is decided in :mod:`chessdl.ui.game`; this module only draws.
"""

from __future__ import annotations

import html
import traceback

import chess
import chess.engine

from ..engine.evaluator import Evaluator
from ..engine.search import mate_in
from .game import REFERENCE_DEPTH, Click, PlayState

#: The two square colours, and the marks laid over them. Kept together because
#: they only make sense as a set: every one of them has to stay legible under
#: the piece glyph drawn on top.
LIGHT = "#f0d9b5"
DARK = "#b58863"
SELECTED = "#7ea25b"
TARGET_LIGHT = "#b9cf8a"
TARGET_DARK = "#93ab63"
CAPTURE_LIGHT = "#e39c8a"
CAPTURE_DARK = "#c87d69"
LAST_LIGHT = "#cdd26a"
LAST_DARK = "#aaa23a"
CHECK = "#e0644f"

#: Marks an empty square the selected piece may move to. The colour alone is
#: readable, but a mark survives a colour-blind reader and a bad screen.
DOT = "·"

#: What an empty square carries as a label: a non-breaking space, never ``""``.
#:
#: This looks like a pointless flourish and is not. ``ButtonView.update`` in
#: ipywidgets 7 -- the version Colab ships -- rewrites the button's text only
#: when there is something to write::
#:
#:     if (description.length || icon.length) {
#:         this.el.textContent = "";
#:         ...
#:         this.el.appendChild(document.createTextNode(description));
#:     }
#:
#: With an empty description and no icon the whole block is skipped, so the
#: label the button already had **stays in the DOM**. A piece that moves away
#: never disappears from the square it left, and every move dot ever drawn stays
#: drawn. The board slowly fills up with pieces that are not there.
#:
#: A single space has ``length == 1``, so the branch runs, the old glyph is
#: cleared, and what gets written is invisible. ipywidgets 8 dropped the guard,
#: which is why this never shows up outside Colab.
EMPTY = " "

#: Square size in pixels. Large enough to click on a laptop trackpad without
#: making the board taller than a notebook cell.
SQUARE_PX = 48

_CSS = f"""
<style>
.chessdl-sq {{
    font-size: {SQUARE_PX - 18}px !important;
    line-height: 1 !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-family: "DejaVu Sans", "Segoe UI Symbol", sans-serif !important;
}}
.chessdl-coord {{
    color: #8a8a8a;
    font-size: 11px;
    text-align: center;
    font-family: monospace;
}}
.chessdl-panel {{ font-family: sans-serif; font-size: 13px; }}
.chessdl-panel table {{ border-collapse: collapse; }}
.chessdl-panel td, .chessdl-panel th {{ padding: 1px 8px 1px 0; text-align: left; }}
</style>
"""


def _needs_widgets():
    try:
        import ipywidgets  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "el tablero interactivo necesita ipywidgets: "
            "pip install 'chessdl[ui]'  (en Colab ya viene instalado)"
        ) from error
    return ipywidgets


#: The accent that marks each source in the panel. The bar itself stays
#: white-on-dark in both -- it means "how much of the position belongs to
#: White", and recolouring that would turn a shared scale into two.
NETWORK_INK = "#3f7fbf"
REFERENCE_INK = "#8a6fbf"


def _bar(reading, ink: str, name: str, note: str) -> str:
    """One evaluation, as a labelled bar, from White's point of view.

    Drawn from ``value`` and not from centipawns on purpose: the network's output
    is already bounded, so the bar is linear in the quantity the model actually
    produces, and a won position cannot push the bar off the end. The centipawn
    figure is printed next to it because that is the unit people read.

    Stockfish's reading goes through the *same* ``tanh(cp/400)`` the labels went
    through before training, so the two bars are the same scale. Showing one in
    centipawns and the other in [-1, 1] would make every position look like a
    disagreement.
    """
    acotado = max(-1.0, min(1.0, reading.value))
    blancas = (acotado + 1.0) / 2.0 * 100.0
    if reading.is_mate:
        lectura = f"mate en {abs(reading.mate)}"
    else:
        lectura = f"{acotado:+.3f} &nbsp; {'+' if reading.cp >= 0 else ''}{reading.cp:.0f} cp"
    return (
        f'<div style="margin:8px 0 0 0;">'
        f'<div style="font-size:12px;margin-bottom:2px;">'
        f'<span style="display:inline-block;width:9px;height:9px;background:{ink};'
        f'border-radius:2px;margin-right:6px;"></span>'
        f"<b>{name}</b> <span style='color:#888;'>{note}</span></div>"
        f'<div style="height:13px;width:100%;background:#3b3b3b;'
        f'border:1px solid {ink};box-sizing:border-box;">'
        f'<div style="height:100%;width:{blancas:.1f}%;background:#f2f2f2;"></div>'
        f"</div>"
        f'<div style="font-family:monospace;font-size:12px;margin-top:2px;">'
        f"{lectura}</div></div>"
    )


def _gap(network, reference) -> str:
    """The distance between the two readings, which is the interesting number.

    It is one term of the error the memoir reports as an average: the network's
    test RMSE of 0,2511 is the root mean of exactly this, over the test split.
    Seeing it on the position in front of you is what turns that figure from a
    number in a table into something with a size.

    A mate is left out rather than turned into a difference: Stockfish pins it
    to the clip bound and the network has never seen one, so subtracting them
    would produce a large number that measures the clipping, not the model.
    """
    if reference is None:
        return ""
    if reference.is_mate:
        return (
            '<div style="font-size:12px;color:#888;padding-top:4px;">'
            "Stockfish ve un mate forzado, que queda fuera de la escala de la "
            "red: el dataset no tiene posiciones asi.</div>"
        )
    diferencia = abs(network.value - reference.value)
    if diferencia < 0.10:
        juicio, color = "coinciden", "#2e7d32"
    elif diferencia < 0.30:
        juicio, color = "se parecen", "#946200"
    else:
        juicio, color = "no coinciden", "#b03a2e"
    return (
        f'<div style="font-size:12px;padding-top:6px;">'
        f"Diferencia: <code>{diferencia:.3f}</code> "
        f'<span style="color:{color};">({juicio})</span> '
        f'<span style="color:#888;">&nbsp;el RMSE de test de la red es 0,251</span>'
        f"</div>"
    )


class PlayUI:
    """An interactive board over a trained evaluator.

    ``evaluators`` maps a name to an :class:`~chessdl.engine.evaluator.Evaluator`.
    Passing more than one puts a selector on the panel, which is the interesting
    case for this project: the two architectures came out a tie on RMSE and a
    tie on centipawn loss at depth two, and switching between them mid-game is
    the only way to get any feel for what that tie means over a board.
    """

    def __init__(
        self,
        evaluators: Evaluator | dict[str, Evaluator],
        depth: int = 2,
        human_color: chess.Color = chess.WHITE,
        fen: str | None = None,
        positions: dict[str, str] | None = None,
        reference: "chess.engine.SimpleEngine | None" = None,
        reference_depth: int = REFERENCE_DEPTH,
    ) -> None:
        self.widgets = _needs_widgets()
        self.positions = dict(positions or {})
        if isinstance(evaluators, dict):
            self.evaluators = dict(evaluators)
        else:
            self.evaluators = {"Red": evaluators}
        primero = next(iter(self.evaluators))

        self.state = PlayState(
            evaluator=self.evaluators[primero],
            human_color=human_color,
            depth=depth,
            board=chess.Board(fen) if fen else chess.Board(),
            reference=reference,
            reference_depth=reference_depth,
        )
        self._build()
        self._refresh()
        if not self.state.human_turn and not self.state.finished:
            self._engine_turn()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        w = self.widgets

        self.squares: dict[int, "w.Button"] = {}
        for square in chess.SQUARES:
            boton = w.Button(
                description=EMPTY,
                layout=w.Layout(
                    width=f"{SQUARE_PX}px", height=f"{SQUARE_PX}px",
                    margin="0px", padding="0px",
                ),
                style=w.ButtonStyle(button_color=LIGHT),
            )
            boton.add_class("chessdl-sq")
            boton.on_click(self._on_square(square))
            self.squares[square] = boton

        self.board_box = w.GridBox(
            children=[],
            layout=w.Layout(
                grid_template_columns=f"repeat(8, {SQUARE_PX}px)",
                grid_gap="0px",
                border="2px solid #8b6f47",
                width="max-content",
            ),
        )
        self.files = w.HTML()
        self.files.add_class("chessdl-coord")

        # --- panel
        self.status = w.HTML()
        self.status.add_class("chessdl-panel")
        self.ranking = w.HTML()
        self.ranking.add_class("chessdl-panel")
        self.moves = w.HTML()
        self.moves.add_class("chessdl-panel")
        # An HTML widget rather than an `Output`: `Output` captures through the
        # running kernel, so outside one -- a test, a headless render -- it
        # captures nothing and the traceback goes to a stderr nobody is reading.
        # Assigning to `.value` works the same everywhere.
        self.errors = w.HTML()

        self.depth_picker = w.ToggleButtons(
            options=[("1 ply", 1), ("2 plies", 2), ("3 plies", 3)],
            value=self.state.depth,
            description="Profundidad:",
            style={"description_width": "initial"},
        )
        self.depth_picker.observe(self._on_depth, names="value")

        self.model_picker = w.ToggleButtons(
            options=list(self.evaluators),
            value=next(iter(self.evaluators)),
            description="Modelo:",
            style={"description_width": "initial"},
        )
        self.model_picker.observe(self._on_model, names="value")
        self.model_picker.layout.display = (
            "" if len(self.evaluators) > 1 else "none"
        )

        self.promo_box = w.HBox([])
        self.promo_box.layout.display = "none"
        self._promo_buttons = []
        for pieza, glifo in (
            (chess.QUEEN, "♕"), (chess.ROOK, "♖"),
            (chess.BISHOP, "♗"), (chess.KNIGHT, "♘"),
        ):
            boton = w.Button(
                description=glifo,
                layout=w.Layout(width="44px", height="44px", margin="0 4px 0 0"),
                style=w.ButtonStyle(button_color="#e8e8e8"),
            )
            boton.add_class("chessdl-sq")
            boton.on_click(self._on_promote(pieza))
            self._promo_buttons.append(boton)

        botones = {
            "Nueva": self._on_new,
            "Deshacer": self._on_undo,
            "Girar": self._on_flip,
            "Sugerir": self._on_hint,
            "Jugar por mi": self._on_play_for_me,
        }
        self.buttons = []
        for etiqueta, handler in botones.items():
            boton = w.Button(
                description=etiqueta,
                layout=w.Layout(width="104px", margin="0 4px 4px 0"),
            )
            boton.on_click(handler)
            self.buttons.append(boton)

        self.fen_field = w.Text(
            value="", placeholder="pega un FEN y apreta Cargar",
            layout=w.Layout(width="330px"),
        )
        self.load_button = w.Button(description="Cargar", layout=w.Layout(width="80px"))
        self.load_button.on_click(self._on_load)
        self.side_picker = w.ToggleButtons(
            options=[("Juego con blancas", chess.WHITE), ("Juego con negras", chess.BLACK)],
            value=self.state.human_color,
            style={"description_width": "initial"},
        )

        self.preset_buttons = []
        for etiqueta, posicion in self.positions.items():
            boton = w.Button(
                description=etiqueta,
                layout=w.Layout(width="auto", margin="0 4px 4px 0"),
                tooltip=posicion,
            )
            boton.on_click(self._on_preset(posicion))
            self.preset_buttons.append(boton)
        presets = w.HBox(self.preset_buttons, layout=w.Layout(flex_flow="row wrap"))
        presets.layout.display = "" if self.preset_buttons else "none"

        panel = w.VBox([
            self.status,
            self.promo_box,
            self.ranking,
            w.HBox(self.buttons, layout=w.Layout(flex_flow="row wrap")),
            self.depth_picker,
            self.model_picker,
            self.side_picker,
            presets,
            w.HBox([self.fen_field, self.load_button]),
            self.moves,
        ], layout=w.Layout(width="420px", margin="0 0 0 18px"))

        self.root = w.VBox([
            w.HTML(_CSS),
            w.HBox(
                [w.VBox([self.board_box, self.files]), panel],
                layout=w.Layout(align_items="flex-start"),
            ),
            self.errors,
        ])

    # --------------------------------------------------------------- handlers

    def _guarded(self, fn):
        """Run a callback and show anything it raises.

        Without this an exception inside a widget callback vanishes: the
        notebook shows nothing and the board simply stops responding, which
        looks like a hung kernel and is the single most confusing way for an
        interactive cell to fail.
        """
        def envuelto(_button=None, _change=None):
            try:
                self.errors.value = ""
                fn()
            except Exception as error:  # noqa: BLE001 - show it, do not classify it
                detalle = html.escape(traceback.format_exc())
                self.errors.value = (
                    f'<div style="color:#b03a2e;font-family:sans-serif;font-size:13px;'
                    f'padding:6px 0;"><b>{html.escape(str(error) or type(error).__name__)}'
                    f"</b><details><summary style='cursor:pointer;color:#777;'>detalle"
                    f"</summary><pre style='font-size:11px;'>{detalle}</pre>"
                    f"</details></div>"
                )
        return envuelto

    def _on_square(self, square: int):
        def handler():
            resultado = self.state.click(square)
            if resultado is Click.IGNORED:
                return
            self._refresh()
            if resultado is Click.MOVED:
                self._engine_turn()
        return self._guarded(handler)

    def _on_promote(self, piece_type: int):
        def handler():
            if self.state.promote(piece_type) is Click.MOVED:
                self._refresh()
                self._engine_turn()
        return self._guarded(handler)

    def _on_depth(self, change) -> None:
        self.state.depth = change["new"]

    def _on_model(self, change) -> None:
        self.state.evaluator = self.evaluators[change["new"]]
        # The cached readings belong to the model that is being replaced, and a
        # panel showing one architecture's evaluation under the other's name
        # would be worse than showing none.
        self.state.invalidate_readings()
        self.state.last_search = None
        self._refresh()

    def _on_new(self, _button=None) -> None:
        def handler():
            self.state.reset(human_color=self.side_picker.value)
            self._refresh()
            if not self.state.human_turn and not self.state.finished:
                self._engine_turn()
        self._guarded(handler)()

    def _on_undo(self, _button=None) -> None:
        self._guarded(lambda: (self.state.undo(), self._refresh()))()

    def _on_flip(self, _button=None) -> None:
        def handler():
            self.state.flipped = not self.state.flipped
            self._refresh()
        self._guarded(handler)()

    def _on_hint(self, _button=None) -> None:
        def handler():
            self._say("Pensando una sugerencia...")
            sugerencia = self.state.hint()
            if sugerencia is not None:
                self.state.last_search = sugerencia
            self._refresh()
        self._guarded(handler)()

    def _on_play_for_me(self, _button=None) -> None:
        def handler():
            if not self.state.human_turn:
                return
            self._say("Pensando...")
            self.state.engine_move()
            self._refresh()
            self._engine_turn()
        self._guarded(handler)()

    def _on_load(self, _button=None) -> None:
        self._guarded(lambda: self._load(self.fen_field.value.strip()))()

    def _on_preset(self, fen: str):
        return self._guarded(lambda: self._load(fen))

    def _load(self, fen: str) -> None:
        """Start from a position, on the side that position is waiting for.

        The side is taken from the FEN rather than from the picker: a puzzle is
        posed to whoever is to move in it, and loading "mate in one, White to
        play" only for the engine to take the mate would be a strange demo.
        """
        if not fen:
            return
        try:
            tablero = chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"Ese FEN no se puede leer: {error}") from error
        self.fen_field.value = fen
        self.side_picker.value = tablero.turn
        self.state.reset(fen=fen, human_color=tablero.turn)
        self._refresh()

    def _engine_turn(self) -> None:
        if self.state.finished or self.state.human_turn:
            return
        self._say("Pensando...")
        self.state.engine_move()
        self._refresh()

    # ----------------------------------------------------------------- render

    def _say(self, mensaje: str) -> None:
        """Put a line on the panel *now*, before a slow call blocks the kernel.

        Widget assignments reach the browser as they happen rather than when the
        callback returns, so this is visible while the search runs.
        """
        self.status.value = f'<div style="padding:4px 0;"><b>{mensaje}</b></div>'

    def _order(self) -> list[int]:
        """Squares in drawing order: top-left first, from the chosen side."""
        filas = range(7, -1, -1) if not self.state.flipped else range(8)
        columnas = range(8) if not self.state.flipped else range(7, -1, -1)
        return [chess.square(f, r) for r in filas for f in columnas]

    def _refresh(self) -> None:
        orden = self._order()
        destinos = self.state.destinations()
        ultimas = set()
        if self.state.last_move is not None:
            ultimas = {self.state.last_move.from_square, self.state.last_move.to_square}
        rey_en_jaque = (
            self.state.board.king(self.state.board.turn)
            if self.state.board.is_check() else None
        )

        for square in chess.SQUARES:
            pieza = self.state.board.piece_at(square)
            clara = (chess.square_rank(square) + chess.square_file(square)) % 2 == 1
            boton = self.squares[square]
            glifo = chess.UNICODE_PIECE_SYMBOLS[pieza.symbol()] if pieza else EMPTY

            if square == rey_en_jaque:
                color = CHECK
            elif square == self.state.selected:
                color = SELECTED
            elif square in destinos:
                if pieza is not None:
                    color = CAPTURE_LIGHT if clara else CAPTURE_DARK
                else:
                    color = TARGET_LIGHT if clara else TARGET_DARK
                    glifo = DOT
            elif square in ultimas:
                color = LAST_LIGHT if clara else LAST_DARK
            else:
                color = LIGHT if clara else DARK

            boton.description = glifo
            boton.style.button_color = color

        self.board_box.children = tuple(self.squares[s] for s in orden)
        columnas = "abcdefgh" if not self.state.flipped else "hgfedcba"
        self.files.value = "".join(
            f'<span style="display:inline-block;width:{SQUARE_PX}px;">{c}</span>'
            for c in columnas
        )

        self._render_panel()

    def _render_panel(self) -> None:
        red, referencia = self.state.readings()
        partes = [
            f'<div style="padding:2px 0;"><b>{self.state.status()}</b></div>',
            _bar(red, NETWORK_INK, self.model_picker.value, "(la red entrenada)"),
        ]
        if referencia is not None:
            partes.append(_bar(
                referencia, REFERENCE_INK, "Stockfish",
                f"(profundidad {self.state.reference_depth}, la del dataset)",
            ))
            partes.append(_gap(red, referencia))
        else:
            partes.append(
                '<div style="font-size:12px;color:#888;padding-top:4px;">'
                "Sin Stockfish: no hay con que comparar. Pasale un motor en "
                "<code>reference=</code> para ver las dos lecturas.</div>"
            )
        partes.append(
            '<div style="font-size:11px;color:#888;padding-top:4px;">'
            "Las dos desde las blancas y en la misma escala "
            "<code>tanh(cp/400)</code>.</div>"
        )
        if self.state.finished:
            partes.append(
                f'<div style="color:#777;">Resultado: '
                f"{self.state.board.result(claim_draw=True)}</div>"
            )
        self.status.value = "".join(partes)

        self.promo_box.children = tuple(self._promo_buttons)
        self.promo_box.layout.display = (
            "" if self.state.pending_promotion is not None else "none"
        )

        self.ranking.value = self._ranking_html()
        self.moves.value = self._moves_html()

    def _ranking_html(self) -> str:
        resultado = self.state.last_search
        if resultado is None:
            return (
                '<div style="color:#777;padding:4px 0;">Las jugadas que considera el '
                "motor aparecen despues de su jugada, o apretando <b>Sugerir</b>.</div>"
            )

        # The ranking belongs to the position the search ran from, not to the
        # one on the board now, so SAN has to be read on a board rewound by the
        # move that was played from it.
        tablero = self.state.board.copy()
        if tablero.move_stack and tablero.peek() == resultado.move:
            tablero.pop()

        filas = []
        for move, value in resultado.ranked[:5]:
            if move not in tablero.legal_moves:      # a hint, not the move played
                continue
            mate = mate_in(float(value))
            if mate is not None:
                # Mates score outside the network's range, so clamping one to
                # +1.0000 would show it as an ordinary winning evaluation and
                # hide the one thing worth seeing.
                lectura = f"#{mate}"
            else:
                lectura = f"{max(-1.0, min(1.0, float(value))):+.4f}"
            marca = " &#9733;" if move == resultado.move else ""
            filas.append(
                f"<tr><td><code>{tablero.san(move)}</code>{marca}</td>"
                f"<td><code>{lectura}</code></td></tr>"
            )
        if not filas:
            return ""

        return (
            f'<div style="padding:4px 0;"><b>Lo que ve el motor</b> '
            f'<span style="color:#777;">({resultado.depth} '
            f'{"ply" if resultado.depth == 1 else "plies"}, '
            f"{resultado.leaves:,} hojas, {resultado.seconds:.2f} s)</span></div>"
            f"<table>{''.join(filas)}</table>"
            f'<div style="color:#777;font-size:11px;padding-top:2px;">'
            "Valores desde el lado que mueve en esa posicion.</div>"
        )

    def _moves_html(self) -> str:
        filas = self.state.move_pairs()
        if not filas:
            return ""
        celdas = "".join(
            f"<tr><td style='color:#999'>{n}.</td>"
            f"<td><code>{blancas}</code></td><td><code>{negras}</code></td></tr>"
            for n, blancas, negras in filas
        )
        return (
            '<div style="padding:6px 0 2px 0;"><b>Partida</b></div>'
            f'<div style="max-height:200px;overflow-y:auto;"><table>{celdas}</table></div>'
        )

    # ------------------------------------------------------------------- show

    def _ipython_display_(self):  # pragma: no cover - display protocol
        from IPython.display import display

        display(self.root)

    def show(self):
        """Display the board. Also happens on its own if the cell ends on it."""
        from IPython.display import display

        display(self.root)
        return self
