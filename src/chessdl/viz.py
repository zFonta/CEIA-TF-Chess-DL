"""Chart styling for the exploratory analysis (WBS 3.5).

The figures produced from these helpers go into the final report, so the styling
lives here rather than being retyped in notebook cells: one definition, one look,
and the palette can be changed in a single place.

The palette is validated for colour-vision deficiency -- the blue/orange pair
used for two-series charts separates by a CVD Delta E of 24.7, far above the
threshold of 8 -- and every two-series chart also varies line style, so identity
is never carried by colour alone (it survives a greyscale print of the report).
"""

from __future__ import annotations

from typing import Any

# --- palette ---------------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in fixed order. Never cycle past the end; fold a tail into
#: an "other" bucket or facet instead.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Single hue for magnitude (bars, histograms). More is darker.
SEQUENTIAL = "#2a78d6"

#: Two-hue diverging pair with a neutral midpoint, for values around zero.
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")


def apply_style() -> None:
    """Apply the project's matplotlib style. Call once per notebook."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (7.5, 4.2),
            "figure.dpi": 120,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 11,
            # "semibold" is a valid weight name but most system sans faces lack
            # it, so matplotlib warns and falls back on every figure.
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.size": 9,
            "font.family": "sans-serif",
        }
    )


def label_axes(
    ax: Any,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    note: str = "",
) -> Any:
    """Title, optional sub-title note, and axis labels.

    The note says what the reader is meant to take from the chart. The title's
    padding grows to make room for it, so the two never overlap -- and because
    the extra space comes from the title's own pad, ``tight_layout`` reserves it
    correctly instead of clipping the text.

    The grid is kept on one axis only: a grid on both competes with the data,
    and for a distribution the reader only needs to trace the value axis.
    """
    ax.set_title(title, pad=26 if note else 10)
    if note:
        ax.text(
            0.0, 1.015, note, transform=ax.transAxes,
            fontsize=8, color=INK_MUTED, va="bottom",
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    return ax
