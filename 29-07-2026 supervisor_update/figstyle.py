"""Shared plotting style for the supervisor-update and paper figures.

Palette is **Okabe–Ito** (Wong, Nature Methods 2011, "Points of view: Color
blindness") — the de facto standard qualitative palette for scientific figures.
All eight colours are distinguishable under deuteranopia, protanopia and
tritanopia, and they survive greyscale printing with distinct luminance.

Yellow (#F0E442) is deliberately excluded from the default cycle: it has too
little contrast against white for lines, markers and thin bars.

Import from any figure script:

    from figstyle import COLORS, MODEL_COLORS, apply_style, save, note
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# --------------------------------------------------------------------------- #
# Okabe–Ito
OI_BLACK = "#000000"
OI_ORANGE = "#E69F00"
OI_SKYBLUE = "#56B4E9"
OI_GREEN = "#009E73"
OI_YELLOW = "#F0E442"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"

#: default qualitative cycle, ordered by how well adjacent pairs separate
COLORS = [OI_BLUE, OI_VERMILLION, OI_GREEN, OI_ORANGE, OI_PURPLE, OI_SKYBLUE]

# --------------------------------------------------------------------------- #
# semantic names used across the figure scripts
FG = "#1A1A1A"        # text and axis labels
MUTED = "#595959"     # captions, secondary tick labels
GREY = "#808080"      # null / reference series
RULE = "#CCCCCC"      # axes spines and grid
PANEL = "#F0F0F0"     # shaded regions behind a subset of the x axis

# Backwards-compatible aliases: the earlier deck palette names still resolve, so
# existing figure scripts keep working and pick up the new colours automatically.
INK = FG
BERRY = OI_BLUE
TEAL = OI_VERMILLION
ROSE = OI_GREEN
CREAM = PANEL

#: stable colour per model so every figure agrees
MODEL_COLORS = {
    "hpg_hier": OI_BLUE,            # our model
    "wdmpnn": OI_VERMILLION,        # literature baseline
    "hpg_hier_octamer": OI_GREEN,
    "hpg_hier_junction": OI_ORANGE,
    "hpg_hier_junction1": OI_PURPLE,
    "chemarch": OI_SKYBLUE,
    "globalarch": "#7F7F7F",
    "frac": "#B0B0B0",
    "null": GREY,
}

#: marker per model, so figures stay readable in greyscale and for print
MODEL_MARKERS = {
    "hpg_hier": "o",
    "wdmpnn": "s",
    "hpg_hier_octamer": "^",
    "hpg_hier_junction": "D",
    "hpg_hier_junction1": "v",
    "chemarch": "P",
    "null": "x",
}

#: hatch per series, for bar charts that must survive greyscale reproduction
MODEL_HATCHES = {
    "hpg_hier": "",
    "wdmpnn": "///",
    "hpg_hier_octamer": "...",
    "hpg_hier_junction": "\\\\\\",
    "hpg_hier_junction1": "xxx",
}

MODEL_LABELS = {
    "hpg_hier": "HPG-hier",
    "wdmpnn": "wDMPNN",
    "hpg_hier_octamer": "octamer",
    "hpg_hier_junction": "junction n=2",
    "hpg_hier_junction1": "junction n=1",
    "chemarch": "ChemArch",
    "globalarch": "GlobalArch",
    "frac": "frac",
}

METRIC_LABELS = {
    "group_mean_r2": r"group-mean $R^2$  (chemistry)",
    "delta_r2": r"$\Delta R^2$  (architecture)",
    "ordering": "pairwise ordering accuracy",
    "overall_r2": r"overall $R^2$",
    "mae": "MAE  (eV)",
    "mean_signed_bias": "mean signed bias  (eV)",
}


def apply_style(context: str = "paper"):
    """Apply the figure style.

    context="paper"  — small type, thin rules, unbolded titles (journal figures)
    context="slide"  — larger type and heavier lines, for projection
    """
    scale = 1.0 if context == "paper" else 1.25
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 — required by most journals
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9 * scale,
        "axes.titlesize": 10 * scale,
        "axes.titleweight": "normal",
        "axes.titlecolor": FG,
        "axes.titlepad": 8,
        "axes.labelsize": 9.5 * scale,
        "axes.labelcolor": FG,
        "axes.edgecolor": "#4D4D4D",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=COLORS),

        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        "xtick.color": FG,
        "ytick.color": FG,
        "xtick.labelsize": 8.5 * scale,
        "ytick.labelsize": 8.5 * scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "lines.linewidth": 1.4 * scale,
        "lines.markersize": 4.5 * scale,
        "legend.frameon": False,
        "legend.fontsize": 8.5 * scale,
        "legend.handlelength": 1.8,
        "figure.facecolor": "white",
        "errorbar.capsize": 2.5,
    })


def save(fig, outdir: Path, name: str, formats=("png", "pdf")):
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = outdir / f"{name}.{ext}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    for p in written:
        print(f"  wrote {p.relative_to(p.parents[2]) if len(p.parents) > 2 else p}")
    return written


def note(ax, text, dy=-0.16):
    """Small caption under an axes — use for provenance and caveats.

    dy is in axes-fraction units; push it further negative when a legend
    already occupies the space directly below the axes.
    """
    ax.annotate(text, xy=(0, dy), xycoords="axes fraction",
                fontsize=7.5, color=MUTED, style="italic", va="top")
