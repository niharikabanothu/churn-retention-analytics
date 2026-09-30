"""
Shared Matplotlib styling so every figure in reports/figures looks consistent.

Design rules applied: one y-axis per chart, thin marks, light recessive grid,
text in neutral ink (never the series colour), a legend only when there are
two or more series, and colours taken in fixed order from a colourblind-safe
palette (blue first, orange second).
"""
import matplotlib

matplotlib.use("Agg")  # headless backend: renders to files, no display needed
import matplotlib.pyplot as plt

from src import config

C = config.COLORS


def apply_style() -> None:
    """Set global rcParams once, before any figure is drawn."""
    plt.rcParams.update({
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelcolor": C["muted"],
        "axes.edgecolor": C["grid"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,        # grid drawn behind the data
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "text.color": C["text"],
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, name: str) -> None:
    """Save a figure into reports/figures and release its memory."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
