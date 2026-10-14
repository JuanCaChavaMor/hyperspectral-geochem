"""Figures (matplotlib, static): PNG for the README and a vector PDF with the same name for the report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots, fixed order
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9, "legend.frameon": False, "axes.axisbelow": True,
})


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def mark_features(ax, features, ymax=None):
    """Vertical markers at the absorption centres."""
    for name, ((_, c, _), _) in features.items():
        ax.axvline(c, color=MUTED, lw=0.8, ls=":")
        ax.text(c, ymax if ymax is not None else ax.get_ylim()[1], name.split()[0], rotation=90, ha="right",
                va="top", fontsize=7, color=MUTED)


def image_axis(ax):
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
