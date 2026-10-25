"""Shared figure styling; importing diagrams does not import model experiments."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
EXPORT_PNG = False
COLORS = dict(target="#2457a7", draft="#8b5bc7", accept="#138a72",
              reject="#d45745", muted="#5a687a")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "svg.hashsalt": "speculative-decoding-figures"})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    if EXPORT_PNG:
        fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)


def box(ax, x, y, width, height, label, color, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=.025,rounding_size=.07",
                              facecolor=color, edgecolor="none"))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            color=text_color, fontsize=11)
