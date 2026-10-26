"""Shared matplotlib style for every figure in the project.

Colors come from a validated colorblind-safe categorical order (blue, orange,
aqua, yellow, magenta, green, violet, red), assigned in fixed order. Every
figure is saved in a light and a dark variant: ``name.png`` and ``name_dark.png``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

FIGURE_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"

THEMES: dict[str, dict[str, object]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],
        "earth": "#2a78d6",
        "moon": "#898781",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
        "earth": "#3987e5",
        "moon": "#c3c2b7",
    },
}


def apply_theme(name: str) -> dict[str, object]:
    t = THEMES[name]
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "axes.edgecolor": t["axis"],
            "axes.labelcolor": t["ink2"],
            "axes.titlecolor": t["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "xtick.labelcolor": t["ink2"],
            "ytick.labelcolor": t["ink2"],
            "text.color": t["ink"],
            "legend.frameon": False,
            "legend.labelcolor": t["ink2"],
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.prop_cycle": matplotlib.cycler(color=t["series"]),
            "font.family": "sans-serif",
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return t


def save_both(
    name: str, draw: Callable[[dict[str, object]], Figure], outdir: Path = FIGURE_DIR
) -> list[Path]:
    """Render `draw(theme)` in light and dark and save both PNGs. Returns the paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme_name, suffix in (("light", ""), ("dark", "_dark")):
        theme = apply_theme(theme_name)
        fig = draw(theme)
        path = outdir / f"{name}{suffix}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_primaries(
    ax, mu: float, theme: dict[str, object], label: bool = True, size: float = 60
) -> None:
    """Earth and Moon markers in the rotating frame (xy projection)."""
    ax.scatter(
        [-mu],
        [0],
        s=size * 2,
        color=theme["earth"],
        zorder=5,
        edgecolor=theme["surface"],
        linewidth=1.5,
    )
    ax.scatter(
        [1 - mu],
        [0],
        s=size,
        color=theme["moon"],
        zorder=5,
        edgecolor=theme["surface"],
        linewidth=1.5,
    )
    if label:
        ax.annotate(
            "Earth",
            (-mu, 0),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color=theme["ink2"],
            fontsize=9,
        )
        ax.annotate(
            "Moon",
            (1 - mu, 0),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=theme["ink2"],
            fontsize=9,
        )
