"""Phase 3: Lagrange point table and figure (system view + zoom near the Moon)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from cislunar import lagrange
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.plotting import save_both

MU = EARTH_MOON.mu
L = EARTH_MOON.length_km
pts = lagrange.lagrange_points(MU)
C = lagrange.jacobi_at_points(MU)

print(f"{'pt':3} {'x':>16} {'y':>14} {'C':>16}  dist from Moon [km]")
for k, r in pts.items():
    d = np.linalg.norm(r - [1 - MU, 0, 0]) * L
    print(f"{k:3} {r[0]:16.12f} {r[1]:14.10f} {C[k]:16.12f}  {d:12.0f}")
print(f"Routh: 27 mu (1-mu) = {27 * MU * (1 - MU):.6f} < 1 -> L4/L5 linearly stable")


def draw(t):
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    s = t["series"]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th) - MU, np.sin(th), color=t["grid"], lw=1, zorder=1)  # Moon's path
    ax.plot(
        [-MU, pts["L4"][0], 1 - MU, pts["L5"][0], -MU],
        [0, pts["L4"][1], 0, pts["L5"][1], 0],
        color=t["axis"],
        lw=1,
        ls="--",
        zorder=1,
    )
    # True scale bodies (tiny!) plus a marker so they are visible.
    ax.add_patch(Circle((-MU, 0), R_EARTH_KM / L, color=t["earth"], zorder=4))
    ax.add_patch(Circle((1 - MU, 0), R_MOON_KM / L, color=t["moon"], zorder=4))
    ax.scatter([-MU], [0], s=110, color=t["earth"], zorder=4)
    ax.scatter([1 - MU], [0], s=45, color=t["moon"], zorder=4)
    ax.annotate(
        "Earth", (-MU, 0), xytext=(0, -18), textcoords="offset points", ha="center", color=t["ink2"]
    )
    ax.annotate(
        "Moon",
        (1 - MU, 0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color=t["ink2"],
    )
    offs = {"L1": (-4, 10), "L2": (6, 10), "L3": (-14, 10), "L4": (8, 6), "L5": (8, -14)}
    for k, r in pts.items():
        color = s[1] if k in ("L1", "L2", "L3") else s[2]
        ax.scatter(
            r[0], r[1], s=70, marker="D", color=color, zorder=6, edgecolor=t["surface"], lw=1.5
        )
        ax.annotate(
            k,
            (r[0], r[1]),
            xytext=offs[k],
            textcoords="offset points",
            color=t["ink"],
            fontweight="bold",
            ha="right" if k == "L1" else "left",
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("x  [Earth–Moon distance = 384,400 km]")
    ax.set_ylabel("y")
    ax.set_title("The five Lagrange points of the Earth–Moon system")
    ax.text(
        0.01,
        0.01,
        "◆ orange: collinear (unstable)   ◆ aqua: triangular (stable for μ < 0.0385)",
        transform=ax.transAxes,
        color=t["ink2"],
        fontsize=8.5,
    )

    # Inset: true-scale zoom around the Moon with L1/L2.
    ins = ax.inset_axes([0.03, 0.68, 0.36, 0.28])
    ins.add_patch(Circle((1 - MU, 0), R_MOON_KM / L, color=t["moon"]))
    for k in ("L1", "L2"):
        ins.scatter(pts[k][0], 0, marker="D", s=40, color=s[1], zorder=5)
        km = abs(pts[k][0] - (1 - MU)) * L
        ins.annotate(
            f"{k}\n{km:,.0f} km",
            (pts[k][0], 0),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=t["ink2"],
        )
    ins.set_xlim(0.80, 1.20)
    ins.set_ylim(-0.07, 0.09)
    ins.set_aspect("equal")
    ins.tick_params(labelsize=7)
    ins.set_title("zoom near the Moon (to scale)", fontsize=8, color=t["ink2"], fontweight="normal")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(t["axis"])
    return fig


print("saved", [p.name for p in save_both("p3_lagrange_points", draw)])
