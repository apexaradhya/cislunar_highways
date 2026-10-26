"""Phase 4: Hill-region panels showing the L1, L2, L3 necks opening as C decreases."""

import matplotlib.pyplot as plt
import numpy as np

from cislunar import hill, lagrange
from cislunar.constants import EARTH_MOON
from cislunar.plotting import save_both

MU = EARTH_MOON.mu
grid = hill.planar_grid(MU, n=901)
C = lagrange.jacobi_at_points(MU)
pts = lagrange.lagrange_points(MU)
found = hill.neck_opening_values(hill.planar_grid(MU), MU)
for k in ("L1", "L2", "L3"):
    print(
        f"{k}: neck opens at C = {found[k]:.8f}   C({k}) = {C[k]:.8f}   diff = {found[k] - C[k]:+.2e}"
    )

cases = [
    (C["L1"] + 0.04, "C > C(L1): trapped near Earth or Moon"),
    (C["L1"] - 0.003, "C just below C(L1): L1 neck opens"),
    (0.5 * (C["L1"] + C["L2"]) - 0.004, "C(L2) < C < C(L1)"),
    (C["L2"] - 0.006, "C < C(L2): escape past the Moon"),
    (0.5 * (C["L3"] + 3.0) + 0.004, "C < C(L3): L3 neck opens"),
    (3.0 - 1e-4, "C < 3: whole plane reachable"),
]


def draw(t):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    XX, YY = np.meshgrid(grid.x, grid.y)
    for ax, (c, label) in zip(axes.flat, cases, strict=True):
        forb = hill.forbidden_mask(grid, c).astype(float)
        ax.contourf(XX, YY, forb, levels=[0.5, 1.5], colors=[t["axis"]], alpha=0.9)
        ax.contour(XX, YY, grid.two_omega, levels=[c], colors=[t["muted"]], linewidths=0.8)
        ax.scatter([-MU], [0], s=50, color=t["earth"], zorder=5)
        ax.scatter([1 - MU], [0], s=22, color=t["moon"], zorder=5)
        for k, r in pts.items():
            ax.scatter(r[0], r[1], marker="D", s=16, color=t["series"][1], zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"{label}\nC = {c:.4f}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(False)
    fig.suptitle(
        "Where can a spacecraft go? Shaded = forbidden (2Ω < C)", color=t["ink"], fontweight="bold"
    )
    fig.tight_layout()
    return fig


print("saved", [p.name for p in save_both("p4_hill_regions", draw)])
