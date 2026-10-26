"""Phase 7: planar Lyapunov families around L1 and L2 (figure + saved data)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize

from cislunar import lagrange, orbits, shooting
from cislunar.constants import EARTH_MOON
from cislunar.plotting import save_both

MU = EARTH_MOON.mu
DAYS = EARTH_MOON.time_days
L_KM = EARTH_MOON.length_km
fams = {
    1: orbits.lyapunov_family(1, MU, np.linspace(2e-3, 0.16, 40)),
    2: orbits.lyapunov_family(2, MU, np.linspace(2e-3, 0.25, 40)),
}
rows = {}
for w, fam in fams.items():
    st = [orbits.stability(o) for o in fam.orbits]
    xL = lagrange.collinear_point(w, MU)
    amp = np.abs(fam.x0 - xL)
    nu = np.array([s.max_index for s in st])
    closure = np.array([orbits.closure_error(o) for o in fam.orbits])
    tracks = [shooting.full_period(o, n_points=400).X[:, :6] for o in fam.orbits]
    rows[w] = dict(amp=amp, nu=nu, tracks=tracks, C=fam.jacobi, T=fam.periods)
    np.savez_compressed(
        f"data/lyapunov_L{w}.npz",
        X0=np.array([o.X0 for o in fam.orbits]),
        period=fam.periods,
        jacobi=fam.jacobi,
        stability_index=nu,
        closure=closure,
        mu=MU,
    )
    print(
        f"L{w}: {len(fam.orbits)} orbits; amplitude up to {amp.max() * L_KM:,.0f} km; "
        f"period {fam.periods.min() * DAYS:.2f}-{fam.periods.max() * DAYS:.2f} d; "
        f"max closure {closure.max():.1e}; nu {nu.min():.0f}-{nu.max():.0f}"
    )


def draw_family(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    cmap = LinearSegmentedColormap.from_list(
        "seq",
        ["#cde2fb", "#3987e5", "#0d366b"]
        if t["surface"] == "#fcfcfb"
        else ["#104281", "#3987e5", "#cde2fb"],
    )
    allC = np.concatenate([r["C"] for r in rows.values()])
    norm = Normalize(allC.min(), allC.max())
    for ax, (w, r) in zip(axes, rows.items(), strict=True):
        for X, c in zip(r["tracks"][::2], r["C"][::2], strict=True):
            ax.plot(X[:, 0], X[:, 1], color=cmap(norm(c)), lw=1.0)
        xL = lagrange.collinear_point(w, MU)
        ax.scatter(xL, 0, marker="D", s=30, color=t["series"][1], zorder=6)
        ax.annotate(
            f"L{w}",
            (xL, 0),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
        ax.scatter(1 - MU, 0, s=50, color=t["moon"], zorder=6)
        ax.annotate(
            "Moon",
            (1 - MU, 0),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=t["ink2"],
        )
        ax.set_aspect("equal")
        ax.set_xlim(0.62, 1.45)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("x (rotating frame, Earth–Moon distances)")
        ax.set_title(
            f"L{w} family: {len(r['C'])} orbits, {r['T'].min() * DAYS:.1f}–{r['T'].max() * DAYS:.1f} days"
        )
    axes[0].set_ylabel("y")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Jacobi constant C (lower = more energy)", color=t["ink2"])
    cb.outline.set_visible(False)
    fig.suptitle(
        "Planar Lyapunov orbit families (every 2nd member shown)", color=t["ink"], fontweight="bold"
    )
    return fig


def draw_stability(t):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for w, r in rows.items():
        c = t["series"][w - 1]
        axes[0].semilogy(r["amp"] * L_KM / 1000, r["nu"], "o-", ms=3, color=c, label=f"L{w}")
        axes[1].plot(r["amp"] * L_KM / 1000, r["T"] * DAYS, "o-", ms=3, color=c, label=f"L{w}")
        T_lin = lagrange.collinear_modes(w, MU).planar_period * DAYS
        axes[1].axhline(T_lin, color=c, lw=0.8, ls=":")
    axes[0].set_xlabel("amplitude along x [thousand km]")
    axes[0].set_ylabel("stability index ν (log)")
    axes[0].set_title("All members unstable (ν ≫ 1)")
    axes[1].set_xlabel("amplitude along x [thousand km]")
    axes[1].set_ylabel("period [days]")
    axes[1].set_title("Period grows from the linear value (dotted)")
    for a in axes:
        a.legend()
    fig.tight_layout()
    return fig


print("saved", [p.name for p in save_both("p7_lyapunov_families", draw_family)])
print("saved", [p.name for p in save_both("p7_lyapunov_stability", draw_stability)])
