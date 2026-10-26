"""Phase 11b figure: the design space (546 CR3BP-corrected candidates) and its Pareto fronts."""

import json

import matplotlib.pyplot as plt
import numpy as np

from cislunar.plotting import save_both

sw = json.load(open("data/design_sweep.json"))
halos = {h["id"]: h for h in sw["halos"]}
rows = sw["candidates"]


def pareto(pts):
    pts = sorted(pts, key=lambda p: p[0])
    out, best = [], np.inf
    for tof, dv in pts:
        if dv < best:
            out.append((tof, dv))
            best = dv
    return np.array(out)


summary = []
for hid, h in halos.items():
    for alt in (185.0, 300.0):
        sel = [r for r in rows if r["halo"] == hid and r["leo_alt_km"] == alt]
        d = min((r["total"] for r in sel if r["kind"] == "direct"), default=np.nan)
        m = min((r["total"] for r in sel if r["kind"] == "manifold"), default=np.nan)
        summary.append(
            dict(
                L=h["L"],
                Az_km=h["Az_km"],
                C=h["jacobi"],
                leo=alt,
                direct=d,
                manifold=m,
                saving=d - m,
            )
        )
print(
    f"{'pt':3} {'Az km':>7} {'C':>7} {'LEO':>4} {'direct':>7} {'manifold':>8} {'saving':>7}  (m/s)"
)
for s in summary:
    print(
        f"L{s['L']:<2} {s['Az_km']:7,.0f} {s['C']:7.4f} {s['leo']:4.0f} {s['direct'] * 1000:7,.0f} {s['manifold'] * 1000:8,.0f} {s['saving'] * 1000:7,.0f}"
    )
with open("data/design_sweep_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)


def draw(t):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, w in zip(axes, (1, 2), strict=True):
        hs = [h for h in halos.values() if h["L"] == w]
        for k, h in enumerate(hs):
            col = t["series"][k]
            sel = [r for r in rows if r["halo"] == h["id"] and r["leo_alt_km"] == 185.0]
            pts = np.array([[r["tof_days"], r["total"] * 1000] for r in sel])
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color=col, alpha=0.45)
            f = pareto(pts.tolist())
            ax.step(
                f[:, 0],
                f[:, 1],
                where="post",
                color=col,
                lw=1.6,
                label=f"Az ≈ {h['Az_km']:,.0f} km (C = {h['jacobi']:.3f})",
            )
        ax.set_xlabel("total time of flight [days]")
        ax.set_ylabel("total Δv [m/s]")
        ax.set_title(f"L{w} halos: every corrected candidate and its Pareto front")
        ax.legend(fontsize=8, loc="upper right")
    n_out = sum(r["total"] > 4.3 for r in rows if r["leo_alt_km"] == 185.0)
    for a in axes:
        a.set_ylim(3380, 4300)  # a few poor candidates up to ~5,350 m/s are off-scale
    axes[0].annotate(
        f"{n_out} candidates above 4,300 m/s not shown",
        (0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        color=t["ink2"],
    )
    fig.suptitle(
        "Design space: LEO (185 km) → halo, direct (left cluster, < 10 d) vs manifold transfers",
        color=t["ink"],
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


print("saved", [p.name for p in save_both("p11b_design_space", draw)])
