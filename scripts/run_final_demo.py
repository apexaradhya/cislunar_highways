"""One-command demo: recompute the headline results from scratch in a few minutes.

Lagrange points -> L2 halo (bifurcation + pseudo-arclength continuation) -> monodromy and
manifold tubes -> direct and low-energy LEO->halo transfers (reduced search: fewer
candidates and no Nelder-Mead refinement) -> results table and figures in docs/figures/demo_*.

The full-resolution run (scripts/run_phase11_transfer.py, ~20 min) is the source of the
published numbers; this demo prints both side by side so any gap from the reduced search is visible.
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from cislunar import lagrange, manifolds, orbits, shooting, transfer
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.plotting import draw_primaries, save_both

T0 = time.perf_counter()
U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
Z0_TARGET = 0.053  # same L2 halo as the full run (Az ~ 20,000 km)


def step(msg):
    print(f"[{time.perf_counter() - T0:6.1f} s] {msg}", flush=True)


step(f"mu = {MU:.10f}; L = {L:,.0f} km; T = {D:.4f} d; V = {U.velocity_km_s:.4f} km/s")
pts, C = lagrange.lagrange_points(MU), lagrange.jacobi_at_points(MU)
for k in ("L1", "L2", "L3"):
    m = lagrange.collinear_modes(int(k[1]), MU)
    step(
        f"{k}: x = {pts[k][0]:.10f}, C = {C[k]:.8f}, lambda = {m.lam:.4f}, planar period {m.planar_period * D:.2f} d"
    )

bif = orbits.find_halo_bifurcation(2, MU, 0.020, 0.030)
step(f"L2 halo bifurcation: x0 = {bif.X0[0]:.8f}, C = {bif.jacobi:.8f}")
fam = orbits.pseudo_arclength_family(
    orbits.halo_from_bifurcation(bif, 2e-3),
    200,
    0.004,
    MU,
    direction=np.array([0, 1, 0]),
    stop=lambda o: o.X0[2] > Z0_TARGET + 0.004,
)
halo = shooting.correct_symmetric(
    min(fam.orbits, key=lambda o: abs(o.X0[2] - Z0_TARGET)).X0, MU, free=(0, 4), targets=(3, 5)
)
st = orbits.stability(halo)
step(
    f"halo: z0 = {halo.X0[2] * L:,.0f} km, period {halo.period * D:.3f} d, C = {halo.jacobi:.6f}, "
    f"closure {orbits.closure_error(halo):.1e}, lambda_u = {st.lambda_unstable:.1f}"
)

s = manifolds.sample_orbit(halo, 40)
tubes = {
    (k, sg): manifolds.generate_tube(
        halo, k, sg, 8.0, r_earth=R_EARTH_KM / L, r_moon=R_MOON_KM / L, samples=s
    )
    for k in "su"
    for sg in (1.0, -1.0)
}
step("manifold tubes generated (4 x 40 strands, 35 days)")

cands = transfer.design_candidates(
    halo, U, leo_alt_km=185.0, n_correct=40, n_direct=6, n_strands=60, duration=40.0
)
best = {
    k: min((c for c in cands if c["kind"] == k), key=lambda c: c["total"])
    for k in ("direct", "manifold")
}
pc = transfer.patched_conic_llo(U, 185.0, 100.0)
step(f"{len(cands)} transfer candidates corrected in the CR3BP")

full = (
    json.load(open("data/transfer_results.json"))
    if __import__("os").path.exists("data/transfer_results.json")
    else None
)
rows = [
    ("Patched conic -> 100 km LLO (estimate)", pc.dv_tli_km_s, pc.dv_loi_km_s, pc.tof_days, None),
    (
        "Direct -> L2 halo (CR3BP)",
        best["direct"]["dv1"],
        best["direct"]["dv2"],
        best["direct"]["tof_days"],
        full and full["direct"]["dv_total_km_s"],
    ),
    (
        "Low-energy (manifold) -> L2 halo (CR3BP)",
        best["manifold"]["dv1"],
        best["manifold"]["dv2"],
        best["manifold"]["tof_days"],
        full and full["manifold"]["dv_total_km_s"],
    ),
]
print("\n" + "-" * 104)
print(
    f"{'design':44} {'TLI m/s':>9} {'2nd burn':>9} {'total m/s':>10} {'TOF days':>9} {'full run total':>16}"
)
print("-" * 104)
for name, a, b, tof, fr in rows:
    print(
        f"{name:44} {a * 1000:9,.0f} {b * 1000:9,.0f} {(a + b) * 1000:10,.0f} {tof:9.1f} "
        f"{(f'{fr * 1000:,.0f}' if fr else '-'):>16}"
    )
print("-" * 104)
print(
    "Demo uses a reduced search (40 corrected manifold candidates, no Nelder-Mead refinement); refinement"
)
print(
    "in the full run lowers the low-energy total by roughly 10 m/s. Full details: docs/results.md.\n"
)


def draw(t):
    fig, ax = plt.subplots(figsize=(9, 6))
    for (k, _sg), tb in tubes.items():
        col = t["series"][0] if k == "s" else t["series"][1]
        for trj in tb[::2]:
            ax.plot(trj.X[:, 0], trj.X[:, 1], color=col, lw=0.5, alpha=0.5)
    H = shooting.full_period(halo, n_points=300).X
    ax.plot(H[:, 0], H[:, 1], color=t["ink"], lw=2)
    draw_primaries(ax, MU, t)
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 1.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(f"Demo: L2 halo (Az ≈ {halo.X0[2] * L:,.0f} km) and its manifold tubes")
    return fig


save_both("demo_tubes", draw)
step("saved docs/figures/demo_tubes.png; done")
