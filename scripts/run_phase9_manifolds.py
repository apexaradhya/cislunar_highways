"""Phase 9: invariant-manifold tubes of L1/L2 halos, asymptotic-rate check, eps sensitivity."""

import matplotlib.pyplot as plt
import numpy as np

from cislunar import cr3bp, lagrange, manifolds, shooting
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.plotting import save_both
from cislunar.shooting import correct_symmetric

MU = EARTH_MOON.mu
L = EARTH_MOON.length_km
D = EARTH_MOON.time_days
Z0 = 0.05  # ~19,000 km vertical amplitude at the far-side crossing: a mid-size halo
N_TRAJ = 60
DURATION = 8.0  # ~35 days: ~2.5 weeks to leave the halo at eps = 1e-5, then the highway
EPS = manifolds.DEFAULT_EPS
R_E, R_M = R_EARTH_KM / L, R_MOON_KM / L


def halo_near(w, z0):
    d = np.load(f"data/halo_L{w}_north.npz")
    i = int(np.argmin(np.abs(d["X0"][:, 2] - z0)))
    return correct_symmetric(d["X0"][i], MU, free=(0, 4), targets=(3, 5))


tubes, halos, tracks = {}, {}, {}
for w in (1, 2):
    o = halo_near(w, Z0)
    halos[w] = o
    s = manifolds.sample_orbit(o, N_TRAJ)
    tracks[w] = shooting.full_period(o, n_points=400).X[:, :3]
    for kind in "su":
        for sign in (1.0, -1.0):
            tubes[(w, kind, sign)] = manifolds.generate_tube(
                o, kind, sign, DURATION, eps=EPS, r_earth=R_E, r_moon=R_M, samples=s
            )
    dC = max(
        np.max(np.abs(cr3bp.jacobi_constant(tr.X, MU) - o.jacobi))
        for k, tb in tubes.items()
        if k[0] == w
        for tr in tb
    )
    print(
        f"L{w} halo: z0={o.X0[2]:.4f} ({o.X0[2] * L:,.0f} km) T={o.period * D:.2f} d C={o.jacobi:.6f} "
        f"lambda_u={s.directions.lambda_u:.1f}; max |C - C_orbit| over {4 * N_TRAJ} trajectories = {dC:.1e}"
    )
    for k, tb in tubes.items():
        if k[0] == w:
            stat = {
                st: sum(tr.status == st for tr in tb) for st in ("time", "earth", "moon", "escape")
            }
            print(
                f"   {'stable' if k[1] == 's' else 'unstable'} {'+' if k[2] > 0 else '-'}: {stat}"
            )

STYLE = {("s", 1.0): (0, 0.45), ("s", -1.0): (0, 0.9), ("u", 1.0): (1, 0.45), ("u", -1.0): (1, 0.9)}
# Branches are named by the direction of the initial displacement along x. Where that leads
# differs by orbit: from L1, -x goes to Earth's realm and +x to the Moon's; from L2, -x goes
# to the Moon's realm and +x to the outer (exterior) realm.
LABEL = {
    ("s", 1.0): "stable, +x branch",
    ("s", -1.0): "stable, −x branch",
    ("u", 1.0): "unstable, +x branch",
    ("u", -1.0): "unstable, −x branch",
}
BRANCH_NOTE = (
    "−x branch: L1 → Earth realm, L2 → Moon realm\n+x branch: L1 → Moon realm, L2 → exterior"
)


def bodies_3d(ax, t):
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    for c, r, col in ((-MU, R_E, t["earth"]), (1 - MU, R_M, t["moon"])):
        ax.plot_surface(
            c + r * np.cos(u) * np.sin(v),
            r * np.sin(u) * np.sin(v),
            r * np.cos(v),
            color=col,
            shade=False,
        )


def draw_hero(t):
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(t["surface"])
    for (_w, kind, sign), tb in tubes.items():
        ci, alpha = STYLE[(kind, sign)]
        for tr in tb[::2]:
            ax.plot(
                tr.X[:, 0], tr.X[:, 1], tr.X[:, 2], color=t["series"][ci], lw=0.5, alpha=alpha * 0.6
            )
    for w in (1, 2):
        X = tracks[w]
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color=t["ink"], lw=2.0)
    bodies_3d(ax, t)
    for k, r in lagrange.lagrange_points(MU).items():
        if k in ("L1", "L2"):
            ax.scatter(*r, marker="D", color=t["series"][3], s=25)
    for key, lab in LABEL.items():
        ci, alpha = STYLE[key]
        ax.plot([], [], color=t["series"][ci], alpha=alpha, lw=2, label=lab)
    ax.plot([], [], color=t["ink"], lw=2, label="halo orbits")
    ax.legend(loc="upper left", fontsize=9)
    ax.text2D(0.01, 0.02, BRANCH_NOTE, transform=ax.transAxes, fontsize=8.5, color=t["ink2"])
    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_zlim(-0.25, 0.25)
    ax.set_box_aspect((0.9, 0.9, 0.5))
    ax.view_init(elev=28, azim=-70)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=8)
    ax.set_title(
        "Gravitational highways: stable and unstable manifold tubes of L1 and L2 halo orbits",
        color=t["ink"],
    )
    return fig


def draw_xy(t):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for (_w, kind, sign), tb in tubes.items():
        ci, alpha = STYLE[(kind, sign)]
        for tr in tb[::2]:
            ax.plot(tr.X[:, 0], tr.X[:, 1], color=t["series"][ci], lw=0.5, alpha=alpha * 0.7)
    for w in (1, 2):
        ax.plot(tracks[w][:, 0], tracks[w][:, 1], color=t["ink"], lw=2)
    ax.scatter([1 - MU], [0], s=40, color=t["moon"], zorder=5)
    for key, lab in LABEL.items():
        ci, alpha = STYLE[key]
        ax.plot([], [], color=t["series"][ci], alpha=alpha, lw=2, label=lab)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.01, 0.02, BRANCH_NOTE, transform=ax.transAxes, fontsize=8.5, color=t["ink2"])
    ax.set_aspect("equal")
    ax.set_xlim(0.45, 1.6)
    ax.set_ylim(-0.42, 0.42)
    ax.set_xlabel("x (rotating frame)")
    ax.set_ylabel("y")
    ax.set_title(
        f"Manifold tubes, xy projection ({DURATION * D:.0f} days of flight, ε = {EPS:g} ≈ {EPS * L:.2f} km)"
    )
    return fig


# --- asymptotic behaviour (L2 halo) ---
o2 = halos[2]
s2 = manifolds.sample_orbit(o2, 40)
lam_u, lam_s = s2.directions.lambda_u, s2.directions.lambda_s
# Start at 1e-12 and stop after 3 periods: with 1/lambda_s ~ 1000 the distance then reaches
# ~1e-3, still in the linear regime (a 4th period would reach ~1 and turn nonlinear).
back = manifolds.stroboscopic_distances(s2, o2, 5, "s", 1e-12, 3, backward=True)
fwd = manifolds.stroboscopic_distances(s2, o2, 5, "s", 1e-6, 3)
# continuous distance to the orbit over one period forward (distance to the moving reference point)
Xs = manifolds.manifold_states(s2, "s", 1.0, 1e-6)[5]
from cislunar.integrators import dopri5  # noqa: E402

tt = np.linspace(0, o2.period, 300)
orb = dopri5(
    cr3bp.make_rhs(MU),
    s2.t[5],
    s2.t[5] + o2.period,
    s2.X[5],
    rtol=1e-13,
    atol=1e-13,
    t_eval=s2.t[5] + tt,
).X
pert = dopri5(cr3bp.make_rhs(MU), 0, o2.period, Xs, rtol=1e-13, atol=1e-13, t_eval=tt).X
dist = np.linalg.norm(pert - orb, axis=1)
print(
    f"L2 halo lambda_s = {lam_s:.6f}; forward 1-period ratio = {fwd[1] / fwd[0]:.6f}; "
    f"backward ratios = {np.round(back[1:] / back[:-1], 3)} (1/lambda_s = {1 / lam_s:.3f})"
)


def draw_rate(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    k = np.arange(len(back))
    axes[0].semilogy(k, back, "o-", color=t["series"][0], label="measured, backward in time")
    axes[0].semilogy(
        k,
        back[0] / lam_s**k,
        "--",
        color=t["muted"],
        lw=1,
        label="prediction: grows by 1/λs per period",
    )
    axes[0].set_xlabel("periods (backward in time)")
    axes[0].set_ylabel("distance from the orbit")
    axes[0].set_title(f"Stable manifold: exact exponential rate (λs = {lam_s:.4f})")
    axes[0].legend()
    axes[1].semilogy(tt / o2.period, dist, color=t["series"][0], label="measured, forward in time")
    axes[1].semilogy(
        tt / o2.period,
        dist[0] * lam_s ** (tt / o2.period),
        "--",
        color=t["muted"],
        lw=1,
        label="average rate ln(λs)/T",
    )
    axes[1].scatter(
        np.arange(len(fwd)), fwd, color=t["series"][1], zorder=5, s=25, label="once per period"
    )
    axes[1].set_xlabel("periods (forward in time)")
    axes[1].set_title("Forward: falls in for ~1 period, then tiny errors take over")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- eps sensitivity (L2 halo, unstable interior branch, one phase) ---
eps_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
DEPART_DIST = 0.02  # "left the orbit" = 0.02 nd (~7,700 km) from the closed orbit curve
T_SENS = 14.0  # long enough for eps = 1e-8 to grow to DEPART_DIST (~2 periods at lambda_u ~ 1000)
t_sens = np.linspace(0, T_SENS, 3000)
# Distance to the closed *curve* (not to a co-moving reference point, which would itself
# drift off the unstable orbit over several periods).
curve = shooting.full_period(o2, n_points=2000).X[:, :3]
sens = {}
for e in eps_list:
    Xs = manifolds.manifold_states(s2, "u", -1.0, e)[10]
    sol = dopri5(cr3bp.make_rhs(MU), 0, T_SENS, Xs, rtol=1e-12, atol=1e-12, t_eval=t_sens)
    d = np.min(np.linalg.norm(sol.X[:, None, :3] - curve[None, :, :], axis=2), axis=1)
    if not (d > DEPART_DIST).any():
        raise RuntimeError(f"eps={e}: never left the orbit within {T_SENS} TU")
    t_dep = sol.t[np.argmax(d > DEPART_DIST)]
    sens[e] = (sol.X[t_sens <= t_dep + 3.0], t_dep)  # keep ~13 days after departure for the plot
deps = np.array([sens[e][1] for e in eps_list])
slope = np.polyfit(np.log10(eps_list), deps, 1)[0]
print(
    f"eps sensitivity: departure time per decade of eps = {-slope * D:.2f} d "
    f"(theory T*ln10/ln(lambda_u) = {o2.period * np.log(10) / np.log(lam_u) * D:.2f} d)"
)


# Whole tubes at two very different eps: if the tube is the same *set*, they overlap.
# Each runs for (its own departure time + 4 TU) so both cover the same stretch of highway.
TUBE_EPS = (1e-6, 1e-4)
whole = {
    e: manifolds.generate_tube(
        o2, "u", -1.0, sens[e][1] + 4.0, eps=e, r_earth=R_E, r_moon=R_M, samples=s2
    )
    for e in TUBE_EPS
}
per_decade_theory = o2.period * np.log(10) / np.log(lam_u)


def draw_eps(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for k, e in enumerate(TUBE_EPS):
        for j, tr in enumerate(whole[e]):
            axes[0].plot(
                tr.X[:, 0],
                tr.X[:, 1],
                color=t["series"][k],
                lw=0.6,
                alpha=0.6,
                label=(
                    f"ε = {e:g} ({e * L * 1000:,.0f} m)"
                    if e * L < 1
                    else f"ε = {e:g} ({e * L:,.0f} km)"
                )
                if j == 0
                else None,
            )
    axes[0].plot(tracks[2][:, 0], tracks[2][:, 1], color=t["ink"], lw=1.5)
    axes[0].scatter([1 - MU], [0], color=t["moon"], s=30, zorder=5)
    axes[0].set_aspect("equal")
    axes[0].set_title("Same tube at ε = 1e-6 and 1e-4 (L2, unstable, −x branch)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].semilogx(eps_list, deps * D, "o-", color=t["series"][0], label="measured")
    axes[1].semilogx(
        eps_list,
        (deps[2] - per_decade_theory * (np.log10(eps_list) - np.log10(eps_list[2]))) * D,
        "--",
        color=t["muted"],
        lw=1,
        label=f"theory: T·ln10 / ln λu = {per_decade_theory * D:.2f} d per decade",
    )
    axes[1].set_xlabel("initial displacement ε (non-dimensional)")
    axes[1].set_ylabel("days to leave the orbit's vicinity")
    axes[1].set_title("ε shifts *when* you leave (and which strand), not the tube")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


for name, fn in (
    ("p9_manifold_tubes_3d", draw_hero),
    ("p9_manifold_tubes_xy", draw_xy),
    ("p9_asymptotic_rate", draw_rate),
    ("p9_eps_sensitivity", draw_eps),
):
    print("saved", [p.name for p in save_both(name, fn)])
