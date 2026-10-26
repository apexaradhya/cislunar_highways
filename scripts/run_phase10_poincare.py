"""Phase 10: Poincare sections of manifold tubes at x = 1 - mu, y < 0 (below the Moon).

1. Planar Lyapunov orbits at the same energy C = 3.17: the L1 unstable tube and the L2
   stable tube both enter the Moon's realm. Each cut of a tube is a closed curve (minus
   strands lost to lunar impact). Where an L1-unstable cut crosses an L2-stable cut, a
   spacecraft can coast from the L1 orbit to the L2 orbit with no burn: a heteroclinic
   connection (Koon, Lo, Marsden & Ross 2000). We report what the curves actually do.
2. The L2 halo's stable tube cut: a closed curve in 4D, shown as two projections.
"""

import functools
import json

import matplotlib.pyplot as plt
import numpy as np

from cislunar import cr3bp, manifolds, orbits, poincare, shooting
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.integrators import dopri5
from cislunar.plotting import save_both
from cislunar.shooting import correct_symmetric

MU = EARTH_MOON.mu
L = EARTH_MOON.length_km
RE, RM = R_EARTH_KM / L, R_MOON_KM / L
C_SEC = 3.17
N = 160
N_HALO = 200
below = poincare.plane_section(0, 1 - MU, extra=lambda X: X[1] < 0)
f = cr3bp.make_rhs(MU)

lyap = {}
for w in (1, 2):
    fam = orbits.lyapunov_family(w, MU, np.linspace(2e-3, 0.06, 16))
    i = int(np.argmin(np.abs(fam.jacobi - C_SEC)))
    lyap[w] = shooting.correct_planar_fixed_energy(fam.orbits[i].X0, MU, C_SEC)

# L1 unstable +x branch and L2 stable -x branch both lead into the Moon's realm.
cuts = {}
for k in (1, 2):
    cuts[("L1u", k)] = poincare.tube_cut(
        lyap[1], "u", +1.0, below, n_traj=N, crossing=k, max_duration=15, r_earth=RE, r_moon=RM
    )
    cuts[("L2s", k)] = poincare.tube_cut(
        lyap[2], "s", -1.0, below, n_traj=N, crossing=k, max_duration=15, r_earth=RE, r_moon=RM
    )
pieces = {key: poincare.split_at_gaps(c, N, [1, 4]) for key, c in cuts.items()}
crossings = {}
for k in (1, 2):
    crossings[k] = poincare.polyline_intersections(pieces[("L1u", k)], pieces[("L2s", k)])
    print(
        f"cut {k}: L1 unstable {len(cuts[('L1u', k)].states)}/{N} strands, L2 stable "
        f"{len(cuts[('L2s', k)].states)}/{N}; curve intersections = {len(crossings[k])}"
    )
mg, cg = poincare.closed_curve_gaps(cuts[("L2s", 1)].states[:, [1, 4]])
print(f"L2 stable first cut: max neighbour gap {mg:.3f}, closing gap {cg:.3f} (closed curve)")

# Verify each 2nd-cut intersection: refine the two starting phases so the strands meet on
# the section in (y, vy); then the full states must agree, including the sign of vx.
cross_a = functools.partial(
    poincare.strand_crossing, lyap[1], "u", 1.0, section=below, crossing=2, r_earth=RE, r_moon=RM
)
cross_b = functools.partial(
    poincare.strand_crossing, lyap[2], "s", -1.0, section=below, crossing=2, r_earth=RE, r_moon=RM
)
connections = []
for p in crossings[2]:
    ca, cb = cuts[("L1u", 2)], cuts[("L2s", 2)]
    ia = int(np.argmin(np.linalg.norm(ca.states[:, [1, 4]] - p, axis=1)))
    ib = int(np.argmin(np.linalg.norm(cb.states[:, [1, 4]] - p, axis=1)))
    r = poincare.refine_connection(cross_a, cross_b, ca.phases[ia], cb.phases[ib])
    if r is None:
        print(f"  intersection at {p}: refinement failed (not counted)")
        continue
    pa, pb, XA, XB = r
    mismatch = float(np.max(np.abs(XA - XB)))
    if mismatch > 1e-8:
        print(
            f"  intersection at {p}: states differ by {mismatch:.1e} (vx sign mismatch?) -> rejected"
        )
        continue
    ta, tb = cross_a(pa)[1], cross_b(pb)[1]
    connections.append(
        dict(
            phase_L1=pa,
            phase_L2=pb,
            section_state=XA.tolist(),
            mismatch=mismatch,
            t_L1_to_section=ta,
            t_section_to_L2=-tb,
        )
    )
    print(
        f"  VERIFIED heteroclinic L1->L2: phases ({pa:.6f}, {pb:.6f}), state mismatch {mismatch:.1e}, "
        f"flight {ta * EARTH_MOON.time_days:.1f} d + {-tb * EARTH_MOON.time_days:.1f} d"
    )
with open("data/heteroclinic_L1_L2_C3.17.json", "w") as fh:
    json.dump(dict(jacobi=C_SEC, mu=MU, connections=connections), fh, indent=2)
# Full connecting trajectory: from the section, forward to L2 and backward to L1.
conn_paths = []
for c in connections:
    Xs = np.array(c["section_state"])
    back = dopri5(
        f,
        0,
        -c["t_L1_to_section"],
        Xs,
        rtol=1e-12,
        atol=1e-12,
        t_eval=np.linspace(0, -c["t_L1_to_section"], 800),
    ).X
    fwd = dopri5(
        f,
        0,
        c["t_section_to_L2"],
        Xs,
        rtol=1e-12,
        atol=1e-12,
        t_eval=np.linspace(0, c["t_section_to_L2"], 800),
    ).X
    conn_paths.append((back, fwd))


def strand_paths(orbit, kind, sign, cut, every=5):
    s = manifolds.sample_orbit(orbit, N)
    starts = manifolds.manifold_states(s, kind, sign)
    paths = []
    for ph, tf in zip(cut.phases[::every], cut.times[::every], strict=True):
        i = int(round(ph * N)) % N
        paths.append(
            dopri5(f, 0, tf, starts[i], rtol=1e-12, atol=1e-12, t_eval=np.linspace(0, tf, 300)).X
        )
    return paths


paths1 = strand_paths(lyap[1], "u", 1.0, cuts[("L1u", 1)])
paths2 = strand_paths(lyap[2], "s", -1.0, cuts[("L2s", 1)])
orbit_tracks = {w: shooting.full_period(lyap[w], n_points=300).X for w in (1, 2)}

d = np.load("data/halo_L2_north.npz")
halo = correct_symmetric(
    d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - 0.05)))], MU, free=(0, 4), targets=(3, 5)
)
cuth = poincare.tube_cut(
    halo, "s", -1.0, below, n_traj=N_HALO, max_duration=12.0, r_earth=RE, r_moon=RM
)
print(
    f"L2 halo stable cut: {len(cuth.states)}/{N_HALO} strands reach the section (others hit the Moon first)"
)


def draw_lyap(t):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axes = axs.flat
    c1, c2 = t["series"][1], t["series"][0]
    ax = axes[0]
    for P in paths1:
        ax.plot(P[:, 0], P[:, 1], color=c1, lw=0.6, alpha=0.6)
    for P in paths2:
        ax.plot(P[:, 0], P[:, 1], color=c2, lw=0.6, alpha=0.6)
    for w in (1, 2):
        ax.plot(orbit_tracks[w][:, 0], orbit_tracks[w][:, 1], color=t["ink"], lw=1.6)
    ax.add_patch(plt.Circle((1 - MU, 0), RM, color=t["moon"], zorder=5))
    ax.plot([1 - MU, 1 - MU], [-0.13, 0], color=t["muted"], lw=1, ls="--")
    ax.annotate(
        "section",
        (1 - MU, -0.12),
        xytext=(4, 0),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=9,
    )
    ax.plot([], [], color=c1, label="L1 unstable tube (leaving L1)")
    ax.plot([], [], color=c2, label="L2 stable tube (arriving at L2)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(0.78, 1.22)
    ax.set_ylim(-0.13, 0.13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Tubes up to their first cut (C = {C_SEC})")
    for ax, k in zip((axes[1], axes[2]), (1, 2), strict=True):
        for key, col in (("L1u", c1), ("L2s", c2)):
            for j, p in enumerate(pieces[(key, k)]):
                lab = None
                if j == 0:
                    n = len(cuts[(key, k)].states)
                    lab = f"{'L1 unstable' if key == 'L1u' else 'L2 stable'} cut {k} ({n}/{N} strands)"
                ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=1.4, label=lab)
        X = crossings[k]
        if len(X):
            ax.scatter(
                X[:, 0],
                X[:, 1],
                s=60,
                facecolor="none",
                edgecolor=t["ink"],
                lw=1.5,
                zorder=6,
                label=f"{len(X)} intersection(s): candidate connection (verified in panel 4)",
            )
        ax.set_xlabel("y on the section")
        ax.set_ylabel("vy on the section")
        ax.legend(fontsize=8)
    n1, n2 = len(crossings[1]), len(crossings[2])
    axes[1].set_title(
        "1st cuts: " + ("no overlap → no direct connection" if n1 == 0 else f"{n1} intersections")
    )
    axes[2].set_title(
        "2nd cuts (zoom near the Moon): " + (f"{n2} intersection(s)" if n2 else "no intersections")
    )
    # Some 2nd-cut strands first wander far from the Moon (|y| up to ~4); zoom to the region
    # where the two cuts can meet.
    axes[2].set_xlim(-0.14, 0.005)
    axes[2].set_ylim(-1.5, 0.8)
    ax = axes[3]
    for w in (1, 2):
        ax.plot(orbit_tracks[w][:, 0], orbit_tracks[w][:, 1], color=t["ink"], lw=1.6)
    for back, fwd in conn_paths:
        ax.plot(back[:, 0], back[:, 1], color=c1, lw=1.4, label="from L1 (unstable manifold)")
        ax.plot(fwd[:, 0], fwd[:, 1], color=c2, lw=1.4, label="to L2 (stable manifold)")
        ax.scatter(
            back[0, 0],
            back[0, 1],
            color=t["ink"],
            s=30,
            zorder=6,
            label="patch point on the section",
        )
    ax.add_patch(plt.Circle((1 - MU, 0), RM, color=t["moon"], zorder=5))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if connections:
        c = connections[0]
        days = (c["t_L1_to_section"] + c["t_section_to_L2"]) * EARTH_MOON.time_days
        ax.set_title(
            f"Verified free L1 → L2 transfer: Δv = 0, {days:.1f} days (mismatch {c['mismatch']:.0e})"
        )
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.set_title("No intersection survived verification")
    fig.tight_layout()
    return fig


def draw_halo(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sc = None
    for ax, (i, j, lab) in zip(axes, ((1, 4, ("y", "vy")), (2, 5, ("z", "vz"))), strict=True):
        for p in poincare.split_at_gaps(cuth, N_HALO, [i, j]):
            ax.plot(p[:, 0], p[:, 1], "-", color=t["muted"], lw=0.8)
        sc = ax.scatter(
            cuth.states[:, i], cuth.states[:, j], c=cuth.phases, cmap="twilight", s=10, zorder=5
        )
        ax.set_xlabel(f"{lab[0]} on the section")
        ax.set_ylabel(f"{lab[1]} on the section")
    axes[0].set_title(
        f"L2 halo stable tube cut, (y, vy) ({len(cuth.states)}/{N_HALO} strands)", fontsize=10
    )
    axes[1].set_title("(z, vz) projection; gap = strands that hit the Moon", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label("starting phase on the halo (t/T)", color=t["ink2"])
    cb.outline.set_visible(False)
    return fig


print("saved", [p.name for p in save_both("p10_lyapunov_heteroclinic", draw_lyap)])
print("saved", [p.name for p in save_both("p10_halo_tube_cut", draw_halo)])
