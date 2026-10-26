"""Phase 8: L1/L2 halo families (north + south), stability, and the 9:2 L2 southern NRHO."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from cislunar import lagrange, orbits, shooting
from cislunar.constants import EARTH_MOON, R_MOON_KM, SYNODIC_MONTH_DAYS
from cislunar.plotting import save_both

MU = EARTH_MOON.mu
L = EARTH_MOON.length_km
D = EARTH_MOON.time_days
BRACKETS = {1: (0.010, 0.017), 2: (0.020, 0.030)}
SURFACE_MARGIN_KM = 100.0  # stop continuation once perilune altitude drops below this


def below_surface(o):
    return orbits.perilune(o, 600) * L < R_MOON_KM + SURFACE_MARGIN_KM


fams, info = {}, {}
for w, (lo, hi) in BRACKETS.items():
    bif = orbits.find_halo_bifurcation(w, MU, lo, hi)
    seed = orbits.halo_from_bifurcation(bif, 2e-3)
    fam = orbits.pseudo_arclength_family(
        seed, 400, 0.004, MU, direction=np.array([0, 1, 0]), stop=below_surface
    )
    rows = []
    for o in fam.orbits:
        st = orbits.stability(o)
        a, b = st.trivial_pair
        rows.append(
            dict(
                X0=o.X0,
                T=o.period,
                C=o.jacobi,
                rp=orbits.perilune(o),
                nu=st.max_index,
                nu_all=sorted(st.indices, key=abs),
                closure=orbits.closure_error(o),
                residual=o.residual,
                trivial_split=max(abs(a - 1), abs(b - 1)),
                recip=max(abs(p * q - 1) for p, q in st.pairs),
            )
        )
    fams[w] = (fam, rows)
    info[w] = bif
    np.savez_compressed(
        f"data/halo_L{w}_north.npz",
        X0=np.array([r["X0"] for r in rows]),
        period=np.array([r["T"] for r in rows]),
        jacobi=np.array([r["C"] for r in rows]),
        perilune=np.array([r["rp"] for r in rows]),
        stability_index=np.array([r["nu"] for r in rows]),
        closure=np.array([r["closure"] for r in rows]),
        trivial_split=np.array([r["trivial_split"] for r in rows]),
        mu=MU,
    )
    print(
        f"L{w} halo bifurcation: x0={bif.X0[0]:.8f} vy0={bif.X0[4]:.8f} T={bif.period:.8f} ({bif.period * D:.3f} d) C={bif.jacobi:.8f}"
    )
    print(
        f"  {len(rows)} orbits; T {min(r['T'] for r in rows) * D:.2f}-{max(r['T'] for r in rows) * D:.2f} d; "
        f"perilune alt down to {min(r['rp'] for r in rows) * L - R_MOON_KM:.0f} km; "
        f"max closure {max(r['closure'] for r in rows):.1e}; max crossing resid {max(r['residual'] for r in rows):.1e}; "
        f"max |lam*1/lam-1| {max(r['recip'] for r in rows):.1e}; trivial split {min(r['trivial_split'] for r in rows):.0e}-{max(r['trivial_split'] for r in rows):.0e}"
    )

# --- 9:2 synodic-resonant L2 southern NRHO ---
T_target = 2 * SYNODIC_MONTH_DAYS / 9 / D
fam2, rows2 = fams[2]
i = int(np.argmin([abs(r["T"] - T_target) for r in rows2]))
nrho = orbits.mirror_z(shooting.correct_symmetric_fixed_period(fam2.orbits[i].X0, MU, T_target))
sol = shooting.full_period(nrho, n_points=20000)
r_moon = np.linalg.norm(sol.X[:, :3] - [1 - MU, 0, 0], axis=1)
rp, ra = orbits.perilune(nrho, 20000) * L, r_moon.max() * L
st = orbits.stability(nrho)
nrho_info = dict(
    X0=nrho.X0.tolist(),
    period_days=nrho.period * D,
    period_nd=nrho.period,
    jacobi=nrho.jacobi,
    perilune_radius_km=rp,
    perilune_altitude_km=rp - R_MOON_KM,
    apolune_radius_km=ra,
    stability_indices=st.indices,
    closure=orbits.closure_error(nrho),
    z_at_apolune=float(sol.X[np.argmax(r_moon), 2]),
)
with open("data/nrho_9_2_south.json", "w") as f:
    json.dump(nrho_info, f, indent=2)
print(
    "9:2 L2 southern NRHO:",
    json.dumps(
        {
            k: (float(f"{v:.6g}") if isinstance(v, float) else v)
            for k, v in nrho_info.items()
            if k != "X0"
        }
    ),
)
print("  X0 =", nrho.X0)
tracks = {
    w: [shooting.full_period(o, n_points=300).X[:, :3] for o in fam.orbits]
    for w, (fam, _) in fams.items()
}
nrho_track = sol.X[::20, :3]


def cmap_for(t):
    return LinearSegmentedColormap.from_list(
        "seq",
        ["#cde2fb", "#3987e5", "#0d366b"]
        if t["surface"] == "#fcfcfb"
        else ["#104281", "#3987e5", "#cde2fb"],
    )


def draw_3d(t):
    fig = plt.figure(figsize=(12, 6.2))
    cmap = cmap_for(t)
    for k, w in enumerate((1, 2)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.set_facecolor(t["surface"])
        _, rows = fams[w]
        norm = Normalize(min(r["C"] for r in rows), max(r["C"] for r in rows))
        for X, r in list(zip(tracks[w], rows, strict=True))[::3]:
            col = cmap(norm(r["C"]))
            ax.plot(X[:, 0], X[:, 1], X[:, 2], color=col, lw=0.8)
            ax.plot(X[:, 0], X[:, 1], -X[:, 2], color=col, lw=0.8, alpha=0.35)  # southern mirror
        u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 12j]
        rm = R_MOON_KM / L
        ax.plot_surface(
            1 - MU + rm * np.cos(u) * np.sin(v),
            rm * np.sin(u) * np.sin(v),
            rm * np.cos(v),
            color=t["moon"],
            shade=False,
        )
        xL = lagrange.collinear_point(w, MU)
        ax.scatter([xL], [0], [0], marker="D", color=t["series"][1], s=25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"L{w} halo family: northern (solid), southern (faint)", fontsize=10)
        ax.view_init(elev=18, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=8)
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis.label.set_color(t["ink2"])
    fig.suptitle(
        "Halo orbit families, rotating frame (colour = Jacobi constant)",
        color=t["ink"],
        fontweight="bold",
    )
    return fig


def draw_projections(t):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    cmap = cmap_for(t)
    for row, w in enumerate((1, 2)):
        _, rows = fams[w]
        norm = Normalize(min(r["C"] for r in rows), max(r["C"] for r in rows))
        for (i, j, lab), ax in zip(
            ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")), axes[row], strict=True
        ):
            for X, r in list(zip(tracks[w], rows, strict=True))[::3]:
                ax.plot(X[:, i], X[:, j], color=cmap(norm(r["C"])), lw=0.8)
            mx = [1 - MU, 0, 0]
            ax.scatter(mx[i], mx[j], s=30, color=t["moon"], zorder=5)
            ax.set_aspect("equal")
            ax.set_xlabel(lab[0])
            ax.set_ylabel(lab[1])
            ax.set_title(f"L{w} northern family, {lab} projection", fontsize=10)
    fig.tight_layout()
    return fig


def draw_stability(t):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for w in (1, 2):
        _, rows = fams[w]
        rp_k = [r["rp"] * L / 1000 for r in rows]
        c = t["series"][w - 1]
        axes[0].plot(
            np.arange(len(rows)),
            [abs(r["X0"][2]) * L / 1000 for r in rows],
            "o-",
            ms=2.5,
            color=c,
            label=f"L{w}",
        )
        axes[1].semilogy(rp_k, [r["nu"] for r in rows], "o-", ms=2.5, color=c, label=f"L{w}")
        axes[2].plot(rp_k, [r["T"] * D for r in rows], "o-", ms=2.5, color=c, label=f"L{w}")
    axes[0].set_xlabel("orbit number along the family (pseudo-arclength)")
    axes[0].set_ylabel("|z0| [thousand km]")
    axes[0].set_title("z0 turns back: stepping in z0 alone would fail")
    axes[1].axhline(1, color=t["muted"], lw=1, ls=":")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("perilune radius [thousand km]  (→ closer to the Moon)")
    axes[1].set_ylabel("max stability index ν (log)")
    axes[1].set_title("Near-Moon halos (NRHOs) are nearly stable")
    axes[2].axhline(T_target * D, color=t["series"][2], lw=1, ls="--")
    axes[2].annotate(
        "9:2 synodic resonance (Gateway)",
        (25, T_target * D),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=9,
    )
    axes[2].scatter([rp / 1000], [nrho.period * D], color=t["series"][2], zorder=6, s=40)
    axes[2].invert_xaxis()
    axes[2].set_xlabel("perilune radius [thousand km]  (→ closer to the Moon)")
    axes[2].set_ylabel("period [days]")
    axes[2].set_title("Period along the family")
    for a in axes:
        a.legend()
    fig.tight_layout()
    return fig


def draw_nrho(t):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), gridspec_kw={"width_ratios": [1, 1.6]})
    X = (nrho_track - [1 - MU, 0, 0]) * L / 1000  # Moon-centred, thousand km
    for ax, (i, j, lab) in zip(axes, ((0, 2, "xz"), (1, 2, "yz")), strict=True):
        ax.plot(X[:, i], X[:, j], color=t["series"][0], lw=1.6)
        ax.add_patch(plt.Circle((0, 0), R_MOON_KM / 1000, color=t["moon"], zorder=5))
        ax.set_aspect("equal")
        ax.set_xlabel(f"{lab[0]} from Moon centre [thousand km]")
        ax.set_ylabel(f"{lab[1]} [thousand km]")
        ax.set_xlim(-22, 22)
    axes[0].set_title("side view (xz)", fontsize=10)
    axes[1].set_title("view from Earth (yz)", fontsize=10)
    fig.suptitle(
        f"9:2 L2 southern NRHO: period {nrho.period * D:.3f} d, perilune radius {rp:,.0f} km "
        f"({rp - R_MOON_KM:,.0f} km altitude), apolune radius {ra:,.0f} km",
        color=t["ink"],
        fontweight="bold",
        fontsize=11,
    )
    return fig


for name, fn in (
    ("p8_halo_families_3d", draw_3d),
    ("p8_halo_projections", draw_projections),
    ("p8_halo_stability", draw_stability),
    ("p8_nrho_9_2", draw_nrho),
):
    print("saved", [p.name for p in save_both(name, fn)])
