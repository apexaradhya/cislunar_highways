"""Phase 11 figures from data/transfer_results.json and data/transfer_trajectories.npz."""

import json

import matplotlib.pyplot as plt
import numpy as np

from cislunar import shooting
from cislunar.constants import EARTH_MOON
from cislunar.plotting import save_both

U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
res = json.load(open("data/transfer_results.json"))
tr = np.load("data/transfer_trajectories.npz")
halo = shooting.SymmetricOrbit(np.array(res["halo_X0"]), res["halo_period_days"] / D, MU, 0, 0.0)
H = shooting.full_period(halo, n_points=600).X[:, :3]
dI, mI = res["direct"], res["manifold"]


def to_inertial(t, X):
    """Earth-centred inertial positions (km), frame aligned with the rotating frame at the patch time."""
    r = X[:, :3] - [-MU, 0, 0]
    c, s = np.cos(t), np.sin(t)
    return np.stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]], axis=1) * L


def draw_rot(t):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    cd, cm = t["series"][1], t["series"][0]
    m_arr = tr["m_t2"] <= mI["tof_arc_days"] / D
    for ax, (i, j) in zip(axes, ((0, 1), (0, 1)), strict=True):
        ax.plot(
            tr["d_X1"][:, i] * L / 1e3,
            tr["d_X1"][:, j] * L / 1e3,
            color=cd,
            lw=1.4,
            label=f"direct: {dI['dv_total_km_s']:.3f} km/s, {dI['tof_total_days']:.1f} d",
        )
        ax.plot(
            tr["m_X1"][:, i] * L / 1e3,
            tr["m_X1"][:, j] * L / 1e3,
            color=cm,
            lw=1.4,
            label=f"manifold: {mI['dv_total_km_s']:.3f} km/s, {mI['tof_total_days']:.1f} d",
        )
        ax.plot(
            tr["m_X2"][m_arr, i] * L / 1e3,
            tr["m_X2"][m_arr, j] * L / 1e3,
            color=cm,
            lw=1.0,
            ls="--",
            label="…riding the stable manifold (no burns)",
        )
        ax.plot(
            H[:, i] * L / 1e3, H[:, j] * L / 1e3, color=t["ink"], lw=1.8, label="target L2 halo"
        )
        ax.scatter([-MU * L / 1e3], [0], s=60, color=t["earth"], zorder=5)
        ax.scatter([(1 - MU) * L / 1e3], [0], s=25, color=t["moon"], zorder=5)
        for X, c in ((tr["d_X1"], cd), (tr["m_X1"], cm)):
            ax.scatter(
                X[-1, i] * L / 1e3,
                X[-1, j] * L / 1e3,
                marker="*",
                s=120,
                color=c,
                zorder=6,
                edgecolor=t["surface"],
            )
        ax.set_aspect("equal")
        ax.set_xlabel("x [thousand km] (rotating frame)")
        ax.set_ylabel(("y" if j == 1 else "z") + " [thousand km]")
    axes[0].set_title("Top view (xy); ★ = second burn")
    axes[1].set_xlim(300, 480)
    axes[1].set_ylim(-90, 90)
    axes[1].set_title("Zoom: arrival near the Moon and the halo")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "LEO → L2 halo: direct vs low-energy (manifold) transfer, full CR3BP",
        color=t["ink"],
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def draw_inertial(t):
    fig, ax = plt.subplots(figsize=(8, 8))
    cd, cm = t["series"][1], t["series"][0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(L / 1e3 * np.cos(th), L / 1e3 * np.sin(th), color=t["grid"], lw=1, label="Moon's orbit")
    for key, c, lab in (("d", cd, "direct"), ("m", cm, "manifold")):
        R1 = to_inertial(tr[f"{key}_t1"], tr[f"{key}_X1"]) / 1e3
        ax.plot(R1[:, 0], R1[:, 1], color=c, lw=1.4, label=f"{lab}: outbound leg")
    m_arr = tr["m_t2"] <= mI["tof_arc_days"] / D + 1.0
    R2 = to_inertial(tr["m_t2"][m_arr], tr["m_X2"][m_arr]) / 1e3
    ax.plot(R2[:, 0], R2[:, 1], color=cm, lw=1.0, ls="--", label="manifold coast to the halo")
    ax.scatter([0], [0], s=80, color=t["earth"], zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [thousand km] (Earth-centred inertial)")
    ax.set_ylabel("Y [thousand km]")
    ax.set_title("The same transfers seen from a non-rotating frame")
    ax.legend(fontsize=8, loc="upper left")
    return fig


def draw_tradeoff(t):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cand = res["candidates"]
    for kind, c in (("direct", t["series"][1]), ("manifold", t["series"][0])):
        pts = np.array([[p["tof_days"], p["dv"]] for p in cand if p["kind"] == kind])
        ax.scatter(
            pts[:, 0],
            pts[:, 1] * 1000,
            s=16,
            color=c,
            alpha=0.6,
            label=f"{kind} candidates (CR3BP-corrected)",
        )
    allp = np.array(sorted([[p["tof_days"], p["dv"]] for p in cand]))
    front, best = [], np.inf
    for tof, dv in allp:
        if dv < best:
            front.append((tof, dv))
            best = dv
    front = np.array(front)
    ax.step(
        front[:, 0],
        front[:, 1] * 1000,
        where="post",
        color=t["ink"],
        lw=1.5,
        label="Pareto front (best Δv for a given TOF)",
    )
    for info, c in ((dI, t["series"][1]), (mI, t["series"][0])):
        ax.scatter(
            info["tof_total_days"],
            info["dv_total_km_s"] * 1000,
            marker="*",
            s=220,
            color=c,
            edgecolor=t["ink"],
            zorder=6,
        )
    pc = res["patched_conic"]
    ax.axhline(pc["total"] * 1000, color=t["muted"], ls=":", lw=1)
    ax.annotate(
        f"patched-conic LEO→low lunar orbit estimate ({pc['total'] * 1000:,.0f} m/s)",
        (1, pc["total"] * 1000),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=8.5,
    )
    ax.set_xlabel("total time of flight [days]")
    ax.set_ylabel("total Δv [m/s]")
    ax.set_title("Trading time for fuel (★ = refined designs)")
    ax.legend(fontsize=8, loc="upper right", scatterpoints=1)
    return fig


def draw_bars(t):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    pc = res["patched_conic"]
    rows = [
        ("Patched conic → LLO\n(estimate)", pc["tli"], pc["loi"], "LOI"),
        ("Direct → L2 halo\n(CR3BP)", dI["dv1_km_s"], dI["dv2_km_s"], "halo insertion"),
        ("Manifold → L2 halo\n(CR3BP)", mI["dv1_km_s"], mI["dv2_km_s"], "manifold insertion"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_lab, a, b, second) in zip(y, rows, strict=True):
        ax.barh(
            yi, a * 1000, color=t["series"][0], height=0.55, edgecolor=t["surface"], linewidth=2
        )
        ax.barh(
            yi,
            b * 1000,
            left=a * 1000,
            color=t["series"][1],
            height=0.55,
            edgecolor=t["surface"],
            linewidth=2,
        )
        ax.text(
            a * 500,
            yi,
            f"TLI {a * 1000:,.0f}",
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=9,
        )
        ax.text(
            a * 1000 + b * 1000 + 30,
            yi,
            f"+ {second} {b * 1000:,.0f} = {(a + b) * 1000:,.0f} m/s",
            va="center",
            color=t["ink"],
            fontsize=9,
        )
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 5600)
    ax.set_xlabel("Δv [m/s]")
    ax.set_title("Δv budget breakdown")
    ax.grid(axis="y", visible=False)
    return fig


def draw_arrival(t):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    tt = tr["m_t2"] * D
    ax.semilogy(tt - mI["tof_arc_days"], tr["m_dist"], color=t["series"][0])
    ax.axvline(0, color=t["muted"], ls="--", lw=1)
    ax.annotate(
        f"nominal arrival: {mI['arrival_distance_km']:.0f} km\n(ideal 3.8 km; 165 days of chaotic\n"
        f"coasting magnify tiny errors)\nclosest {mI['closest_approach_km']:.0f} km, clean-up "
        f"{mI['insertion_dv_m_s']:.1f} m/s",
        (0, tr["m_dist"].max() / 10),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.5,
        color=t["ink2"],
    )
    ax.set_xlabel("days relative to arrival")
    ax.set_ylabel("distance to the halo orbit [km]")
    ax.set_title("Asymptotic arrival, then drift-off within ~1 period without station-keeping")
    return fig


for name, fn in (
    ("p11_transfer_rotating", draw_rot),
    ("p11_transfer_inertial", draw_inertial),
    ("p11_dv_vs_tof", draw_tradeoff),
    ("p11_dv_breakdown", draw_bars),
    ("p11_arrival_distance", draw_arrival),
):
    print("saved", [p.name for p in save_both(name, fn)])
