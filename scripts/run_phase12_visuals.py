"""Phase 12: Plotly interactive 3D (single HTML file) and an animation of the transfer in both frames."""

import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter, PillowWriter

from cislunar import manifolds, shooting
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.plotting import apply_theme

U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
res = json.load(open("data/transfer_results.json"))
tr = np.load("data/transfer_trajectories.npz")
halo = shooting.SymmetricOrbit(np.array(res["halo_X0"]), res["halo_period_days"] / D, MU, 0, 0.0)
H = shooting.full_period(halo, n_points=400).X[:, :3]

# ---------------- Plotly 3D ----------------
fig = go.Figure()
s = manifolds.sample_orbit(halo, 40)
for kind, sign, color, name in (
    ("s", -1.0, "#3987e5", "stable manifold (Moon side)"),
    ("u", -1.0, "#d95926", "unstable manifold (Moon side)"),
):
    tube = manifolds.generate_tube(
        halo, kind, sign, 8.0, r_earth=R_EARTH_KM / L, r_moon=R_MOON_KM / L, samples=s
    )
    for k, trj in enumerate(tube):
        fig.add_trace(
            go.Scatter3d(
                x=trj.X[:, 0] * L,
                y=trj.X[:, 1] * L,
                z=trj.X[:, 2] * L,
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.35,
                name=name,
                legendgroup=name,
                showlegend=k == 0,
                hoverinfo="skip",
            )
        )
for key, color, name in (
    ("d", "#eb6834", "direct transfer"),
    ("m", "#1baf7a", "low-energy transfer"),
):
    X = np.vstack(
        [
            tr[f"{key}_X1"],
            tr[f"{key}_X2"][
                tr[f"{key}_t2"] <= res["direct" if key == "d" else "manifold"]["tof_arc_days"] / D
            ],
        ]
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0] * L,
            y=X[:, 1] * L,
            z=X[:, 2] * L,
            mode="lines",
            line=dict(color=color, width=5),
            name=name,
        )
    )
fig.add_trace(
    go.Scatter3d(
        x=H[:, 0] * L,
        y=H[:, 1] * L,
        z=H[:, 2] * L,
        mode="lines",
        line=dict(color="#0b0b0b", width=6),
        name="L2 halo",
    )
)
u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
for cx, r, col, name in (
    (-MU * L, R_EARTH_KM, "#2a78d6", "Earth"),
    ((1 - MU) * L, R_MOON_KM, "#898781", "Moon"),
):
    fig.add_trace(
        go.Surface(
            x=cx + r * np.cos(u) * np.sin(v),
            y=r * np.sin(u) * np.sin(v),
            z=r * np.cos(v),
            showscale=False,
            colorscale=[[0, col], [1, col]],
            name=name,
            hoverinfo="name",
        )
    )
fig.update_layout(
    title="LEO → L2 halo transfers and manifold tubes (rotating frame, km)",
    scene=dict(aspectmode="data", xaxis_title="x [km]", yaxis_title="y [km]", zaxis_title="z [km]"),
    legend=dict(itemsizing="constant"),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig.write_html("docs/figures/p12_interactive_3d.html", include_plotlyjs=True, full_html=True)
print("saved docs/figures/p12_interactive_3d.html")


# ---------------- animation: both frames side by side ----------------
def inertial(t, X):
    r = X[:, :3] - [-MU, 0, 0]
    c, s_ = np.cos(t), np.sin(t)
    return np.stack([c * r[:, 0] - s_ * r[:, 1], s_ * r[:, 0] + c * r[:, 1]], axis=1) * L / 1e3


tm = np.concatenate([tr["m_t1"], tr["m_t2"]])
Xm = np.vstack([tr["m_X1"], tr["m_X2"]])
keep = tm <= res["manifold"]["tof_arc_days"] / D
tm, Xm = tm[keep], Xm[keep]
tq = np.linspace(tm[0], tm[-1], 1500)  # dense: the path has tight loops
Xq = np.stack([np.interp(tq, tm, Xm[:, k]) for k in range(3)], axis=1)
Rrot = (Xq[:, :2] - [-MU, 0]) * L / 1e3
Rin = inertial(tq, Xq)
moon_in = np.stack([np.cos(tq), np.sin(tq)], axis=1) * L / 1e3

t = apply_theme("dark")
figa, axes = plt.subplots(1, 2, figsize=(12, 6))
lines, heads = [], []
for ax, title in zip(
    axes, ("Rotating frame (Moon fixed)", "Earth-centred inertial frame"), strict=True
):
    ax.set_aspect("equal")
    ax.set_xlim(-480, 480)
    ax.set_ylim(-480, 480)
    ax.set_title(title)
    ax.set_xlabel("thousand km")
    ax.scatter([0], [0], s=60, color=t["earth"], zorder=5)
    (ln,) = ax.plot([], [], color=t["series"][0], lw=1.2)
    (hd,) = ax.plot([], [], "o", color="#ffffff", ms=5)
    lines.append(ln)
    heads.append(hd)
axes[0].scatter([L / 1e3], [0], s=25, color=t["moon"], zorder=5)
axes[0].plot((H[:, 0] + MU) * L / 1e3, H[:, 1] * L / 1e3, color=t["ink"], lw=1)
(moon_dot,) = axes[1].plot([], [], "o", color=t["moon"], ms=6)
axes[1].plot(
    L / 1e3 * np.cos(np.linspace(0, 7, 200)),
    L / 1e3 * np.sin(np.linspace(0, 7, 200)),
    color=t["grid"],
    lw=1,
)
label = figa.text(0.5, 0.02, "", ha="center", color=t["ink"], fontsize=12)


def update(i):
    lines[0].set_data(Rrot[: i + 1, 0], Rrot[: i + 1, 1])
    lines[1].set_data(Rin[: i + 1, 0], Rin[: i + 1, 1])
    heads[0].set_data([Rrot[i, 0]], [Rrot[i, 1]])
    heads[1].set_data([Rin[i, 0]], [Rin[i, 1]])
    moon_dot.set_data([moon_in[i, 0]], [moon_in[i, 1]])
    label.set_text(
        f"Low-energy transfer, day {(tq[i] - tq[0]) * D:5.1f} of {(tq[-1] - tq[0]) * D:.0f}"
    )
    return [*lines, *heads, moon_dot, label]


frames = range(0, len(tq), 5)
from matplotlib.animation import FuncAnimation  # noqa: E402

anim = FuncAnimation(figa, update, frames=frames, blit=False)
anim.save("docs/figures/p12_transfer_both_frames.gif", writer=PillowWriter(fps=20), dpi=80)
anim.save(
    "docs/figures/p12_transfer_both_frames.mp4", writer=FFMpegWriter(fps=30, bitrate=3000), dpi=120
)
print("saved docs/figures/p12_transfer_both_frames.gif / .mp4")
