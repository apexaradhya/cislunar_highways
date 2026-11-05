"""Simple point-and-click app: pick a halo orbit and watch its manifold "highways" get computed.

Run with:  uv run streamlit run app.py      (or: streamlit run app.py)
Everything on screen is computed live by the project's own code in src/cislunar/.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from cislunar import lagrange, manifolds, orbits
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.shooting import SymmetricOrbit, full_period

U = EARTH_MOON
MU, L, DAYS = U.mu, U.length_km, U.time_days

st.set_page_config(page_title="Cislunar Highways", layout="wide")
st.title("Cislunar Highways")
st.caption(
    "Pick an orbit near the Moon and see the gravitational 'highways' (invariant manifolds) "
    "that lead onto it and away from it. Everything is computed live from the project's code."
)


@st.cache_data
def load_family(point: str):
    d = np.load(f"data/halo_{point}_north.npz")
    return d["X0"], d["period"]


@st.cache_data
def compute(point: str, index: int, south: bool, n_strands: int, days: float, kinds: tuple):
    X0s, periods = load_family(point)
    orbit = SymmetricOrbit(X0s[index].copy(), float(periods[index]), MU, 0, 0.0)
    if south:
        orbit = orbits.mirror_z(orbit)
    halo = full_period(orbit, n_points=400).X[:, :3]
    samples = manifolds.sample_orbit(orbit, n_strands)
    lam_u = float(samples.directions.lambda_u)
    tubes = {}
    for kind in kinds:
        for sign in (1.0, -1.0):
            tubes[(kind, sign)] = [
                trj.X[:, :3]
                for trj in manifolds.generate_tube(
                    orbit,
                    kind,
                    sign,
                    days / DAYS,
                    r_earth=R_EARTH_KM / L,
                    r_moon=R_MOON_KM / L,
                    samples=samples,
                    n_out=300,
                )
            ]
    info = {
        "Vertical size (km)": abs(orbit.X0[2]) * L,
        "Period (days)": orbit.period * DAYS,
        "Jacobi constant C": orbit.jacobi,
        "Instability λu": lam_u,
    }
    return halo, tubes, info


# ---- sidebar controls ----
with st.sidebar:
    st.header("Settings")
    point = st.radio("Which balance point?", ["L2", "L1"], horizontal=True)
    X0s, periods = load_family(point)
    sizes = np.abs(X0s[:, 2]) * L
    index = st.select_slider(
        "Orbit size (vertical amplitude, km)",
        options=list(range(len(sizes))),
        value=len(sizes) // 3,
        format_func=lambda i: f"{sizes[i]:,.0f}",
    )
    south = st.toggle("Southern orbit (mirror image)", value=False)
    show = st.multiselect(
        "Highways to draw",
        ["Stable (leads onto the orbit)", "Unstable (leads away)"],
        default=["Stable (leads onto the orbit)", "Unstable (leads away)"],
    )
    n_strands = st.slider("Number of strands per tube", 10, 60, 24, step=2)
    days = st.slider("How far to follow them (days)", 5, 60, 30)
    st.caption("More strands or days = prettier but slower (a few seconds to ~1 minute).")

kinds = tuple(
    k for k, label in (("s", "Stable"), ("u", "Unstable")) if any(label in s for s in show)
)
with st.spinner("Computing orbit and manifold tubes..."):
    halo, tubes, info = compute(point, int(index), south, n_strands, float(days), kinds)

# ---- numbers ----
cols = st.columns(len(info))
for col, (name, val) in zip(cols, info.items(), strict=True):
    col.metric(name, f"{val:,.0f}" if val > 100 else f"{val:,.4f}" if val < 10 else f"{val:,.1f}")

# ---- 3D plot (km, rotating frame) ----
fig = go.Figure()
colours = {"s": "#5ec8ff", "u": "#ffad5c"}  # explorer tokens --stable / --unstable
names = {"s": "Stable highway", "u": "Unstable highway"}
for (kind, _sign), strands in tubes.items():
    xs, ys, zs = [], [], []
    for P in strands:
        xs += list(P[:, 0] * L) + [None]
        ys += list(P[:, 1] * L) + [None]
        zs += list(P[:, 2] * L) + [None]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=colours[kind], width=2),
            opacity=0.5,
            name=names[kind],
            legendgroup=kind,
            showlegend=not any(t.name == names[kind] for t in fig.data),
        )
    )
fig.add_trace(
    go.Scatter3d(
        x=halo[:, 0] * L,
        y=halo[:, 1] * L,
        z=halo[:, 2] * L,
        mode="lines",
        line=dict(color="#e8edf6", width=6),
        name=f"{point} halo orbit",
    )
)
pts = lagrange.lagrange_points(MU)
for name in ("L1", "L2"):
    p = pts[name]
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] * L],
            y=[p[1] * L],
            z=[0],
            mode="markers+text",
            text=[name],
            textposition="top center",
            marker=dict(size=4, color="#ff5c8d"),
            showlegend=False,
        )
    )
for name, x, size, col in (("Earth", -MU, 12, "#5ec8ff"), ("Moon", 1 - MU, 6, "#8e98ad")):
    fig.add_trace(
        go.Scatter3d(
            x=[x * L],
            y=[0],
            z=[0],
            mode="markers+text",
            text=[name],
            textposition="bottom center",
            marker=dict(size=size, color=col),
            showlegend=False,
        )
    )
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#04060b",
    font=dict(color="#e8edf6"),
    height=720,
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        aspectmode="data",
        xaxis_title="x (km)",
        yaxis_title="y (km)",
        zaxis_title="z (km)",
    ),
    legend=dict(x=0.01, y=0.99),
)
st.plotly_chart(fig, width="stretch")

st.markdown(
    "**How to read this:** the view rotates with the Earth and Moon, so they stay still. "
    "The white loop is the halo orbit. **Blue** strands are the stable manifold: a spacecraft "
    "placed on one coasts onto the orbit with the engine off. **Amber** strands are the "
    "unstable manifold: they peel away from the orbit. Drag to rotate, scroll to zoom."
)
