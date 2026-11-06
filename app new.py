"""Point-and-click app for the project.

Tab 1, Mission plan: pick a target orbit and get the burns, timing and fuel for the direct
route and the "highway" (manifold) route, plus the route itself in 3D.
Tab 2, Highways explorer: pick a halo orbit and watch its manifold tubes get computed live.

Run with:  uv run streamlit run app.py      (or: streamlit run app.py)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud only installs requirements.txt, not this project, so find the project's own
# code (the folder holding cislunar/lagrange.py) and its data folder wherever they sit in the repo.
HERE = Path(__file__).resolve().parent


def _find(pattern):
    hits = sorted(HERE.rglob(pattern), key=lambda q: len(q.parts))
    return hits[0] if hits else None


_pkg = _find("cislunar/lagrange.py")
_data = _find("data/design_sweep.json")
if _pkg is None or _data is None:
    st.error(
        "The app cannot find the project files it needs.\n\n"
        f"- `src/cislunar/lagrange.py` (the physics code): {'found' if _pkg else 'MISSING'}\n"
        f"- `data/design_sweep.json` (the computed routes): {'found' if _data else 'MISSING'}\n\n"
        "Upload the whole `src` folder and the whole `data` folder to the same GitHub repo as `app.py`, "
        "keeping their folder structure."
    )
    st.stop()
sys.path.insert(0, str(_pkg.parent.parent))
for _name in [m for m in sys.modules if m == "cislunar" or m.startswith("cislunar.")]:
    del sys.modules[_name]  # drop any half-found "cislunar" so the real one is loaded
ROOT = _data.parent.parent

try:
    from cislunar import lagrange, manifolds, orbits, transfer
    from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
    from cislunar.shooting import SymmetricOrbit, full_period
except Exception as exc:  # show the real reason instead of Streamlit's redacted error
    _needed = [
        "__init__",
        "constants",
        "units",
        "cr3bp",
        "integrators",
        "rootfind",
        "stm",
        "lagrange",
        "shooting",
        "orbits",
        "manifolds",
        "transfer",
    ]
    _have = {f.stem for f in _pkg.parent.glob("*.py")}
    _missing = [n + ".py" for n in _needed if n not in _have]
    st.error(
        f"Loading the physics code failed: `{type(exc).__name__}: {exc}`\n\n"
        f"Code folder used: `{_pkg.parent.relative_to(HERE)}`\n\n"
        + (
            f"Files missing from it: {', '.join(_missing)}"
            if _missing
            else "All the code files are there."
        )
    )
    st.stop()

U = EARTH_MOON
MU, L, DAYS = U.mu, U.length_km, U.time_days
G0 = 9.80665  # m/s^2, standard gravity (defines specific impulse in seconds)
STABLE, UNSTABLE, BURN, INK, DIM = "#5ec8ff", "#ffad5c", "#ff5c8d", "#e8edf6", "#8e98ad"

TIMELINE_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@600&family=IBM+Plex+Mono:wght@400&display=swap');
body { margin: 0; background: #04060b; overflow-x: auto; }
svg.tl { width: 100%; min-width: 640px; height: auto; display: block; margin: 4px 0 14px; }
svg.tl .grid { stroke: rgba(150,172,214,.14); }
svg.tl .tick { font: 400 11px 'IBM Plex Mono', monospace; fill: #8e98ad; }
svg.tl .rname { font: 600 15px 'Saira Condensed', sans-serif; letter-spacing: .04em; fill: #e8edf6; }
svg.tl .leg { stroke: #ff5c8d; stroke-width: 3; }
svg.tl .coast { stroke: #5ec8ff; stroke-width: 3; }
svg.tl .burn { fill: #ff5c8d; }
svg.tl .arrive { fill: #04060b; stroke: #e8edf6; stroke-width: 2; }
svg.tl .blab { font: 400 12px 'IBM Plex Mono', monospace; fill: #e8edf6; }
svg.tl .blab.dim { fill: #8e98ad; }
</style>"""

st.set_page_config(page_title="Cislunar Highways", layout="wide")
st.html(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
.eyebrow { font: 500 11px/1.4 'IBM Plex Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: #8e98ad; margin: 18px 0 2px; }
.masthead h1 { font: 700 44px/1 'Saira Condensed', sans-serif; letter-spacing: .02em; color: #e8edf6; margin: 4px 0 6px; padding: 0; }
.masthead p { font: 400 15px/1.55 'IBM Plex Sans', sans-serif; color: #c9d1e0; max-width: 62ch; margin: 0; }
.lede { font: 600 30px/1.15 'Saira Condensed', sans-serif; letter-spacing: .01em; color: #e8edf6; max-width: 30ch; margin: 2px 0 18px; padding: 0; }
.sec { font: 600 20px/1.2 'Saira Condensed', sans-serif; letter-spacing: .03em; color: #e8edf6; margin: 30px 0 8px; padding: 0; border-top: 1px solid rgba(150,172,214,.2); padding-top: 18px; }
.body { font: 400 14px/1.55 'IBM Plex Sans', sans-serif; color: #c9d1e0; max-width: 70ch; margin: 0 0 10px; }
.note { font: 400 12px/1.5 'IBM Plex Sans', sans-serif; color: #8e98ad; max-width: 72ch; margin: 8px 0 0; }
table.cmp, table.sheet { border-collapse: collapse; font: 400 13px/1.4 'IBM Plex Mono', monospace; color: #e8edf6; font-variant-numeric: tabular-nums; }
table.cmp th, table.cmp td, table.sheet th, table.sheet td { padding: 7px 14px 7px 0; border-bottom: 1px solid rgba(150,172,214,.14); text-align: right; }
table.cmp thead th, table.sheet thead th { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: #8e98ad; font-weight: 500; }
table.cmp tbody th { text-align: left; font: 400 14px 'IBM Plex Sans', sans-serif; color: #c9d1e0; padding-right: 28px; }
table.sheet th, table.sheet td { text-align: left; }
table.cmp td.good { color: #5ec8ff; }
table.cmp td.bad { color: #ffad5c; }
table.cmp td.u { color: #8e98ad; text-align: left; }
@media (max-width: 640px) { .lede { font-size: 24px; } .masthead h1 { font-size: 34px; } table.cmp, table.sheet { font-size: 12px; } }
</style>"""
)
st.html(
    '<div class="masthead"><p class="eyebrow">Earth–Moon transfer planner</p>'
    "<h1>Cislunar Highways</h1>"
    "<p>Plan a trip from low Earth orbit to a halo orbit near the Moon, two ways: straight "
    "there, or by the slow route where gravity does most of the work.</p></div>"
)

with st.expander("Why the slow route is worth knowing about"):
    st.markdown(
        """
Near the Moon there are special orbits (halo orbits) that are great places to park a space
station, like NASA's planned Gateway. Getting onto one normally takes two big engine burns:
one to **leave Earth orbit** and one to **match the halo orbit's speed and direction** when you get there.

The **invariant manifolds** ("highways") are the set of paths that *coast onto the halo orbit
by themselves*, with the engine off. If the spacecraft first steers onto a highway, gravity
does most of that matching for free, so the second burn becomes much smaller.

The catch: highway routes are slow (weeks to months instead of days), and they are
**sensitive**: a tiny error in the first burn grows quickly, so small correction burns are
needed along the way.

**What you get out of this app** (the numbers a mission planner needs):
1. **The burn plan:** when to fire the engine, how much speed change (Δv) each burn needs, and
   roughly which way to point.
2. **Fuel:** how much propellant that costs for your spacecraft, using the rocket equation.
3. **The trade-off:** fuel saved versus extra days of travel, direct route versus highway.
4. **Error budget:** how far off you end up if the first burn is slightly wrong, and how big
   the fix burn is.
"""
    )


# ---------------------------------------------------------------------------
# Data and computations (cached so each choice is only computed once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_sweep():
    with open(ROOT / "data/design_sweep.json") as f:
        return json.load(f)


@st.cache_data
def load_predictability():
    with open(ROOT / "data/predictability.json") as f:
        return json.load(f)


def halo_orbit(h: dict, mu: float) -> SymmetricOrbit:
    return SymmetricOrbit(np.array(h["X0"]), h["period_days"] / DAYS, mu, 0, 0.0)


def best_candidate(sweep: dict, halo_id: int, leo: float, kind: str) -> dict:
    return min(
        (
            c
            for c in sweep["candidates"]
            if c["halo"] == halo_id and c["leo_alt_km"] == leo and c["kind"] == kind
        ),
        key=lambda c: c["total"],
    )


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@st.cache_data
def rebuild_route(halo_id: int, leo: float, kind: str):
    """Rebuild a stored transfer in the full CR3BP: path, burn sizes, timing and directions."""
    sweep = load_sweep()
    mu = sweep["mu"]
    orbit = halo_orbit(sweep["halos"][halo_id], mu)
    c = best_candidate(sweep, halo_id, leo, kind)
    if kind == "direct":
        X_patch = transfer.orbit_point(orbit, c["phase"])
    else:
        X_patch = transfer.manifold_point(orbit, c["sign"], c["phase"], c["tau"])
    tr = transfer.correct_patch(X_patch, R_EARTH_KM + leo, U)

    _, leg1 = transfer.propagate(tr.X_dep, 0.0, tr.tof_leg1, mu, n=1500)
    coast = None
    if c["tau"] > 0:
        _, coast = transfer.propagate(tr.X_patch_after, 0.0, c["tau"], mu, n=2500)

    # Burn directions, measured against the direction of travel just before each burn.
    r_in, v_in = transfer.rot_to_earth_inertial(tr.X_dep, mu, tr.tau_dep)
    v_leo = transfer.leo_velocity_for(r_in, v_in, R_EARTH_KM + leo, U)
    a1 = angle_deg(v_in - v_leo, v_leo)
    _, v_before = transfer.earth_relative_km(tr.X_patch_before[None, :], U)
    dv2_vec = tr.X_patch_after[3:6] - tr.X_patch_before[3:6]
    a2 = angle_deg(dv2_vec, v_before[0])
    return dict(
        dv1=tr.dv1_km_s * 1000,
        dv2=tr.dv2_km_s * 1000,
        leg1_days=tr.tof_leg1 * DAYS,
        coast_days=c["tau"] * DAYS,
        angle1=a1,
        angle2=a2,
        leg1=leg1[:, :3],
        coast=None if coast is None else coast[:, :3],
        burn1=tr.X_dep[:3],
        burn2=tr.X_patch_before[:3],
    )


@st.cache_data
def halo_path(X0: tuple, period: float):
    orbit = SymmetricOrbit(np.array(X0), period, MU, 0, 0.0)
    return full_period(orbit, n_points=400).X[:, :3]


@st.cache_data
def load_family(point: str):
    d = np.load(ROOT / f"data/halo_{point}_north.npz")
    return d["X0"], d["period"]


@st.cache_data
def compute_tubes(point: str, index: int, south: bool, n_strands: int, days: float, kinds: tuple):
    X0s, periods = load_family(point)
    orbit = SymmetricOrbit(X0s[index].copy(), float(periods[index]), MU, 0, 0.0)
    if south:
        orbit = orbits.mirror_z(orbit)
    halo = full_period(orbit, n_points=400).X[:, :3]
    samples = manifolds.sample_orbit(orbit, n_strands)
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
        "Instability λu": float(samples.directions.lambda_u),
    }
    return halo, tubes, info


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def add_line(fig, P, colour, name, width=3, opacity=1.0, legend=True, dash=None):
    fig.add_trace(
        go.Scatter3d(
            x=P[:, 0] * L,
            y=P[:, 1] * L,
            z=P[:, 2] * L,
            mode="lines",
            line=dict(color=colour, width=width, dash=dash),
            opacity=opacity,
            name=name,
            showlegend=legend,
        )
    )


def add_point(fig, p, colour, text, size=6, pos="top center"):
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] * L],
            y=[p[1] * L],
            z=[p[2] * L],
            mode="markers+text",
            text=[text],
            textposition=pos,
            marker=dict(size=size, color=colour),
            showlegend=False,
        )
    )


def add_bodies(fig):
    pts = lagrange.lagrange_points(MU)
    for name in ("L1", "L2"):
        add_point(fig, [pts[name][0], pts[name][1], 0.0], DIM, name, size=3)
    add_point(fig, [-MU, 0, 0], STABLE, "Earth", size=12, pos="bottom center")
    add_point(fig, [1 - MU, 0, 0], DIM, "Moon", size=6, pos="bottom center")


def style(fig, height=640):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#04060b",
        font=dict(color=INK),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode="data", xaxis_title="x (km)", yaxis_title="y (km)", zaxis_title="z (km)"
        ),
        legend=dict(x=0.01, y=0.99),
    )


def fuel_kg(dv_m_s: float, dry_kg: float, isp_s: float) -> float:
    """Rocket equation: propellant needed for a spacecraft of dry mass dry_kg."""
    return dry_kg * (math.exp(dv_m_s / (isp_s * G0)) - 1.0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
tab_plan, tab_tubes = st.tabs(["Mission plan", "Highway map"])

with tab_plan:
    sweep = load_sweep()
    halos = sweep["halos"]
    side, main = st.columns([1, 2.9], gap="large")

    with side:
        st.html('<p class="eyebrow">Target</p>')
        hid = st.selectbox(
            "Halo orbit",
            options=[h["id"] for h in halos],
            index=4,
            format_func=lambda i: f"L{halos[i]['L']} · {halos[i]['Az_km']:,.0f} km tall",
        )
        leo = st.radio(
            "Parking orbit before departure",
            [185.0, 300.0],
            horizontal=True,
            format_func=lambda a: f"{a:.0f} km",
        )
        st.html('<p class="eyebrow">Spacecraft</p>')
        dry = st.number_input("Dry mass, kg", 100, 100_000, 1000, step=100)
        isp = st.number_input(
            "Engine Isp, s",
            100,
            500,
            320,
            step=5,
            help="Specific impulse of your engine. 320 s is only a placeholder; put in your engine's number.",
        )
        st.html(
            '<p class="note">Model: Earth and Moon only (no Sun), burns treated as instant. '
            "Each route is the cheapest one the project's design sweep found for this orbit, "
            "rebuilt here in the full three-body model.</p>"
        )

    with st.spinner("Rebuilding both routes..."):
        routes = {k: rebuild_route(hid, leo, k) for k in ("direct", "manifold")}
    d, m = routes["direct"], routes["manifold"]
    tot = {k: r["dv1"] + r["dv2"] for k, r in routes.items()}
    trip = {k: r["leg1_days"] + r["coast_days"] for k, r in routes.items()}
    fuel = {k: fuel_kg(tot[k], dry, isp) for k in routes}
    h = halos[hid]

    with main:
        st.html(
            f'<p class="eyebrow">L{h["L"]} halo · {h["Az_km"]:,.0f} km tall · '
            f"period {h['period_days']:.1f} d · from a {leo:.0f} km parking orbit</p>"
            f'<h2 class="lede">The highway saves {fuel["direct"] - fuel["manifold"]:,.0f} kg of propellant '
            f"and arrives {trip['manifold'] - trip['direct']:,.0f} days later.</h2>"
        )

        def row(name, a, b, unit, fmt="{:,.0f}", lower_is_better=True):
            diff = b - a
            good = (diff < 0) == lower_is_better
            cls = "good" if good and abs(diff) > 0 else "bad" if abs(diff) > 0 else ""
            sign = "−" if diff < 0 else "+"
            return (
                f"<tr><th>{name}</th><td>{fmt.format(a)}</td><td>{fmt.format(b)}</td>"
                f'<td class="{cls}">{sign}{fmt.format(abs(diff))}</td><td class="u">{unit}</td></tr>'
            )

        st.html(
            '<table class="cmp"><thead><tr><th></th><th>Direct</th><th>Highway</th>'
            "<th>Difference</th><th></th></tr></thead><tbody>"
            + row("Burn 1, leave Earth orbit", d["dv1"], m["dv1"], "m/s")
            + row("Burn 2, at the far end", d["dv2"], m["dv2"], "m/s")
            + row("Total Δv", tot["direct"], tot["manifold"], "m/s")
            + row(f"Propellant for {dry:,} kg dry", fuel["direct"], fuel["manifold"], "kg")
            + row("Trip time", trip["direct"], trip["manifold"], "days", "{:,.1f}")
            + "</tbody></table>"
        )

        # Timeline, both routes on the same day scale so the time cost is visible.
        span = max(trip.values())
        x0, x1 = 120, 980

        def X(day):
            return x0 + (x1 - x0) * day / span

        ticks = "".join(
            f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="18" y2="150"/>'
            f'<text class="tick" x="{X(t):.1f}" y="168" text-anchor="middle">day {t}</text>'
            for t in range(0, int(span) + 1, 30)
        )

        def burn(day, y, label, above):
            ty = y - 12 if above else y + 22
            return (
                f'<circle class="burn" cx="{X(day):.1f}" cy="{y}" r="5"/>'
                f'<text class="blab" x="{X(day) + 8:.1f}" y="{ty}">{label}</text>'
            )

        yd, yh = 52, 118
        svg = (
            '<svg class="tl" viewBox="0 0 1000 180" role="img" '
            'aria-label="Burn timeline for both routes">'
            + ticks
            + f'<text class="rname" x="0" y="{yd + 4}">Direct</text>'
            + f'<line class="leg" x1="{X(0):.1f}" x2="{X(d["leg1_days"]):.1f}" y1="{yd}" y2="{yd}"/>'
            + burn(0, yd, f"burn 1 · {d['dv1']:,.0f} m/s", True)
            + burn(
                d["leg1_days"],
                yd,
                f"burn 2 · {d['dv2']:,.0f} m/s · arrive day {d['leg1_days']:.1f}",
                False,
            )
            + f'<text class="rname" x="0" y="{yh + 4}">Highway</text>'
            + f'<line class="leg" x1="{X(0):.1f}" x2="{X(m["leg1_days"]):.1f}" y1="{yh}" y2="{yh}"/>'
            + f'<line class="coast" x1="{X(m["leg1_days"]):.1f}" x2="{X(trip["manifold"]):.1f}" y1="{yh}" y2="{yh}"/>'
            + burn(0, yh, f"burn 1 · {m['dv1']:,.0f} m/s", True)
            + burn(m["leg1_days"], yh, f"burn 2 · {m['dv2']:,.0f} m/s", False)
            + f'<text class="blab dim" x="{X((m["leg1_days"] + trip["manifold"]) / 2):.1f}" y="{yh - 10}" '
            f'text-anchor="middle">engine off for {m["coast_days"]:.0f} days</text>'
            + f'<circle class="arrive" cx="{X(trip["manifold"]):.1f}" cy="{yh}" r="5"/>'
            + f'<text class="blab" x="{X(trip["manifold"]):.1f}" y="{yh + 22}" text-anchor="end">'
            f"arrive day {trip['manifold']:.0f}</text>" + "</svg>"
        )
        st.html('<h3 class="sec">When to fire the engine</h3>')
        # st.html strips inline SVG, so the timeline goes in its own small frame with its own styles.
        components.html(TIMELINE_CSS + svg, height=200)

        def point(angle):
            if angle < 30:
                return f"along the velocity, {angle:.0f}° off"
            if angle > 150:
                return f"against the velocity, {180 - angle:.0f}° off"
            return f"{angle:.0f}° from the velocity"

        sheet = "".join(
            f"<tr><td>{route}</td><td>{n}</td><td>day {day:.1f}</td><td>{dv:,.0f} m/s</td>"
            f"<td>{point(a)}</td></tr>"
            for route, n, day, dv, a in (
                ("Direct", "1", 0.0, d["dv1"], d["angle1"]),
                ("Direct", "2", d["leg1_days"], d["dv2"], d["angle2"]),
                ("Highway", "1", 0.0, m["dv1"], m["angle1"]),
                ("Highway", "2", m["leg1_days"], m["dv2"], m["angle2"]),
            )
        )
        st.html(
            '<table class="sheet"><thead><tr><th>Route</th><th>Burn</th><th>When</th>'
            "<th>Δv</th><th>Engine pointing</th></tr></thead><tbody>" + sheet + "</tbody></table>"
            '<p class="note">Pointing is measured against the velocity relative to Earth just before '
            "the burn. On the highway route nothing fires after burn 2: the spacecraft coasts onto "
            "the halo orbit.</p>"
        )

        st.html('<h3 class="sec">If burn 1 is a little off</h3>')
        pred = load_predictability()
        if h["L"] == 2 and abs(h["Az_km"] - 20370) < 50 and leo == 185.0:
            mc_d, mc_m = pred["monte_carlo"]["direct"], pred["monte_carlo"]["manifold"]
            st.html(
                f'<p class="body">We flew each route {mc_m["n"]:,} times with a random 1 m/s error in '
                "burn 1, then fixed each one with a single correction burn.</p>"
                '<table class="cmp"><thead><tr><th></th><th>Direct</th><th>Highway</th><th></th></tr>'
                "</thead><tbody>"
                f"<tr><th>Typical miss without a fix</th><td>{mc_d['miss_median_km']:,.0f}</td>"
                f'<td>{mc_m["miss_median_km"]:,.0f}</td><td class="u">km</td></tr>'
                f"<tr><th>Typical fix burn</th><td>{mc_d['tcm_median_m_s']:.1f}</td>"
                f'<td>{mc_m["tcm_median_m_s"]:.1f}</td><td class="u">m/s</td></tr>'
                f"<tr><th>Fix burn, worst 1 in 20</th><td>{mc_d['tcm_95_m_s']:.1f}</td>"
                f'<td>{mc_m["tcm_95_m_s"]:.1f}</td><td class="u">m/s</td></tr>'
                "</tbody></table>"
                '<p class="note">Add the worst-case fix to the propellant budget. These numbers are for '
                "the project's main transfer to this orbit.</p>"
            )
        else:
            st.html(
                '<p class="body">The error study was only run for the project\'s main case, the '
                "L2 halo 20,370 km tall from a 185 km parking orbit. Select that to see it.</p>"
            )

        st.html('<h3 class="sec">The two routes, drawn</h3>')
        fig = go.Figure()
        add_line(fig, halo_path(tuple(h["X0"]), h["period_days"] / DAYS), INK, "Halo orbit", 5)
        add_line(fig, d["leg1"], DIM, "Direct", 3, dash="dash")
        add_line(fig, m["leg1"], BURN, "Highway, burn 1 to burn 2", 3)
        if m["coast"] is not None:
            add_line(fig, m["coast"], STABLE, "Highway, engine off", 3)
        add_point(fig, m["burn1"], BURN, "burn 1", size=5)
        add_point(fig, m["burn2"], BURN, "burn 2", size=5)
        add_bodies(fig)
        style(fig, 600)
        st.plotly_chart(fig, width="stretch")
        st.html(
            '<p class="note">Seen from a frame that turns with the Moon, so Earth and Moon stay put. '
            "The long blue loops are the highway: months of coasting while gravity lines the "
            "spacecraft up with the halo orbit. Drag to turn, scroll to zoom.</p>"
        )

with tab_tubes:
    c1, c2, c3 = st.columns(3)
    point = c1.radio("Which balance point?", ["L2", "L1"], horizontal=True)
    X0s, periods = load_family(point)
    sizes = np.abs(X0s[:, 2]) * L
    index = c1.select_slider(
        "Orbit size (vertical amplitude, km)",
        options=list(range(len(sizes))),
        value=len(sizes) // 3,
        format_func=lambda i: f"{sizes[i]:,.0f}",
    )
    south = c2.toggle("Southern orbit (mirror image)", value=False)
    show = c2.multiselect(
        "Highways to draw",
        ["Stable (leads onto the orbit)", "Unstable (leads away)"],
        default=["Stable (leads onto the orbit)", "Unstable (leads away)"],
    )
    n_strands = c3.slider("Number of strands per tube", 10, 60, 24, step=2)
    days = c3.slider("How far to follow them (days)", 5, 60, 30)
    st.caption("More strands or days = prettier but slower (a few seconds to ~1 minute).")

    kinds = tuple(
        k for k, label in (("s", "Stable"), ("u", "Unstable")) if any(label in s for s in show)
    )
    with st.spinner("Computing orbit and manifold tubes..."):
        halo, tubes, info = compute_tubes(point, int(index), south, n_strands, float(days), kinds)

    cols = st.columns(len(info))
    for col, (name, val) in zip(cols, info.items(), strict=True):
        col.metric(
            name, f"{val:,.0f}" if val > 100 else f"{val:,.4f}" if val < 10 else f"{val:,.1f}"
        )

    fig = go.Figure()
    colours = {"s": STABLE, "u": UNSTABLE}
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
                showlegend=not any(t.name == names[kind] for t in fig.data),
            )
        )
    add_line(fig, halo, INK, f"{point} halo orbit", 6)
    add_bodies(fig)
    style(fig, 720)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "**How to read this:** the white loop is the halo orbit. **Blue** strands are the stable "
        "manifold: a spacecraft placed on one coasts onto the orbit with the engine off. **Amber** "
        "strands are the unstable manifold: they peel away from the orbit."
    )
