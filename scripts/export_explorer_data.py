"""Export every precomputed result the cinematic explorer needs into web/explorer_data.json.

Nothing here is invented for the visuals: every curve is a trajectory computed by the
package (Phases 3-11c). Arrays are Float32, base64-encoded, positions in non-dimensional
rotating-frame units (1 = 384,400 km).
"""

import base64
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cislunar import hill, lagrange, manifolds, shooting, transfer  # noqa: E402
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM  # noqa: E402
from cislunar.cr3bp import make_rhs  # noqa: E402
from cislunar.integrators import dopri5  # noqa: E402

U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
f = make_rhs(MU)


def b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def resample(X, n):
    idx = np.linspace(0, len(X) - 1, n).astype(int)
    return X[idx]


def halo_near(w, z0):
    d = np.load(f"data/halo_L{w}_north.npz")
    return shooting.correct_symmetric(
        d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - z0)))], MU, free=(0, 4), targets=(3, 5)
    )


out: dict = {}
pts = lagrange.lagrange_points(MU)
C = lagrange.jacobi_at_points(MU)
out["meta"] = dict(
    mu=MU,
    L_km=L,
    T_days=D,
    V_km_s=U.velocity_km_s,
    R_E=R_EARTH_KM / L,
    R_M=R_MOON_KM / L,
    lagrange={k: v.tolist() for k, v in pts.items()},
    C_points=C,
)

# --- zero-velocity curves at many energies (for the energy dial / walls) ---
g = hill.planar_grid(MU, half_width=1.4, n=561)
levels = np.concatenate(
    [
        np.linspace(C["L1"] + 0.06, C["L1"] - 0.002, 10),
        np.linspace(C["L1"] - 0.004, C["L3"] - 0.002, 14),
        np.linspace(C["L3"] - 0.004, 3.0005, 8),
    ]
)
fig, ax = plt.subplots()
hl = []
for c in levels:
    cs = ax.contour(g.x, g.y, g.two_omega, levels=[c])
    segs = [s for s in cs.allsegs[0] if len(s) > 3]
    hl.append(dict(C=float(c), segs=[b64(resample(s, min(len(s), 400))) for s in segs]))
plt.close(fig)
out["hill"] = hl

# --- halo families (every 3rd member) ---
fam = {}
for w in (1, 2):
    d = np.load(f"data/halo_L{w}_north.npz")
    orbs = []
    for i in range(0, len(d["X0"]), 3):
        o = shooting.SymmetricOrbit(d["X0"][i], float(d["period"][i]), MU, 0, 0.0)
        X = shooting.full_period(o, n_points=120).X[:, :3]
        orbs.append(dict(C=float(d["jacobi"][i]), T=float(d["period"][i]), xyz=b64(X)))
    fam[f"L{w}"] = orbs
out["halo_families"] = fam

# --- target halos and their manifold tubes ---
tubes = {}
for w, z0 in ((1, 0.05), (2, 0.053)):
    o = halo_near(w, z0)
    s = manifolds.sample_orbit(o, 40)
    entry = dict(
        orbit=b64(shooting.full_period(o, n_points=300).X[:, :3]),
        C=o.jacobi,
        T=o.period,
        lambda_u=s.directions.lambda_u,
        branches={},
    )
    for kind in "su":
        for sign in (1.0, -1.0):
            tb = manifolds.generate_tube(
                o,
                kind,
                sign,
                8.0,
                r_earth=R_EARTH_KM / L,
                r_moon=R_MOON_KM / L,
                samples=s,
                n_out=None,
            )
            strands = []
            for trj in tb:
                X = trj.X[:, :3]
                if kind == "s":
                    X = X[::-1]  # store in the direction of motion (toward the halo)
                strands.append(b64(resample(X, 140)))
            entry["branches"][f"{kind}{'+' if sign > 0 else '-'}"] = strands
    tubes[f"L{w}"] = entry
out["tubes"] = tubes

# --- pickable stable strands (L2, Moon-side branch), 96 phases, ~45 days each, motion order ---
o2 = halo_near(2, 0.053)
s96 = manifolds.sample_orbit(o2, 96)
pick = []
for X0 in manifolds.manifold_states(s96, "s", -1.0):
    sol = dopri5(f, 0, -10.0, X0, rtol=1e-11, atol=1e-11, t_eval=np.linspace(0, -10.0, 220))
    pick.append(b64(sol.X[::-1, :3]))
out["picker"] = dict(
    phases=(s96.t / o2.period).tolist(), strands=pick, days=10.0 * D, halo_points=b64(s96.X[:, :3])
)

# --- transfers (rotating states + times; inertial conversion happens in the page) ---
res = json.load(open("data/transfer_results.json"))
tr = np.load("data/transfer_trajectories.npz")
trs = {}
for key, name in (("d", "direct"), ("m", "manifold")):
    info = res[name]
    t1, X1, t2, X2 = tr[f"{key}_t1"], tr[f"{key}_X1"], tr[f"{key}_t2"], tr[f"{key}_X2"]
    keep = t2 <= info["tof_arc_days"] / D + 2.0
    t = np.concatenate([t1, t2[keep]])
    X = np.vstack([X1[:, :3], X2[keep, :3]])
    n = 700 if key == "d" else 2400
    idx = np.unique(
        np.concatenate(
            [
                np.linspace(0, len(t1) - 1, n // 4).astype(int),
                len(t1) + np.linspace(0, keep.sum() - 1, 3 * n // 4).astype(int),
            ]
        )
    )
    trs[name] = dict(
        t=b64(t[idx]),
        xyz=b64(X[idx]),
        t_burn2=0.0,
        t_dep=float(t1[0]),
        arc_days=info["tof_arc_days"],
        dv1=info["dv1_km_s"],
        dv2=info["dv2_km_s"],
        total=info["dv_total_km_s"],
        tof_days=info["tof_total_days"],
    )
out["transfers"] = trs
out["patched_conic"] = res["patched_conic"]

# --- butterfly fan ---
fanf = np.load("data/butterfly_fan.npz")
nt = len(fanf["t"])
out["fan"] = dict(
    t=b64(fanf["t"]),
    n=int(fanf["states"].shape[1]),
    frames=nt,
    xyz=b64(fanf["states"]),
    ref=b64(fanf["reference"]),
    predict=json.load(open("data/predictability.json")),
)

# --- NRHO ---
nr = json.load(open("data/nrho_9_2_south.json"))
on = shooting.SymmetricOrbit(np.array(nr["X0"]), nr["period_nd"], MU, 0, 0.0)
out["nrho"] = dict(
    xyz=b64(shooting.full_period(on, n_points=400).X[:, :3]),
    **{
        k: nr[k]
        for k in ("period_days", "perilune_radius_km", "perilune_altitude_km", "apolune_radius_km")
    },
)

# --- heteroclinic L1 -> L2 connection ---
het = json.load(open("data/heteroclinic_L1_L2_C3.17.json"))["connections"][0]
Xs = np.array(het["section_state"])
back = dopri5(
    f,
    0,
    -het["t_L1_to_section"],
    Xs,
    rtol=1e-12,
    atol=1e-12,
    t_eval=np.linspace(0, -het["t_L1_to_section"], 400),
).X[::-1, :3]
fwd = dopri5(
    f,
    0,
    het["t_section_to_L2"],
    Xs,
    rtol=1e-12,
    atol=1e-12,
    t_eval=np.linspace(0, het["t_section_to_L2"], 400),
).X[:, :3]
out["heteroclinic"] = dict(
    xyz=b64(np.vstack([back, fwd])), days=(het["t_L1_to_section"] + het["t_section_to_L2"]) * D
)

# --- design sweep: all candidates + trajectories for the Pareto front ---
try:
    sw = json.load(open("data/design_sweep.json"))
except FileNotFoundError:
    sw = None
if sw:
    cands = [c for c in sw["candidates"] if c["leo_alt_km"] == 185.0]
    order = sorted(cands, key=lambda c: c["tof_days"])
    front, best = [], np.inf
    for c in order:
        if c["total"] < best - 1e-9:
            front.append(c)
            best = c["total"]
    halos = {h["id"]: h for h in sw["halos"]}
    front_tr = []
    for c in front:
        h = halos[c["halo"]]
        o = shooting.SymmetricOrbit(np.array(h["X0"]), h["period_days"] / D, MU, 0, 0.0)
        try:
            Xp = (
                transfer.manifold_point(o, c["sign"], c["phase"], c["tau"])
                if c["kind"] == "manifold"
                else transfer.orbit_point(o, c["phase"])
            )
            t_ = transfer.correct_patch(Xp, R_EARTH_KM + 185.0, U)
        except Exception:  # noqa: BLE001 (skip any candidate that fails to rebuild)
            continue
        _, X1 = transfer.propagate(t_.X_dep, t_.tau_dep, 0.0, MU, n=200)
        _, X2 = transfer.propagate(t_.X_patch_after, 0.0, max(c["tau"], 1e-6), MU, n=500)
        front_tr.append(
            dict(
                c,
                leg=b64(X1[:, :3]),
                arc=b64(X2[:, :3]),
                orbit=b64(shooting.full_period(o, n_points=200).X[:, :3]),
                rebuilt_total=t_.dv_total_km_s,
            )
        )
    out["sweep"] = dict(
        halos=[
            {k: h[k] for k in ("id", "L", "period_days", "jacobi", "Az_km")} for h in sw["halos"]
        ],
        points=[
            [c["tof_days"], c["total"], c["halo"], 1 if c["kind"] == "manifold" else 0]
            for c in cands
        ],
        front=front_tr,
    )
with open("web/explorer_data.json", "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print("explorer data size: %.2f MB" % (len(json.dumps(out)) / 1e6))
