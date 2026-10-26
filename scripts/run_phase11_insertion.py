"""Phase 11 post-processing: insertion delta-v at the closest approach to the halo.

The manifold coast is ~165 days through a chaotic region, so integration error (~1e-12)
is amplified enormously and the arc does not land exactly on the halo. A real mission
would fly small trajectory-correction burns. Here we estimate the final clean-up: the
impulse that turns the state at closest approach into the nearest halo state.
"""

import json

import numpy as np

from cislunar import shooting
from cislunar.constants import EARTH_MOON

U = EARTH_MOON
MU, L, V, D = U.mu, U.length_km, U.velocity_km_s, U.time_days
res = json.load(open("data/transfer_results.json"))
tr = np.load("data/transfer_trajectories.npz")
halo = shooting.SymmetricOrbit(np.array(res["halo_X0"]), res["halo_period_days"] / D, MU, 0, 0.0)
H = shooting.full_period(halo, n_points=20000).X[:, :6]

for key in ("d", "m"):
    X2 = tr[f"{key}_X2"]
    t2 = tr[f"{key}_t2"]
    tof_arc = res["direct" if key == "d" else "manifold"]["tof_arc_days"] / D
    # Only look up to one halo period past nominal arrival.
    win = t2 <= tof_arc + halo.period
    dpos = np.linalg.norm(X2[win, None, :3] - H[None, :, :3], axis=2)
    i, j = np.unravel_index(np.argmin(dpos), dpos.shape)
    dv_ins = np.linalg.norm(X2[win][i, 3:6] - H[j, 3:6]) * V * 1000
    name = "direct" if key == "d" else "manifold"
    res[name]["closest_approach_km"] = float(dpos[i, j] * L)
    res[name]["closest_approach_day"] = float((t2[win][i] - tof_arc) * D)
    res[name]["insertion_dv_m_s"] = float(dv_ins)
    print(
        f"{name}: closest approach {dpos[i, j] * L:.1f} km at {(t2[win][i] - tof_arc) * D:+.1f} d from nominal arrival; "
        f"clean-up insertion dv = {dv_ins:.1f} m/s"
    )
with open("data/transfer_results.json", "w") as fh:
    json.dump(res, fh, indent=2)
