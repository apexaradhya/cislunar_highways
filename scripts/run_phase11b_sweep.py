"""Phase 11b: design-space sweep. For several L1/L2 halos and two LEO altitudes, screen and
CR3BP-correct direct and manifold transfers; save every candidate (knobs -> delta-v, TOF).

Knobs: libration point, halo size (z0 -> C), manifold branch and phase, time along the
manifold, LEO altitude. eps is fixed at the project default (Phase 9 showed it only
relabels strands). Output: data/design_sweep.json (used by the Pareto figure and the explorer).
"""

import json
import time

import numpy as np

from cislunar import transfer
from cislunar.constants import EARTH_MOON
from cislunar.shooting import correct_symmetric

U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
HALOS = [(1, 0.02), (1, 0.06), (1, 0.12), (2, 0.02), (2, 0.053), (2, 0.10), (2, 0.15)]
LEO_ALTS = (185.0, 300.0)

rows, halos = [], []
for w, z0 in HALOS:
    d = np.load(f"data/halo_L{w}_north.npz")
    o = correct_symmetric(
        d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - z0)))], MU, free=(0, 4), targets=(3, 5)
    )
    hid = len(halos)
    halos.append(
        dict(
            id=hid,
            L=w,
            X0=o.X0.tolist(),
            period_days=o.period * D,
            jacobi=o.jacobi,
            Az_km=abs(o.X0[2]) * L,
        )
    )
    for alt in LEO_ALTS:
        t0 = time.perf_counter()
        cands = transfer.design_candidates(o, U, leo_alt_km=alt)
        for c in cands:
            c["halo"] = hid
        rows += cands
        best = {
            k: min((c["total"] for c in cands if c["kind"] == k), default=np.nan)
            for k in ("direct", "manifold")
        }
        print(
            f"L{w} halo z0={o.X0[2] * L:6.0f} km (C={o.jacobi:.4f}), LEO {alt:.0f} km: {len(cands)} candidates, "
            f"best direct {best['direct']:.3f}, best manifold {best['manifold']:.3f} km/s [{time.perf_counter() - t0:.0f}s]",
            flush=True,
        )
with open("data/design_sweep.json", "w") as fh:
    json.dump(dict(mu=MU, halos=halos, candidates=rows), fh)
print(f"saved {len(rows)} candidates")
