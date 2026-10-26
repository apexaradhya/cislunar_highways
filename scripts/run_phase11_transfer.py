"""Phase 11: LEO -> L2 halo transfers. Direct baseline vs stable-manifold (low-energy) transfer.

Pipeline (same for both, so the comparison is fair):
  1. candidate patch points: points ON the halo (direct) / points on its stable manifold
  2. screen each with the tangential-departure conic (fast, two-body Earth)
  3. correct the best distinct candidates in the full CR3BP (outbound leg shooting)
  4. refine the best few with Nelder-Mead on the CR3BP-corrected total delta-v
  5. assemble the winner end to end in the CR3BP and check Jacobi-constant bookkeeping
"""

import json

import numpy as np

from cislunar import cr3bp, shooting, transfer
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.shooting import correct_symmetric

U = EARTH_MOON
MU, L, V, D = U.mu, U.length_km, U.velocity_km_s, U.time_days
LEO_ALT_KM = 185.0  # parking-orbit altitude (spec: 185-300 km); a parameter
R0 = R_EARTH_KM + LEO_ALT_KM
Z0_HALO = 0.053  # same L2 northern halo as Phases 9-10 (Az ~ 20,000 km)
N_CORRECT = 60  # distinct manifold candidates corrected in the CR3BP
N_REFINE = 3

d = np.load("data/halo_L2_north.npz")
halo = correct_symmetric(
    d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - Z0_HALO)))], MU, free=(0, 4), targets=(3, 5)
)
print(
    f"Target: L2 northern halo, z0 = {halo.X0[2] * L:,.0f} km, period {halo.period * D:.2f} d, C = {halo.jacobi:.6f}"
)

baseline = transfer.patched_conic_llo(U, LEO_ALT_KM, 100.0)
print(
    f"Patched-conic LEO->LLO estimate: TLI {baseline.dv_tli_km_s:.3f} + LOI {baseline.dv_loi_km_s:.3f} "
    f"= {baseline.dv_total_km_s:.3f} km/s, {baseline.tof_days:.2f} d"
)


def distinct_best(c, sc, n, key_days=3.0):
    order = np.argsort(sc.total)
    seen, out = set(), []
    for k in order:
        if not np.isfinite(sc.total[k]):
            break
        key = (c.sign[k], round(c.phase[k] * 80), round(c.tau[k] * D / key_days))
        if key in seen:
            continue
        seen.add(key)
        out.append(int(k))
        if len(out) >= n:
            break
    return out


def try_correct(X):
    try:
        return transfer.correct_patch(X, R0, U)
    except (transfer.TransferError, RuntimeError, np.linalg.LinAlgError):
        return None


# ---- direct: patch ON the halo ----
cd = transfer.orbit_patch_candidates(halo, 400)
scd = transfer.screen_candidates(cd, R0, U)
direct = []
for k in distinct_best(cd, scd, 20, key_days=1e9):
    tr = try_correct(cd.X[k])
    if tr is not None:
        direct.append(dict(phase=float(cd.phase[k]), tr=tr, tof_days=tr.tof_leg1 * D))
direct.sort(key=lambda r: r["tr"].dv_total_km_s)
print(f"direct: {len(direct)} corrected; best {direct[0]['tr'].dv_total_km_s:.3f} km/s")

# ---- manifold: patch on the stable manifold ----
cm = transfer.manifold_patch_candidates(halo, U, n_strands=80, duration=40.0)
scm = transfer.screen_candidates(cm, R0, U)
print(f"manifold: {len(cm.X):,} patch points screened")
manifold = []
for k in distinct_best(cm, scm, N_CORRECT):
    tr = try_correct(cm.X[k])
    if tr is not None:
        manifold.append(
            dict(
                sign=float(cm.sign[k]),
                phase=float(cm.phase[k]),
                tau=float(cm.tau[k]),
                tr=tr,
                tof_days=(tr.tof_leg1 + cm.tau[k]) * D,
                screen=float(scm.total[k]),
            )
        )
manifold.sort(key=lambda r: r["tr"].dv_total_km_s)
print(f"manifold: {len(manifold)} corrected; best {manifold[0]['tr'].dv_total_km_s:.3f} km/s")

# ---- refinement (Nelder-Mead on CR3BP-corrected delta-v) ----
refined_d = []
for r in direct[:2]:
    x, fval = transfer.refine(
        lambda x: transfer.correct_patch(transfer.orbit_point(halo, x[0]), R0, U).dv_total_km_s,
        np.array([r["phase"]]),
        np.array([0.01]),
        max_evals=40,
    )
    tr = transfer.correct_patch(transfer.orbit_point(halo, x[0]), R0, U)
    refined_d.append(dict(phase=float(x[0]), tr=tr, tof_days=tr.tof_leg1 * D))
    print(f"  direct refined: {r['tr'].dv_total_km_s:.3f} -> {tr.dv_total_km_s:.3f} km/s")
refined_m = []
for r in manifold[:N_REFINE]:
    sign = r["sign"]

    def obj(x, s=sign):
        return transfer.correct_patch(
            transfer.manifold_point(halo, s, x[0], x[1]), R0, U
        ).dv_total_km_s

    x, fval = transfer.refine(
        obj, np.array([r["phase"], r["tau"]]), np.array([0.005, 0.05]), max_evals=80
    )
    Xp = transfer.manifold_point(halo, sign, x[0], x[1])
    tr = transfer.correct_patch(Xp, R0, U)
    refined_m.append(
        dict(
            sign=sign,
            phase=float(x[0]),
            tau=float(x[1]),
            tr=tr,
            tof_days=(tr.tof_leg1 + x[1]) * D,
            X_patch=Xp,
        )
    )
    print(
        f"  manifold refined: {r['tr'].dv_total_km_s:.3f} -> {tr.dv_total_km_s:.3f} km/s (TOF {(tr.tof_leg1 + x[1]) * D:.1f} d)"
    )
best_d = min(refined_d, key=lambda r: r["tr"].dv_total_km_s)
best_m = min(refined_m, key=lambda r: r["tr"].dv_total_km_s)


# ---- end-to-end assembly in the CR3BP ----
def assemble(tr, tau_arc, label):
    """Leg 1 (LEO -> patch), burn 2, coast along the target arc for tau_arc, then 1 halo period more."""
    t1, X1 = transfer.propagate(tr.X_dep, tr.tau_dep, 0.0, MU, n=3000)
    t2, X2 = transfer.propagate(tr.X_patch_after, 0.0, tau_arc + halo.period, MU, n=6000)
    r_in, v_in = transfer.rot_to_earth_inertial(tr.X_dep, MU, tr.tau_dep)
    X_leo = transfer.earth_inertial_to_rot(
        r_in, transfer.leo_velocity_for(r_in, v_in, R0, U), MU, tr.tau_dep
    )
    C1, C2 = cr3bp.jacobi_constant(X1, MU), cr3bp.jacobi_constant(X2, MU)
    curve = shooting.full_period(halo, n_points=3000).X[:, :3]
    dist = np.min(np.linalg.norm(X2[:, None, :3] - curve[None, :, :], axis=2), axis=1) * L
    t_arr = tau_arc
    i_arr = int(np.argmin(np.abs(t2 - t_arr)))
    moon1 = np.min(np.linalg.norm(X1[:, :3] - [1 - MU, 0, 0], axis=1)) * L - R_MOON_KM
    moon2 = np.min(np.linalg.norm(X2[:, :3] - [1 - MU, 0, 0], axis=1)) * L - R_MOON_KM
    earth1 = np.min(np.linalg.norm(X1[:, :3] - [-MU, 0, 0], axis=1)) * L - R_EARTH_KM
    info = dict(
        label=label,
        dv1_km_s=tr.dv1_km_s,
        dv2_km_s=tr.dv2_km_s,
        dv_total_km_s=tr.dv_total_km_s,
        tof_leg1_days=tr.tof_leg1 * D,
        tof_arc_days=tau_arc * D,
        tof_total_days=(tr.tof_leg1 + tau_arc) * D,
        jacobi_leo=float(cr3bp.jacobi_constant(X_leo, MU)),
        jacobi_leg1=float(C1[0]),
        jacobi_leg1_drift=float(np.ptp(C1)),
        jacobi_after_burn2=float(C2[0]),
        jacobi_arc_drift=float(np.ptp(C2)),
        jacobi_halo=halo.jacobi,
        arrival_distance_km=float(dist[i_arr]),
        min_distance_to_halo_km=float(dist.min()),
        min_lunar_altitude_leg1_km=float(moon1),
        min_lunar_altitude_arc_km=float(moon2),
        min_earth_altitude_leg1_km=float(earth1),
        patch_miss_km=tr.miss * L,
    )
    return info, (t1, X1, t2, X2, dist)


info_d, traj_d = assemble(best_d["tr"], 0.0, "direct")
info_m, traj_m = assemble(best_m["tr"], best_m["tau"], "manifold")
for info in (info_d, info_m):
    print(
        json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in info.items()})
    )

np.savez_compressed(
    "data/transfer_trajectories.npz",
    d_t1=traj_d[0],
    d_X1=traj_d[1],
    d_t2=traj_d[2],
    d_X2=traj_d[3],
    m_t1=traj_m[0],
    m_X1=traj_m[1],
    m_t2=traj_m[2],
    m_X2=traj_m[3],
    m_dist=traj_m[4],
    mu=MU,
)
pareto = [
    dict(tof_days=r["tof_days"], dv=r["tr"].dv_total_km_s, kind="manifold") for r in manifold
] + [dict(tof_days=r["tof_days"], dv=r["tr"].dv_total_km_s, kind="direct") for r in direct]
with open("data/transfer_results.json", "w") as fh:
    json.dump(
        dict(
            leo_alt_km=LEO_ALT_KM,
            halo_X0=halo.X0.tolist(),
            halo_period_days=halo.period * D,
            halo_jacobi=halo.jacobi,
            patched_conic=dict(
                tli=baseline.dv_tli_km_s,
                loi=baseline.dv_loi_km_s,
                total=baseline.dv_total_km_s,
                tof_days=baseline.tof_days,
            ),
            direct=info_d,
            manifold=info_m,
            manifold_choice=dict(sign=best_m["sign"], phase=best_m["phase"], tau=best_m["tau"]),
            direct_choice=dict(phase=best_d["phase"]),
            candidates=pareto,
        ),
        fh,
        indent=2,
    )
print("saved data/transfer_results.json")
