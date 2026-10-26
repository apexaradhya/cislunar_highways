"""LEO-to-halo transfer design: direct baseline and manifold-based low-energy transfer.

PROJECT_SPEC.md Phase 11. Three ingredients.

1. Frames (spec 1.10). Earth-centred non-rotating ("inertial") frame aligned with the
   rotating frame at reference time tau = 0 (the patch time). At time tau the rotating
   frame has turned by angle tau, so for an Earth-relative position r_rot:
       r_in = Rz(tau) r_rot,   v_in = Rz(tau) (v_rot + w x r_rot),   w = (0, 0, 1).
   A velocity *difference* at a given point is the same in both frames (the w x r terms
   cancel), so impulsive delta-v can be computed in either.

2. Screening conics (two-body Earth, km and km/s). For a patch point with Earth-relative
   position r_p and inertial velocity v_m (the velocity we must match), consider conics that
   leave a circular LEO of radius r0 *tangentially* (perigee = r0, the cheapest way to leave)
   and pass through r_p at true anomaly nu, in the plane spanned by r_p and v_m.
   Perigee r0 and the point (r_p, nu) fix the conic:
       e = (r_p - r0) / (r0 - r_p cos nu),   p = r0 (1 + e)
   with speeds at r_p: v_r = sqrt(mu/p) e sin nu, v_theta = sqrt(mu/p) (1 + e cos nu)
   (Vallado 2013, Fundamentals of Astrodynamics and Applications, 4th ed., sec. 2.3).
       dv1 = sqrt(mu/p)(1 + e) - sqrt(mu/r0)       (burn in LEO, tangential)
       dv2 = | v_m - (v_r r_hat + v_theta theta_hat) |   (match at the patch point)
   We minimize dv1 + dv2 over nu. Time of flight from Kepler's equation.
   This ignores the Moon on the outbound leg; the best candidates are then corrected in
   the full CR3BP (3).

3. CR3BP correction. Single shooting: departure position fixed on the LEO sphere, free
   departure velocity (3 unknowns), target the patch position after the conic's time of
   flight (3 equations); Newton with the position-velocity block of the STM.

Patched-conic baseline to a low lunar orbit (textbook estimate, e.g. Curtis 2014,
Orbital Mechanics for Engineering Students, 3rd ed., sec. 8.4; Bate, Mueller & White 1971,
ch. 7): Hohmann-like ellipse from LEO to the Moon's distance, hyperbolic arrival at the Moon
with v_inf = |v_moon - v_apogee|, capture into a circular orbit at periapsis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import cr3bp, stm
from cislunar.constants import GM_EARTH_KM3_S2, GM_MOON_KM3_S2, R_EARTH_KM, R_MOON_KM, SystemUnits
from cislunar.integrators import dopri5

FloatArray = NDArray[np.float64]
W = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------
def _rz(a: float) -> FloatArray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_to_earth_inertial(
    X: FloatArray, mu: float, tau: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Rotating-frame state -> Earth-centred inertial (r, v), non-dimensional (spec 1.10)."""
    r = X[:3] - np.array([-mu, 0.0, 0.0])
    v = X[3:6] + np.cross(W, r)
    R = _rz(tau)
    return R @ r, R @ v


def earth_inertial_to_rot(
    r_in: FloatArray, v_in: FloatArray, mu: float, tau: float = 0.0
) -> FloatArray:
    """Inverse of rot_to_earth_inertial."""
    R = _rz(-tau)
    r = R @ r_in
    v = R @ v_in - np.cross(W, r)
    return np.concatenate([r + np.array([-mu, 0.0, 0.0]), v])


def circular_speed_km_s(radius_km: float, gm: float = GM_EARTH_KM3_S2) -> float:
    return math.sqrt(gm / radius_km)


# ---------------------------------------------------------------------------
# Patched-conic baseline (LEO -> low lunar orbit)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchedConicBaseline:
    leo_alt_km: float
    llo_alt_km: float
    dv_tli_km_s: float
    dv_loi_km_s: float
    tof_days: float
    v_inf_km_s: float

    @property
    def dv_total_km_s(self) -> float:
        return self.dv_tli_km_s + self.dv_loi_km_s


def patched_conic_llo(
    units: SystemUnits, leo_alt_km: float = 185.0, llo_alt_km: float = 100.0
) -> PatchedConicBaseline:
    """Estimate (not mission-grade): Hohmann LEO -> Moon distance, capture into circular LLO."""
    r0 = R_EARTH_KM + leo_alt_km
    ra = units.length_km
    a = 0.5 * (r0 + ra)
    v_circ = circular_speed_km_s(r0)
    v_peri = math.sqrt(GM_EARTH_KM3_S2 * (2.0 / r0 - 1.0 / a))
    v_apo = math.sqrt(GM_EARTH_KM3_S2 * (2.0 / ra - 1.0 / a))
    v_moon = math.sqrt((GM_EARTH_KM3_S2 + GM_MOON_KM3_S2) / ra)  # Moon's circular speed about Earth
    v_inf = abs(v_moon - v_apo)
    rp = R_MOON_KM + llo_alt_km
    v_hyp_peri = math.sqrt(v_inf**2 + 2.0 * GM_MOON_KM3_S2 / rp)
    dv_loi = v_hyp_peri - circular_speed_km_s(rp, GM_MOON_KM3_S2)
    tof = math.pi * math.sqrt(a**3 / GM_EARTH_KM3_S2) / 86400.0
    return PatchedConicBaseline(leo_alt_km, llo_alt_km, v_peri - v_circ, dv_loi, tof, v_inf)


# ---------------------------------------------------------------------------
# Screening: tangential-departure conics through a patch point (vectorized)
# ---------------------------------------------------------------------------
@dataclass
class ConicScreen:
    dv1: FloatArray  # km/s, LEO departure burn
    dv2: FloatArray  # km/s, patch-point burn
    nu: FloatArray  # true anomaly at the patch point (rad)
    e: FloatArray
    p: FloatArray  # semi-latus rectum, km
    tof_s: FloatArray  # perigee -> patch time of flight, s

    @property
    def total(self) -> FloatArray:
        return self.dv1 + self.dv2


def _time_since_perigee(nu: FloatArray, e: FloatArray, p: FloatArray, gm: float) -> FloatArray:
    """Kepler's equation, elliptic and hyperbolic branches (nu in (0, 2pi) for ellipses)."""
    out = np.full(np.shape(nu), np.nan)
    ell = e < 1.0
    hyp = e > 1.0
    if np.any(ell):
        en, nn, pn = e[ell], nu[ell], p[ell]
        a = pn / (1.0 - en**2)
        E = 2.0 * np.arctan2(
            np.sqrt(1.0 - en) * np.sin(nn / 2.0), np.sqrt(1.0 + en) * np.cos(nn / 2.0)
        )
        E = np.mod(E, 2.0 * np.pi)
        out[ell] = (E - en * np.sin(E)) * np.sqrt(a**3 / gm)
    if np.any(hyp):
        en, nn, pn = e[hyp], nu[hyp], p[hyp]
        a = pn / (en**2 - 1.0)
        F = 2.0 * np.arctanh(np.sqrt((en - 1.0) / (en + 1.0)) * np.tan(nn / 2.0))
        out[hyp] = (en * np.sinh(F) - F) * np.sqrt(a**3 / gm)
    return out


def screen_tangent_conics(
    r_p: FloatArray, v_m: FloatArray, r0_km: float, n_nu: int = 721, gm: float = GM_EARTH_KM3_S2
) -> ConicScreen:
    """Best tangential-departure conic for each patch point. r_p, v_m: (N, 3) km and km/s."""
    r_p = np.atleast_2d(r_p)
    v_m = np.atleast_2d(v_m)
    rn = np.linalg.norm(r_p, axis=1)
    r_hat = r_p / rn[:, None]
    vperp = v_m - np.sum(v_m * r_hat, axis=1)[:, None] * r_hat
    th_hat = vperp / np.linalg.norm(vperp, axis=1)[:, None]
    vm_r = np.sum(v_m * r_hat, axis=1)
    vm_t = np.sum(v_m * th_hat, axis=1)  # >= 0 by construction
    nu = np.linspace(1e-3, 2.0 * np.pi - 1e-3, n_nu)[None, :]
    rr = rn[:, None]
    denom = r0_km - rr * np.cos(nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = (rr - r0_km) / denom
        p = r0_km * (1.0 + e)
        valid = (denom > 0) & (e >= 0) & (1.0 + e * np.cos(nu) > 0)
        # Hyperbolas only reach nu < arccos(-1/e): 1 + e cos nu > 0 encodes it.
        sq = np.sqrt(gm / p)
        v_r = sq * e * np.sin(nu)
        v_t = sq * (1.0 + e * np.cos(nu))
        dv2 = np.sqrt((vm_r[:, None] - v_r) ** 2 + (vm_t[:, None] - v_t) ** 2)
        dv1 = sq * (1.0 + e) - math.sqrt(gm / r0_km)
    total = np.where(valid, dv1 + dv2, np.inf)
    k = np.argmin(total, axis=1)
    idx = np.arange(len(rn))
    nu_b, e_b, p_b = nu[0, k], e[idx, k], p[idx, k]
    tof = _time_since_perigee(nu_b, e_b, p_b, gm)
    return ConicScreen(dv1[idx, k], dv2[idx, k], nu_b, e_b, p_b, tof)


def conic_departure(
    r_p: FloatArray,
    v_m: FloatArray,
    r0_km: float,
    nu: float,
    e: float,
    p: float,
    gm: float = GM_EARTH_KM3_S2,
) -> tuple[FloatArray, FloatArray]:
    """Perigee (LEO) position and post-burn velocity of the screening conic (km, km/s, inertial)."""
    r_hat = r_p / np.linalg.norm(r_p)
    vperp = v_m - (v_m @ r_hat) * r_hat
    th_hat = vperp / np.linalg.norm(vperp)
    r_dep = r0_km * (math.cos(nu) * r_hat - math.sin(nu) * th_hat)
    v_dep = math.sqrt(gm / p) * (1.0 + e) * (math.sin(nu) * r_hat + math.cos(nu) * th_hat)
    return r_dep, v_dep


# ---------------------------------------------------------------------------
# CR3BP correction and end-to-end assembly
# ---------------------------------------------------------------------------
@dataclass
class Transfer:
    X_dep: FloatArray  # rotating state right after the LEO burn (time tau_dep)
    tau_dep: float  # departure time relative to the patch time (negative)
    X_patch_before: FloatArray  # arriving state at the patch point
    X_patch_after: FloatArray  # state after the patch burn (on the target arc)
    dv1_km_s: float
    dv2_km_s: float
    tof_leg1: float  # non-dimensional
    iterations: int
    miss: float  # final position miss at the patch point (non-dimensional)

    @property
    def dv_total_km_s(self) -> float:
        return self.dv1_km_s + self.dv2_km_s


class TransferError(RuntimeError):
    pass


def leo_velocity_for(
    r_in: FloatArray, v_in: FloatArray, r0_km: float, units: SystemUnits
) -> FloatArray:
    """Circular LEO velocity at r_in in the plane containing the departure velocity (non-dim)."""
    r_hat = r_in / np.linalg.norm(r_in)
    t = v_in - (v_in @ r_hat) * r_hat
    v_c = circular_speed_km_s(r0_km) / units.velocity_km_s
    return v_c * t / np.linalg.norm(t)


def correct_leg(
    X_patch: FloatArray,
    r_dep_km: FloatArray,
    v_dep_km_s: FloatArray,
    tof_s: float,
    r0_km: float,
    units: SystemUnits,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> Transfer:
    """Correct the screening conic in the full CR3BP so it hits the patch position.

    Inertial frame = rotating frame at the patch time (tau = 0). Departure at tau = -TOF.
    """
    mu = units.mu
    tof = tof_s / units.time_s
    tau_dep = -tof
    r_in = r_dep_km / units.length_km
    v_in = v_dep_km_s / units.velocity_km_s
    X0 = earth_inertial_to_rot(r_in, v_in, mu, tau_dep)
    target = X_patch[:3]

    def shoot(X: FloatArray, T: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        Xf, Phi = stm.split(stm.propagate_stm(X, 0.0, T, mu, rtol=1e-12, atol=1e-12).X_final)
        return Xf, Phi, Xf[:3] - target

    # Unknowns: departure velocity (3) and time of flight (1); equations: patch position (3).
    # With TOF fixed, the 3x3 block Phi_rv is nearly singular on these near-parabolic arcs
    # (smallest singular value ~1e-3), so Newton demands huge steps. Freeing TOF and taking
    # the minimum-norm step (pseudo-inverse) is well conditioned. Backtracking line search:
    # halve the step until the miss decreases.
    Xf, Phi, miss = shoot(X0, tof)
    for it in range(max_iter):  # noqa: B007 (iteration count is reported)
        if np.linalg.norm(miss) < tol:
            break
        J = np.column_stack([Phi[0:3, 3:6], Xf[3:6]])
        step = np.linalg.lstsq(J, -miss, rcond=None)[0]
        for _ in range(12):
            X_try = X0.copy()
            X_try[3:6] += step[:3]
            T_try = tof + step[3]
            if T_try > 0:
                Xf_t, Phi_t, miss_t = shoot(X_try, T_try)
                if np.linalg.norm(miss_t) < np.linalg.norm(miss):
                    break
            step *= 0.5
        else:
            raise TransferError(f"line search failed: miss {np.linalg.norm(miss):.2e}")
        X0, tof, Xf, Phi, miss = X_try, T_try, Xf_t, Phi_t, miss_t
    else:
        raise TransferError(f"leg correction failed: miss {np.linalg.norm(miss):.2e}")
    tau_dep = -tof
    # Delta-v at departure: from circular LEO (in the departure plane) to the corrected velocity.
    r_in0, v_in0 = rot_to_earth_inertial(X0, mu, tau_dep)
    v_leo = leo_velocity_for(r_in0, v_in0, r0_km, units)
    dv1 = float(np.linalg.norm(v_in0 - v_leo)) * units.velocity_km_s
    dv2 = float(np.linalg.norm(X_patch[3:6] - Xf[3:6])) * units.velocity_km_s
    return Transfer(X0, tau_dep, Xf, X_patch.copy(), dv1, dv2, tof, it, float(np.linalg.norm(miss)))


def leo_radius_nd(alt_km: float, units: SystemUnits) -> float:
    return (R_EARTH_KM + alt_km) / units.length_km


def propagate(
    X0: FloatArray, t0: float, tf: float, mu: float, n: int = 2000, tol: float = 1e-12
) -> tuple[FloatArray, FloatArray]:
    sol = dopri5(cr3bp.make_rhs(mu), t0, tf, X0, rtol=tol, atol=tol, t_eval=np.linspace(t0, tf, n))
    return sol.t, sol.X


# ---------------------------------------------------------------------------
# Candidate generation, correction and refinement
# ---------------------------------------------------------------------------
def earth_relative_km(X: FloatArray, units: SystemUnits) -> tuple[FloatArray, FloatArray]:
    """Rotating states (N, 6) -> Earth-relative inertial position (km), velocity (km/s), tau = 0."""
    r = X[:, :3] - np.array([-units.mu, 0.0, 0.0])
    v = X[:, 3:6] + np.cross(W, r)
    return r * units.length_km, v * units.velocity_km_s


@dataclass
class PatchCandidates:
    """Patch points (rotating states) with labels: branch sign, start phase, time along the arc."""

    X: FloatArray  # (N, 6)
    sign: FloatArray  # (N,) manifold branch (0 for points on the orbit itself)
    phase: FloatArray  # (N,) phase on the orbit, 0..1
    tau: FloatArray  # (N,) time from patch to arrival at the orbit (>= 0), non-dimensional


def manifold_patch_candidates(
    orbit,
    units: SystemUnits,
    n_strands: int = 80,
    duration: float = 40.0,
    dt: float = 0.02,
    branches: tuple[float, ...] = (1.0, -1.0),
    eps: float | None = None,
) -> PatchCandidates:
    """Sample the stable manifold backward in time: every point is a possible patch point."""
    from cislunar import manifolds

    eps = manifolds.DEFAULT_EPS if eps is None else eps
    s = manifolds.sample_orbit(orbit, n_strands)
    stops = manifolds.make_stop_events(
        units.mu, R_EARTH_KM / units.length_km, R_MOON_KM / units.length_km, r_escape=3.0
    )
    f = cr3bp.make_rhs(units.mu)
    Xs, sg, ph, ta = [], [], [], []
    for sign in branches:
        for t_i, X0 in zip(s.t, manifolds.manifold_states(s, "s", sign, eps), strict=True):
            sol = dopri5(
                f,
                0.0,
                -duration,
                X0,
                rtol=1e-11,
                atol=1e-11,
                events=stops,
                t_eval=np.arange(0.0, -duration, -dt),
            )
            n = len(sol.t)
            Xs.append(sol.X)
            sg.append(np.full(n, sign))
            ph.append(np.full(n, t_i / orbit.period))
            ta.append(-sol.t)
    return PatchCandidates(
        np.vstack(Xs), np.concatenate(sg), np.concatenate(ph), np.concatenate(ta)
    )


def orbit_patch_candidates(orbit, n: int = 400) -> PatchCandidates:
    """Points on the periodic orbit itself (direct insertion targets)."""
    from cislunar.shooting import full_period

    X = full_period(orbit, n_points=n + 1).X[:-1, :6]
    return PatchCandidates(X, np.zeros(n), np.arange(n) / n, np.zeros(n))


def screen_candidates(
    c: PatchCandidates, r0_km: float, units: SystemUnits, chunk: int = 4000, n_nu: int = 361
) -> ConicScreen:
    rp, vm = earth_relative_km(c.X, units)
    parts = [
        screen_tangent_conics(rp[a : a + chunk], vm[a : a + chunk], r0_km, n_nu=n_nu)
        for a in range(0, len(rp), chunk)
    ]
    cat = {
        k: np.concatenate([getattr(p, k) for p in parts])
        for k in ("dv1", "dv2", "nu", "e", "p", "tof_s")
    }
    return ConicScreen(**cat)


def correct_patch(X_patch: FloatArray, r0_km: float, units: SystemUnits) -> Transfer:
    """Fine conic screen for one patch point, then CR3BP correction of the outbound leg."""
    rp, vm = earth_relative_km(X_patch[None, :], units)
    sc = screen_tangent_conics(rp, vm, r0_km, n_nu=2001)
    r_dep, v_dep = conic_departure(rp[0], vm[0], r0_km, sc.nu[0], sc.e[0], sc.p[0])
    return correct_leg(X_patch, r_dep, v_dep, float(sc.tof_s[0]), r0_km, units)


def manifold_point(
    orbit, sign: float, phase: float, tau: float, eps: float | None = None
) -> FloatArray:
    """Stable-manifold state at continuous (phase, tau); tau >= 0 is the time left to the orbit."""
    from cislunar import manifolds

    eps = manifolds.DEFAULT_EPS if eps is None else eps
    M = stm.split(
        stm.propagate_stm(orbit.X0, 0.0, orbit.period, orbit.mu, rtol=1e-13, atol=1e-13).X_final
    )[1]
    v0 = manifolds.stable_unstable_directions(M).v_s
    if v0[0] < 0:
        v0 = -v0
    t_i = (phase % 1.0) * orbit.period
    if t_i > 0:
        Xg, Phi = stm.split(
            stm.propagate_stm(orbit.X0, 0.0, t_i, orbit.mu, rtol=1e-13, atol=1e-13).X_final
        )
    else:
        Xg, Phi = orbit.X0.copy(), np.eye(6)
    v = Phi @ v0
    v /= np.linalg.norm(v[:3])
    X0 = Xg + sign * eps * v
    if tau <= 0:
        return X0
    return dopri5(cr3bp.make_rhs(orbit.mu), 0.0, -tau, X0, rtol=1e-12, atol=1e-12).X_final


def orbit_point(orbit, phase: float) -> FloatArray:
    t_i = (phase % 1.0) * orbit.period
    if t_i == 0:
        return orbit.X0.copy()
    return dopri5(cr3bp.make_rhs(orbit.mu), 0.0, t_i, orbit.X0, rtol=1e-13, atol=1e-13).X_final


def refine(
    objective, x0: FloatArray, scale: FloatArray, max_evals: int = 80
) -> tuple[FloatArray, float]:
    """Nelder-Mead (scipy) on a CR3BP-corrected delta-v objective; failures score +inf."""
    from scipy.optimize import minimize

    def safe(x: FloatArray) -> float:
        try:
            return float(objective(x))
        except (TransferError, RuntimeError, np.linalg.LinAlgError):
            return math.inf

    res = minimize(
        safe,
        x0,
        method="Nelder-Mead",
        options=dict(
            maxfev=max_evals,
            xatol=1e-6,
            fatol=1e-5,
            initial_simplex=np.vstack([x0, x0 + np.diag(scale)]),
        ),
    )
    return res.x, float(res.fun)
