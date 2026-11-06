"""Predictability: how fast do small errors grow, and what does that cost? (Phase 11c)

1. Butterfly fan. Many copies of a state on an unstable periodic orbit, each displaced by
   a tiny random error, propagated together. Linear theory (Phase 8) predicts the spread
   grows on average by lambda_u per period, so the time to grow from sigma0 to d is
       t(d) ~= T ln(d / sigma0) / ln(lambda_u)                         (prediction)
   This is a *forecast horizon*: beyond it, the initial state no longer determines where
   the spacecraft is (deterministic chaos, Parker & Chua 1989, ch. 1).

2. Burn-error Monte Carlo. Random errors in the trans-lunar injection burn, propagated in
   the full CR3BP; the trajectory-correction manoeuvre (TCM) that re-targets the patch
   position is computed with the STM of the remaining leg (linear targeting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import cr3bp, stm
from cislunar.constants import SystemUnits
from cislunar.integrators import dopri5

FloatArray = NDArray[np.float64]


def make_batch_rhs(mu: float):
    """f(t, Y) for N independent CR3BP states stacked in one vector of length 6N."""

    def rhs(t: float, Y: FloatArray) -> FloatArray:
        return cr3bp.equations_of_motion(t, Y.reshape(-1, 6), mu).ravel()

    return rhs


def propagate_batch(
    X0s: FloatArray, t_final: float, mu: float, t_eval: FloatArray, tol: float = 1e-11
) -> FloatArray:
    """Propagate (N, 6) states together; returns (len(t_eval), N, 6).

    One shared adaptive step for all copies (the error norm covers all of them), which is
    as accurate as the worst copy needs and much faster than N separate integrations.
    """
    sol = dopri5(make_batch_rhs(mu), 0.0, t_final, X0s.ravel(), rtol=tol, atol=tol, t_eval=t_eval)
    return sol.X.reshape(len(t_eval), -1, 6)


@dataclass
class Fan:
    t: FloatArray  # (n_t,)
    states: FloatArray  # (n_t, N, 6)
    reference: FloatArray  # (n_t, 6) unperturbed trajectory
    sigma0_nd: float  # rms initial position error (non-dimensional)
    dX0: FloatArray  # (N, 6) initial errors

    def spread(self) -> FloatArray:
        """rms position distance from the reference trajectory, per time (non-dimensional)."""
        d = np.linalg.norm(self.states[:, :, :3] - self.reference[:, None, :3], axis=2)
        return np.sqrt(np.mean(d**2, axis=1))


def butterfly_fan(
    X0: FloatArray,
    mu: float,
    units: SystemUnits,
    n: int,
    sigma_r_km: float,
    sigma_v_m_s: float,
    duration: float,
    n_out: int = 400,
    seed: int = 0,
) -> Fan:
    """n copies of X0 with Gaussian position/velocity errors (per axis), propagated together."""
    rng = np.random.default_rng(seed)
    dr = rng.normal(scale=sigma_r_km / units.length_km, size=(n, 3))
    dv = rng.normal(scale=sigma_v_m_s / 1000.0 / units.velocity_km_s, size=(n, 3))
    X0s = np.concatenate([X0 + np.hstack([dr, dv]), X0[None, :]])  # last row = reference
    t = np.linspace(0.0, duration, n_out)
    S = propagate_batch(X0s, duration, mu, t)
    sigma0 = float(np.sqrt(np.mean(np.sum(dr**2, axis=1))))
    return Fan(t, S[:, :-1], S[:, -1], sigma0, np.hstack([dr, dv]))


def predicted_time_to_reach(d: float, sigma0: float, period: float, lambda_u: float) -> float:
    """t = T ln(d/sigma0) / ln(lambda_u): when a sigma0-sized error has grown to size d."""
    return period * np.log(d / sigma0) / np.log(lambda_u)


def unstable_amplitude(dX: FloatArray, M: FloatArray) -> tuple[FloatArray, float]:
    """Size of each error's component along the unstable eigenvector, in position units.

    Write dX = sum_i c_i v_i over the eigenvectors of M. With w_u the *left* eigenvector
    (w_u^T M = lambda_u w_u^T), c_u = (w_u . dX) / (w_u . v_u). Only this component grows
    like lambda_u^(t/T); the others stay bounded or shrink. v_u is scaled to unit position
    part, so |c_u| is directly the position-error amplitude. Returns (|c_u| per error, lambda_u).
    """
    w, Vr = np.linalg.eig(M)
    iu = int(np.argmax(np.abs(w)))
    lam = float(w[iu].real)
    v_u = Vr[:, iu].real
    v_u /= np.linalg.norm(v_u[:3])
    wl, Vl = np.linalg.eig(M.T)
    w_u = Vl[:, int(np.argmin(np.abs(wl - lam)))].real
    return np.abs(dX @ w_u) / abs(w_u @ v_u), lam


@dataclass
class BurnMonteCarlo:
    miss_km: FloatArray  # patch-point position miss without correction (km)
    tcm_m_s: FloatArray  # TCM delta-v to re-target the patch position (m/s), linear targeting
    tcm_check_km: FloatArray  # nonlinear residual miss after applying the linear TCM (km)


def burn_error_monte_carlo(
    X_dep: FloatArray,
    tof: float,
    X_patch: FloatArray,
    units: SystemUnits,
    sigma_m_s: float,
    n: int,
    t_tcm: float,
    n_check: int = 20,
    seed: int = 0,
) -> BurnMonteCarlo:
    """Gaussian errors (per axis, sigma) on the departure burn; TCM at t_tcm re-targets the patch.

    The TCM uses the STM of the remaining leg: dv = -Phi_rv^-1 (r_f - r_patch), with the
    STM from the nominal trajectory (linear targeting). The first n_check samples are
    re-propagated nonlinearly with the TCM applied to verify it.
    """
    mu = units.mu
    rng = np.random.default_rng(seed)
    errs = rng.normal(scale=sigma_m_s / 1000.0 / units.velocity_km_s, size=(n, 3))
    X0s = np.repeat(X_dep[None, :], n, axis=0)
    X0s[:, 3:6] += errs
    S = propagate_batch(X0s, tof, mu, np.array([0.0, t_tcm, tof]))
    miss = np.linalg.norm(S[2, :, :3] - X_patch[:3], axis=1) * units.length_km
    nom_tcm = dopri5(cr3bp.make_rhs(mu), 0.0, t_tcm, X_dep, rtol=1e-12, atol=1e-12).X_final
    Phi = stm.split(
        stm.propagate_stm(nom_tcm, 0.0, tof - t_tcm, mu, rtol=1e-12, atol=1e-12).X_final
    )[1]
    # Linear map from a state error at t_tcm to the final position error, then solve for dv.
    dX = S[1] - nom_tcm  # (n, 6) state errors at the TCM time
    dr_f = dX @ Phi[0:3, :].T  # predicted final position errors
    dv = -np.linalg.solve(Phi[0:3, 3:6], dr_f.T).T
    tcm = np.linalg.norm(dv, axis=1) * units.velocity_km_s * 1000.0
    check = []
    for k in range(min(n_check, n)):
        Xc = S[1, k].copy()
        Xc[3:6] += dv[k]
        Xf = dopri5(cr3bp.make_rhs(mu), 0.0, tof - t_tcm, Xc, rtol=1e-12, atol=1e-12).X_final
        check.append(np.linalg.norm(Xf[:3] - X_patch[:3]) * units.length_km)
    return BurnMonteCarlo(miss, tcm, np.array(check))
