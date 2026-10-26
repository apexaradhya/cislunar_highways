"""Differential correction (single shooting) for xz-plane-symmetric periodic orbits.

PROJECT_SPEC.md 1.7; Howell (1984), "Three-dimensional, periodic, 'halo' orbits",
Celestial Mechanics 32, 53-71.

Mirror theorem: if an orbit crosses the xz-plane (y = 0) perpendicularly twice,
it is periodic and symmetric. So start at X0 = [x0, 0, z0, 0, vy0, 0], integrate
to the next y = 0 crossing (t = T/2), and drive (vx, vz) there to zero.

Linearizing the crossing state with respect to a free initial variable q_j,
the crossing time also moves (y must stay 0), and that adds a correction:
    d(vx_f)/dq_j = Phi[3, j] - (ax_f / vy_f) Phi[1, j]
    d(vz_f)/dq_j = Phi[5, j] - (az_f / vy_f) Phi[1, j]
(0-based indices, state order x, y, z, vx, vy, vz). Derivation: dy_f = Phi[1,j] dq + vy_f dt = 0
gives dt = -Phi[1,j] dq / vy_f, and dvx_f = Phi[3,j] dq + ax_f dt.

Newton: solve M dq = -[vx_f, vz_f], update q, repeat until |vx_f|, |vz_f| < tol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import cr3bp, stm
from cislunar.integrators import Event, Solution

FloatArray = NDArray[np.float64]

CROSSING_TOL = 1e-11  # spec 1.7: |vx_f|, |vz_f| < 1e-11
MAX_ITER = 25
INTEGRATION_TOL = 1e-13
# Do not accept a y = 0 crossing earlier than this (non-dimensional time): protects
# against detecting the start point itself, which lies on y = 0.
MIN_HALF_PERIOD = 1e-3


class CorrectionError(RuntimeError):
    pass


@dataclass
class SymmetricOrbit:
    X0: FloatArray  # initial state on the xz-plane
    period: float
    mu: float
    iterations: int
    residual: float  # max(|vx_f|, |vz_f|) at the half-period crossing

    @property
    def jacobi(self) -> float:
        return float(cr3bp.jacobi_constant(self.X0, self.mu))


def half_period_crossing(
    X0: FloatArray, mu: float, t_max: float = 20.0, tol: float = INTEGRATION_TOL
) -> Solution:
    """Integrate state+STM from X0 to the next crossing of y = 0."""
    ev = Event(lambda t, Y: Y[1], direction=0, terminal=True)
    # Step past the start plane first: the event function is exactly 0 at t = 0.
    sol = stm.propagate_stm(X0, 0.0, t_max, mu, rtol=tol, atol=tol, events=[ev])
    if sol.status != "event" or sol.t_final < MIN_HALF_PERIOD:
        raise CorrectionError("no y = 0 crossing found")
    return sol


def crossing_jacobian(
    X: FloatArray, mu: float, free: Sequence[int], targets: Sequence[int]
) -> tuple[FloatArray, FloatArray, Solution]:
    """Residual (targets at the half-period crossing) and its Jacobian w.r.t. `free`.

    Includes the crossing-time correction terms (module docstring).
    """
    sol = half_period_crossing(X, mu)
    Xf, Phi = stm.split(sol.X_final)
    acc = cr3bp.equations_of_motion(0.0, Xf, mu)[3:]
    vy_f = Xf[4]
    M = np.empty((len(targets), len(free)))
    for r, ti in enumerate(targets):
        a_t = acc[ti - 3]
        for c, fj in enumerate(free):
            M[r, c] = Phi[ti, fj] - (a_t / vy_f) * Phi[1, fj]
    return Xf[list(targets)], M, sol


def correct_symmetric(
    X0_guess: FloatArray,
    mu: float,
    free: Sequence[int] = (0, 4),
    targets: Sequence[int] = (3, 5),
    tol: float = CROSSING_TOL,
    max_iter: int = MAX_ITER,
    max_step: float = 0.05,
) -> SymmetricOrbit:
    """Newton differential correction for a symmetric periodic orbit.

    free:    indices of initial-state entries to adjust (subset of x0=0, z0=2, vy0=4).
    targets: indices of crossing-state entries to zero (vx=3, vz=5).
    Planar Lyapunov: free=(4,), targets=(3,). Halo with fixed z0: free=(0, 4), targets=(3, 5).
    `max_step` caps each Newton update (non-dimensional) to keep early iterations sane.
    """
    X = np.array(X0_guess, dtype=float)
    X[[1, 3, 5]] = 0.0
    free, targets = list(free), list(targets)
    best: SymmetricOrbit | None = None
    for it in range(1, max_iter + 1):
        resid, M, sol = crossing_jacobian(X, mu, free, targets)
        err = float(np.max(np.abs(resid)))
        if best is not None:
            # Polishing step taken after convergence: keep whichever is better.
            if err < best.residual:
                best = SymmetricOrbit(X.copy(), 2.0 * sol.t_final, mu, it - 1, err)
            return best
        if err < tol:
            # Converged. Unstable orbits amplify any leftover residual by ~lambda_u over a
            # period, so take one more (nearly free) Newton step to squeeze it further.
            best = SymmetricOrbit(X.copy(), 2.0 * sol.t_final, mu, it - 1, err)
        dq = np.linalg.lstsq(M, -resid, rcond=None)[0]
        scale = np.max(np.abs(dq))
        if scale > max_step:
            dq *= max_step / scale
        X[free] += dq
    raise CorrectionError(f"no convergence after {max_iter} iterations (residual {err:.2e})")


def full_period(
    orbit: SymmetricOrbit, n_points: int | None = None, tol: float = INTEGRATION_TOL
) -> Solution:
    """Propagate state+STM over one full period (dense samples if n_points is given)."""
    t_eval = None if n_points is None else np.linspace(0.0, orbit.period, n_points)
    return stm.propagate_stm(
        orbit.X0, 0.0, orbit.period, orbit.mu, rtol=tol, atol=tol, t_eval=t_eval
    )


def correct_symmetric_fixed_period(
    X0_guess: FloatArray,
    mu: float,
    period: float,
    tol: float = CROSSING_TOL,
    max_iter: int = MAX_ITER,
) -> SymmetricOrbit:
    """Symmetric 3D orbit with a prescribed period: unknowns (x0, z0, vy0).

    Equations: vx_f = 0, vz_f = 0, t_f - period/2 = 0. The crossing time responds to
    q_j as dt_f/dq_j = -Phi[1, j] / vy_f (from keeping y_f = 0), which gives the
    third row of the Newton matrix.
    """
    X = np.array(X0_guess, dtype=float)
    X[[1, 3, 5]] = 0.0
    free = [0, 2, 4]
    for it in range(1, max_iter + 1):
        resid2, M2, sol = crossing_jacobian(X, mu, free, (3, 5))
        _, Phi = stm.split(sol.X_final)
        vy_f = sol.X_final[4]
        dt = np.array([-Phi[1, j] / vy_f for j in free])
        G = np.concatenate([resid2, [sol.t_final - 0.5 * period]])
        if np.max(np.abs(G)) < tol:
            return SymmetricOrbit(X, 2.0 * sol.t_final, mu, it - 1, float(np.max(np.abs(resid2))))
        X[free] -= np.linalg.solve(np.vstack([M2, dt]), G)
    raise CorrectionError(f"fixed-period correction failed (residual {np.max(np.abs(G)):.2e})")


def correct_planar_fixed_energy(
    X0_guess: FloatArray, mu: float, jacobi: float, tol: float = CROSSING_TOL
) -> SymmetricOrbit:
    """Planar symmetric periodic orbit with a prescribed Jacobi constant.

    Unknowns (x0, vy0); equations vx_f = 0 and C(x0, vy0) - jacobi = 0, where
    dC/dx0 = 2 dOmega/dx (at y=z=0) and dC/dvy0 = -2 vy0.
    """
    X = np.array(X0_guess, dtype=float)
    X[[1, 2, 3, 5]] = 0.0
    for it in range(1, MAX_ITER + 1):
        resid, M, sol = crossing_jacobian(X, mu, (0, 4), (3,))
        dC = np.array([2.0 * cr3bp.grad_pseudo_potential(X, mu)[0], -2.0 * X[4]])
        G = np.array([resid[0], cr3bp.jacobi_constant(X, mu) - jacobi])
        if np.max(np.abs(G)) < tol:
            return SymmetricOrbit(X, 2.0 * sol.t_final, mu, it - 1, float(abs(resid[0])))
        X[[0, 4]] -= np.linalg.solve(np.vstack([M, dC]), G)
    raise CorrectionError(f"fixed-energy correction failed (residual {np.max(np.abs(G)):.2e})")
