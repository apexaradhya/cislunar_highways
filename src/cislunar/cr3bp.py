"""Circular Restricted Three-Body Problem dynamics in the rotating frame.

Non-dimensional units: primary separation = 1, total mass = 1, frame rotation rate = 1.
Big primary m1 at (-mu, 0, 0), small primary m2 at (1 - mu, 0, 0).

State X = [x, y, z, vx, vy, vz]. All functions accept a single state of shape (6,)
or a stack of states of shape (..., 6) and broadcast over the leading axes.

Equations (PROJECT_SPEC.md 1.2; Szebehely 1967, ch. 1; Koon et al. 2011, ch. 2):
    x'' =  2 y' + dOmega/dx
    y'' = -2 x' + dOmega/dy
    z'' =         dOmega/dz

Pseudo-potential convention (used everywhere in this package):
    Omega = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2 + mu(1-mu)/2
The constant mu(1-mu)/2 does not affect the motion; it shifts every Jacobi
constant by mu(1-mu). With it, C(L4) = C(L5) = 3 exactly.

Jacobi constant: C = 2 Omega - |v|^2   (the only integral of motion).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _distances(
    x: FloatArray, y: FloatArray, z: FloatArray, mu: float
) -> tuple[FloatArray, FloatArray]:
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    return r1, r2


def primary_distances(X: ArrayLike, mu: float) -> tuple[FloatArray, FloatArray]:
    """Distances (r1 to the big primary, r2 to the small primary) for position(s) X[..., :3]."""
    X = np.asarray(X, dtype=float)
    return _distances(X[..., 0], X[..., 1], X[..., 2], mu)


def pseudo_potential(X: ArrayLike, mu: float) -> FloatArray:
    """Omega(x, y, z) including the constant mu(1-mu)/2 (see module docstring)."""
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r1, r2 = _distances(x, y, z, mu)
    return 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2 + 0.5 * mu * (1.0 - mu)


def grad_pseudo_potential(X: ArrayLike, mu: float) -> FloatArray:
    """Gradient (dOmega/dx, dOmega/dy, dOmega/dz); shape (..., 3)."""
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r1, r2 = _distances(x, y, z, mu)
    k1 = (1.0 - mu) / r1**3
    k2 = mu / r2**3
    ox = x - k1 * (x + mu) - k2 * (x - 1.0 + mu)
    oy = y - k1 * y - k2 * y
    oz = -k1 * z - k2 * z
    return np.stack([ox, oy, oz], axis=-1)


def equations_of_motion(t: float, X: ArrayLike, mu: float) -> FloatArray:
    """Time derivative dX/dt of the CR3BP state (autonomous: t is unused).

    Signature f(t, X, mu) so it plugs into any ODE integrator.
    """
    X = np.asarray(X, dtype=float)
    g = grad_pseudo_potential(X, mu)
    vx, vy, vz = X[..., 3], X[..., 4], X[..., 5]
    ax = 2.0 * vy + g[..., 0]
    ay = -2.0 * vx + g[..., 1]
    az = g[..., 2]
    return np.stack([vx, vy, vz, ax, ay, az], axis=-1)


def jacobi_constant(X: ArrayLike, mu: float) -> FloatArray:
    """C = 2 Omega - (vx^2 + vy^2 + vz^2)  (PROJECT_SPEC.md 1.3)."""
    X = np.asarray(X, dtype=float)
    v2 = X[..., 3] ** 2 + X[..., 4] ** 2 + X[..., 5] ** 2
    return 2.0 * pseudo_potential(X, mu) - v2


def jacobi_rate(X: ArrayLike, mu: float) -> FloatArray:
    """Analytic dC/dt along the flow: 2 grad(Omega).v - 2 v.a.

    Substituting the EOM, a = grad(Omega) + 2(vy, -vx, 0), so
    dC/dt = -4 (vx vy - vy vx) = 0 identically. Evaluating it numerically is a
    check that the EOM and the Jacobi constant are mutually consistent.
    """
    X = np.asarray(X, dtype=float)
    v = X[..., 3:6]
    a = equations_of_motion(0.0, X, mu)[..., 3:6]
    g = grad_pseudo_potential(X, mu)
    return 2.0 * np.sum(g * v, axis=-1) - 2.0 * np.sum(v * a, axis=-1)


def hessian_pseudo_potential(X: ArrayLike, mu: float) -> FloatArray:
    """Second derivatives of Omega, shape (..., 3, 3) (PROJECT_SPEC.md 1.4)."""
    X = np.asarray(X, dtype=float)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r1, r2 = _distances(x, y, z, mu)
    a1 = (1.0 - mu) / r1**3
    a2 = mu / r2**3
    b1 = 3.0 * (1.0 - mu) / r1**5
    b2 = 3.0 * mu / r2**5
    dx1, dx2 = x + mu, x - 1.0 + mu
    oxx = 1.0 - a1 - a2 + b1 * dx1**2 + b2 * dx2**2
    oyy = 1.0 - a1 - a2 + b1 * y**2 + b2 * y**2
    ozz = -a1 - a2 + b1 * z**2 + b2 * z**2
    oxy = b1 * dx1 * y + b2 * dx2 * y
    oxz = b1 * dx1 * z + b2 * dx2 * z
    oyz = b1 * y * z + b2 * y * z
    row0 = np.stack([oxx, oxy, oxz], axis=-1)
    row1 = np.stack([oxy, oyy, oyz], axis=-1)
    row2 = np.stack([oxz, oyz, ozz], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


# Coriolis block of the linearization: d(acceleration)/d(velocity).
CORIOLIS_K: FloatArray = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def a_matrix(X: ArrayLike, mu: float) -> FloatArray:
    """Jacobian A = df/dX of the EOM, shape (..., 6, 6).

    A = [[0, I], [Omega_XX, K]]  (PROJECT_SPEC.md 1.4). Used for dPhi/dt = A Phi.
    """
    X = np.asarray(X, dtype=float)
    H = hessian_pseudo_potential(X, mu)
    A = np.zeros(X.shape[:-1] + (6, 6))
    A[..., 0:3, 3:6] = np.eye(3)
    A[..., 3:6, 0:3] = H
    A[..., 3:6, 3:6] = CORIOLIS_K
    return A


def make_rhs(mu: float):
    """Fast f(t, X) for one 6-state, for the integrators.

    Same equations as `equations_of_motion`, written with scalar math to avoid
    numpy overhead on 3-element arrays (~5x faster). Tested against it.
    """
    one_m_mu = 1.0 - mu

    def rhs(t: float, X: FloatArray) -> FloatArray:
        x, y, z, vx, vy, vz = X
        dx1 = x + mu
        dx2 = x - one_m_mu
        r1s = dx1 * dx1 + y * y + z * z
        r2s = dx2 * dx2 + y * y + z * z
        k1 = one_m_mu / (r1s * r1s**0.5)
        k2 = mu / (r2s * r2s**0.5)
        return np.array(
            [
                vx,
                vy,
                vz,
                2.0 * vy + x - k1 * dx1 - k2 * dx2,
                -2.0 * vx + y - (k1 + k2) * y,
                -(k1 + k2) * z,
            ]
        )

    return rhs
