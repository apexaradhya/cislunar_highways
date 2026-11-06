"""State transition matrix (STM) propagation (PROJECT_SPEC.md 1.4, Phase 6).

The STM Phi(t, t0) = dX(t)/dX(t0) says how a small change in the initial state
maps to a change in the state at time t. It obeys the variational equation
    dPhi/dt = A(X(t)) Phi,   Phi(t0, t0) = I,
with A = [[0, I], [Omega_XX, K]] from cr3bp.a_matrix. We integrate the 6 state
equations and the 36 STM entries together as one 42-dimensional ODE
(state first, then Phi flattened row by row).

Structure checks (proved in docs/math_notes.md):
* trace(A) = 0, so by Liouville's formula det(Phi) = exp(int trace A) = 1.
* The CR3BP is Hamiltonian in canonical coordinates (q, p) with
  p = (vx - y, vy + x, vz). In (q, v) coordinates this means
      Phi^T Jv Phi = Jv,   Jv = [[W, I], [-I, 0]],   W = [[0,-2,0],[2,0,0],[0,0,0]],
  i.e. Phi is symplectic with respect to the "Coriolis-twisted" form Jv
  (Jv = T^T J T with T = d(q,p)/d(q,v)).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from cislunar.integrators import Event, Solution, dopri5

FloatArray = NDArray[np.float64]

W_CORIOLIS: FloatArray = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
J_VELOCITY: FloatArray = np.block([[W_CORIOLIS, np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
J_CANONICAL: FloatArray = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])


def make_rhs_stm(mu: float):
    """f(t, Y) for the 42-dim augmented system Y = [X, vec(Phi)]."""
    one_m_mu = 1.0 - mu

    def rhs(t: float, Y: FloatArray) -> FloatArray:
        x, y, z, vx, vy, vz = Y[:6]
        dx1 = x + mu
        dx2 = x - one_m_mu
        r1s = dx1 * dx1 + y * y + z * z
        r2s = dx2 * dx2 + y * y + z * z
        r1 = r1s**0.5
        r2 = r2s**0.5
        k1 = one_m_mu / (r1s * r1)
        k2 = mu / (r2s * r2)
        b1 = 3.0 * k1 / r1s  # 3(1-mu)/r1^5
        b2 = 3.0 * k2 / r2s  # 3 mu / r2^5
        oxx = 1.0 - k1 - k2 + b1 * dx1 * dx1 + b2 * dx2 * dx2
        oyy = 1.0 - k1 - k2 + (b1 + b2) * y * y
        ozz = -k1 - k2 + (b1 + b2) * z * z
        oxy = (b1 * dx1 + b2 * dx2) * y
        oxz = (b1 * dx1 + b2 * dx2) * z
        oyz = (b1 + b2) * y * z
        P = Y[6:].reshape(6, 6)
        dP = np.empty((6, 6))
        dP[0:3] = P[3:6]  # d(position rows)/dt = velocity rows
        H = np.array([[oxx, oxy, oxz], [oxy, oyy, oyz], [oxz, oyz, ozz]])
        dP[3:6] = H @ P[0:3]
        dP[3] += 2.0 * P[4]
        dP[4] -= 2.0 * P[3]
        out = np.empty(42)
        out[0:6] = (
            vx,
            vy,
            vz,
            2.0 * vy + x - k1 * dx1 - k2 * dx2,
            -2.0 * vx + y - (k1 + k2) * y,
            -(k1 + k2) * z,
        )
        out[6:] = dP.ravel()
        return out

    return rhs


def initial_augmented(X0: FloatArray, Phi0: FloatArray | None = None) -> FloatArray:
    Phi0 = np.eye(6) if Phi0 is None else Phi0
    return np.concatenate([np.asarray(X0, dtype=float), np.asarray(Phi0, dtype=float).ravel()])


def split(Y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Augmented state(s) -> (X, Phi). Works for shape (42,) or (N, 42)."""
    Y = np.asarray(Y)
    return Y[..., :6], Y[..., 6:].reshape(Y.shape[:-1] + (6, 6))


def propagate_stm(
    X0: FloatArray,
    t0: float,
    tf: float,
    mu: float,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    events: Sequence[Event] = (),
    t_eval: FloatArray | None = None,
) -> Solution:
    """Integrate state + STM from t0 to tf. Use `split(sol.X)` to unpack.

    Event functions receive the full 42-vector; X is its first 6 entries.
    """
    return dopri5(
        make_rhs_stm(mu),
        t0,
        tf,
        initial_augmented(X0),
        rtol=rtol,
        atol=atol,
        events=events,
        t_eval=t_eval,
    )


def stm_finite_difference(
    X0: FloatArray, t0: float, tf: float, mu: float, delta: float = 1e-7, rtol: float = 1e-13
) -> FloatArray:
    """Independent STM estimate by central differences (6 x 2 extra propagations)."""
    from cislunar.cr3bp import make_rhs

    f = make_rhs(mu)
    Phi = np.empty((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = delta
        xp = dopri5(f, t0, tf, X0 + d, rtol=rtol, atol=rtol).X_final
        xm = dopri5(f, t0, tf, X0 - d, rtol=rtol, atol=rtol).X_final
        Phi[:, j] = (xp - xm) / (2 * delta)
    return Phi
