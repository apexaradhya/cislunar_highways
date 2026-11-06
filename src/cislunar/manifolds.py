"""Stable and unstable invariant manifolds of unstable periodic orbits (PROJECT_SPEC.md 1.9).

For a periodic orbit gamma with monodromy matrix M = Phi(T, 0):
* the real eigenvalue lambda_u > 1 has eigenvector v_u: perturbations along it grow
  by lambda_u every period (unstable direction);
* lambda_s = 1/lambda_u < 1 with eigenvector v_s: perturbations shrink (stable).
The eigenvector is carried to any point gamma(t_i) by the STM, v(t_i) = Phi(t_i, 0) v(0),
because Phi maps M's eigenvectors at t=0 to those of the monodromy based at t_i.

Manifold initial states: X = gamma(t_i) +- eps * v(t_i) / |position part of v(t_i)|.
Unstable manifolds are propagated forward in time; stable manifolds backward. Their
union over t_i forms the tube-shaped surfaces W^u(gamma), W^s(gamma).
(Gomez, Koon, Lo, Marsden, Masdemont & Ross 2004, Nonlinearity 17, 1571-1606;
Koon, Lo, Marsden & Ross 2011, ch. 4 and 7; Parker & Anderson 2014, ch. 2.)

Energy: since gradC^T M = gradC^T (C is conserved), every eigenvector with lambda != 1
satisfies gradC . v = 0, so the eps-displacement changes C only at O(eps^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import stm
from cislunar.cr3bp import make_rhs, primary_distances
from cislunar.integrators import Event, Solution, dopri5
from cislunar.shooting import SymmetricOrbit

FloatArray = NDArray[np.float64]

# Default displacement onto the local manifold: 1e-5 non-dimensional ~ 3.8 km.
# - far above integration noise (~1e-12), so the eigen-direction dominates;
# - small enough that the linear approximation's error (O(eps^2)) changes C by ~1e-10;
# - large enough that L1/L2 halo tubes leave the orbit within ~2-3 weeks.
# Changing eps by 10x only re-labels which strand of the same tube starts at a given
# phase (shift of T ln10 / ln lambda_u); measured in scripts/run_phase9_manifolds.py.
DEFAULT_EPS = 1e-5


@dataclass(frozen=True)
class EigenDirections:
    lambda_u: float
    lambda_s: float
    v_u: FloatArray  # unit eigenvector at t = 0
    v_s: FloatArray


def stable_unstable_directions(M: FloatArray) -> EigenDirections:
    """Real eigenpair with the largest |lambda| (unstable) and its reciprocal (stable)."""
    w, V = np.linalg.eig(M)
    real = np.abs(w.imag) < 1e-9 * np.abs(w)
    if not real.any():
        raise ValueError("no real eigenvalues")
    iu = int(np.argmax(np.where(real, np.abs(w), -np.inf)))
    is_ = int(np.argmin(np.where(real, np.abs(w), np.inf)))
    lam_u, lam_s = float(w[iu].real), float(w[is_].real)
    if abs(lam_u) <= 1.0 + 1e-6:
        raise ValueError(f"orbit is not unstable (|lambda|max = {abs(lam_u)})")
    v_u = V[:, iu].real / np.linalg.norm(V[:, iu].real)
    v_s = V[:, is_].real / np.linalg.norm(V[:, is_].real)
    return EigenDirections(lam_u, lam_s, v_u, v_s)


@dataclass
class OrbitSamples:
    """The orbit and its STM sampled at n equally spaced times over one period."""

    t: FloatArray  # (n,)
    X: FloatArray  # (n, 6)
    Phi: FloatArray  # (n, 6, 6)
    monodromy: FloatArray
    directions: EigenDirections


def sample_orbit(orbit: SymmetricOrbit, n: int = 200, tol: float = 1e-13) -> OrbitSamples:
    """Propagate state+STM over one period, storing n samples at t_i = i T / n."""
    t = np.linspace(0.0, orbit.period, n + 1)
    sol = stm.propagate_stm(orbit.X0, 0.0, orbit.period, orbit.mu, rtol=tol, atol=tol, t_eval=t)
    X, Phi = stm.split(sol.X)
    return OrbitSamples(t[:-1], X[:-1], Phi[:-1], Phi[-1], stable_unstable_directions(Phi[-1]))


def manifold_states(
    samples: OrbitSamples, kind: str, sign: float, eps: float = DEFAULT_EPS
) -> FloatArray:
    """Initial states on the local stable ('s') or unstable ('u') manifold, branch `sign` (+1/-1).

    v(t_i) = Phi(t_i, 0) v(0), scaled so its position part has unit length
    (PROJECT_SPEC.md 1.9 step 2); X = gamma(t_i) + sign * eps * v(t_i).

    The sign of v(0) is fixed so that its x-component is >= 0; with that convention
    sign=+1 starts on the +x side of the orbit at t=0. Which branch then heads to the
    Moon or away from it depends on the orbit, and is reported by `classify_branch`.
    """
    v0 = samples.directions.v_u if kind == "u" else samples.directions.v_s
    if v0[0] < 0:
        v0 = -v0
    V = samples.Phi @ v0  # (n, 6)
    V /= np.linalg.norm(V[:, :3], axis=1, keepdims=True)
    return samples.X + sign * eps * V


@dataclass
class ManifoldTrajectory:
    t: FloatArray
    X: FloatArray
    status: str  # "time", "earth", "moon", "escape", or "section"
    phase: float  # t_i / T of its starting point on the orbit


def make_stop_events(
    mu: float, r_earth: float, r_moon: float, r_escape: float = 5.0, extra: tuple[Event, ...] = ()
) -> list[Event]:
    """Terminal events: impact Earth / Moon (radius in non-dimensional units), escape."""
    return [
        Event(lambda t, X: primary_distances(X, mu)[0] - r_earth, direction=-1, terminal=True),
        Event(lambda t, X: primary_distances(X, mu)[1] - r_moon, direction=-1, terminal=True),
        Event(lambda t, X: np.linalg.norm(X[:3]) - r_escape, direction=1, terminal=True),
        *extra,
    ]


STATUS_BY_EVENT = ("earth", "moon", "escape")


def propagate_manifold(
    X_start: FloatArray,
    mu: float,
    duration: float,
    kind: str,
    events: list[Event],
    n_out: int | None = 400,
    tol: float = 1e-12,
) -> tuple[Solution, str]:
    """Propagate one manifold trajectory: forward if unstable ('u'), backward if stable ('s')."""
    tf = duration if kind == "u" else -duration
    sol = dopri5(make_rhs(mu), 0.0, tf, X_start, rtol=tol, atol=tol, events=events)
    status = "time"
    if sol.status == "event":
        idx = sol.events[-1].index
        status = STATUS_BY_EVENT[idx] if idx < len(STATUS_BY_EVENT) else "section"
    if n_out is not None and len(sol.t) > n_out:
        # Resample to a manageable size for plotting (keep the exact end point).
        keep = np.unique(np.linspace(0, len(sol.t) - 1, n_out).astype(int))
        sol = Solution(
            sol.t[keep], sol.X[keep], sol.events, sol.status, sol.n_steps, sol.n_rejected, sol.n_fev
        )
    return sol, status


def generate_tube(
    orbit: SymmetricOrbit,
    kind: str,
    sign: float,
    duration: float,
    n_traj: int = 100,
    eps: float = DEFAULT_EPS,
    r_earth: float = 0.0,
    r_moon: float = 0.0,
    extra_events: tuple[Event, ...] = (),
    samples: OrbitSamples | None = None,
    n_out: int | None = 400,
) -> list[ManifoldTrajectory]:
    """A whole manifold tube: n_traj trajectories started at equally spaced phases."""
    samples = samples if samples is not None else sample_orbit(orbit, n_traj)
    starts = manifold_states(samples, kind, sign, eps)
    events = make_stop_events(orbit.mu, r_earth, r_moon, extra=extra_events)
    out = []
    for ti, Xs in zip(samples.t, starts, strict=True):
        sol, status = propagate_manifold(Xs, orbit.mu, duration, kind, events, n_out=n_out)
        out.append(ManifoldTrajectory(sol.t, sol.X, status, ti / orbit.period))
    return out


def stroboscopic_distances(
    samples: OrbitSamples,
    orbit: SymmetricOrbit,
    i: int,
    kind: str,
    eps: float,
    n_periods: int,
    backward: bool = False,
    tol: float = 1e-13,
) -> FloatArray:
    """|X(+-kT) - gamma(t_i)| for k = 0..n_periods, starting from gamma(t_i) + eps v(t_i).

    Forward in time, a point on the stable manifold closes in by lambda_s per period and
    one on the unstable manifold moves away by lambda_u (PROJECT_SPEC.md Phase 9).
    Backward in time the roles swap. Only the *growing* case can be followed for many
    periods: in the shrinking case any tiny unstable component (roundoff, O(eps^2)
    curvature) is amplified by lambda_u per period and soon dominates.
    """
    X = manifold_states(samples, kind, +1.0, eps)[i]
    target = samples.X[i]
    f = make_rhs(orbit.mu)
    span = -orbit.period if backward else orbit.period
    out = [np.linalg.norm(X - target)]
    for _ in range(n_periods):
        X = dopri5(f, 0.0, span, X, rtol=tol, atol=tol).X_final
        out.append(np.linalg.norm(X - target))
    return np.array(out)
