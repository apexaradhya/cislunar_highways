"""Periodic-orbit families: planar Lyapunov, halo, continuation, stability.

Monodromy matrix M = Phi(T, 0). For a periodic orbit of an autonomous
Hamiltonian system, its eigenvalues come in reciprocal pairs (lambda, 1/lambda)
(symplecticity), and one pair is exactly (1, 1): a shift along the orbit stays
on the orbit, and a change of energy changes the period. Stability index for each
non-trivial pair: nu = (lambda + 1/lambda)/2. If |nu| > 1 the orbit is unstable
(PROJECT_SPEC.md 1.8; Howell 1984; Koon et al. 2011, sec. 4.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cislunar import lagrange, stm
from cislunar.shooting import (
    CorrectionError,
    SymmetricOrbit,
    correct_symmetric,
    crossing_jacobian,
    full_period,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass
class Stability:
    monodromy: FloatArray
    eigenvalues: ComplexArray
    trivial_pair: tuple[complex, complex]
    pairs: list[tuple[complex, complex]]  # non-trivial reciprocal pairs
    indices: list[float]  # stability index nu for each non-trivial pair

    @property
    def max_index(self) -> float:
        return max(abs(n) for n in self.indices)

    @property
    def lambda_unstable(self) -> float:
        """Largest real |eigenvalue| (the unstable multiplier lambda_u)."""
        return float(np.max(np.abs(self.eigenvalues)))


def pair_eigenvalues(
    eig: ComplexArray,
) -> tuple[tuple[complex, complex], list[tuple[complex, complex]]]:
    """Group 6 eigenvalues into 3 reciprocal pairs and pick out the trivial (1, 1) pair.

    Greedy matching: repeatedly take the eigenvalue of largest modulus and pair it with
    the remaining eigenvalue closest to its reciprocal.
    """
    remaining = list(eig)
    pairs = []
    while remaining:
        remaining.sort(key=lambda v: -abs(v))
        a = remaining.pop(0)
        j = int(np.argmin([abs(b - 1.0 / a) for b in remaining]))
        pairs.append((a, remaining.pop(j)))
    k = int(np.argmin([abs(a - 1) + abs(b - 1) for a, b in pairs]))
    trivial = pairs.pop(k)
    return trivial, pairs


def stability(orbit: SymmetricOrbit) -> Stability:
    sol = full_period(orbit)
    _, M = stm.split(sol.X_final)
    eig = np.linalg.eigvals(M)
    trivial, pairs = pair_eigenvalues(eig)
    idx = [float(np.real(0.5 * (a + b))) for a, b in pairs]
    return Stability(M, eig, trivial, pairs, idx)


def closure_error(orbit: SymmetricOrbit) -> float:
    """max |X(T) - X(0)| after propagating one full period."""
    return float(np.max(np.abs(full_period(orbit).X_final[:6] - orbit.X0)))


@dataclass
class Family:
    name: str
    orbits: list[SymmetricOrbit] = field(default_factory=list)

    @property
    def periods(self) -> FloatArray:
        return np.array([o.period for o in self.orbits])

    @property
    def jacobi(self) -> FloatArray:
        return np.array([o.jacobi for o in self.orbits])

    @property
    def x0(self) -> FloatArray:
        return np.array([o.X0[0] for o in self.orbits])


# Which x-axis crossing to parametrize each family by: the side *away* from the Moon.
# Continuing the L2 family by its Moon-side crossing walks x0 into the Moon, where the
# corrector jumps onto unrelated small lunar orbits.
LYAPUNOV_SIDE = {1: -1.0, 2: +1.0}
# A real family changes smoothly; a period jump larger than this between neighbouring
# members means the corrector converged onto a different family.
MAX_PERIOD_JUMP = 0.25


class FamilyJumpError(RuntimeError):
    pass


def lyapunov_orbit(which: int, mu: float, amplitude: float) -> SymmetricOrbit:
    """Planar Lyapunov orbit around L1/L2 through x0 = xL + side * amplitude.

    Seeded from the linear solution (lagrange.linear_lyapunov_state); x0 stays fixed
    and only vy0 is corrected (1-variable shooting, target vx_f = 0). Only valid for
    small amplitudes; use `lyapunov_family` beyond ~0.005.
    """
    X0 = lagrange.linear_lyapunov_state(which, mu, LYAPUNOV_SIDE[which] * amplitude)
    return correct_symmetric(X0, mu, free=(4,), targets=(3,))


def _lyapunov_guess(which: int, mu: float, fam: Family, a: float) -> FloatArray:
    """Seed for amplitude a: linear solution, with vy0 extrapolated from the family so far."""
    side = LYAPUNOV_SIDE[which]
    xL = lagrange.collinear_point(which, mu)
    X0 = lagrange.linear_lyapunov_state(which, mu, side * a)
    if len(fam.orbits) >= 2:
        o1, o2 = fam.orbits[-2], fam.orbits[-1]
        a1, a2 = side * (o1.X0[0] - xL), side * (o2.X0[0] - xL)
        X0[4] = o2.X0[4] + (o2.X0[4] - o1.X0[4]) * (a - a2) / (a2 - a1)
    elif len(fam.orbits) == 1:
        prev = fam.orbits[-1]
        X0[4] = prev.X0[4] * a / (side * (prev.X0[0] - xL))
    return X0


def lyapunov_family(which: int, mu: float, amplitudes: FloatArray, max_halvings: int = 8) -> Family:
    """Natural-parameter continuation in x0 (PROJECT_SPEC.md Phase 7).

    Each orbit is seeded by linear extrapolation of vy0 from the previous two.
    If a step fails to converge or the period jumps (the corrector landed on a
    different family), the step is halved and retried. Intermediate orbits found
    this way are kept, so the returned family may contain more orbits than requested.
    """
    fam = Family(f"L{which} Lyapunov")
    side = LYAPUNOV_SIDE[which]
    xL = lagrange.collinear_point(which, mu)

    def try_step(a: float) -> bool:
        try:
            orbit = correct_symmetric(
                _lyapunov_guess(which, mu, fam, a), mu, free=(4,), targets=(3,)
            )
        except CorrectionError:
            return False
        if fam.orbits and abs(orbit.period - fam.orbits[-1].period) > MAX_PERIOD_JUMP:
            return False
        fam.orbits.append(orbit)
        return True

    for target in amplitudes:
        halvings = 0
        while True:
            a_prev = side * (fam.orbits[-1].X0[0] - xL) if fam.orbits else 0.0
            step = target - a_prev
            for _ in range(halvings):
                step *= 0.5
            if try_step(a_prev + step):
                if halvings == 0:
                    break
                halvings -= 1  # success: try a bigger step toward the target again
            else:
                halvings += 1
                if halvings > max_halvings:
                    raise FamilyJumpError(f"continuation stalled near amplitude {a_prev:.5f}")
    return fam


# ---------------------------------------------------------------------------
# Halo orbits (PROJECT_SPEC.md Phase 8)
# ---------------------------------------------------------------------------
# Along a planar Lyapunov family, a second pair of monodromy eigenvalues (the
# "vertical" pair) sits on the unit circle with stability index nu_v < 1 for small
# orbits. Where nu_v crosses +1, the monodromy matrix gains an extra eigenvalue 1:
# a new direction of periodic motion appears, and the 3D halo family branches off
# (Howell 1984; Grebow 2006, M.S. thesis, Purdue, sec. 3.3). We locate that crossing,
# step off in z0, and continue the halo family by pseudo-arclength continuation.


def vertical_index(orbit: SymmetricOrbit) -> float:
    """Stability index of the non-trivial pair closest to the unit circle."""
    return min(stability(orbit).indices, key=abs)


def find_halo_bifurcation(
    which: int, mu: float, a_lo: float, a_hi: float, tol: float = 1e-7
) -> SymmetricOrbit:
    """Lyapunov orbit (on the family's x-crossing side) where nu_v crosses +1.

    Requires nu_v - 1 to change sign between amplitudes a_lo and a_hi.
    Bisection; each probe is reached by continuation from a_lo.
    """
    fam_lo = lyapunov_family(which, mu, np.linspace(min(2e-3, a_lo), a_lo, 6))
    fam_hi = lyapunov_family(which, mu, np.linspace(min(2e-3, a_lo), a_hi, 10))
    lo, hi = fam_lo.orbits[-1], fam_hi.orbits[-1]
    g_lo, g_hi = vertical_index(lo) - 1.0, vertical_index(hi) - 1.0
    if g_lo * g_hi > 0:
        raise ValueError("nu_v - 1 does not change sign on the bracket")
    while abs(hi.X0[0] - lo.X0[0]) > tol:
        X0 = 0.5 * (lo.X0 + hi.X0)
        mid = correct_symmetric(X0, mu, free=(4,), targets=(3,))
        g = vertical_index(mid) - 1.0
        if g * g_lo > 0:
            lo, g_lo = mid, g
        else:
            hi, g_hi = mid, g
    # Return the closer-to-bifurcation end (linear interpolation of g picks it).
    return lo if abs(g_lo) < abs(g_hi) else hi


def halo_from_bifurcation(bif: SymmetricOrbit, z0: float) -> SymmetricOrbit:
    """Step off the bifurcating Lyapunov orbit to a small halo with given z0 (sign = class)."""
    X0 = bif.X0.copy()
    X0[2] = z0
    return correct_symmetric(X0, bif.mu, free=(0, 4), targets=(3, 5))


HALO_FREE = (0, 2, 4)  # (x0, z0, vy0)
HALO_TARGETS = (3, 5)  # (vx, vz) at the half-period crossing


def _null_vector(M: FloatArray) -> FloatArray:
    """Unit vector spanning the null space of a full-rank 2x3 matrix (via SVD)."""
    _, _, Vt = np.linalg.svd(M)
    return Vt[-1]


def pseudo_arclength_family(
    seed: SymmetricOrbit,
    n_orbits: int,
    ds: float,
    mu: float,
    direction: FloatArray | None = None,
    stop: object = None,
    tol: float = 1e-11,
    max_iter: int = 12,
    ds_max: float | None = None,
) -> Family:
    """Pseudo-arclength continuation of a symmetric 3D family (Keller 1977; Doedel et al. AUTO).

    Unknowns q = (x0, z0, vy0); equations F(q) = (vx_f, vz_f) = 0 plus the arclength
    constraint (q - q_prev) . tau_prev = ds, where tau is the unit tangent to the
    solution curve (null vector of dF/dq). Unlike stepping in a fixed parameter, this
    follows the curve through turning points (folds) in any single coordinate.

    direction: initial tangent hint (its sign picks which way to go along the family).
    stop: optional callable(orbit) -> bool; the first orbit for which it is True is
        *not* kept, and continuation ends (e.g. "perilune below the lunar surface").
    The step adapts: halved on failure, grown 1.2x after easy convergence, capped at
    |ds_max| (default 2|ds|).
    """
    fam = Family("halo")
    q = seed.X0[list(HALO_FREE)].copy()
    _, M, _ = crossing_jacobian(seed.X0, mu, HALO_FREE, HALO_TARGETS)
    tau = _null_vector(M)
    if direction is not None and tau @ direction < 0:
        tau = -tau
    fam.orbits.append(seed)
    ds_cur = ds
    ds_cap = abs(ds_max) if ds_max is not None else 2.0 * abs(ds)
    while len(fam.orbits) < n_orbits:
        q_pred = q + ds_cur * tau
        X = seed.X0.copy()
        converged = False
        try:
            for it in range(max_iter):  # noqa: B007 (iteration count used after the loop)
                X[list(HALO_FREE)] = q_pred
                resid, M, sol = crossing_jacobian(X, mu, HALO_FREE, HALO_TARGETS)
                arc = (q_pred - q) @ tau - ds_cur
                G = np.concatenate([resid, [arc]])
                if np.max(np.abs(resid)) < tol and abs(arc) < 1e-12:
                    converged = True
                    break
                DG = np.vstack([M, tau])
                q_pred = q_pred - np.linalg.solve(DG, G)
        except (CorrectionError, np.linalg.LinAlgError):
            converged = False
        if not converged:
            ds_cur *= 0.5
            if abs(ds_cur) < 1e-7 * abs(ds):
                break
            continue
        # Polish at fixed z0 (2-variable corrector incl. its extra Newton step): unstable
        # orbits amplify the leftover crossing residual by ~lambda_u over one period.
        try:
            orbit = correct_symmetric(X, mu, free=(0, 4), targets=HALO_TARGETS, tol=tol)
        except CorrectionError:
            orbit = SymmetricOrbit(
                X.copy(), 2.0 * sol.t_final, mu, it, float(np.max(np.abs(resid)))
            )
        q_pred = orbit.X0[list(HALO_FREE)].copy()
        if abs(orbit.period - fam.orbits[-1].period) > MAX_PERIOD_JUMP:
            ds_cur *= 0.5
            continue
        new_tau = _null_vector(M)
        if new_tau @ tau < 0:
            new_tau = -new_tau
        if stop is not None and stop(orbit):
            break
        q, tau = q_pred, new_tau
        fam.orbits.append(orbit)
        if it <= 3:
            ds_cur = float(np.sign(ds_cur)) * min(abs(ds_cur) * 1.2, ds_cap)
    return fam


def mirror_z(orbit: SymmetricOrbit) -> SymmetricOrbit:
    """The CR3BP is symmetric under z -> -z: northern <-> southern halo classes."""
    X0 = orbit.X0.copy()
    X0[2] = -X0[2]
    X0[5] = -X0[5]
    return SymmetricOrbit(X0, orbit.period, orbit.mu, orbit.iterations, orbit.residual)


def perilune(orbit: SymmetricOrbit, n_points: int = 2000) -> float:
    """Closest approach to the Moon's center over one period (non-dimensional).

    Dense samples, then the minimum is refined with a local quadratic fit.
    """
    sol = full_period(orbit, n_points=n_points)
    X = sol.X[:, :3]
    r = np.linalg.norm(X - np.array([1.0 - orbit.mu, 0.0, 0.0]), axis=1)
    i = int(np.argmin(r))
    if 0 < i < len(r) - 1:
        r0, r1, r2 = r[i - 1], r[i], r[i + 1]
        denom = r0 - 2 * r1 + r2
        if denom > 0:
            return float(r1 - 0.125 * (r2 - r0) ** 2 / denom)
    return float(r[i])
