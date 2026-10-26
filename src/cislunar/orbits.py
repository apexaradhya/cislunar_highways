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
from cislunar.shooting import CorrectionError, SymmetricOrbit, correct_symmetric, full_period

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
