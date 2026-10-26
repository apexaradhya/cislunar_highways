"""Equilibrium (Lagrange) points of the CR3BP and their linear stability.

Equilibria: v = 0 and a = 0, i.e. grad(Omega) = 0 (PROJECT_SPEC.md 1.5).

Collinear points L1, L2, L3 (y = z = 0) are roots of
    f(x) = x - (1-mu)(x+mu)/|x+mu|^3 - mu(x-1+mu)/|x-1+mu|^3 = 0
on three intervals separated by the singularities at the primaries:
    L1 in (-mu, 1-mu)   between Earth and Moon
    L2 in (1-mu, 2)     beyond the Moon
    L3 in (-2, -mu)     beyond Earth, opposite the Moon
f -> -inf just right of each primary and +inf just left of it (and f is
monotonically increasing on each interval, f' = 1 + 2c > 0), so each interval
holds exactly one root. We find it with our own Brent solver and finish with
one Newton step.

Triangular points L4, L5 form equilateral triangles with the primaries:
    (1/2 - mu, +-sqrt(3)/2, 0)            (Szebehely 1967, sec. 4.4)

Linear stability of L4/L5 (Routh 1875; Szebehely 1967, sec. 5.6): the planar
characteristic equation is s^4 + s^2 + (27/4) mu (1-mu) = 0, which has purely
imaginary roots iff 27 mu (1-mu) < 1, i.e. mu < mu_Routh = (1 - sqrt(23/27))/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import cr3bp
from cislunar.rootfind import brent

FloatArray = NDArray[np.float64]

# Distance kept from each singularity when bracketing: f is ~ -1/eps^2 there,
# so any eps far below the true root distance (>~0.03 for Earth-Moon, >~0.01
# for Sun-Earth) is safe. This one is safe for mu down to ~1e-15.
_BRACKET_EPS = 1e-9
# Outer bracket limit for L2/L3. For any mu in (0, 0.5], L2 and L3 lie within
# distance ~1.2 of the barycenter (Szebehely 1967, Table 4.1), so 2 is generous.
_OUTER = 2.0

POINT_NAMES = ("L1", "L2", "L3", "L4", "L5")


def collinear_force(x: float, mu: float) -> float:
    """f(x): x-acceleration of a particle at rest on the x-axis (zero at L1-L3)."""
    d1 = x + mu
    d2 = x - 1.0 + mu
    return x - (1.0 - mu) * d1 / abs(d1) ** 3 - mu * d2 / abs(d2) ** 3


def collinear_c(x: float, mu: float) -> float:
    """c = (1-mu)/|x+mu|^3 + mu/|x-1+mu|^3  (PROJECT_SPEC.md 1.6). Note f'(x) = 1 + 2c."""
    return (1.0 - mu) / abs(x + mu) ** 3 + mu / abs(x - 1.0 + mu) ** 3


def collinear_point(which: int, mu: float) -> float:
    """x-coordinate of L1 (which=1), L2 (2) or L3 (3)."""
    brackets = {
        1: (-mu + _BRACKET_EPS, 1.0 - mu - _BRACKET_EPS),
        2: (1.0 - mu + _BRACKET_EPS, _OUTER),
        3: (-_OUTER, -mu - _BRACKET_EPS),
    }
    a, b = brackets[which]
    x = brent(lambda s: collinear_force(s, mu), a, b, xtol=1e-16)
    # One Newton step polishes Brent's last bit: f'(x) = 1 + 2c(x).
    x -= collinear_force(x, mu) / (1.0 + 2.0 * collinear_c(x, mu))
    return x


def triangular_point(which: int, mu: float) -> FloatArray:
    """Position of L4 (which=4, y > 0) or L5 (which=5, y < 0)."""
    sign = 1.0 if which == 4 else -1.0
    return np.array([0.5 - mu, sign * math.sqrt(3.0) / 2.0, 0.0])


def lagrange_points(mu: float) -> dict[str, FloatArray]:
    """Positions (3-vectors) of all five equilibrium points."""
    pts = {f"L{k}": np.array([collinear_point(k, mu), 0.0, 0.0]) for k in (1, 2, 3)}
    pts["L4"] = triangular_point(4, mu)
    pts["L5"] = triangular_point(5, mu)
    return pts


def equilibrium_state(name: str, mu: float) -> FloatArray:
    """6-state [x, y, z, 0, 0, 0] at a Lagrange point."""
    return np.concatenate([lagrange_points(mu)[name], np.zeros(3)])


def jacobi_at_points(mu: float) -> dict[str, float]:
    """Jacobi constant at each equilibrium (v = 0, so C = 2 Omega)."""
    return {name: float(cr3bp.jacobi_constant(equilibrium_state(name, mu), mu)) for name in POINT_NAMES}


def routh_critical_mu() -> float:
    """mu_Routh = (1 - sqrt(23/27))/2, the root of 27 mu (1-mu) = 1 below 1/2."""
    return 0.5 * (1.0 - math.sqrt(23.0 / 27.0))


def triangular_is_linearly_stable(mu: float, tol: float = 1e-10) -> bool:
    """Numerical test: all eigenvalues of A at L4 have |Re| <= tol (spectrally stable)."""
    eig = np.linalg.eigvals(cr3bp.a_matrix(equilibrium_state("L4", mu), mu))
    return bool(np.max(np.abs(eig.real)) <= tol)


# ---------------------------------------------------------------------------
# Linear dynamics at the collinear points (PROJECT_SPEC.md 1.6, Phase 5)
# ---------------------------------------------------------------------------
# With c as above, Omega_xx = 1 + 2c, Omega_yy = 1 - c, Omega_zz = -c, all cross terms 0.
# Linearized motion about Li (xi = x - xL, eta = y, zeta = z):
#     xi''  - 2 eta' = (1 + 2c) xi
#     eta'' + 2 xi'  = (1 - c) eta
#     zeta''         = -c zeta
# Trying e^{st} in the planar pair gives s^4 + (2 - c) s^2 + (1 + c - 2c^2) = 0, a
# quadratic in u = s^2 with discriminant (2-c)^2 - 4(1+c-2c^2) = 9c^2 - 8c > 0 for c > 1:
#     u_+ = (c - 2 + sqrt(9c^2 - 8c)) / 2 > 0  -> s = +-lambda   (saddle)
#     u_- = (c - 2 - sqrt(9c^2 - 8c)) / 2 < 0  -> s = +-i omega_p (planar center)
# and the vertical oscillation has frequency nu = sqrt(c)  (vertical center).
# (Koon, Lo, Marsden & Ross 2011, sec. 2.7; Szebehely 1967, sec. 5.4.)


@dataclass(frozen=True)
class CollinearModes:
    name: str
    x: float
    c: float
    lam: float  # real (saddle) exponent
    omega_p: float  # in-plane oscillation frequency
    nu: float  # out-of-plane oscillation frequency
    k: float  # in-plane ellipse ratio: eta amplitude = k * xi amplitude

    @property
    def planar_period(self) -> float:
        return 2.0 * math.pi / self.omega_p

    @property
    def vertical_period(self) -> float:
        return 2.0 * math.pi / self.nu


def collinear_modes(which: int, mu: float) -> CollinearModes:
    """Analytic linear exponents and frequencies at L1/L2/L3."""
    x = collinear_point(which, mu)
    c = collinear_c(x, mu)
    disc = math.sqrt(9.0 * c * c - 8.0 * c)
    u_plus = 0.5 * (c - 2.0 + disc)
    u_minus = 0.5 * (c - 2.0 - disc)
    omega_p = math.sqrt(-u_minus)
    # In-plane periodic linear solution xi = A cos(w t), eta = -k A sin(w t), with
    # k = (w^2 + 1 + 2c) / (2 w) from the xi equation.
    k = (omega_p**2 + 1.0 + 2.0 * c) / (2.0 * omega_p)
    return CollinearModes(f"L{which}", x, c, math.sqrt(u_plus), omega_p, math.sqrt(c), k)


def linear_lyapunov_state(which: int, mu: float, amplitude: float) -> FloatArray:
    """Initial state on the linearized planar periodic orbit around Li.

    xi(0) = A_x (on the x-axis), eta(0) = 0, xi'(0) = 0, eta'(0) = -k omega_p A_x.
    Used to seed the differential corrector for planar Lyapunov orbits (Phase 7).
    """
    m = collinear_modes(which, mu)
    return np.array([m.x + amplitude, 0.0, 0.0, 0.0, -m.k * m.omega_p * amplitude, 0.0])
