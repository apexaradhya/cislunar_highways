"""Phase 3 tests: Lagrange points."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from cislunar import cr3bp, lagrange
from cislunar.constants import EARTH_MOON

MU_EM = EARTH_MOON.mu
MU_SE = 3.0035e-6  # Sun-(Earth+Moon) mass parameter quoted in PROJECT_SPEC.md (Phase 3)
# Spec: force residual < 1e-13. A root in float64 can do ~|f'| * ulp(x) ~ 1e-15.
RESIDUAL_TOL = 1e-13
# Agreement with scipy brentq at xtol=1e-15: both converge to within a few ulp.
SCIPY_TOL = 1e-14
# Literature sanity values (Koon, Lo, Marsden & Ross 2011, sec. 2.5, Earth-Moon) quoted
# to 4 decimals, so only checked to 1e-4.
LITERATURE = {"L1": 0.8369, "L2": 1.1557, "L3": -1.0051}
LIT_TOL = 1e-4


@pytest.mark.parametrize("mu", [MU_EM, MU_SE, 0.1, 0.5])
def test_full_force_residual_is_tiny(mu: float) -> None:
    for name, r in lagrange.lagrange_points(mu).items():
        X = np.concatenate([r, np.zeros(3)])
        acc = cr3bp.equations_of_motion(0.0, X, mu)[3:]
        assert np.max(np.abs(acc)) < RESIDUAL_TOL, name


@pytest.mark.parametrize("mu", [MU_EM, MU_SE])
def test_collinear_points_match_scipy_brentq(mu: float) -> None:
    eps = 1e-9
    brackets = {1: (-mu + eps, 1 - mu - eps), 2: (1 - mu + eps, 2.0), 3: (-2.0, -mu - eps)}
    for k, (a, b) in brackets.items():
        ref = brentq(
            lambda x: lagrange.collinear_force(x, mu),
            a,
            b,
            xtol=1e-15,
            rtol=4 * np.finfo(float).eps,
        )  # scipy minimum
        assert lagrange.collinear_point(k, mu) == pytest.approx(ref, abs=SCIPY_TOL)


def test_earth_moon_points_match_literature() -> None:
    pts = lagrange.lagrange_points(MU_EM)
    for name, x_ref in LITERATURE.items():
        assert pts[name][0] == pytest.approx(x_ref, abs=LIT_TOL)


def test_collinear_ordering() -> None:
    pts = lagrange.lagrange_points(MU_EM)
    assert pts["L3"][0] < -MU_EM < pts["L1"][0] < 1 - MU_EM < pts["L2"][0]


@pytest.mark.parametrize("name", ["L4", "L5"])
def test_triangular_points_are_equilateral(name: str) -> None:
    p = lagrange.lagrange_points(MU_EM)[name]
    earth, moon = np.array([-MU_EM, 0, 0]), np.array([1 - MU_EM, 0, 0])
    for side in (np.linalg.norm(p - earth), np.linalg.norm(p - moon), np.linalg.norm(moon - earth)):
        assert side == pytest.approx(1.0, abs=1e-15)


def test_jacobi_ordering() -> None:
    # Energy ordering of the equilibria (Koon et al. 2011, sec. 2.6): C1 > C2 > C3 > C4 = C5 = 3.
    C = lagrange.jacobi_at_points(MU_EM)
    assert C["L1"] > C["L2"] > C["L3"] > C["L4"]
    assert C["L4"] == pytest.approx(3.0, abs=1e-14)
    assert C["L5"] == pytest.approx(3.0, abs=1e-14)


def test_routh_critical_value() -> None:
    mu0 = lagrange.routh_critical_mu()
    assert 27 * mu0 * (1 - mu0) == pytest.approx(1.0, abs=1e-15)
    assert mu0 == pytest.approx(0.03852, abs=1e-5)  # spec quotes ~0.03852


@pytest.mark.parametrize(
    ("mu", "stable"),
    [
        (MU_SE, True),
        (MU_EM, True),
        (0.03, True),
        (0.038, True),
        (0.039, False),
        (0.1, False),
        (0.5, False),
    ],
)
def test_routh_criterion_numerically(mu: float, stable: bool) -> None:
    assert lagrange.triangular_is_linearly_stable(mu) is stable
    assert (27 * mu * (1 - mu) < 1) is stable


def test_collinear_points_are_unstable() -> None:
    for name in ("L1", "L2", "L3"):
        eig = np.linalg.eigvals(cr3bp.a_matrix(lagrange.equilibrium_state(name, MU_EM), MU_EM))
        assert np.max(eig.real) > 1e-3, name
    assert math.isfinite(lagrange.collinear_c(lagrange.collinear_point(1, MU_EM), MU_EM))
