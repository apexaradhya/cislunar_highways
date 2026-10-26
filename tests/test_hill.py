"""Phase 4 tests: necks open exactly at C(L1), C(L2), C(L3)."""

import numpy as np
import pytest

from cislunar import hill, lagrange
from cislunar.constants import EARTH_MOON

MU = EARTH_MOON.mu
# Grid spacing h = 3/1500 = 2e-3. Near a saddle, 2*Omega differs from its saddle value by
# ~ |d2(2 Omega)/dx2| * (h/2)^2 <= 2*(1+2c)*(1e-3)^2 ~ 2.4e-5 for Earth-Moon L1 (c ~ 5.1).
# So the grid-detected opening value can differ from C(Li) by up to ~2.4e-5.
NECK_TOL = 3e-5


@pytest.fixture(scope="module")
def grid() -> hill.PlanarGrid:
    return hill.planar_grid(MU)


def test_neck_values_match_lagrange_jacobi_constants(grid) -> None:
    found = hill.neck_opening_values(grid, MU)
    C = lagrange.jacobi_at_points(MU)
    for name in ("L1", "L2", "L3"):
        # Along the neck's narrowest line, Li is a *minimum* of 2*Omega. Grid samples miss that
        # exact minimum, so the grid sees the neck open slightly early: found >= C(Li).
        assert found[name] >= C[name] - 1e-12
        assert found[name] == pytest.approx(C[name], abs=NECK_TOL), name


def test_realm_sequence(grid) -> None:
    C = lagrange.jacobi_at_points(MU)
    above = hill.connectivity(grid, C["L1"] + 0.01, MU)
    assert not above.earth_moon and not above.moon_exterior
    between12 = hill.connectivity(grid, 0.5 * (C["L1"] + C["L2"]), MU)
    assert between12.earth_moon and not between12.moon_exterior
    between23 = hill.connectivity(grid, 0.5 * (C["L2"] + C["L3"]), MU)
    assert between23.earth_moon and between23.moon_exterior and between23.forbidden_components == 1
    below3 = hill.connectivity(grid, 0.5 * (C["L3"] + 3.0), MU)
    assert below3.forbidden_components == 2


def test_no_forbidden_region_below_three(grid) -> None:
    # min over the plane of 2 Omega is 3, attained at L4/L5 (with our Omega convention).
    assert np.min(grid.two_omega) >= 3.0 - 1e-12
    assert not hill.forbidden_mask(grid, 2.999).any()
