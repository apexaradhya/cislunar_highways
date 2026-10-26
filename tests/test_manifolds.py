"""Phase 9 tests: eigen-directions, energy of manifold states, Jacobi along tubes, asymptotics."""

import numpy as np
import pytest

from cislunar import cr3bp, manifolds, orbits
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM
from cislunar.shooting import correct_symmetric

MU = EARTH_MOON.mu
L_KM = EARTH_MOON.length_km
JACOBI_TOL = 1e-8  # spec Phase 9: manifold trajectory C equals orbit C to ~1e-8
# A moderately unstable L2 halo (lambda_u ~ 100) keeps the stroboscopic tests well above the
# noise floor: integration error ~1e-13 grows by lambda_u per period.
Z0_TEST_HALO = 0.17


@pytest.fixture(scope="module")
def halo() -> orbits.SymmetricOrbit:
    # L2 northern halo: the stored family member nearest z0 = 0.17, re-corrected.
    d = np.load("data/halo_L2_north.npz")
    i = int(np.argmin(np.abs(d["X0"][:, 2] - Z0_TEST_HALO)))
    return correct_symmetric(d["X0"][i], MU, free=(0, 4), targets=(3, 5))


@pytest.fixture(scope="module")
def samples(halo) -> manifolds.OrbitSamples:
    return manifolds.sample_orbit(halo, 40)


def test_eigen_directions(samples, halo) -> None:
    d = samples.directions
    assert d.lambda_u > 1 > d.lambda_s > 0
    assert d.lambda_u * d.lambda_s == pytest.approx(1.0, abs=1e-6)
    g = cr3bp.jacobi_gradient(halo.X0, MU)
    # gradC^T M = gradC^T  ==>  gradC . v = 0 for any eigenvector with lambda != 1.
    assert abs(g @ d.v_u) < 1e-9 and abs(g @ d.v_s) < 1e-9


def test_displacement_changes_jacobi_only_at_second_order(samples, halo) -> None:
    dC = {}
    for eps in (1e-5, 1e-4):
        X = manifolds.manifold_states(samples, "u", +1.0, eps)
        dC[eps] = np.max(np.abs(cr3bp.jacobi_constant(X, MU) - halo.jacobi))
    assert dC[1e-5] < 1e-8
    assert 60 < dC[1e-4] / dC[1e-5] < 140  # O(eps^2): 10x eps -> ~100x change


@pytest.mark.parametrize(("kind", "sign"), [("s", 1.0), ("s", -1.0), ("u", 1.0), ("u", -1.0)])
def test_jacobi_conserved_along_tubes(halo, kind, sign) -> None:
    s = manifolds.sample_orbit(halo, 8)
    tube = manifolds.generate_tube(
        halo,
        kind,
        sign,
        6.0,
        eps=1e-6,
        r_earth=R_EARTH_KM / L_KM,
        r_moon=R_MOON_KM / L_KM,
        samples=s,
    )
    for tr in tube:
        assert np.max(np.abs(cr3bp.jacobi_constant(tr.X, MU) - halo.jacobi)) < JACOBI_TOL


def test_stable_manifold_approaches_orbit_at_rate_lambda_s(samples, halo) -> None:
    # Forward: one period of contraction is observable before the unstable component
    # (amplified ~lambda_u per period) takes over; see stroboscopic_distances docstring.
    lam_s = samples.directions.lambda_s
    d = manifolds.stroboscopic_distances(samples, halo, 5, "s", 1e-6, 1)
    assert d[1] / d[0] == pytest.approx(lam_s, rel=0.02)


def test_stable_manifold_rate_over_several_periods_backward(samples, halo) -> None:
    # Backward in time the stable direction is the growing one, so the exponential rate
    # 1/lambda_s can be followed for several periods without contamination.
    lam_s = samples.directions.lambda_s
    d = manifolds.stroboscopic_distances(samples, halo, 5, "s", 1e-12, 3, backward=True)
    np.testing.assert_allclose(d[1:] / d[:-1], 1.0 / lam_s, rtol=0.02)


def test_unstable_manifold_departs_at_rate_lambda_u(samples, halo) -> None:
    lam_u = samples.directions.lambda_u
    d = manifolds.stroboscopic_distances(samples, halo, 5, "u", 1e-10, 2)
    np.testing.assert_allclose(d[1:] / d[:-1], lam_u, rtol=0.02)
