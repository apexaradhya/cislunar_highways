"""Phase 1 tests: constants, units, EOM, Jacobi constant, A-matrix."""

import numpy as np
import pytest

from cislunar import cr3bp
from cislunar.constants import EARTH_MOON, SECONDS_PER_DAY
from cislunar.units import days_to_time, km_to_length, length_to_km, time_to_days

# Spec 1.1 value of mu (also the JPL Three-Body Periodic Orbit Catalog value).
MU_SPEC = 0.0121505856
# "Match to 7 digits" = 7 significant figures -> relative tolerance 5e-7.
MU_REL_TOL = 5e-7
# Central-difference Jacobian error ~ h^2 * f''' ~ 1e-12 plus roundoff eps/h ~ 1e-10
# with h = 1e-6, so 1e-6 is a comfortable bound that still catches any wrong term.
FD_JACOBIAN_TOL = 1e-6
FD_STEP = 1e-6
# dC/dt is analytically zero; terms are O(1-100) near the primaries, so roundoff ~1e-13.
JACOBI_RATE_TOL = 1e-12
SEED = 20261026

MU = EARTH_MOON.mu


def random_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random states in a box around the system, kept away from the singular primaries."""
    states = []
    while len(states) < n:
        X = np.concatenate([rng.uniform(-1.5, 1.5, 3) * [1, 1, 0.5], rng.uniform(-1, 1, 3)])
        r1, r2 = cr3bp.primary_distances(X, MU)
        if r1 > 0.05 and r2 > 0.05:
            states.append(X)
    return np.array(states)


def test_mu_matches_spec_to_7_digits() -> None:
    assert EARTH_MOON.mu == pytest.approx(MU_SPEC, rel=MU_REL_TOL)


def test_units_are_physically_sensible() -> None:
    # Spec 1.1 gives T ~ 4.34 days and V ~ 1.0245 km/s as sanity checks (not pass/fail values).
    assert EARTH_MOON.time_days == pytest.approx(4.34, abs=0.01)
    assert EARTH_MOON.velocity_km_s == pytest.approx(1.0245, abs=0.001)
    # The frame rotates once per sidereal month: 2*pi*T should be ~27.3 days.
    assert 2 * np.pi * EARTH_MOON.time_s / SECONDS_PER_DAY == pytest.approx(27.3, abs=0.2)


def test_unit_round_trips() -> None:
    assert km_to_length(length_to_km(0.123)) == pytest.approx(0.123, rel=1e-15)
    assert days_to_time(time_to_days(2.5)) == pytest.approx(2.5, rel=1e-15)


def test_eom_single_and_batched_agree() -> None:
    rng = np.random.default_rng(SEED)
    Xs = random_states(10, rng)
    batched = cr3bp.equations_of_motion(0.0, Xs, MU)
    single = np.array([cr3bp.equations_of_motion(0.0, X, MU) for X in Xs])
    assert batched.shape == (10, 6)
    np.testing.assert_array_equal(batched, single)


def test_eom_known_value_at_rest_on_axis() -> None:
    # At rest at x=0.5 on the x-axis: ax = x - (1-mu)(x+mu)/r1^3 - mu(x-1+mu)/r2^3.
    x = 0.5
    expected_ax = (
        x - (1 - MU) * (x + MU) / abs(x + MU) ** 3 - MU * (x - 1 + MU) / abs(x - 1 + MU) ** 3
    )
    f = cr3bp.equations_of_motion(0.0, np.array([x, 0, 0, 0, 0, 0]), MU)
    assert f[3] == pytest.approx(expected_ax, rel=1e-15)
    assert f[4] == 0.0 and f[5] == 0.0


def test_a_matrix_matches_finite_difference_jacobian() -> None:
    rng = np.random.default_rng(SEED)
    for X in random_states(8, rng):
        A = cr3bp.a_matrix(X, MU)
        A_fd = np.zeros((6, 6))
        for j in range(6):
            dX = np.zeros(6)
            dX[j] = FD_STEP
            fp = cr3bp.equations_of_motion(0.0, X + dX, MU)
            fm = cr3bp.equations_of_motion(0.0, X - dX, MU)
            A_fd[:, j] = (fp - fm) / (2 * FD_STEP)
        np.testing.assert_allclose(A, A_fd, atol=FD_JACOBIAN_TOL, rtol=FD_JACOBIAN_TOL)


def test_hessian_is_symmetric() -> None:
    rng = np.random.default_rng(SEED)
    H = cr3bp.hessian_pseudo_potential(random_states(5, rng), MU)
    np.testing.assert_array_equal(H, np.swapaxes(H, -1, -2))


def test_gradient_matches_finite_difference_of_potential() -> None:
    rng = np.random.default_rng(SEED)
    for X in random_states(5, rng):
        g = cr3bp.grad_pseudo_potential(X, MU)
        for j in range(3):
            dX = np.zeros(6)
            dX[j] = FD_STEP
            fd = (cr3bp.pseudo_potential(X + dX, MU) - cr3bp.pseudo_potential(X - dX, MU)) / (
                2 * FD_STEP
            )
            assert g[j] == pytest.approx(fd, abs=FD_JACOBIAN_TOL)


def test_jacobi_rate_is_zero() -> None:
    rng = np.random.default_rng(SEED)
    rates = cr3bp.jacobi_rate(random_states(50, rng), MU)
    assert np.max(np.abs(rates)) < JACOBI_RATE_TOL


def test_jacobi_at_triangular_point_is_three() -> None:
    # With the mu(1-mu)/2 convention, C(L4) = 3 exactly (Szebehely 1967, sec. 4.5).
    L4 = np.array([0.5 - MU, np.sqrt(3) / 2, 0, 0, 0, 0])
    assert cr3bp.jacobi_constant(L4, MU) == pytest.approx(3.0, abs=1e-14)
