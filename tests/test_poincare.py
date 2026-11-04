"""Phase 10 tests: tube cuts on the section x = 1 - mu are closed curves and obey time reversal."""

import numpy as np
import pytest

from cislunar import cr3bp, orbits, poincare, shooting
from cislunar.constants import EARTH_MOON, R_EARTH_KM, R_MOON_KM

MU = EARTH_MOON.mu
RE, RM = R_EARTH_KM / EARTH_MOON.length_km, R_MOON_KM / EARTH_MOON.length_km
C_SECTION = (
    3.17  # between C(L2) and C(L1), both necks open
)
N = 24
REVERSAL = np.array([1, -1, 1, -1, 1, -1.0])  # (x, y, z, vx, vy, vz) -> (x, -y, z, -vx, vy, -vz)
SYMMETRY_TOL = 1e-7


@pytest.fixture(scope="module")
def l2_lyapunov() -> shooting.SymmetricOrbit:
    fam = orbits.lyapunov_family(2, MU, np.linspace(2e-3, 0.06, 16))
    i = int(np.argmin(np.abs(fam.jacobi - C_SECTION)))
    return shooting.correct_planar_fixed_energy(fam.orbits[i].X0, MU, C_SECTION)


@pytest.fixture(scope="module")
def cuts(l2_lyapunov):
    below = poincare.plane_section(0, 1 - MU, extra=lambda X: X[1] < 0)
    above = poincare.plane_section(0, 1 - MU, extra=lambda X: X[1] > 0)
    s = poincare.tube_cut(l2_lyapunov, "s", -1.0, below, n_traj=N, r_earth=RE, r_moon=RM)
    u = poincare.tube_cut(l2_lyapunov, "u", -1.0, above, n_traj=N, r_earth=RE, r_moon=RM)
    return s, u


def test_fixed_energy_corrector(l2_lyapunov) -> None:
    assert l2_lyapunov.jacobi == pytest.approx(C_SECTION, abs=1e-12)
    assert orbits.closure_error(l2_lyapunov) < 1e-10


def test_crossings_lie_on_section_with_orbit_energy(cuts) -> None:
    for cut in cuts:
        assert len(cut.states) == N  # every strand reaches the section
        assert np.max(np.abs(cut.states[:, 0] - (1 - MU))) < 1e-12
        assert np.max(np.abs(cr3bp.jacobi_constant(cut.states, MU) - C_SECTION)) < 1e-9


def test_cut_is_a_closed_curve(cuts) -> None:
    s, _ = cuts
    max_gap, closing = poincare.closed_curve_gaps(s.states[:, [1, 4]])
    # Ordered by phase, consecutive points are neighbours and the last joins the first.
    assert closing < 1.5 * max_gap
    assert max_gap < 0.2 * np.ptp(s.states[:, 4])


def test_time_reversal_maps_unstable_cut_onto_stable_cut(cuts) -> None:
    s, u = cuts
    for k, p in enumerate(u.phases):
        j = int(np.argmin(np.abs(((1 - p) % 1) - s.phases)))
        # Relative to the state size: some strands cross on a fast (~2 km/s) low lunar pass.
        diff = np.max(np.abs(u.states[k] * REVERSAL - s.states[j]))
        assert diff / np.max(np.abs(s.states[j])) < SYMMETRY_TOL
        # Crossing time error ~ (state error) / |vx at the crossing|, with |vx| ~ 0.1 here.
        assert u.times[k] == pytest.approx(-s.times[j], abs=SYMMETRY_TOL)


def test_integrator_is_exactly_time_reversible() -> None:
    # Dormand-Prince applied to R(X0) backward reproduces R(forward result) to the last bit,
    # because every arithmetic step is mirrored under the sign flips in R.
    from cislunar.integrators import dopri5

    f = cr3bp.make_rhs(MU)
    X0 = np.array([1.12, 0.01, 0.0, 0.02, -0.2, 0.0])
    fwd = dopri5(f, 0.0, 3.0, X0, rtol=1e-12, atol=1e-12).X_final
    bwd = dopri5(f, 0.0, -3.0, X0 * REVERSAL, rtol=1e-12, atol=1e-12).X_final
    assert np.max(np.abs(bwd * REVERSAL - fwd)) == 0.0


def test_polyline_intersections_and_gap_splitting() -> None:
    sq = [np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)]
    line = [np.array([[-1, 0.5], [2, 0.5]])]
    pts = poincare.polyline_intersections(sq, line)
    np.testing.assert_allclose(sorted(pts[:, 0]), [0.0, 1.0])
    cut = poincare.SectionCut(np.zeros((5, 6)), np.zeros(5), np.array([0.0, 0.1, 0.2, 0.6, 0.7]))
    assert [len(p) for p in poincare.split_at_gaps(cut, 10, [1, 4])] == [3, 2]
