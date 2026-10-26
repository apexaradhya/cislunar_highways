"""Phase 7 tests: planar Lyapunov orbits and natural-parameter continuation."""

import json
from pathlib import Path

import numpy as np
import pytest

from cislunar import cr3bp, lagrange, orbits, shooting
from cislunar.constants import EARTH_MOON

MU = EARTH_MOON.mu
CLOSURE_TOL = 1e-10  # spec Phase 7
JACOBI_TOL = 1e-11  # drift along one period at integration tol 1e-13
# The trivial eigenvalue 1 is defective (a 2x2 Jordan block): a perturbation eps in M
# splits it into 1 +- sqrt(eps*||M||), but leaves the *mean* of the pair at 1 + O(eps).
# So we test the mean tightly and the flow direction f(X0) as an exact eigenvector.
TRIVIAL_MEAN_TOL = 1e-8
FLOW_EIGVEC_TOL = 1e-7
RECIPROCAL_TOL = 1e-6  # spec Phase 8 value, also applied here
CATALOG = Path(__file__).parent / "data" / "jpl_lyapunov.json"


@pytest.fixture(scope="module")
def l1_family() -> orbits.Family:
    return orbits.lyapunov_family(1, MU, np.linspace(2e-3, 0.08, 8))


@pytest.fixture(scope="module")
def l2_family() -> orbits.Family:
    return orbits.lyapunov_family(2, MU, np.linspace(2e-3, 0.08, 8))


@pytest.mark.parametrize("which", [1, 2])
def test_small_orbit_period_tends_to_linear_period(which: int) -> None:
    T_lin = lagrange.collinear_modes(which, MU).planar_period
    d_small = orbits.lyapunov_orbit(which, MU, 1e-4).period - T_lin
    d_large = orbits.lyapunov_orbit(which, MU, 1e-3).period - T_lin
    assert abs(d_small) / T_lin < 1e-5
    # Nonlinear period correction is O(A^2): 10x amplitude -> ~100x correction.
    assert 70 < d_large / d_small < 130


@pytest.mark.parametrize("fam_name", ["l1_family", "l2_family"])
def test_family_orbits_close_and_conserve_jacobi(fam_name: str, request) -> None:
    fam = request.getfixturevalue(fam_name)
    for o in fam.orbits:
        assert o.residual < shooting.CROSSING_TOL
        sol = shooting.full_period(o)
        X = sol.X[:, :6]
        assert np.max(np.abs(X[-1] - o.X0)) < CLOSURE_TOL
        C = cr3bp.jacobi_constant(X, MU)
        assert np.max(np.abs(C - C[0])) < JACOBI_TOL


@pytest.mark.parametrize("fam_name", ["l1_family", "l2_family"])
def test_monodromy_structure(fam_name: str, request) -> None:
    fam = request.getfixturevalue(fam_name)
    for o in fam.orbits[::3]:
        st = orbits.stability(o)
        a, b = st.trivial_pair
        assert abs(0.5 * (a + b) - 1.0) < TRIVIAL_MEAN_TOL
        f = cr3bp.equations_of_motion(0.0, o.X0, MU)
        assert np.linalg.norm(st.monodromy @ f - f) / np.linalg.norm(f) < FLOW_EIGVEC_TOL
        g = cr3bp.jacobi_gradient(o.X0, MU)  # energy conservation: gradC is a left eigenvector
        assert np.linalg.norm(g @ st.monodromy - g) / np.linalg.norm(g) < FLOW_EIGVEC_TOL
        for p, q in st.pairs:
            assert abs(p * q - 1.0) < RECIPROCAL_TOL
        assert st.max_index > 1.0  # planar Lyapunov orbits near L1/L2 are unstable


def test_family_is_smooth_and_energy_decreases(l1_family, l2_family) -> None:
    for fam in (l1_family, l2_family):
        assert np.all(np.diff(fam.periods) > 0)  # period grows with amplitude
        assert np.all(np.diff(fam.jacobi) < 0)  # bigger orbit = more energy = lower C


def test_half_period_symmetry(l1_family) -> None:
    o = l1_family.orbits[-1]
    sol = shooting.half_period_crossing(o.X0, MU)
    Xh = sol.X_final[:6]
    assert abs(Xh[1]) < 1e-12 and abs(Xh[3]) < shooting.CROSSING_TOL


@pytest.mark.skipif(
    not CATALOG.exists(),
    reason="JPL catalog entries not available (network blocked); see PROGRESS.md",
)
def test_matches_jpl_catalog() -> None:
    entries = json.loads(CATALOG.read_text())
    for e in entries:
        mu_cat = e["mu"]
        X0 = np.array(e["state"], dtype=float)
        orbit = shooting.correct_symmetric(X0, mu_cat, free=(4,), targets=(3,))
        assert orbit.period == pytest.approx(e["period"], rel=1e-6)
        assert cr3bp.jacobi_constant(orbit.X0, mu_cat) == pytest.approx(e["jacobi"], abs=1e-6)
