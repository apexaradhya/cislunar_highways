"""Phase 8 tests: halo bifurcation, halo family, monodromy, mirror symmetry, 9:2 NRHO."""

import json
from pathlib import Path

import numpy as np
import pytest

from cislunar import cr3bp, orbits, shooting
from cislunar.constants import EARTH_MOON, SYNODIC_MONTH_DAYS

MU = EARTH_MOON.mu
CLOSURE_TOL = 1e-10  # spec Phase 8
CROSSING_TOL = 1e-11  # spec Phase 8
RECIPROCAL_TOL = 1e-6  # spec Phase 8
# Spec asks for the trivial pair (1, 1) to 1e-6. But eigenvalue 1 is defective (2x2 Jordan
# block), so its individual eigenvalues are ill-conditioned: an error eps in M moves them
# by ~sqrt(eps*||M||) (Trefethen & Bau 1997, Lecture 25), i.e. ~1e-6..1e-5 here for any
# halo, near the Moon or not. We therefore test what the trivial pair *means*, with
# well-conditioned O(eps) checks: M f = f (flow direction), gradC^T M = gradC^T (energy),
# and mean of the pair = 1. The raw split is reported in PROGRESS.md / data files.
TRIVIAL_MEAN_TOL = 1e-8
FLOW_EIGVEC_TOL = 1e-7
ENERGY_EIGVEC_TOL = 1e-7
CATALOG = Path(__file__).parent / "data" / "jpl_halo.json"


@pytest.fixture(scope="module")
def l1_bifurcation() -> shooting.SymmetricOrbit:
    return orbits.find_halo_bifurcation(1, MU, 0.010, 0.017)


@pytest.fixture(scope="module")
def l1_halos(l1_bifurcation) -> orbits.Family:
    seed = orbits.halo_from_bifurcation(l1_bifurcation, 2e-3)
    return orbits.pseudo_arclength_family(
        seed, 12, 0.02, MU, direction=np.array([0, 1, 0]), ds_max=0.04
    )


def test_bifurcation_has_unit_vertical_index(l1_bifurcation) -> None:
    assert orbits.vertical_index(l1_bifurcation) == pytest.approx(1.0, abs=1e-4)
    assert l1_bifurcation.X0[2] == 0.0  # still planar


def test_halo_seed_branches_off(l1_bifurcation) -> None:
    halo = orbits.halo_from_bifurcation(l1_bifurcation, 1e-3)
    assert halo.residual < CROSSING_TOL
    # Close to the bifurcation the halo barely differs from the Lyapunov orbit.
    assert abs(halo.period - l1_bifurcation.period) < 1e-4


def test_halo_family_closure_and_monodromy(l1_halos) -> None:
    assert len(l1_halos.orbits) == 12
    assert (
        max(abs(o.X0[2]) for o in l1_halos.orbits) > 0.1
    )  # continuation travelled well beyond the bifurcation
    for o in l1_halos.orbits:
        assert o.residual < CROSSING_TOL
        assert orbits.closure_error(o) < CLOSURE_TOL
        st = orbits.stability(o)
        for p, q in st.pairs:
            assert abs(p * q - 1) < RECIPROCAL_TOL
        a, b = st.trivial_pair
        assert abs(0.5 * (a + b) - 1) < TRIVIAL_MEAN_TOL
        f = cr3bp.equations_of_motion(0.0, o.X0, MU)
        assert np.linalg.norm(st.monodromy @ f - f) / np.linalg.norm(f) < FLOW_EIGVEC_TOL
        g = cr3bp.jacobi_gradient(o.X0, MU)
        assert np.linalg.norm(g @ st.monodromy - g) / np.linalg.norm(g) < ENERGY_EIGVEC_TOL


def test_jacobi_conserved_on_halo(l1_halos) -> None:
    sol = shooting.full_period(l1_halos.orbits[-1], n_points=500)
    C = cr3bp.jacobi_constant(sol.X[:, :6], MU)
    assert np.max(np.abs(C - C[0])) < 1e-11


def test_southern_mirror_is_also_periodic(l1_halos) -> None:
    north = l1_halos.orbits[-1]
    south = orbits.mirror_z(north)
    assert orbits.closure_error(south) < CLOSURE_TOL
    assert south.jacobi == pytest.approx(north.jacobi, abs=1e-15)
    sol_n = shooting.full_period(north, n_points=50).X[:, :3]
    sol_s = shooting.full_period(south, n_points=50).X[:, :3]
    np.testing.assert_allclose(sol_s[:, 2], -sol_n[:, 2], atol=1e-9)


def test_fixed_period_corrector_hits_9_2_resonance() -> None:
    target = 2 * SYNODIC_MONTH_DAYS / 9 / EARTH_MOON.time_days
    # Start from a nearby L2 NRHO-like guess (a family member with period ~6.5 d).
    guess = np.array([1.0220, 0.0, 0.1821, 0.0, -0.1033, 0.0])
    o = shooting.correct_symmetric_fixed_period(guess, MU, target)
    assert o.period == pytest.approx(target, abs=1e-11)
    assert orbits.closure_error(o) < CLOSURE_TOL


@pytest.mark.skipif(
    not CATALOG.exists(),
    reason="JPL catalog entries not available (network blocked); see PROGRESS.md",
)
def test_matches_jpl_catalog() -> None:
    for e in json.loads(CATALOG.read_text()):
        X0 = np.array(e["state"], dtype=float)
        o = shooting.correct_symmetric(X0, e["mu"], free=(0, 4), targets=(3, 5))
        assert o.period == pytest.approx(e["period"], rel=1e-6)
        np.testing.assert_allclose(o.X0, X0, rtol=1e-6, atol=1e-9)
