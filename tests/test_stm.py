"""Phase 6 tests: STM against finite differences, symplecticity, det = 1, Jacobi."""

import numpy as np
import pytest

from cislunar import cr3bp, stm
from cislunar.constants import EARTH_MOON
from cislunar.integrators import dopri5

MU = EARTH_MOON.mu
SEED = 7
X_SAMPLE = np.array([0.5, 0.0, 0.05, 0.0, 0.9, 0.05])
X_NEAR_L1 = np.array([0.82, 0.0, 0.02, 0.0, 0.15, 0.0])  # strongly unstable region
# Central differences with delta = 1e-7: truncation ~ delta^2 |X'''| ~ 1e-14, but each
# propagation carries ~1e-13 * ||Phi|| integration error, divided by 2*delta -> ~1e-6 relative.
FD_REL_TOL = 1e-6
# Phi^T Jv Phi - Jv is O(||Phi||^2 * integration error); normalized by ||Phi||^2.
SYMPLECTIC_TOL = 1e-10
DET_TOL = 1e-9  # spec value
JACOBI_TOL = 1e-11


def test_fast_rhs_matches_reference_eom() -> None:
    rng = np.random.default_rng(SEED)
    f = cr3bp.make_rhs(MU)
    for _ in range(20):
        X = np.concatenate([rng.uniform(-1.2, 1.2, 3), rng.uniform(-1, 1, 3)])
        np.testing.assert_allclose(
            f(0.0, X), cr3bp.equations_of_motion(0.0, X, MU), rtol=1e-13, atol=1e-13
        )


def test_fast_stm_rhs_matches_a_matrix() -> None:
    rng = np.random.default_rng(SEED)
    g = stm.make_rhs_stm(MU)
    for _ in range(10):
        X = np.concatenate([rng.uniform(-1.2, 1.2, 3), rng.uniform(-1, 1, 3)])
        P = rng.normal(size=(6, 6))
        out = g(0.0, stm.initial_augmented(X, P))
        np.testing.assert_allclose(
            out[6:].reshape(6, 6), cr3bp.a_matrix(X, MU) @ P, rtol=1e-12, atol=1e-12
        )


def test_a_matrix_is_infinitesimally_symplectic() -> None:
    # Phi^T Jv Phi = Jv for all t  <=>  A^T Jv + Jv A = 0 at every state.
    rng = np.random.default_rng(SEED)
    for _ in range(10):
        X = np.concatenate([rng.uniform(-1.2, 1.2, 3), rng.uniform(-1, 1, 3)])
        A = cr3bp.a_matrix(X, MU)
        np.testing.assert_allclose(A.T @ stm.J_VELOCITY + stm.J_VELOCITY @ A, 0, atol=1e-12)
        assert abs(np.trace(A)) < 1e-15


@pytest.mark.parametrize(("X0", "T"), [(X_SAMPLE, 3.0), (X_NEAR_L1, 2.0)])
def test_stm_matches_finite_differences(X0, T) -> None:
    sol = stm.propagate_stm(X0, 0.0, T, MU, rtol=1e-13, atol=1e-13)
    _, Phi = stm.split(sol.X_final)
    Phi_fd = stm.stm_finite_difference(X0, 0.0, T, MU)
    rel = np.max(np.abs(Phi - Phi_fd)) / np.max(np.abs(Phi))
    assert rel < FD_REL_TOL


@pytest.mark.parametrize(("X0", "T"), [(X_SAMPLE, 10.0), (X_NEAR_L1, 3.0)])
def test_symplectic_and_unit_determinant(X0, T) -> None:
    sol = stm.propagate_stm(X0, 0.0, T, MU, rtol=1e-13, atol=1e-13)
    _, Phi = stm.split(sol.X_final)
    scale = np.max(np.abs(Phi)) ** 2
    assert np.max(np.abs(Phi.T @ stm.J_VELOCITY @ Phi - stm.J_VELOCITY)) / scale < SYMPLECTIC_TOL
    assert abs(np.linalg.det(Phi) - 1.0) < DET_TOL


def test_jacobi_conserved_in_augmented_propagation() -> None:
    sol = stm.propagate_stm(X_SAMPLE, 0.0, 10.0, MU, rtol=1e-13, atol=1e-13)
    X, _ = stm.split(sol.X)
    C = cr3bp.jacobi_constant(X, MU)
    assert np.max(np.abs(C - C[0])) < JACOBI_TOL


def test_state_part_unaffected_by_stm() -> None:
    a = stm.propagate_stm(X_SAMPLE, 0.0, 5.0, MU, rtol=1e-13, atol=1e-13).X_final[:6]
    b = dopri5(cr3bp.make_rhs(MU), 0.0, 5.0, X_SAMPLE, rtol=1e-13, atol=1e-13).X_final
    assert np.max(np.abs(a - b)) < 1e-10
