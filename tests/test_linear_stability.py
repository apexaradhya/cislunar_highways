"""Phase 5 tests: linear stability at the collinear points."""

import numpy as np
import pytest

from cislunar import cr3bp, lagrange
from cislunar.constants import EARTH_MOON

MU = EARTH_MOON.mu
# Spec: numeric eigenvalues match analytic ones to 1e-10. LAPACK's eig on a well-scaled 6x6
# matrix gives ~10 eps * ||A|| ~ 1e-14 here (||A|| ~ 10), so 1e-10 has a big safety margin.
EIG_TOL = 1e-10


@pytest.mark.parametrize("which", [1, 2, 3])
def test_numeric_eigenvalues_match_analytic(which: int) -> None:
    m = lagrange.collinear_modes(which, MU)
    A = cr3bp.a_matrix(np.array([m.x, 0, 0, 0, 0, 0]), MU)
    eig = np.linalg.eigvals(A)
    expected = [m.lam, -m.lam, 1j * m.omega_p, -1j * m.omega_p, 1j * m.nu, -1j * m.nu]
    for e in expected:
        assert np.min(np.abs(eig - e)) < EIG_TOL


@pytest.mark.parametrize("which", [1, 2, 3])
def test_saddle_center_center_structure(which: int) -> None:
    m = lagrange.collinear_modes(which, MU)
    eig = np.linalg.eigvals(cr3bp.a_matrix(np.array([m.x, 0, 0, 0, 0, 0]), MU))
    real = eig[np.abs(eig.imag) < EIG_TOL]
    imag = eig[np.abs(eig.real) < EIG_TOL]
    assert len(real) == 2 and len(imag) == 4  # one real pair, two imaginary pairs
    assert real.real.max() > 0 > real.real.min()


@pytest.mark.parametrize("which", [1, 2, 3])
def test_hessian_at_collinear_point_matches_c(which: int) -> None:
    m = lagrange.collinear_modes(which, MU)
    H = cr3bp.hessian_pseudo_potential(np.array([m.x, 0, 0]), MU)
    np.testing.assert_allclose(np.diag(H), [1 + 2 * m.c, 1 - m.c, -m.c], rtol=1e-14)
    assert np.max(np.abs(H - np.diag(np.diag(H)))) == 0.0


@pytest.mark.parametrize("which", [1, 2])
def test_linear_lyapunov_state_solves_linear_equations(which: int) -> None:
    # xi = A cos(wt), eta = -kA sin(wt) must satisfy the planar linear equations exactly.
    m = lagrange.collinear_modes(which, MU)
    t = np.linspace(0, 2 * np.pi / m.omega_p, 50)
    A, w, k, c = 1e-3, m.omega_p, m.k, m.c
    xi, eta = A * np.cos(w * t), -k * A * np.sin(w * t)
    xi_d, eta_d = -A * w * np.sin(w * t), -k * A * w * np.cos(w * t)
    xi_dd, eta_dd = -w * w * xi, -w * w * eta
    np.testing.assert_allclose(xi_dd - 2 * eta_d - (1 + 2 * c) * xi, 0, atol=1e-15)
    np.testing.assert_allclose(eta_dd + 2 * xi_d - (1 - c) * eta, 0, atol=1e-15)
    X0 = lagrange.linear_lyapunov_state(which, MU, A)
    assert X0[4] == pytest.approx(eta_d[0], rel=1e-14)
