"""Phase 11 tests: frames, screening conics, patched-conic baseline, CR3BP leg correction."""

import math

import numpy as np
import pytest

from cislunar import cr3bp, transfer
from cislunar.constants import EARTH_MOON, GM_EARTH_KM3_S2, R_EARTH_KM, make_system
from cislunar.integrators import dopri5

U = EARTH_MOON
MU = U.mu
R0 = R_EARTH_KM + 185.0
SEED = 11


def test_frame_round_trip_and_spec_formula() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(10):
        X = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)])
        tau = rng.uniform(-5, 5)
        r, v = transfer.rot_to_earth_inertial(X, MU, tau)
        np.testing.assert_allclose(transfer.earth_inertial_to_rot(r, v, MU, tau), X, atol=1e-14)
        # Spec 1.10 at tau = 0: v_in = (vx - y, vy + x + mu, vz).
        r0, v0 = transfer.rot_to_earth_inertial(X, MU, 0.0)
        np.testing.assert_allclose(v0, [X[3] - X[1], X[4] + X[0] + MU, X[5]], atol=1e-15)


def test_delta_v_is_frame_independent() -> None:
    rng = np.random.default_rng(SEED)
    X = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)])
    Y = X.copy()
    Y[3:6] += rng.normal(size=3) * 0.01
    dv_rot = np.linalg.norm(Y[3:6] - X[3:6])
    _, vx = transfer.rot_to_earth_inertial(X, MU, 0.7)
    _, vy = transfer.rot_to_earth_inertial(Y, MU, 0.7)
    assert np.linalg.norm(vy - vx) == pytest.approx(dv_rot, rel=1e-13)


def test_patched_conic_baseline_formulas() -> None:
    b = transfer.patched_conic_llo(U, 185.0, 100.0)
    # Independent vis-viva check of the TLI burn.
    a = 0.5 * (R0 + U.length_km)
    v_p = math.sqrt(GM_EARTH_KM3_S2 * (2 / R0 - 1 / a))
    assert b.dv_tli_km_s == pytest.approx(v_p - math.sqrt(GM_EARTH_KM3_S2 / R0), rel=1e-14)
    # Sanity: textbook ballpark for TLI (~3.1 km/s) and a ~5-day Hohmann coast.
    assert 3.0 < b.dv_tli_km_s < 3.2 and 4.5 < b.tof_days < 5.5


def test_screening_conic_hits_patch_point_in_two_body_limit() -> None:
    # With a vanishing Moon, the CR3BP is Earth two-body: the uncorrected conic must hit the point.
    u2 = make_system(GM_EARTH_KM3_S2, 1e-9 * GM_EARTH_KM3_S2, U.length_km)
    rp, vm = np.array([200000.0, 150000.0, 20000.0]), np.array([-0.3, 0.5, 0.05])
    sc = transfer.screen_tangent_conics(rp[None], vm[None], R0)
    r_dep, v_dep = transfer.conic_departure(rp, vm, R0, sc.nu[0], sc.e[0], sc.p[0])
    assert np.linalg.norm(r_dep) == pytest.approx(R0, rel=1e-12)
    assert abs(r_dep @ v_dep) < 1e-9 * np.linalg.norm(r_dep) * np.linalg.norm(v_dep)  # tangential
    tof = sc.tof_s[0] / u2.time_s
    X0 = transfer.earth_inertial_to_rot(r_dep / u2.length_km, v_dep / u2.velocity_km_s, u2.mu, -tof)
    Xf = dopri5(cr3bp.make_rhs(u2.mu), 0, tof, X0, rtol=1e-12, atol=1e-12).X_final
    r_f, _ = transfer.rot_to_earth_inertial(Xf, u2.mu, 0.0)
    assert np.linalg.norm(r_f * u2.length_km - rp) < 1e-2  # km


def test_leg_correction_and_jacobi_bookkeeping() -> None:
    # Patch point roughly 300,000 km out with a modest velocity: correct in the full CR3BP.
    Xp = np.array([0.75, -0.25, 0.05, -0.1, -0.35, 0.02])
    tr = transfer.correct_patch(Xp, R0, U)
    assert tr.miss < 1e-10
    t, X = transfer.propagate(tr.X_dep, tr.tau_dep, 0.0, MU, n=500)
    np.testing.assert_allclose(X[-1, :3], Xp[:3], atol=1e-9)
    C = cr3bp.jacobi_constant(X, MU)
    assert np.ptp(C) < 1e-9  # coast between burns conserves C
    assert 3.0 < tr.dv1_km_s < 3.3  # leaving LEO for ~lunar distance
    # The patch burn changes C to the patch state's value.
    assert cr3bp.jacobi_constant(Xp, MU) != pytest.approx(C[-1], abs=1e-6)
