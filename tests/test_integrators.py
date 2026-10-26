"""Phase 2 tests: RK4 and DOPRI5 accuracy, events, scipy cross-check, Jacobi drift."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cislunar import cr3bp
from cislunar.constants import EARTH_MOON
from cislunar.integrators import Event, dopri5, rk4

MU = EARTH_MOON.mu
# Sample trajectory used throughout Phase 2: starts between Earth and Moon, loops in the
# Earth region without close lunar passes (keeps the comparison well conditioned).
X_SAMPLE = np.array([0.5, 0.0, 0.05, 0.0, 0.9, 0.05])

# Measured convergence slopes must be within this of the theoretical order.
ORDER_SLACK = 0.25
# Spec Phase 2: own DOPRI5 vs scipy DOP853 (both at 1e-13) over 10 TU.
SCIPY_AGREEMENT_TOL = 1e-8
# Spec Phase 2: Jacobi drift over 20 TU at tight tolerance.
JACOBI_DRIFT_TOL = 1e-9
# Spec Phase 2: event crossings located to ~1e-12.
EVENT_TOL = 1e-12


def oscillator(t: float, X: np.ndarray) -> np.ndarray:
    return np.array([X[1], -X[0]])


def growth(t: float, X: np.ndarray) -> np.ndarray:
    """y' = y cos t, exact y = exp(sin t). Non-periodic sampling avoids the error
    cancellation RK methods show on the harmonic oscillator over whole periods."""
    return X * math.cos(t)


T_GROWTH = 3.0
Y_GROWTH = math.exp(math.sin(T_GROWTH))


def eom(t: float, X: np.ndarray) -> np.ndarray:
    return cr3bp.equations_of_motion(t, X, MU)


def slope(hs, errs) -> float:
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def test_rk4_is_fourth_order() -> None:
    ns = np.array([50, 100, 200, 400])
    errs = [abs(rk4(growth, 0.0, T_GROWTH, np.array([1.0]), n)[1][-1, 0] - Y_GROWTH) for n in ns]
    assert slope(T_GROWTH / ns, errs) == pytest.approx(4.0, abs=ORDER_SLACK)


def test_dopri5_is_fifth_order_with_fixed_steps() -> None:
    # Huge tolerances -> no step is ever rejected and h stays pinned at max_step.
    hs = T_GROWTH / np.array([10, 20, 40, 80])
    errs = []
    for h in hs:
        sol = dopri5(growth, 0.0, T_GROWTH, np.array([1.0]), rtol=1e10, atol=1e10, max_step=h, h0=h)
        errs.append(abs(sol.X_final[0] - Y_GROWTH))
    assert slope(hs, errs) == pytest.approx(5.0, abs=ORDER_SLACK)


def test_dopri5_error_tracks_tolerance() -> None:
    tols = [1e-6, 1e-8, 1e-10, 1e-12]
    errs = []
    for tol in tols:
        sol = dopri5(oscillator, 0.0, 10.0, np.array([1.0, 0.0]), rtol=tol, atol=tol)
        errs.append(np.max(np.abs(sol.X_final - [math.cos(10.0), -math.sin(10.0)])))
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:], strict=False))
    # Global error grows roughly linearly with number of steps; 100*tol bounds it here.
    assert all(e < 100 * tol for e, tol in zip(errs, tols, strict=True))


def test_dense_output_matches_exact_solution() -> None:
    t_eval = np.linspace(0, 10, 101)
    sol = dopri5(oscillator, 0.0, 10.0, np.array([1.0, 0.0]), rtol=1e-12, atol=1e-12, t_eval=t_eval)
    np.testing.assert_allclose(sol.t, t_eval)
    # 4th-order interpolant is less accurate than the steps themselves; 1e-9 is loose but real.
    np.testing.assert_allclose(sol.X[:, 0], np.cos(t_eval), atol=1e-9)


def test_dopri5_matches_scipy_dop853_on_cr3bp() -> None:
    ours = dopri5(eom, 0.0, 10.0, X_SAMPLE, rtol=1e-13, atol=1e-13)
    ref = solve_ivp(eom, (0.0, 10.0), X_SAMPLE, method="DOP853", rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(ours.X_final - ref.y[:, -1])) < SCIPY_AGREEMENT_TOL


def test_jacobi_drift_over_20_time_units() -> None:
    sol = dopri5(eom, 0.0, 20.0, X_SAMPLE, rtol=1e-13, atol=1e-13)
    C = cr3bp.jacobi_constant(sol.X, MU)
    assert np.max(np.abs(C - C[0])) < JACOBI_DRIFT_TOL


def test_backward_integration_returns_to_start() -> None:
    fwd = dopri5(eom, 0.0, 5.0, X_SAMPLE, rtol=1e-13, atol=1e-13)
    back = dopri5(eom, 5.0, 0.0, fwd.X_final, rtol=1e-13, atol=1e-13)
    assert np.max(np.abs(back.X_final - X_SAMPLE)) < 1e-9


def test_event_location_on_analytic_problem() -> None:
    # x = cos t crosses zero (falling) at t = pi/2.
    ev = Event(lambda t, X: X[0], direction=-1, terminal=True)
    sol = dopri5(oscillator, 0.0, 10.0, np.array([1.0, 0.0]), events=[ev])
    assert sol.status == "event"
    assert sol.events[0].t == pytest.approx(math.pi / 2, abs=EVENT_TOL)


def test_y_crossing_located_to_1e12_and_matches_scipy() -> None:
    ev = Event(lambda t, X: X[1], direction=-1, terminal=True)
    sol = dopri5(eom, 0.0, 20.0, X_SAMPLE, rtol=1e-13, atol=1e-13, events=[ev])
    hit = sol.events[0]
    assert abs(hit.X[1]) < EVENT_TOL

    def g(t, X):
        return X[1]

    g.terminal, g.direction = True, -1
    ref = solve_ivp(eom, (0.0, 20.0), X_SAMPLE, method="DOP853", rtol=1e-13, atol=1e-13, events=g)
    assert hit.t == pytest.approx(ref.t_events[0][0], abs=1e-10)


def test_event_count_stops_on_nth_crossing() -> None:
    ev = Event(lambda t, X: X[0], direction=0, terminal=True, count=3)
    sol = dopri5(oscillator, 0.0, 20.0, np.array([1.0, 0.0]), events=[ev])
    # Zeros of cos t at pi/2, 3pi/2, 5pi/2.
    assert sol.t_final == pytest.approx(5 * math.pi / 2, abs=EVENT_TOL)
