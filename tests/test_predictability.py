"""Phase 11c tests: batch propagation, unstable-component projection, error-growth forecast."""

import numpy as np
import pytest

from cislunar import cr3bp, predictability, stm
from cislunar.constants import EARTH_MOON
from cislunar.integrators import dopri5
from cislunar.shooting import correct_symmetric, full_period

MU = EARTH_MOON.mu
L = EARTH_MOON.length_km


@pytest.fixture(scope="module")
def halo():
    d = np.load("data/halo_L2_north.npz")
    return correct_symmetric(
        d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - 0.053)))], MU, free=(0, 4), targets=(3, 5)
    )


def test_batch_propagation_matches_individual() -> None:
    rng = np.random.default_rng(3)
    X0s = np.array([[0.8, 0.0, 0.05, 0.0, 0.2, 0.0]]) + rng.normal(scale=1e-3, size=(4, 6))
    t = np.linspace(0, 2.0, 5)
    S = predictability.propagate_batch(X0s, 2.0, MU, t, tol=1e-12)
    for k in range(4):
        ref = dopri5(cr3bp.make_rhs(MU), 0, 2.0, X0s[k], rtol=1e-12, atol=1e-12).X_final
        np.testing.assert_allclose(S[-1, k], ref, atol=1e-9)


def test_unstable_amplitude_recovers_component(halo) -> None:
    M = stm.split(full_period(halo).X_final)[1]
    w, V = np.linalg.eig(M)
    iu = int(np.argmax(np.abs(w)))
    v_u = V[:, iu].real / np.linalg.norm(V[:, iu].real[:3])
    others = [V[:, k].real for k in range(6) if k != iu and abs(w[k].imag) < 1e-9]
    dX = 3e-6 * v_u + sum(1e-5 * o for o in others)  # unstable part plus distractions
    amp, lam = predictability.unstable_amplitude(dX[None, :], M)
    assert lam == pytest.approx(abs(w[iu]), rel=1e-12)
    assert amp[0] == pytest.approx(3e-6, rel=1e-6)


def test_error_growth_forecast(halo) -> None:
    # Forecast from linear theory vs a 60-member ensemble: thresholds 10 km .. 10,000 km.
    M = stm.split(full_period(halo).X_final)[1]
    fan = predictability.butterfly_fan(
        halo.X0, MU, EARTH_MOON, 60, 0.001, 0.01, 2 * halo.period, n_out=400, seed=1
    )
    cu, lam = predictability.unstable_amplitude(fan.dX0, M)
    s0 = np.sqrt(np.mean(cu**2))
    spread = fan.spread() * L
    for d_km in (10.0, 100.0, 1000.0, 10000.0):
        t_meas = fan.t[np.argmax(spread > d_km)]
        t_pred = predictability.predicted_time_to_reach(d_km / L, s0, halo.period, lam)
        # Within ~2 days (0.46 TU): the growth is exponential only on average over a period.
        assert abs(t_meas - t_pred) * EARTH_MOON.time_days < 2.0
