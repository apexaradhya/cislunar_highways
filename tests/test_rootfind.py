import math

import pytest
from scipy.optimize import brentq

from cislunar.rootfind import BracketError, brent

ROOT_TOL = 1e-14  # Brent converges to ~2 eps|x| + xtol/2


@pytest.mark.parametrize(
    ("f", "a", "b"),
    [
        (lambda x: math.cos(x) - x, 0.0, 1.0),
        (lambda x: x**3 - 2 * x - 5, 2.0, 3.0),  # Wallis' classic test equation
        (lambda x: math.exp(x) - 10.0, 0.0, 5.0),
    ],
)
def test_brent_matches_scipy(f, a, b) -> None:
    assert brent(f, a, b) == pytest.approx(brentq(f, a, b, xtol=1e-15), abs=ROOT_TOL)


def test_brent_rejects_bad_bracket() -> None:
    with pytest.raises(BracketError):
        brent(lambda x: x**2 + 1, -1.0, 1.0)


def test_brent_triple_root() -> None:
    # (x-0.3)^3: f is flat near the root, so |f| < eps only pins x to ~eps^(1/3) ~ 6e-6.
    # scipy's brentq exceeds its default 100 iterations here; ours must still converge.
    assert brent(lambda x: (x - 0.3) ** 3, 0.0, 1.0) == pytest.approx(0.3, abs=1e-5)
