"""Zero-velocity curves and Hill regions (PROJECT_SPEC.md Phase 4).

Since C = 2 Omega - v^2 and v^2 >= 0, a spacecraft with Jacobi constant C can
only be where 2 Omega(x, y, z) >= C. Where 2 Omega < C it would need an
imaginary speed: that is the *forbidden region*. Its boundary 2 Omega = C is
the zero-velocity curve (Szebehely 1967, ch. 4; Koon et al. 2011, sec. 2.6).

As C decreases (more energy), "necks" in the allowed region open at the
collinear points, in the order L1, L2, L3. L1, L2 and L3 are saddles of
2 Omega, so the neck opens exactly when C drops below C(Li):
    C > C1        Earth and Moon realms are separate
    C1 > C > C2   Earth <-> Moon possible through the L1 neck
    C2 > C > C3   the Moon realm also opens to the exterior through L2
    C3 > C > 3    the L3 neck opens; the forbidden region splits in two around L4/L5
    C < 3         no planar forbidden region left

Neck detection here is purely geometric (connected components of the allowed
region on a grid, labelled with scipy.ndimage.label), so it is an independent
check of the Lagrange-point Jacobi constants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from cislunar import cr3bp

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlanarGrid:
    x: FloatArray
    y: FloatArray
    two_omega: FloatArray  # 2*Omega(x, y, 0), shape (len(y), len(x))


def planar_grid(mu: float, half_width: float = 1.5, n: int = 1501) -> PlanarGrid:
    """2*Omega on an n x n grid over [-w, w]^2. Odd n puts a row exactly on y = 0."""
    x = np.linspace(-half_width, half_width, n)
    y = np.linspace(-half_width, half_width, n)
    XX, YY = np.meshgrid(x, y)
    pos = np.stack([XX, YY, np.zeros_like(XX)], axis=-1)
    with np.errstate(divide="ignore"):
        two_omega = 2.0 * cr3bp.pseudo_potential(pos, mu)
    return PlanarGrid(x, y, np.nan_to_num(two_omega, nan=np.inf, posinf=np.inf))


def forbidden_mask(grid: PlanarGrid, C: float) -> NDArray[np.bool_]:
    """True where 2 Omega < C (motion impossible at this Jacobi constant)."""
    return grid.two_omega < C


def _cell(grid: PlanarGrid, x: float, y: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(grid.y - y))), int(np.argmin(np.abs(grid.x - x)))


@dataclass(frozen=True)
class Connectivity:
    earth_moon: bool  # L1 neck open
    moon_exterior: bool  # L2 neck open (Moon realm reaches the outer region)
    forbidden_components: int  # 1 before the L3 neck opens, 2 after, 0 below C = 3


def connectivity(grid: PlanarGrid, C: float, mu: float) -> Connectivity:
    """Which realms are connected through the allowed region at Jacobi constant C."""
    allowed = ~forbidden_mask(grid, C)
    labels, _ = ndimage.label(allowed)  # 4-connectivity: no sneaking diagonally through a wall
    earth = labels[_cell(grid, -mu + 0.05, 0.0)]
    moon = labels[_cell(grid, 1.0 - mu + 0.02, 0.0)]
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    exterior = set(border[border > 0].tolist())
    _, n_forbidden = ndimage.label(~allowed)
    return Connectivity(
        earth_moon=bool(earth == moon and earth > 0),
        moon_exterior=bool(moon in exterior),
        forbidden_components=int(n_forbidden),
    )


def _bisect_threshold(pred: Callable[[float], bool], c_hi: float, c_lo: float, tol: float = 1e-9) -> float:
    """Largest C at which pred becomes True, given pred(c_hi) False and pred(c_lo) True."""
    assert not pred(c_hi) and pred(c_lo)
    while c_hi - c_lo > tol:
        mid = 0.5 * (c_hi + c_lo)
        if pred(mid):
            c_lo = mid
        else:
            c_hi = mid
    return 0.5 * (c_hi + c_lo)


def neck_opening_values(grid: PlanarGrid, mu: float) -> dict[str, float]:
    """Find, purely from geometry, the C values at which the L1, L2 and L3 necks open."""
    return {
        "L1": _bisect_threshold(lambda C: connectivity(grid, C, mu).earth_moon, 3.5, 3.1),
        "L2": _bisect_threshold(lambda C: connectivity(grid, C, mu).moon_exterior, 3.5, 3.1),
        "L3": _bisect_threshold(lambda C: connectivity(grid, C, mu).forbidden_components >= 2, 3.1, 3.001),
    }
