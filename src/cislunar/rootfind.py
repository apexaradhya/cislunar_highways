"""Scalar root finding: Brent's method (written from scratch).

Brent, R. P. (1973), *Algorithms for Minimization without Derivatives*, ch. 4.
Combines bisection (guaranteed), the secant method and inverse quadratic
interpolation (fast). The bracket [a, b] always contains a sign change, so the
method cannot diverge. Follows the classic `zeroin` structure (Forsythe,
Malcolm & Moler 1977, sec. 7.2).
"""

from __future__ import annotations

from collections.abc import Callable

MACHINE_EPS = 2.220446049250313e-16  # float64 unit roundoff * 2 (numpy.finfo(float).eps)


class BracketError(ValueError):
    """Raised when f(a) and f(b) do not have opposite signs."""


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-15,
    max_iter: int = 200,
) -> float:
    """Return x in [a, b] with f(x) ~ 0, given f(a) * f(b) <= 0."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise BracketError(f"f(a)={fa:.3e} and f(b)={fb:.3e} have the same sign")

    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:  # make sure the root lies between b and c
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):  # b is always the best estimate so far
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2.0 * MACHINE_EPS * abs(b) + 0.5 * xtol
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0.0:
            return b
        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:  # secant step
                p = 2.0 * m * s
                q = 1.0 - s
            else:  # inverse quadratic interpolation
                q_, r = fa / fc, fb / fc
                p = s * (2.0 * m * q_ * (q_ - r) - (b - a) * (r - 1.0))
                q = (q_ - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q  # accept interpolation
            else:
                d = e = m  # fall back to bisection
        else:
            d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else (tol if m > 0 else -tol)
        fb = f(b)
    raise RuntimeError("brent: maximum iterations exceeded")
