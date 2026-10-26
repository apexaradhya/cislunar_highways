"""ODE integrators written from scratch: RK4, Dormand-Prince 5(4), event location.

RK4
    Classic fourth-order Runge-Kutta with a fixed step (global error O(h^4)).

DOPRI5
    Dormand, J. R. & Prince, P. J. (1980), "A family of embedded Runge-Kutta
    formulae", J. Comput. Appl. Math. 6(1), 19-26. A 7-stage, 5th-order method
    with an embedded 4th-order solution used to estimate the local error. FSAL
    ("first same as last"): the last stage equals the first stage of the next step.
    Step-size control and the initial-step heuristic follow Hairer, Norsett &
    Wanner, *Solving ODEs I* (2nd ed., 1993), sec. II.4, including the PI
    ("Lund-stabilized") controller of their code DOPRI5.
    Dense output: the 4th-order continuous extension of Shampine (1986), using
    the coefficients d1..d7 of Hairer's dopri5.f.

Event location
    An event is a scalar g(t, X). A sign change across an accepted step brackets
    a root. It is refined with Brent's method on the dense output, then
    polished with secant iterations on *re-integrated* states (so the result
    does not inherit the interpolant's error). Crossings like y = 0 end up
    located to ~1e-13.

Time can run backward (tf < t0); all logic uses the signed step h.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cislunar.rootfind import brent

FloatArray = NDArray[np.float64]
RHS = Callable[[float, FloatArray], FloatArray]

# ---------------------------------------------------------------------------
# Dormand-Prince 5(4) Butcher tableau (Dormand & Prince 1980, Table 2).
# ---------------------------------------------------------------------------
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_E = _B5 - _B4
# Dense-output coefficients (Hairer's dopri5.f, after Shampine 1986).
_D = np.array(
    [
        -12715105075 / 11282082432,
        0.0,
        87487479700 / 32700410799,
        -10690763975 / 1880347072,
        701980252875 / 199316789632,
        -1453857185 / 822651844,
        69997945 / 29380423,
    ]
)

# Step-size controller constants (Hairer et al. 1993, DOPRI5 defaults).
_SAFETY = 0.9
_FAC_MIN = 0.2  # step may shrink by at most 5x
_FAC_MAX = 10.0  # step may grow by at most 10x
_PI_BETA = 0.04  # Lund stabilization exponent
_ERR_EXPONENT = 0.2 - 0.75 * _PI_BETA  # 1/(order+1) corrected for the PI term


def rk4(
    f: RHS, t0: float, tf: float, X0: FloatArray, n_steps: int
) -> tuple[FloatArray, FloatArray]:
    """Fixed-step classical RK4. Returns (t[n_steps+1], X[n_steps+1, n])."""
    h = (tf - t0) / n_steps
    t = t0 + h * np.arange(n_steps + 1)
    X = np.empty((n_steps + 1, len(X0)))
    X[0] = X0
    y = np.asarray(X0, dtype=float)
    for i in range(n_steps):
        ti = t[i]
        k1 = f(ti, y)
        k2 = f(ti + h / 2, y + h / 2 * k1)
        k3 = f(ti + h / 2, y + h / 2 * k2)
        k4 = f(ti + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[i + 1] = y
    return t, X


@dataclass
class Event:
    """Scalar event function g(t, X) = 0.

    direction: +1 only rising crossings, -1 only falling, 0 both.
    terminal: stop integrating after `count` matching crossings.
    """

    fn: Callable[[float, FloatArray], float]
    direction: int = 0
    terminal: bool = False
    count: int = 1


@dataclass
class EventHit:
    index: int  # which event in the list
    t: float
    X: FloatArray


@dataclass
class Solution:
    t: FloatArray
    X: FloatArray
    events: list[EventHit] = field(default_factory=list)
    status: str = "completed"  # "completed" | "event" | "max_steps"
    n_steps: int = 0
    n_rejected: int = 0
    n_fev: int = 0

    @property
    def t_final(self) -> float:
        return float(self.t[-1])

    @property
    def X_final(self) -> FloatArray:
        return self.X[-1]


@dataclass
class _Step:
    """One accepted step, with what dense output needs."""

    t: float
    h: float
    y: FloatArray
    y_new: FloatArray
    rcont: tuple[FloatArray, ...]

    def __call__(self, t: float) -> FloatArray:
        theta = (t - self.t) / self.h
        th1 = 1.0 - theta
        r1, r2, r3, r4, r5 = self.rcont
        return r1 + theta * (r2 + th1 * (r3 + theta * (r4 + th1 * r5)))


def _error_norm(
    err: FloatArray, y: FloatArray, y_new: FloatArray, rtol: float, atol: float
) -> float:
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    return math.sqrt(float(np.mean((err / scale) ** 2)))


def _initial_step(
    f: RHS, t0: float, y0: FloatArray, f0: FloatArray, direction: float, rtol: float, atol: float
) -> float:
    """Initial step heuristic (Hairer et al. 1993, sec. II.4, 'Starting step size')."""
    scale = atol + rtol * np.abs(y0)
    d0 = math.sqrt(float(np.mean((y0 / scale) ** 2)))
    d1 = math.sqrt(float(np.mean((f0 / scale) ** 2)))
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    f1 = f(t0 + direction * h0, y0 + direction * h0 * f0)
    d2 = math.sqrt(float(np.mean(((f1 - f0) / scale) ** 2))) / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / 5)
    return min(100 * h0, h1)


def dopri5(
    f: RHS,
    t0: float,
    tf: float,
    X0: FloatArray,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    events: Sequence[Event] = (),
    t_eval: FloatArray | None = None,
    max_steps: int = 1_000_000,
    max_step: float = math.inf,
    h0: float | None = None,
    polish_events: bool = True,
) -> Solution:
    """Adaptive Dormand-Prince 5(4) integration of dX/dt = f(t, X) from t0 to tf.

    Returns every accepted step, or the states at `t_eval` (via dense output) if
    given. Event crossings are located as described in the module docstring.
    """
    y = np.array(X0, dtype=float)
    direction = 1.0 if tf >= t0 else -1.0
    t = t0
    k1 = f(t, y)
    n_fev = 1
    if h0 is None:
        h_abs = _initial_step(f, t0, y, k1, direction, rtol, atol)
        n_fev += 1
    else:
        h_abs = abs(h0)
    h_abs = min(h_abs, max_step, abs(tf - t0)) if tf != t0 else 0.0

    ts: list[float] = [t]
    ys: list[FloatArray] = [y.copy()]
    hits: list[EventHit] = []
    counts = [0] * len(events)
    g_prev = [ev.fn(t, y) for ev in events]
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        out_t: list[float] = []
        out_y: list[FloatArray] = []
        eval_i = 0
        while eval_i < len(t_eval) and t_eval[eval_i] == t0:
            out_t.append(t0)
            out_y.append(y.copy())
            eval_i += 1

    err_old = 1e-4  # PI controller memory (Hairer's FACOLD initial value)
    n_steps = n_rej = 0
    status = "completed"
    rejected_last = False
    K = np.empty((7, len(y)))
    K[0] = k1

    while direction * (tf - t) > 0.0:
        if n_steps >= max_steps:
            status = "max_steps"
            break
        h = direction * min(h_abs, abs(tf - t))
        # --- stages ---
        for s in range(1, 7):
            ys_ = y + h * (np.asarray(_A[s]) @ K[:s])
            K[s] = f(t + _C[s] * h, ys_)
        n_fev += 6
        y_new = ys_  # stage 7 argument is the 5th-order solution (FSAL)
        err = _error_norm(h * (_E @ K), y, y_new, rtol, atol)

        if err <= 1.0:
            # --- accept ---
            fac11 = err**_ERR_EXPONENT if err > 0 else 0.0
            fac = fac11 / err_old**_PI_BETA
            fac = min(1.0 / _FAC_MIN, max(1.0 / _FAC_MAX, fac / _SAFETY))
            h_new_abs = abs(h) / fac
            if rejected_last:
                h_new_abs = min(h_new_abs, abs(h))
            err_old = max(err, 1e-4)
            rejected_last = False
            n_steps += 1

            ydiff = y_new - y
            bspl = h * K[0] - ydiff
            rcont = (y, ydiff, bspl, ydiff - h * K[6] - bspl, h * (_D @ K))
            step = _Step(t, h, y, y_new, rcont)
            t_new = t + h

            # --- events ---
            stop_at: EventHit | None = None
            if events:
                stop_at = _locate_events(
                    f, events, step, g_prev, counts, hits, rtol, atol, polish_events
                )
            if t_eval is not None:
                t_lim = stop_at.t if stop_at is not None else t_new
                while eval_i < len(t_eval) and direction * (t_lim - t_eval[eval_i]) >= 0.0:
                    out_t.append(float(t_eval[eval_i]))
                    out_y.append(step(t_eval[eval_i]))
                    eval_i += 1
            if stop_at is not None:
                ts.append(stop_at.t)
                ys.append(stop_at.X.copy())
                status = "event"
                break

            t, y = t_new, y_new
            K[0] = K[6]
            ts.append(t)
            ys.append(y.copy())
            h_abs = min(h_new_abs, max_step)
        else:
            # --- reject: shrink and retry ---
            h_abs = abs(h) / min(1.0 / _FAC_MIN, err**_ERR_EXPONENT / _SAFETY)
            rejected_last = True
            n_rej += 1
            if h_abs < 1e-14 * max(1.0, abs(t)):
                raise RuntimeError(f"dopri5: step size underflow at t={t}")

    if t_eval is not None:
        T, X = np.array(out_t), np.array(out_y).reshape(len(out_t), len(y))
    else:
        T, X = np.array(ts), np.array(ys)
    return Solution(T, X, hits, status, n_steps, n_rej, n_fev)


def _locate_events(
    f: RHS,
    events: Sequence[Event],
    step: _Step,
    g_prev: list[float],
    counts: list[int],
    hits: list[EventHit],
    rtol: float,
    atol: float,
    polish: bool,
) -> EventHit | None:
    """Find event roots inside an accepted step; return the terminal hit, if any."""
    t_new = step.t + step.h
    found: list[EventHit] = []
    for i, ev in enumerate(events):
        g0 = g_prev[i]
        g1 = ev.fn(t_new, step.y_new)
        g_prev[i] = g1
        if g0 == 0.0 or g0 * g1 > 0.0:
            continue
        rising = g1 > g0
        if ev.direction == 1 and not rising or ev.direction == -1 and rising:
            continue
        if g1 == 0.0:
            t_root = t_new
        else:
            t_root = brent(lambda s, fn=ev.fn: fn(s, step(s)), step.t, t_new, xtol=1e-15)
        X_root = step(t_root)
        if polish:
            t_root, X_root = _polish_root(f, ev.fn, step, t_root, rtol, atol)
        found.append(EventHit(i, t_root, X_root))
    found.sort(key=lambda hit: (hit.t - step.t) / step.h)
    for hit in found:
        hits.append(hit)
        counts[hit.index] += 1
        ev = events[hit.index]
        if ev.terminal and counts[hit.index] >= ev.count:
            return hit
    return None


def _polish_root(
    f: RHS,
    g: Callable[[float, FloatArray], float],
    step: _Step,
    t_guess: float,
    rtol: float,
    atol: float,
    iters: int = 3,
) -> tuple[float, FloatArray]:
    """Secant polish of an event root, integrating from the step start each time.

    The dense output is only 4th order; re-integrating the short arc from
    step.t (an accepted, accurate state) removes that interpolation error.
    """

    def state_at(s: float) -> FloatArray:
        if s == step.t:
            return step.y
        sub = dopri5(f, step.t, s, step.y, rtol=rtol, atol=atol, polish_events=False)
        return sub.X_final

    t_a = t_guess
    X_a = state_at(t_a)
    g_a = g(t_a, X_a)
    dt = 1e-6 * abs(step.h) if step.h != 0 else 1e-9
    t_b = t_a + dt
    X_b = state_at(t_b)
    g_b = g(t_b, X_b)
    for _ in range(iters):
        if g_b == g_a:
            break
        t_c = t_b - g_b * (t_b - t_a) / (g_b - g_a)
        t_a, X_a, g_a = t_b, X_b, g_b
        t_b = t_c
        X_b = state_at(t_b)
        g_b = g(t_b, X_b)
        if abs(g_b) < 1e-15:
            break
    if abs(g_b) <= abs(g_a):
        return t_b, X_b
    return t_a, X_a
