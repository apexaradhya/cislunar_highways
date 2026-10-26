"""Poincare sections of manifold tubes (PROJECT_SPEC.md Phase 10).

A Poincare section is a surface Sigma, e.g. the plane x = 1 - mu through the Moon.
Recording where trajectories pierce Sigma reduces a flow to a map: each manifold
tube, cut by Sigma, leaves a closed curve (a tube's cross-section). Points inside
the curve of a stable tube are on trajectories that pass *through* the tube's
interior (transit orbits); points on the curve are exactly asymptotic to the orbit
(Koon, Lo, Marsden & Ross 2000, Chaos 10, 427; Koon et al. 2011, ch. 3).

Time-reversal symmetry: the CR3BP is invariant under
    R: (x, y, z, vx, vy, vz, t) -> (x, -y, z, -vx, vy, -vz, -t),
which maps a symmetric orbit's unstable manifold onto its stable manifold. On a
section x = const this means the stable cut is the mirror image (y, z, vy, vz) ->
(-y, z, vy, -vz) of the unstable cut: a strong, exact numerical check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cislunar import cr3bp, manifolds
from cislunar.integrators import Event, dopri5
from cislunar.shooting import SymmetricOrbit

FloatArray = NDArray[np.float64]


def plane_section(axis: int, value: float, direction: int = 0, extra=None) -> Event:
    """Section X[axis] = value. `extra(X) -> bool` filters crossings (e.g. y < 0)."""
    if extra is None:
        return Event(lambda t, X: X[axis] - value, direction=direction)

    def g(t: float, X: FloatArray) -> float:
        return X[axis] - value

    ev = Event(g, direction=direction)
    ev.accept = extra  # type: ignore[attr-defined]
    return ev


@dataclass
class SectionCut:
    states: FloatArray  # (n, 6) crossing states, ordered by orbit phase
    times: FloatArray  # (n,) flight time to the crossing (negative for stable tubes)
    phases: FloatArray  # (n,) starting phase t_i / T on the orbit


def tube_cut(
    orbit: SymmetricOrbit,
    kind: str,
    sign: float,
    section: Event,
    n_traj: int = 200,
    max_duration: float = 10.0,
    crossing: int = 1,
    eps: float = manifolds.DEFAULT_EPS,
    r_earth: float = 0.0,
    r_moon: float = 0.0,
    tol: float = 1e-12,
) -> SectionCut:
    """Intersect a manifold tube with a section: the `crossing`-th accepted crossing of each strand.

    Strands that hit a primary, escape, or never cross within max_duration are dropped.
    """
    samples = manifolds.sample_orbit(orbit, n_traj)
    starts = manifolds.manifold_states(samples, kind, sign, eps)
    accept = getattr(section, "accept", None)
    stops = manifolds.make_stop_events(orbit.mu, r_earth, r_moon)
    tf = max_duration if kind == "u" else -max_duration
    f = cr3bp.make_rhs(orbit.mu)
    states, times, phases = [], [], []
    for ti, X0 in zip(samples.t, starts, strict=True):
        sol = dopri5(f, 0.0, tf, X0, rtol=tol, atol=tol, events=[*stops, section])
        hits = [h for h in sol.events if h.index == len(stops) and (accept is None or accept(h.X))]
        if len(hits) >= crossing:
            h = hits[crossing - 1]
            # Discard if the strand was stopped (impact/escape) before this crossing.
            states.append(h.X)
            times.append(h.t)
            phases.append(ti / orbit.period)
    return SectionCut(np.array(states), np.array(times), np.array(phases))


def closed_curve_gaps(points: FloatArray) -> tuple[float, float]:
    """For points ordered along a curve: (max gap between neighbours, closing gap last->first).

    For a tube cut that is a genuine closed curve, the closing gap is comparable to the
    typical neighbour gap instead of being as large as the curve itself.
    """
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.max(d)), float(np.linalg.norm(points[-1] - points[0]))


def hausdorff(a: FloatArray, b: FloatArray) -> float:
    """Symmetric Hausdorff distance between two point sets."""
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(D.min(axis=1).max(), D.min(axis=0).max()))


def split_at_gaps(cut: SectionCut, n_traj: int, cols: list[int]) -> list[FloatArray]:
    """Pieces of the cut curve between missing strands (strands lost to impacts etc.).

    Consecutive phases differ by 1/n_traj; a larger jump means strands are missing there,
    so the curve must not be drawn (or intersected) across it. The cut is periodic in
    phase, so the last piece joins the first when nothing is missing at the wrap-around.
    """
    if len(cut.phases) == 0:
        return []
    P = cut.states[:, cols]
    step = 1.0 / n_traj
    breaks = np.where(np.diff(cut.phases) > 1.5 * step)[0]
    pieces = np.split(P, breaks + 1)
    wraps = (cut.phases[0] + 1.0 - cut.phases[-1]) <= 1.5 * step
    if wraps and len(pieces) > 1:
        pieces = [np.vstack([pieces[-1], pieces[0]]), *pieces[1:-1]]
    elif wraps:
        pieces = [np.vstack([P, P[:1]])]
    return [p for p in pieces if len(p) >= 2]


def polyline_intersections(a: list[FloatArray], b: list[FloatArray]) -> FloatArray:
    """All intersection points between two sets of 2D polylines (segment-by-segment test)."""
    out = []
    for pa in a:
        for pb in b:
            p, r = pa[:-1], np.diff(pa, axis=0)  # segments p + s r
            q, s = pb[:-1], np.diff(pb, axis=0)  # segments q + u s
            rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
            qp = q[None, :, :] - p[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
                uu = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
            hit = (np.abs(rxs) > 0) & (tt >= 0) & (tt <= 1) & (uu >= 0) & (uu <= 1)
            for i, j in zip(*np.nonzero(hit), strict=True):
                out.append(p[i] + tt[i, j] * r[i])
    return np.array(out).reshape(-1, 2)


def strand_crossing(
    orbit: SymmetricOrbit,
    kind: str,
    sign: float,
    phase: float,
    section: Event,
    crossing: int = 1,
    max_duration: float = 15.0,
    eps: float = manifolds.DEFAULT_EPS,
    r_earth: float = 0.0,
    r_moon: float = 0.0,
    tol: float = 1e-12,
) -> tuple[FloatArray, float] | None:
    """Section crossing of the manifold strand starting at an arbitrary phase (0..1).

    Same construction as manifold_states, but for one continuous phase value: the orbit
    and STM are propagated to t = phase*T, the eigenvector is transported and scaled.
    Returns (crossing state, flight time) or None if the strand never gets there.
    """
    from cislunar import stm  # local import to keep module dependencies light

    M_sol = stm.propagate_stm(orbit.X0, 0.0, orbit.period, orbit.mu, rtol=1e-13, atol=1e-13)
    dirs = manifolds.stable_unstable_directions(stm.split(M_sol.X_final)[1])
    v0 = dirs.v_u if kind == "u" else dirs.v_s
    if v0[0] < 0:
        v0 = -v0
    t_i = (phase % 1.0) * orbit.period
    if t_i > 0:
        Xg, Phi = stm.split(
            stm.propagate_stm(orbit.X0, 0.0, t_i, orbit.mu, rtol=1e-13, atol=1e-13).X_final
        )
    else:
        Xg, Phi = orbit.X0, np.eye(6)
    v = Phi @ v0
    v /= np.linalg.norm(v[:3])
    X0 = Xg + sign * eps * v
    accept = getattr(section, "accept", None)
    stops = manifolds.make_stop_events(orbit.mu, r_earth, r_moon)
    tf = max_duration if kind == "u" else -max_duration
    sol = dopri5(
        cr3bp.make_rhs(orbit.mu), 0.0, tf, X0, rtol=tol, atol=tol, events=[*stops, section]
    )
    hits = [h for h in sol.events if h.index == len(stops) and (accept is None or accept(h.X))]
    if len(hits) < crossing:
        return None
    return hits[crossing - 1].X, hits[crossing - 1].t


def refine_connection(
    cross_a,  # callable phase -> (state, time) | None, e.g. functools.partial(strand_crossing, ...)
    cross_b,
    pa: float,
    pb: float,
    cols: tuple[int, int] = (1, 4),
    tol: float = 1e-10,
    max_iter: int = 30,
    dp: float = 1e-7,
) -> tuple[float, float, FloatArray, FloatArray] | None:
    """Newton on (pa, pb) so the two strands hit the section at the same (y, vy).

    Finite-difference Jacobian in the phases. In the planar problem at equal Jacobi
    constant, equal (y, vy) on x = const fixes |vx|; the caller checks its sign.
    """
    c = list(cols)
    for _ in range(max_iter):
        A, B = cross_a(pa), cross_b(pb)
        if A is None or B is None:
            return None
        F = A[0][c] - B[0][c]
        if np.max(np.abs(F)) < tol:
            return pa, pb, A[0], B[0]
        Ap, Bp = cross_a(pa + dp), cross_b(pb + dp)
        if Ap is None or Bp is None:
            return None
        J = np.column_stack([(Ap[0][c] - A[0][c]) / dp, -(Bp[0][c] - B[0][c]) / dp])
        step = np.linalg.solve(J, -F)
        if np.max(np.abs(step)) > 0.02:  # stay local: phases are periodic and strands diverge
            step *= 0.02 / np.max(np.abs(step))
        pa, pb = pa + step[0], pb + step[1]
    return None
