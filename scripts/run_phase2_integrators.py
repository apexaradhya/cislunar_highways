"""Phase 2 figures: convergence orders, Jacobi drift vs tolerance, sample trajectory."""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from cislunar import cr3bp
from cislunar.constants import EARTH_MOON
from cislunar.integrators import dopri5, rk4
from cislunar.plotting import draw_primaries, save_both

MU = EARTH_MOON.mu
X0 = np.array([0.5, 0.0, 0.05, 0.0, 0.9, 0.05])
T_GROWTH = 3.0
Y_GROWTH = math.exp(math.sin(T_GROWTH))


def eom(t, X):
    return cr3bp.equations_of_motion(t, X, MU)


def growth(t, X):
    return X * math.cos(t)


# --- convergence data ---
hs = T_GROWTH / np.array([10, 20, 40, 80, 160, 320])
err_rk4 = [
    abs(rk4(growth, 0, T_GROWTH, np.array([1.0]), round(T_GROWTH / h))[1][-1, 0] - Y_GROWTH)
    for h in hs
]
err_dp5 = [
    abs(
        dopri5(
            growth, 0, T_GROWTH, np.array([1.0]), rtol=1e10, atol=1e10, max_step=h, h0=h
        ).X_final[0]
        - Y_GROWTH
    )
    for h in hs
]
mask = np.array(err_dp5) > 1e-14  # drop points that hit roundoff before fitting
s_rk4 = np.polyfit(np.log(hs), np.log(err_rk4), 1)[0]
s_dp5 = np.polyfit(np.log(hs[mask]), np.log(np.array(err_dp5)[mask]), 1)[0]

# --- Jacobi drift vs tolerance (20 TU) ---
tols = np.logspace(-6, -13, 8)
drift, nsteps = [], []
for tol in tols:
    sol = dopri5(eom, 0, 20, X0, rtol=tol, atol=tol)
    C = cr3bp.jacobi_constant(sol.X, MU)
    drift.append(np.max(np.abs(C - C[0])))
    nsteps.append(sol.n_steps)
ref = solve_ivp(eom, (0, 10), X0, method="DOP853", rtol=1e-13, atol=1e-13)
ours = dopri5(eom, 0, 10, X0, rtol=1e-13, atol=1e-13)
scipy_diff = np.max(np.abs(ours.X_final - ref.y[:, -1]))
traj = dopri5(eom, 0, 20, X0, rtol=1e-13, atol=1e-13, t_eval=np.linspace(0, 20, 4000))

print(f"RK4 slope           : {s_rk4:.3f}  (theory 4)")
print(f"DOPRI5 slope        : {s_dp5:.3f}  (theory 5)")
print(f"vs scipy DOP853     : {scipy_diff:.3e}  (spec < 1e-8)")
for tol, d, n in zip(tols, drift, nsteps, strict=True):
    print(f"  tol {tol:.0e}: max |dC| = {d:.3e}, steps = {n}")


def fig_convergence(t):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    c = t["series"]
    ax.loglog(hs, err_rk4, "o-", color=c[0], label=f"RK4 (slope {s_rk4:.2f})")
    ax.loglog(
        hs[mask],
        np.array(err_dp5)[mask],
        "s-",
        color=c[1],
        label=f"Dormand–Prince 5 (slope {s_dp5:.2f})",
    )
    ax.set_xlabel("step size h (non-dimensional)")
    ax.set_ylabel("global error at t = 3")
    ax.set_title("Halving the step: error falls as h⁴ and h⁵")
    ax.legend()
    return fig


def fig_drift(t):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(tols, drift, "o-", color=t["series"][0], label="max |C(t) − C(0)| over 20 TU")
    ax.loglog(tols, tols, "--", color=t["muted"], linewidth=1, label="y = tolerance")
    ax.axhline(1e-9, color=t["series"][1], linewidth=1, linestyle=":")
    ax.annotate(
        "spec limit 1e-9",
        (tols[0], 1e-9),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=9,
    )
    ax.invert_xaxis()
    ax.set_xlabel("integrator tolerance (rtol = atol)")
    ax.set_ylabel("Jacobi constant drift")
    ax.set_title("Tighter tolerance → energy conserved to ~12 digits")
    ax.legend(loc="lower left")
    return fig


def fig_traj(t):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(traj.X[:, 0], traj.X[:, 1], color=t["series"][0], linewidth=1.2)
    ax.scatter([X0[0]], [X0[1]], color=t["series"][1], zorder=6, s=40)
    ax.annotate(
        "start",
        (X0[0], X0[1]),
        xytext=(6, 6),
        textcoords="offset points",
        color=t["ink2"],
        fontsize=9,
    )
    draw_primaries(ax, MU, t)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Earth–Moon distances, rotating frame)")
    ax.set_ylabel("y")
    ax.set_title("Sample CR3BP trajectory, 20 time units ≈ 87 days")
    return fig


for name, fn in [
    ("p2_convergence", fig_convergence),
    ("p2_jacobi_drift", fig_drift),
    ("p2_sample_trajectory", fig_traj),
]:
    print("saved", [str(p.name) for p in save_both(name, fn)])
