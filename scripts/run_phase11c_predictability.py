"""Phase 11c: predictability. Butterfly fan on the L2 halo + TLI burn-error Monte Carlo."""

import json

import matplotlib.pyplot as plt
import numpy as np

from cislunar import predictability as P  # noqa: N812 (short name in formulas)
from cislunar import stm
from cislunar.constants import EARTH_MOON
from cislunar.plotting import save_both
from cislunar.shooting import correct_symmetric, full_period

U = EARTH_MOON
MU, L, D = U.mu, U.length_km, U.time_days
N_FAN = 500
SIGMA_R_KM, SIGMA_V_M_S = 0.001, 0.01  # 1 m and 1 cm/s per axis: state-of-the-art navigation
SIGMA_TLI_M_S = 1.0  # 1 m/s per axis on the 3.1 km/s injection burn (0.03%)
N_MC = 1000

d = np.load("data/halo_L2_north.npz")
halo = correct_symmetric(
    d["X0"][int(np.argmin(np.abs(d["X0"][:, 2] - 0.053)))], MU, free=(0, 4), targets=(3, 5)
)
M = stm.split(full_period(halo).X_final)[1]
fan = P.butterfly_fan(
    halo.X0, MU, U, N_FAN, SIGMA_R_KM, SIGMA_V_M_S, 3 * halo.period, n_out=600, seed=2026
)
cu, lam = P.unstable_amplitude(fan.dX0, M)
s0 = float(np.sqrt(np.mean(cu**2)))
spread = fan.spread() * L
pred = s0 * L * lam ** (fan.t / halo.period)
thresholds = [10.0, 100.0, 1000.0, 10000.0]
table = []
for dk in thresholds:
    tm = fan.t[np.argmax(spread > dk)] * D
    tp = P.predicted_time_to_reach(dk / L, s0, halo.period, lam) * D
    table.append(dict(threshold_km=dk, measured_days=float(tm), predicted_days=float(tp)))
    print(f"spread {dk:>7,.0f} km: measured {tm:5.1f} d, predicted {tp:5.1f} d")
print(
    f"(lambda_u = {lam:.1f}, halo period {halo.period * D:.2f} d, rms unstable amplitude {s0 * L * 1000:.0f} m)"
)

tr = np.load("data/transfer_trajectories.npz")
mc = {}
for key, name in (("m", "manifold"), ("d", "direct")):
    t1, X1 = tr[f"{key}_t1"], tr[f"{key}_X1"]
    r = P.burn_error_monte_carlo(
        X1[0], t1[-1] - t1[0], X1[-1], U, SIGMA_TLI_M_S, N_MC, 1.0 / D, n_check=20, seed=7
    )
    mc[name] = r
    print(
        f"{name}: miss median {np.median(r.miss_km):,.0f} km (95%: {np.percentile(r.miss_km, 95):,.0f}); "
        f"TCM at day 1: median {np.median(r.tcm_m_s):.1f} m/s (95%: {np.percentile(r.tcm_m_s, 95):.1f}); "
        f"nonlinear residual after TCM <= {r.tcm_check_km.max():.1f} km"
    )
with open("data/predictability.json", "w") as fh:
    json.dump(
        dict(
            lambda_u=lam,
            period_days=halo.period * D,
            sigma_r_km=SIGMA_R_KM,
            sigma_v_m_s=SIGMA_V_M_S,
            unstable_rms_m=s0 * L * 1000,
            forecast=table,
            monte_carlo={
                k: dict(
                    sigma_tli_m_s=SIGMA_TLI_M_S,
                    n=N_MC,
                    miss_median_km=float(np.median(v.miss_km)),
                    miss_95_km=float(np.percentile(v.miss_km, 95)),
                    tcm_median_m_s=float(np.median(v.tcm_m_s)),
                    tcm_95_m_s=float(np.percentile(v.tcm_m_s, 95)),
                    residual_max_km=float(v.tcm_check_km.max()),
                )
                for k, v in mc.items()
            },
        ),
        fh,
        indent=2,
    )
# Fan snapshot data for the cinematic explorer (subsampled).
np.savez_compressed(
    "data/butterfly_fan.npz",
    t=fan.t[::3],
    states=fan.states[::3, :200, :3].astype(np.float32),
    reference=fan.reference[::3, :3],
    mu=MU,
)


def draw_fan(t):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    snaps = [1.0, 1.5, 2.0]  # 7-day cloud (~40 km) would be invisible at this scale
    H = full_period(halo, n_points=400).X
    ax.plot(H[:, 0], H[:, 1], color=t["ink"], lw=1.5)
    for k, frac in enumerate(snaps):
        i = int(np.argmin(np.abs(fan.t - frac * halo.period)))
        P_ = fan.states[i, :, :2]
        ax.scatter(
            P_[:, 0],
            P_[:, 1],
            s=3,
            color=t["series"][k],
            alpha=0.7,
            label=f"after {fan.t[i] * D:.0f} d (spread {spread[i]:,.0f} km)",
        )
    ax.scatter([1 - MU], [0], color=t["moon"], s=30)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{N_FAN} spacecraft, each nudged by ~1 m and ~1 cm/s")
    ax.legend(fontsize=8, markerscale=4, loc="lower left")
    ax = axes[1]
    ax.semilogy(fan.t * D, spread, color=t["series"][0], label="measured spread (rms)")
    ax.semilogy(
        fan.t * D, pred, "--", color=t["muted"], label=f"forecast: |c_u| · λu^(t/T), λu = {lam:.0f}"
    )
    for row in table:
        ax.scatter(row["measured_days"], row["threshold_km"], color=t["series"][1], s=20, zorder=5)
    ax.set_ylim(1e-3, 1e6)
    ax.set_xlabel("days")
    ax.set_ylabel("spread [km]")
    ax.set_title("Linear theory predicts the blow-up to within ~0.5 day")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def draw_mc(t):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, r in mc.items():
        c = t["series"][0] if name == "manifold" else t["series"][1]
        axes[0].hist(
            r.miss_km,
            bins=40,
            color=c,
            alpha=0.6,
            label=f"{name} (median {np.median(r.miss_km):,.0f} km)",
        )
        axes[1].hist(
            r.tcm_m_s,
            bins=40,
            color=c,
            alpha=0.6,
            label=f"{name} (median {np.median(r.tcm_m_s):.1f} m/s)",
        )
    axes[0].set_xlabel("miss at the second burn, no correction [km]")
    axes[0].set_title(f"A {SIGMA_TLI_M_S:.0f} m/s TLI error ({N_MC} Monte Carlo runs)")
    axes[1].set_xlabel("correction burn 1 day after TLI [m/s]")
    axes[1].set_title("…costs a few m/s to fix early")
    for a in axes:
        a.set_ylabel("runs")
        a.legend(fontsize=8)
    fig.tight_layout()
    return fig


print("saved", [p.name for p in save_both("p11c_butterfly_fan", draw_fan)])
print("saved", [p.name for p in save_both("p11c_tli_monte_carlo", draw_mc)])
