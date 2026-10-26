"""Phase 5: table of linear exponents/frequencies at L1, L2, L3 (analytic vs numpy.eig)."""

import numpy as np

from cislunar import cr3bp, lagrange
from cislunar.constants import EARTH_MOON

MU = EARTH_MOON.mu
DAYS = EARTH_MOON.time_days

hdr = f"{'pt':3} {'c':>10} {'lambda':>10} {'omega_p':>10} {'nu':>10} {'2pi/w_p':>9} {'[days]':>7} {'2pi/nu':>8} {'[days]':>7} {'e-fold':>7} {'max|eig-analytic|':>18}"
lines = [hdr, "-" * len(hdr)]
for k in (1, 2, 3):
    m = lagrange.collinear_modes(k, MU)
    eig = np.linalg.eigvals(cr3bp.a_matrix(np.array([m.x, 0, 0, 0, 0, 0]), MU))
    expected = [m.lam, -m.lam, 1j * m.omega_p, -1j * m.omega_p, 1j * m.nu, -1j * m.nu]
    err = max(np.min(np.abs(eig - e)) for e in expected)
    lines.append(
        f"{m.name:3} {m.c:10.6f} {m.lam:10.6f} {m.omega_p:10.6f} {m.nu:10.6f} "
        f"{m.planar_period:9.5f} {m.planar_period * DAYS:7.2f} {m.vertical_period:8.5f} "
        f"{m.vertical_period * DAYS:7.2f} {DAYS / m.lam:7.2f} {err:18.2e}"
    )
lines.append("e-fold = time [days] for a small deviation to grow by a factor e = 1/lambda.")
table = "\n".join(lines)
print(table)
with open("docs/figures/p5_linear_table.txt", "w") as f:
    f.write(table + "\n")
