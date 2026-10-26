"""Phase 6: STM validation numbers (finite differences, symplecticity, det, Jacobi)."""

import numpy as np

from cislunar import cr3bp, stm
from cislunar.constants import EARTH_MOON

MU = EARTH_MOON.mu
cases = {
    "sample loop, T=3": (np.array([0.5, 0.0, 0.05, 0.0, 0.9, 0.05]), 3.0),
    "near L1, T=2": (np.array([0.82, 0.0, 0.02, 0.0, 0.15, 0.0]), 2.0),
}
lines = []
for name, (X0, T) in cases.items():
    sol = stm.propagate_stm(X0, 0.0, T, MU, rtol=1e-13, atol=1e-13)
    X, Phi = stm.split(sol.X)
    P = Phi[-1]
    fd = stm.stm_finite_difference(X0, 0.0, T, MU)
    rel = np.max(np.abs(P - fd)) / np.max(np.abs(P))
    symp = np.max(np.abs(P.T @ stm.J_VELOCITY @ P - stm.J_VELOCITY)) / np.max(np.abs(P)) ** 2
    det = abs(np.linalg.det(P) - 1)
    C = cr3bp.jacobi_constant(X, MU)
    lines.append(
        f"{name:18}: ||Phi||max={np.max(np.abs(P)):9.3e}  FD rel={rel:.2e}  symplectic={symp:.2e}  "
        f"|det-1|={det:.2e}  Jacobi drift={np.max(np.abs(C - C[0])):.2e}"
    )
print("\n".join(lines))
with open("docs/figures/p6_stm_checks.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
