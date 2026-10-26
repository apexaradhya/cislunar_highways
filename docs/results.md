# Results

All numbers below come from the scripts in `scripts/` and the data files in `data/`. Units: the Earth–Moon CR3BP with μ = 0.0121505839, length unit 384,400 km, time unit 4.3425 days, velocity unit 1.0245 km/s.

## 1. Validation summary

| Check | Result | Requirement |
|---|---|---|
| μ from GM values vs spec value | agrees to 7 significant digits | 7 digits |
| A-matrix vs finite-difference Jacobian | < 1e-6 | ~1e-6 |
| Analytic dC/dt at 50 random states | < 1e-12 | ~1e-12 |
| Convergence order (own RK4 / DOPRI5) | 4.17 / 4.96 | 4 / 5 |
| Own DOPRI5 vs scipy DOP853, 10 TU | 5.2e-12 | < 1e-8 |
| Jacobi drift, 20 TU, tol 1e-13 | 7.4e-13 | < 1e-9 |
| y = 0 event location | \|y\| < 1e-12 | ~1e-12 |
| Lagrange-point force residual | < 1e-13 | < 1e-13 |
| Lagrange points vs scipy brentq | < 1e-14 | — |
| Hill-region neck opening vs C(Lᵢ) | +9.4e-6, +7.4e-7, +2.7e-6 | within grid bound 3e-5 |
| Linearization eigenvalues, numeric vs analytic | 8.9e-16 | 1e-10 |
| STM vs central differences | 5.5e-9, 1.7e-9 (relative) | ~1e-6 |
| STM symplectic / det Φ = 1 | 1e-15 / 1.6e-12 | det to 1e-9 |
| Lyapunov and halo closure | ≤ 1e-11 / ≤ 2.3e-12 | < 1e-10 |
| Symmetry-crossing residual | ≤ 2.4e-13 | < 1e-11 |
| Monodromy reciprocal pairs | ≤ 1e-9 | 1e-6 |
| Trivial pair (see §6) | mean 1 to < 1e-8; M f = f and ∇Cᵀ M = ∇Cᵀ to < 1e-7 | (1, 1) to 1e-6 |
| Jacobi along 480 manifold trajectories | ≤ 4.6e-10 | ~1e-8 |
| Stable-manifold rate, 3 periods backward | 976.0, 980.6, 978.5 vs 1/λ_s = 980.6 | ≈ λ_s per period |
| Time-reversal symmetry of the integrator | bit-exact | — |
| Transfer Jacobi bookkeeping | drift ≤ 2.8e-10 between burns; C after patch burn = halo C | consistent |
| JPL Three-Body Periodic Orbit Catalog | **not run** (host blocked from this environment) | ≥ 3 halos to 1e-6 |

## 2. Lagrange points and linear dynamics

| | x | Jacobi C | distance to Moon | λ (saddle) | 2π/ω_p | e-folding time |
|---|---|---|---|---|---|---|
| L1 | 0.836915134104 | 3.200344049 | 58,019 km | 2.9321 | 11.69 d | 1.48 d |
| L2 | 1.155682158932 | 3.184163395 | 64,515 km | 2.1587 | 14.65 d | 2.01 d |
| L3 | −1.005062645105 | 3.024150096 | 766,075 km | 0.1779 | 27.00 d | 24.4 d |
| L4/L5 | 0.487849416, ±√3/2 | 3 (exact) | 384,400 km | stable: 27μ(1−μ) = 0.324 < 1 | | |

## 3. Periodic orbits

- **Planar Lyapunov families** (40 orbits each):
  - L1: periods 11.69–26.84 d, amplitudes up to 61,504 km.
  - L2: periods 14.65–33.48 d, amplitudes up to 96,100 km.
  - Every member is unstable (stability index ν between 50 and 1,335).
- **Halo bifurcations:**
  - L1: x₀ = 0.82339089, T = 11.911 d, C = 3.18635484.
  - L2: x₀ = 1.18089853, T = 14.832 d, C = 3.16412188.
- **Halo families:** 57 orbits (L1) and 53 orbits (L2), continued down to perilunes 100 km above the surface. z₀ has a fold along the L2 family, which pseudo-arclength continuation passes through. Near-Moon members are nearly stable (ν → 1).
- **9:2 southern L2 NRHO** (period fixed at 2/9 of a synodic month):

  | Quantity | Value |
  |---|---|
  | Period | 6.56235 d |
  | Perilune radius | 3,249 km (1,512 km altitude) |
  | Apolune radius | 71,222 km, over the south pole |
  | Stability indices | −1.32, 0.68 |

  Davis et al. (AAS 2020) list Gateway's osculating perilune radius as 3,195–3,557 km and apolune radius as 70,005–71,849 km. Our values fall inside both ranges. Those published ranges come from a web-search summary; the paper itself could not be opened from this environment, so confirm them before quoting.

## 4. Manifolds, sections, and a free L1→L2 transfer

- **Target halo:** L2 northern, Az ≈ 20,370 km, T = 14.73 d, C = 3.152169, λ_u = 980.6.
- **ε sensitivity:** each decade of ε shifts the departure time by 5.00 d (theory T·ln10/ln λ_u = 4.92 d). Whole tubes computed at ε = 1e-6 and 1e-4 coincide. Default ε = 1e-5 (3.8 km).
- **Poincaré section x = 1 − μ, y < 0, at C = 3.17:** the first cuts of the L1-unstable and L2-stable tubes do not intersect; the second cuts intersect once. Refining that intersection gives a **zero-Δv L1 → L2 Lyapunov transfer of 48.5 d**, with full-state mismatch 3.6e-10.

## 5. LEO → L2 halo transfers (185 km LEO, full CR3BP)

| Design | TLI | Second burn | Total Δv | Time of flight |
|---|---|---|---|---|
| Patched conic → 100 km lunar orbit (textbook estimate) | 3,135 | 824 | **3,959 m/s** | 5.0 d |
| Direct → L2 halo | 3,161 | 941 | **4,103 m/s** | 6.1 d |
| Low-energy (stable manifold) → L2 halo | 3,107 | 325 (+0.2 clean-up) | **3,432 m/s** | 168.4 d |

- The low-energy route saves **671 m/s (16 %)** and takes 27× longer.
- The Pareto front for this halo: ≈ 3,660 m/s at 24 d, 3,617 m/s at 74 d, and 3,450 m/s at 143 d.
- **Design sweep** (546 corrected candidates):

  | Target | Direct | Best manifold | Saving |
  |---|---|---|---|
  | L2 halos, Az 8,900–57,200 km | 4,030–4,112 m/s | 3,442–3,604 m/s | 446–664 m/s |
  | L1 halos, Az 8,400–46,900 km | 3,689–3,740 m/s | 3,583–3,651 m/s | 38–145 m/s |

  A 300 km parking orbit saves about 30 m/s over 185 km in every case.
- **Why the saving is "only" 16 %:** the TLI burn (~3.1 km/s) is unavoidable. In the Earth–Moon CR3BP no halo manifold comes closer than ~42,000 km to Earth within 170 days, so a manifold transfer still needs a full TLI plus a matching burn. The savings come only at arrival. Real ballistic lunar transfers such as CAPSTONE's also use the Sun's gravity to shed energy, which this model leaves out.

## 6. Predictability

- **Butterfly fan:** 500 copies on the L2 halo, each off by 1 m and 1 cm/s per axis. Projected onto the unstable eigenvector, the starting errors have an rms size of 986 m. Velocity errors dominate.

  | Spread | Forecast (\|c_u\|·λ_u^(t/T)) | Measured |
  |---|---|---|
  | 10 km | 5.0 d | 4.8 d |
  | 100 km | 9.9 d | 9.4 d |
  | 1,000 km | 14.8 d | 14.8 d |
  | 10,000 km | 19.7 d | 19.6 d |

- **TLI error of 1 m/s per axis** (1,000 runs):

  | Transfer | Miss at the patch point (median) | Correction 1 day later (median / 95 %) | Residual after correction |
  |---|---|---|---|
  | Low-energy | 1,729 km | 6.4 / 19.7 m/s | ≤ 2.1 km (nonlinear check) |
  | Direct | 3,706 km | 5.6 / 16.0 m/s | ≤ 4.2 km |

## 7. Limitations

- **Model.** The CR3BP assumes the Moon moves on a circle and the spacecraft has no mass. It has no Sun, no solar radiation pressure and no non-spherical gravity. Real missions need an ephemeris model. Leaving out the Sun is the biggest single gap for low-energy transfers.
- **Impulsive burns.** Every manoeuvre is instantaneous. Finite burns of a few minutes at 3 km/s cost slightly more (gravity losses).
- **Patched-conic baseline.** It is a textbook estimate, not a mission-grade number. The fair comparison is the CR3BP direct transfer, which uses the same method as the low-energy design.
- **Free parking-orbit plane.** The LEO plane is chosen to contain the departure velocity; launch-site latitude and timing are ignored.
- **ε dependence.** ε changes the departure timing (5.00 d per decade) and which strand starts at a given phase, not the tube. All transfer numbers use ε = 1e-5.
- **Long chaotic coasts.** The 165-day coast magnifies ~1e-12 integration error about 10⁸ times. The arc arrives 254 km from the halo (closest approach 44 km) instead of 3.8 km, and a 0.2 m/s clean-up fixes it. A real mission would fly trajectory-correction manoeuvres along the way.
- **Search, not proof of optimality.** Transfers come from screening, correcting the best candidates and local refinement. Better solutions may exist elsewhere in the design space.
- **Differences from published values.**
  - μ from IERS/spec GM values is 0.01215058392; the JPL catalog uses 0.0121505856 (7-digit agreement).
  - The JPL catalog comparison was not run because the host is blocked here; the test is written and skips with a reason.
  - The Gateway NRHO comparison relies on unverified secondary numbers.
  - The trivial monodromy eigenvalues split by up to 9e-6 rather than 1e-6. The eigenvalue 1 is defective, so its individual eigenvalues are ill-conditioned; the well-conditioned properties it encodes are met to 1e-7 (§1).
