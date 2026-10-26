# PROGRESS

Running log of phase status, test numbers, and decisions. Newest phase at the bottom.

## Decisions (defaults; change any of them)
| # | Decision | Why |
|---|---|---|
| D1 | Pseudo-potential includes the constant `mu(1-mu)/2`. | Matches the spec; gives C(L4)=C(L5)=3 exactly. Before comparing C with any catalog, check that catalog's convention. |
| D2 | GM values: Earth 398600.4418 (IERS 2010), Moon 4902.8000 (spec). Gives mu = 0.01215058392. | The spec/JPL-catalog mu 0.0121505856 agrees to 7 significant digits. Catalog comparisons will use the catalog's own mu, so nothing gets fudged. |
| D3 | L = 384400 km (conventional mean distance). | This only affects km/day conversions, not non-dimensional results. |
| D4 | Tooling: uv + hatchling, Python 3.13, ruff, pytest, Makefile. | Python 3.13 is what's installed; the spec asks for >= 3.11. |
| D5 | Scope addition (approved): design sweep + Pareto front, chaos/Monte Carlo predictability, cinematic three.js explorer. | Requested during planning. Built after the core phases are validated. |

## Open items
- JPL periodic-orbit API (`ssd-api.jpl.nasa.gov`) is blocked by this environment's network policy. Needed by Phase 7.
- No git remote yet: the cloud container is ephemeral, so commits must be pushed somewhere.

## Phase log

### Phase 0: Setup ✅
- uv project (Python 3.13), `src/` layout, pytest, ruff, Makefile (`test`, `figures`, `lint`, `format`, `demo`). Dependencies pinned in `uv.lock`.
- Acceptance: `pytest` runs and `import cislunar` works.

### Phase 1: Constants, units, dynamics, Jacobi constant ✅
- `constants.py`: cited GM values; `make_system()` computes mu, L, T, V. Result: mu = 0.01215058392, T = 4.3425 d, V = 1.0245 km/s.
- `cr3bp.py`: EOM, Ω, ∇Ω, Hessian, A-matrix, Jacobi constant, analytic dC/dt. All of them broadcast over (..., 6).
- Tests: mu matches the spec to 7 significant digits; A-matrix vs central-difference Jacobian within 1e-6; dC/dt < 1e-12 on 50 random states; C(L4) = 3 exactly.

### Phase 2: Integrators and event detection ✅
- `rootfind.py`: Brent's method (from scratch). `integrators.py`: RK4; DOPRI5 with PI step control, FSAL and 4th-order dense output; event detection (Brent on the dense output, then a secant polish on re-integrated states).
- Measured:
  - RK4 order 4.17; DOPRI5 order 4.96.
  - Own DOPRI5 vs scipy DOP853 (1e-13, 10 TU): **5.2e-12**. Spec: < 1e-8.
  - Jacobi drift over 20 TU at tol 1e-13: **7.4e-13**. Spec: < 1e-9.
  - y = 0 crossing: |y| < 1e-12, and it matches scipy's event time to 1e-10.
- Decision: the triple-root Brent test compares against the exact root (0.3), because scipy's brentq fails to converge on it within its default 100 iterations.
- Decision: the RK4 order test uses y' = y cos t rather than the harmonic oscillator. Sampling the oscillator over whole periods cancels the leading error term, and the measured slope came out 5.0.
- Figures: `p2_convergence`, `p2_jacobi_drift`, `p2_sample_trajectory` (each in light and dark).

### Phase 3: Lagrange points ✅
- `lagrange.py`: own Brent + one Newton polish for L1–L3; closed form for L4/L5; Jacobi constant at each point; Routh threshold; numerical stability test via the eigenvalues of A.
- Earth–Moon:

  | Point | x | C | Distance from the Moon |
  |---|---|---|---|
  | L1 | 0.836915134104 | 3.200344049361 | 58,019 km |
  | L2 | 1.155682158932 | 3.184163394831 | 64,515 km |
  | L3 | −1.005062645105 | 3.024150096215 | — |
  | L4/L5 | — | 3 (exactly) | — |

- Tests:
  - Force residual < 1e-13 for mu ∈ {Earth–Moon, Sun–Earth, 0.1, 0.5}.
  - Matches scipy brentq to 1e-14.
  - Matches the literature values (Koon et al. 2011) to 1e-4.
  - L4/L5 are equilateral to 1e-15.
- Routh: 27μ(1−μ) = 0.324 for Earth–Moon, so L4/L5 are stable. Threshold mu_Routh = 0.0385209. The numerical eigenvalue test agrees on 7 values of mu (stable at Sun–Earth, Earth–Moon, 0.03, 0.038; unstable at 0.039, 0.1, 0.5).
- Figure: `p3_lagrange_points` (with a to-scale zoom near the Moon).

### Phase 4: Hill regions ✅
- `hill.py`: 2Ω grid, forbidden mask, realm connectivity (via `scipy.ndimage.label`, used for connected-component bookkeeping only), and bisection for the neck-opening C.
- Geometric neck-opening C compared with C(Lᵢ) from Phase 3:
  - L1: +9.4e-6
  - L2: +7.4e-7
  - L3: +2.7e-6

  All are within the grid bound of 3e-5, derived from the grid spacing h = 2e-3.
- Decision: the bias is one-sided (found ≥ C(Lᵢ)), because the grid misses the exact minimum of 2Ω across the neck. The test asserts this direction.
- Figure: `p4_hill_regions` (6 panels from C > C1 to C < 3).

### Phase 5: Linear stability at the collinear points ✅
- `lagrange.collinear_modes()` gives the analytic λ, ω_p, ν and the ellipse ratio k. `linear_lyapunov_state()` gives the linear seed for Phase 7.
- Table (`docs/figures/p5_linear_table.txt`):

  | Point | λ | ω_p | ν | Planar period | e-fold time |
  |---|---|---|---|---|---|
  | L1 | 2.932056 | 2.334386 | 2.268831 | 11.69 d | 1.48 d |
  | L2 | 2.158674 | 1.862646 | 1.786176 | 14.65 d | 2.01 d |
  | L3 | 0.177875 | 1.010420 | 1.005331 | 27.00 d | 24.4 d |

- numpy.eig vs analytic: max error 8.9e-16 (spec: 1e-10). Saddle × center × center confirmed at all three points. Ω_xx = 1+2c, Ω_yy = 1−c, Ω_zz = −c to 1e-14.

### Phase 6: STM propagation ✅
- `stm.py`: 42-dim augmented RHS (fast scalar path, checked against `a_matrix`), `propagate_stm`, `split`, finite-difference STM. `cr3bp.make_rhs` is a fast single-state EOM, checked against the reference EOM to 1e-13.
- Checks (`docs/figures/p6_stm_checks.txt`):
  - Finite-difference STM vs propagated STM: 5.5e-9 and 1.7e-9 relative (spec ~1e-6).
  - Symplectic ΦᵀJ_vΦ = J_v to 1e-15 relative.
  - |det Φ − 1| ≤ 1.6e-12 (spec 1e-9).
  - Jacobi drift ≤ 2e-14.
- Decision: implemented **both** checks. Symplecticity uses J_v = [[W, I], [−I, 0]] in velocity coordinates, where W = [[0,−2,0],[2,0,0],[0,0,0]] is the Coriolis twist. This is because p = v + ω×r, so J_v = TᵀJT. Its infinitesimal form `AᵀJ_v + J_vA = 0` is also tested at random states.

### Phase 7: Planar Lyapunov orbits ⚠️ (complete except the JPL catalog comparison)
- `shooting.py`: symmetric differential corrector, implementing spec 1.7 including the (a/v_y) crossing-time terms. It works for any choice of free variables and targets (planar or halo), and adds one polishing Newton step after convergence.
- `orbits.py`: Lyapunov seeds, adaptive natural-parameter continuation (halves the step on failure), monodromy, reciprocal-pair matching, stability index.
- Results: 40 orbits per family. Data saved in `data/lyapunov_L{1,2}.npz`.

  | Family | Period | Max amplitude | Stability index ν | Max closure |
  |---|---|---|---|---|
  | L1 | 11.69–26.84 d | 61,504 km | 56–1335 | 9e-13 |
  | L2 | 14.65–33.48 d | 96,100 km | 50–726 | 1e-11 |

- Closure spec is < 1e-10 for both.
- Small-orbit period → linear period. At A = 1e-4 the relative difference is < 1e-5, and the correction scales as A² (measured ratio of ~100 for 10× the amplitude).
- Trivial monodromy pair: mean = 1 to < 1e-8, and M f = f to < 1e-7. The *individual* eigenvalues spread by up to ~3e-5 on the largest orbits.
  - Decision: the eigenvalue 1 is defective (a Jordan block), so roundoff ε splits it by √(ε‖M‖). The test therefore checks the mean and the flow eigenvector. The 1e-6 spec on individual eigenvalues is met only for smaller orbits; documented rather than loosened.
- Decision: the L2 family is parametrized by its **far-side** x-crossing. The Moon-side parametrization marched into the Moon and silently jumped to a small lunar orbit (period 0.02); a period-jump guard now catches this.
- **Pending:** `tests/test_lyapunov.py::test_matches_jpl_catalog` is skipped until catalog entries are provided (`tests/data/jpl_lyapunov.json`).
- Figures: `p7_lyapunov_families`, `p7_lyapunov_stability`.
