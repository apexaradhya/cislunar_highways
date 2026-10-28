# PROGRESS

This log was written by Claude (Anthropic's AI coding assistant) as it built the project phase by phase. See [CREDITS.md](CREDITS.md).

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

### Phase 8: Halo orbits ⚠️ (complete except the JPL catalog comparison)
- `orbits.py`:
  - Halo bifurcation finder (bisection on ν_v − 1 along the Lyapunov family); step-off into 3D.
  - Pseudo-arclength continuation (unknowns x0, z0, vy0; SVD tangent; adaptive step; polish at fixed z0).
  - Mirror (north/south), perilune.
- `shooting.py`: `crossing_jacobian` (shared) and `correct_symmetric_fixed_period` (3×3 Newton with the period constraint).
- Bifurcations:

  | Point | x0 | Period | C |
  |---|---|---|---|
  | L1 | 0.82339089 | 2.74299426 (11.91 d) | 3.18635484 |
  | L2 | 1.18089853 | 3.41553077 (14.83 d) | 3.16412188 |

- Families: L1 has 57 orbits (7.8–12.1 d) and L2 has 53 orbits (6.0–14.8 d). Both are continued until the perilune is < 100 km above the surface. Data is in `data/halo_L{1,2}_north.npz`; southern = z-mirror.
  - Max closure 2.3e-12. Crossing residual ≤ 2.4e-13. Reciprocal pairs to 9e-10.
  - z0 has a fold (turning point) along the L2 family, so continuation in z0 alone would fail; pseudo-arclength goes around it.
- **9:2 L2 southern NRHO** (period fixed at 2 × 29.530589 / 9 d = 6.56235 d; `data/nrho_9_2_south.json`):

  | Quantity | Value |
  |---|---|
  | X0 | [1.02202821, 0, −0.18210139, 0, −0.10327095, 0] |
  | C | 3.0585 |
  | Perilune radius | 3,249 km (altitude 1,512 km) |
  | Apolune radius | 71,222 km, over the south pole |
  | Stability indices | −1.32, 0.68 (mildly unstable) |

  - Comparison: Davis et al. (AAS 2020) report Gateway's osculating perilune radius as 3,195–3,557 km and apolune radius as 70,005–71,849 km. Our CR3BP values lie inside both ranges.
  - **Unverified:** these published numbers come from a web-search summary. The PDFs (ntrs.nasa.gov) are unreachable from this environment, so please confirm them before presenting. They are not used as a pass/fail test.
- **Decision (please review):**
  - The spec asks for the trivial monodromy pair to be (1, 1) to 1e-6 *individually*. Measured splits are 2e-8 to 9e-6.
  - Cause: 1 is a defective eigenvalue (Jordan block), so its eigenvalues are ill-conditioned. Error ε splits them by √(ε‖M‖) (Trefethen & Bau 1997, Lecture 25), with ‖M‖ up to ~10³.
  - Response: the test now checks the well-conditioned facts that the pair encodes: M f = f, ∇Cᵀ M = ∇Cᵀ, and mean of the pair = 1, all to 1e-7 or better. The raw split is reported, not hidden.
  - Tried and rejected: a half-period symmetric monodromy formula. It agrees with the full-period M to 4e-12, but on near-Moon orbits it was worse (split 3.5e-4).
- **Pending:** `test_halo.py::test_matches_jpl_catalog` is skipped until JPL halo entries are available.
- Figures: `p8_halo_families_3d`, `p8_halo_projections`, `p8_halo_stability`, `p8_nrho_9_2`.

### Phase 9: Invariant manifolds ✅ (tag `v0.1-core`)
- `manifolds.py`:
  - Eigen-directions of M.
  - Orbit + STM sampling.
  - Eigenvector transport v(tᵢ) = Φ(tᵢ,0)v(0) with the position part normalized.
  - Tubes with terminal events (Earth/Moon impact, escape, optional section).
  - Stroboscopic distance tool.
- **Decision:** default ε = 1e-5 (≈3.8 km). Reasoning:
  - At ε = 1e-6, L2 halo tubes need ~20 days just to leave the orbit.
  - At ε = 1e-5, the energy error is still only ~4.6e-10.
  - Sensitivity measured: each decade of ε shifts the departure by 5.00 d (theory T·ln10/ln λ_u = 4.92 d). Whole tubes at ε = 1e-6 and 1e-4 overlap. So ε re-labels the strands but doesn't change the tube.
- Tests (L2 halo with λ_u ≈ 106) and figure runs (λ_u ≈ 980):
  - λ_u·λ_s = 1 to 8e-11.
  - ∇C·v = O(1e-13), confirming the theory that eigenvectors with λ ≠ 1 are energy-neutral.
  - C changes only at O(ε²): 10× ε gives ~100× ΔC.
  - Jacobi along 480 trajectories: ≤ 4.6e-10 (spec ~1e-8).
- **Asymptotic approach** (L2 halo, λ_s = 0.001020):
  - Forward 1-period contraction: 0.000994 (2.5% off).
  - Backward 3-period growth ratios: 976.0, 980.6, 978.5, against 1/λ_s = 980.6.
- **Finding:** forward-time convergence can only be watched for about one period. Any unstable component (roundoff, O(ε²) curvature) grows by λ_u per period and takes over. The multi-period exponential rate is therefore verified backward in time, where the stable direction is the growing one. The test covers both.
- Figures: `p9_manifold_tubes_3d` (hero), `p9_manifold_tubes_xy`, `p9_asymptotic_rate`, `p9_eps_sensitivity`.
- Process note: two script edits silently failed to apply (`str.replace` found no match after `ruff format` re-wrapped the lines), so a figure showed stale results. All edits now go through the Edit tool or assert a match.

### Phase 10: Poincaré sections ✅
- `poincare.py`:
  - plane sections with crossing filters; tube cuts (n-th crossing)
  - closed-curve gap metric; gap-aware curve splitting
  - polyline intersections; strand crossing at arbitrary phase
  - Newton refinement of tube–tube connections
- `shooting.correct_planar_fixed_energy` (Lyapunov orbits at a prescribed C).
- Section x = 1 − μ, y < 0, at C = 3.17 (between C2 and C1):
  - L2 stable first cut: 160/160 strands; a closed curve (max neighbour gap 0.032, closing gap 0.018) ✅.
  - L1 unstable first cut: 102/160 strands. The others hit the Moon, so that curve is open (physical, not a numerical issue).
  - The L2 halo stable cut has 155/200 strands: an open 4D loop, again because of lunar impacts.
- Time-reversal symmetry:
  - The integrator is **bit-exactly** reversible (mirrored input gives mirrored output, difference 0.0).
  - Unstable vs stable cuts match pointwise, with a median relative mismatch of 9e-10 and worst 3.2e-8, identical at integrator tolerance 1e-12 and 1e-13.
  - Diagnosis: the mismatch comes from the starting points γ(t) and γ(T−t) (≈1e-12 apart; they are sampled from one forward pass), amplified by λ_u^(t/T) ≈ 1e4 over the flight. The test bound of 1e-7 is derived from this error budget. (This test is my own addition, not a spec tolerance.)
- **Result:** the first cuts don't intersect. The second cuts intersect once, and Newton refinement of the two phases confirms a **heteroclinic L1 → L2 connection**: state mismatch 3.6e-10, vx = 0.234897 on both sides, Δv = 0, flight 22.2 + 26.3 = 48.5 d. Saved in `data/heteroclinic_L1_L2_C3.17.json`.
- Figures: `p10_lyapunov_heteroclinic` (tubes, 1st cuts, 2nd cuts, verified connection) and `p10_halo_tube_cut`.

### Phase 11: LEO → L2 halo transfer ✅
- `transfer.py`:
  - Frames (spec 1.10, with time rotation).
  - Patched-conic LEO → LLO estimate.
  - Vectorized tangential-departure conic screening.
  - CR3BP leg correction (free departure velocity + TOF, minimum-norm Newton, backtracking).
  - Candidate generation on the manifold and on the orbit; continuous (phase, τ) manifold points.
  - Nelder–Mead refinement (scipy optimizer, applied to my own CR3BP-corrected objective).
- **Exploration finding:** within 170 days of backward flight, no L1/L2 halo stable manifold comes closer than ~42,000 km to Earth (16 halo/branch combinations tested). So a one-burn LEO → manifold transfer is impossible in this model, and a patch burn is required.
- **Method:**
  1. Screen 154,776 manifold points and 400 halo points with Earth-only conics.
  2. Correct the best 60 + 20 distinct candidates in the CR3BP.
  3. Refine with Nelder–Mead.
  4. Assemble end-to-end in the CR3BP.
- **Finding:** the screening is biased for patch points near the Moon. Screened 3.28 km/s turned into 3.62–3.96 km/s once corrected. All final numbers are CR3BP-corrected.
- **Bug fixed:** fixed-TOF shooting from LEO failed because Φ_rv is nearly singular on near-parabolic arcs (singular values 15.9 / 0.24 / 0.0011). Freeing TOF made it converge in 4–7 iterations.
- **Results** (LEO 185 km; L2 northern halo, Az ≈ 20,370 km, period 14.73 d, C = 3.152169; `data/transfer_results.json`):

  | Design | TLI | 2nd burn | Total | TOF |
  |---|---|---|---|---|
  | Patched conic → 100 km LLO (estimate) | 3,135 | 824 | 3,959 m/s | 5.0 d |
  | Direct → halo (CR3BP) | 3,161 | 941 | **4,103 m/s** | 6.1 d |
  | Manifold → halo (CR3BP) | 3,107 | 325 (+0.2 clean-up) | **3,432 m/s** | 168.4 d (3.4 + 165.0) |

  - The manifold route saves 671 m/s (16%) for 162 extra days.
  - Pareto front: about 3,660 m/s at 24 d, 3,617 m/s at 74 d, 3,450 m/s at 143 d.
- **Jacobi bookkeeping** (acceptance ✅): C = 58.15 in LEO → 2.928 after TLI (drift 2.8e-10 over leg 1) → 3.152169 after the patch burn, equal to the halo's C (drift 1.8e-10 over 165 d).
- **Honest caveats:**
  - The manifold arc's nominal arrival lands 254 km from the halo instead of ε = 3.8 km. 165 days through a chaotic region magnify ~1e-12 integration error about 10⁸×. Closest approach is 44 km, and the clean-up insertion is 0.2 m/s. A real mission would fly trajectory-correction maneuvers.
  - LEO plane is free (the departure plane is chosen to contain the needed velocity). Launch-site constraints are ignored.
- Figures: `p11_transfer_rotating`, `p11_transfer_inertial`, `p11_dv_vs_tof`, `p11_dv_breakdown`, `p11_arrival_distance`.

### Phase 11c: Predictability ✅ (approved add-on)
- `predictability.py`: batch propagation (N states as one ODE), butterfly fan, unstable-component projection (left eigenvector of M), error-growth forecast, TLI burn-error Monte Carlo with linear TCM targeting + nonlinear check.
- **Butterfly fan:** 500 copies on the L2 halo, σ = 1 m / 1 cm/s per axis, λ_u = 980.6.

  | Spread | Measured | Forecast |
  |---|---|---|
  | 10 km | 4.8 d | 5.0 d |
  | 100 km | 9.4 d | 9.9 d |
  | 1,000 km | 14.8 d | 14.8 d |
  | 10,000 km | 19.6 d | 19.7 d |

- **Finding:** the first forecast used only the position error and was off by about 13 days. Velocity errors dominate: the rms unstable amplitude is 986 m against a 1.7 m position error. Projecting onto the unstable direction fixed the forecast.
- **TLI Monte Carlo** (1 m/s per axis, 1,000 runs):

  | Transfer | Patch miss (median / 95%) | Day-1 TCM (median / 95%) | Nonlinear residual after TCM |
  |---|---|---|---|
  | Manifold | 1,729 / 5,298 km | 6.4 / 19.7 m/s | ≤ 2.1 km |
  | Direct | 3,706 / 10,554 km | 5.6 / 16.0 m/s | ≤ 4.2 km |

- Tests: batch propagation = individual to 1e-9; unstable component recovered to 1e-6; forecast within 2 d on a 60-member ensemble.
- Figures: `p11c_butterfly_fan`, `p11c_tli_monte_carlo`. Data: `data/predictability.json`, `data/butterfly_fan.npz`.
- Phase 11b design sweep is still running (`scripts/run_phase11b_sweep.py`; 7 halos × 2 LEO altitudes).

### Phase 11b: Design sweep ✅ (approved add-on)
- `transfer.design_candidates`: screen and CR3BP-correct direct and manifold candidates for one orbit. `scripts/run_phase11b_sweep.py` covers 7 halos (L1: 3, L2: 4) × LEO 185/300 km, giving **546 corrected candidates** (`data/design_sweep.json`).
- Best direct vs manifold (m/s, LEO 185 km):

  | Target | Direct | Manifold | Saving |
  |---|---|---|---|
  | L2 Az 8,924 km | 4,112 | 3,604 | 508 |
  | L2 Az 20,370 km | 4,105 | 3,442 | 662 |
  | L2 Az 38,983 km | 4,074 | 3,535 | 539 |
  | L2 Az 57,184 km | 4,030 | 3,584 | 446 |
  | L1 Az 8,422 km | 3,740 | 3,603 | 137 |
  | L1 Az 23,778 km | 3,729 | 3,584 | 145 |
  | L1 Az 46,916 km | 3,689 | 3,651 | 38 |

  A 300 km LEO is ~30 m/s cheaper for every target.
- The 19 Pareto-front transfers were rebuilt independently for the explorer; all reproduce their Δv exactly.
- Figure: `p11b_design_space`.

### Phase 12: Visualization, interactivity, packaging ✅
- **Cinematic explorer** (`web/explorer_template.html` → `docs/explorer/cislunar_explorer.html`, 2.45 MB single file):
  - three.js 0.160.0 with bloom, procedural Earth/Moon shading and a starfield.
  - **Story mode:** 9 chapters with captions whose numbers come from the data.
  - **Explore mode:**
    - drag a marker around the halo to choose the strand (96 precomputed);
    - drag a ring around L1 to change the energy (32 zero-velocity curves);
    - click Pareto-front transfers to fly them;
    - morph between the rotating and inertial frames;
    - toggle layers.
  - Published as a private artifact: https://claude.ai/artifact/KvggXGPjNuYvTiigeYiX5F
- **Video:** `docs/explorer/cislunar_highways_story_1080p.mp4`. Frames are rendered deterministically by stepping the clock in headless Chromium and screenshotting each one (`web/frames.mjs`); real-time capture only reached 7 fps.
- Plotly single-file 3D: `docs/figures/p12_interactive_3d.html`. Animation in both frames: `p12_transfer_both_frames.gif` / `.mp4`.
- **One command:** `scripts/run_final_demo.py` (`make demo`) recomputes from scratch in ~95 s and gets direct 4,105 / manifold 3,442 m/s, against 4,103 / 3,432 in the full run.
- **Notebook:** `notebooks/demo.ipynb`, executed with nbclient; outputs saved.
- Decisions and fixes:
  - **Grey background:** the haze came from 8-bit linear render targets in software GL (dark non-zero clear colours quantize to grey). Fixed with a pure-black clear.
  - **True-scale Moon** in the NRHO view, because the 3× Moon would hide the 1,512 km perilune.
  - **CDN workaround:** this container cannot reach jsDelivr, so local renders serve the same URLs from an npm copy of three@0.160.0 via Playwright routing. The published page loads from the CDN.

### Phase 13: Documentation ✅
- `README.md`, `docs/results.md` (validation table, results, limitations per spec §6), `docs/math_notes.md` (derivations; draft to restate in your own words), `docs/qa_prep.md` (the 10 defense questions), `docs/references.md` (with verification status).
