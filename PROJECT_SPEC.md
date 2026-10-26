# Cislunar Trajectory Design in the CR3BP: Master Prompt and Build Guide

## PART 1: KICKOFF PROMPT

You are my senior engineering partner on a from-scratch scientific computing project: designing low-energy Earth-to-Moon trajectories in the Circular Restricted Three-Body Problem (CR3BP). I am a high school student with calculus-level math who wants to genuinely understand every piece, because I will present and defend this project.

First, read PROJECT_SPEC.md completely. It is the source of truth for scope, math, architecture, validation tests and acceptance criteria.

Working rules:
1. Before writing any code, enter plan mode and give me a short plan: repo layout, tooling choices, and the phase order from the spec. Wait for my approval.
2. Work one phase at a time, in order. Each phase ends with passing tests, saved figures, a git commit, and an update to PROGRESS.md. Do not start the next phase until the current phase's acceptance criteria pass. If a validation fails, debug it; never loosen a tolerance just to get a green test.
3. Write the core numerical methods yourself (integrators, root finders for Lagrange points, differential correction, manifold generation). Use scipy/numpy only as an independent cross-check, never as a substitute for the core algorithm, unless the spec says otherwise.
4. Do not invent numerical constants or published reference values. Derive them, compute them, or look them up and cite the source in a comment. If you cannot verify a number, say so and ask me.
5. After each phase, write a short "Learning note" (in docs/learning_notes.md) explaining the concept in plain language with an analogy, the key equation, and one question I should be able to answer. Keep it to about 150 words per phase.
6. Keep code clean: type hints, docstrings that cite the equation or paper being implemented, no magic numbers, deterministic outputs, and a pytest suite.
7. If you hit a decision that is genuinely mine (scope, which extension to pursue, a tradeoff), ask me briefly with concrete options. Otherwise make a sensible default, note it in PROGRESS.md, and continue.
8. At the end of each phase, give me a 3-sentence summary: what now works, what the test numbers were, and what comes next.

## PART 2: FULL PROJECT SPEC

### 0. Project summary
Title: Low-Energy Cislunar Transfer Design in the Earth-Moon CR3BP

One-sentence pitch: Build, from scratch and validated against published data, a numerical toolchain that finds Lagrange points, computes halo orbits around them, generates their invariant manifolds ("gravitational highways"), and uses those manifolds to design a fuel-efficient transfer from low Earth orbit to a lunar halo orbit.

What makes it impressive:
* The core algorithms are written by hand and cross-checked against independent references.
* Every result is validated by a conserved quantity or a published value, not just plotted.
* It ends in a real engineering deliverable: a transfer with a computed delta-v budget compared against a direct-transfer baseline.
* It uses real mission concepts (halo orbits as used by JWST, Gateway's near-rectilinear halo orbit, ballistic lunar transfers as used by CAPSTONE).

Language/tooling: Python 3.11+, numpy, scipy, matplotlib, plotly (interactive 3D), pytest, numba (optional speed-up), jupyter (for demo notebooks). Use `uv` or `venv` + `pyproject.toml`.

### 1. Background math

#### 1.1 The model
Two massive bodies (Earth `m1`, Moon `m2`) move in circles about their common barycenter. A third body of negligible mass (the spacecraft) moves in their field. Work in non-dimensional units in the rotating frame:
* Mass parameter: `mu = m2 / (m1 + m2)`. For Earth-Moon, `mu ≈ 0.0121505856` (compute it from GM values: GM_earth ≈ 398600.4418 km^3/s^2, GM_moon ≈ 4902.8000 km^3/s^2; cite the source in code).
* Unit of length `L` = Earth-Moon distance (≈ 384,400 km). Unit of time `T` = 1/n where `n = sqrt((GM1+GM2)/L^3)` (≈ 4.34 days). Unit of velocity = `L/T` (≈ 1.0245 km/s). Compute these from constants; do not hardcode.
* Earth is at `(-mu, 0, 0)`, Moon at `(1-mu, 0, 0)`. The frame rotates at angular rate 1 about +z.

Distances to the primaries:
```
r1 = sqrt((x + mu)^2 + y^2 + z^2)        # to Earth
r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2)    # to Moon
```

#### 1.2 Equations of motion
```
x'' =  2 y' + x - (1-mu)(x+mu)/r1^3 - mu (x-1+mu)/r2^3
y'' = -2 x' + y - (1-mu) y / r1^3   - mu y / r2^3
z'' =              -(1-mu) z / r1^3 - mu z / r2^3
```
Pseudo-potential:
```
Omega(x,y,z) = (x^2 + y^2)/2 + (1-mu)/r1 + mu/r2 + mu(1-mu)/2
```
(The constant term `mu(1-mu)/2` is a convention. Pick one convention, state it in the docstring, and keep it everywhere, because it shifts every Jacobi-constant value you compare against the literature.)

#### 1.3 Jacobi constant (the only conserved quantity)
```
C = 2 * Omega(x,y,z) - (x'^2 + y'^2 + z'^2)
```
This is your primary correctness check. A numerical trajectory whose `C` drifts is a wrong trajectory.

#### 1.4 Variational equations and the State Transition Matrix (STM)
State `X = [x, y, z, vx, vy, vz]`. Linearization matrix:
```
A(X) = [[ 0_3 ,  I_3 ],
        [ Oxx ,  K   ]]

K = [[ 0, 2, 0],
     [-2, 0, 0],
     [ 0, 0, 0]]

Oxx = Hessian of Omega:
  Oxx_xx = 1 - (1-mu)/r1^3 - mu/r2^3 + 3(1-mu)(x+mu)^2/r1^5 + 3 mu (x-1+mu)^2/r2^5
  Oxx_yy = 1 - (1-mu)/r1^3 - mu/r2^3 + 3(1-mu) y^2/r1^5     + 3 mu y^2/r2^5
  Oxx_zz =   - (1-mu)/r1^3 - mu/r2^3 + 3(1-mu) z^2/r1^5     + 3 mu z^2/r2^5
  Oxx_xy = 3(1-mu)(x+mu) y/r1^5 + 3 mu (x-1+mu) y/r2^5
  Oxx_xz = 3(1-mu)(x+mu) z/r1^5 + 3 mu (x-1+mu) z/r2^5
  Oxx_yz = 3(1-mu) y z/r1^5     + 3 mu y z/r2^5
```
The STM `Phi(t, t0)` obeys `dPhi/dt = A(X(t)) Phi`, `Phi(t0,t0) = I`. Integrate the 6 state equations plus the 36 STM entries together as one 42-dimensional ODE.

#### 1.5 Lagrange (equilibrium) points
Set all velocities and accelerations to zero.
* Collinear points L1, L2, L3 lie on the x-axis. Solve the 1D equation `x - (1-mu)(x+mu)/|x+mu|^3 - mu(x-1+mu)/|x-1+mu|^3 = 0`. Brackets: L1 in `(-mu, 1-mu)`, L2 in `(1-mu, 2)`, L3 in `(-2, -mu)` (stay a small epsilon away from the singularities at the primaries). Use Brent's method or Newton with a bracketing fallback, written yourself.
* Triangular points L4, L5: `x = 1/2 - mu`, `y = ±sqrt(3)/2`, `z = 0`.
* L4/L5 are linearly stable only when `27 mu (1-mu) < 1` (Routh criterion, `mu < ~0.03852`). Earth-Moon satisfies this. Verify numerically.

Reference values to sanity-check (approximate, verify against a cited source before using them as pass/fail): Earth-Moon L1 `x ≈ 0.8369`, L2 `x ≈ 1.1557`, L3 `x ≈ -1.0051`.

#### 1.6 Linear dynamics at a collinear point
Let `c = (1-mu)/|xL+mu|^3 + mu/|xL-1+mu|^3`. At a collinear point:
```
Oxx_xx = 1 + 2c,   Oxx_yy = 1 - c,   Oxx_zz = -c
```
* Out-of-plane motion decouples: `z'' = -c z`, so the vertical frequency is `nu = sqrt(c)`.
* In-plane motion has characteristic equation `s^4 + (2 - c) s^2 + (1 + c - 2c^2) = 0`. For collinear points this gives one real pair `±lambda` (saddle, the instability) and one imaginary pair `±i*omega_p` (the oscillation). So each collinear point is a saddle x center x center.

#### 1.7 Periodic orbits via differential correction
Planar Lyapunov orbits and 3D halo orbits are symmetric about the xz-plane. Start from `X0 = [x0, 0, z0, 0, vy0, 0]`. Integrate until the next crossing of `y = 0` (at half period, `t = T/2`). Symmetry demands that at that crossing `vx = 0` and `vz = 0`.

Hold `z0` fixed. Free variables: `(x0, vy0)`. Targets: `(vx_f, vz_f) = (0, 0)`. Newton update:
```
M = [[ Phi41 - (ax_f/vy_f) Phi21 ,  Phi45 - (ax_f/vy_f) Phi25 ],
     [ Phi61 - (az_f/vy_f) Phi21 ,  Phi65 - (az_f/vy_f) Phi25 ]]

M * [dx0, dvy0]^T = -[vx_f, vz_f]^T
```
where `Phi_ij` are STM entries at the crossing (1-based: state order x,y,z,vx,vy,vz), and `ax_f, az_f` are the accelerations at the crossing. The `(a/vy)` terms correct for the fact that the crossing time itself changes when you perturb the initial state. Iterate until `|vx_f|, |vz_f| < 1e-11`.

#### 1.8 Monodromy matrix and stability
After finding a periodic orbit of period `T`, the monodromy matrix is `Phi(T, 0)`. Its six eigenvalues come in reciprocal pairs `(lambda, 1/lambda)`, and one pair is always exactly `(1, 1)` (the "trivial" pair). Check both numerically. The stability index is `nu = (lambda + 1/lambda)/2`. If any `|nu| > 1`, the orbit is unstable and has stable/unstable manifolds. Halo orbits near L1/L2 are unstable, with one dominant real pair `lambda_u > 1`, `lambda_s = 1/lambda_u`.

#### 1.9 Invariant manifolds
Let `v_u` and `v_s` be the unstable and stable eigenvectors of the monodromy matrix. To get the manifold from many points along the orbit:
1. Propagate the orbit and the STM over one period, storing `X_gamma(t_i)` and `Phi(t_i, 0)` at about 100-300 points.
2. Transport the eigenvector: `v(t_i) = Phi(t_i, 0) v(0)`, then normalize its position part (first 3 components) to unit length.
3. Displace: `X_manifold = X_gamma(t_i) ± eps * v(t_i)`, with `eps` roughly 1e-6 to 1e-4 non-dimensional (about 0.4 km to 40 km; document your choice and test sensitivity).
4. Propagate unstable manifolds forward in time and stable manifolds backward in time, for both signs `±`.

The union over `t_i` forms tube-like surfaces. Trajectories on the stable manifold flow into the halo orbit; those on the unstable manifold flow away from it.

#### 1.10 Rotating-to-inertial velocity conversion (for delta-v)
To compare against Earth-orbit speeds you need Earth-centered inertial velocity:
```
r_rel_Earth = (x + mu,  y,  z)
v_rel_Earth_inertial = (vx - y,  vy + x + mu,  vz)
```
(Rotation rate is 1 in these units.) Convert to km and km/s with the unit constants from 1.1.

### 2. Repository layout
```
cislunar/
├── PROJECT_SPEC.md
├── PROGRESS.md                 # running log: phase status, test numbers, decisions
├── README.md                   # polished project overview with figures
├── pyproject.toml
├── docs/
│   ├── math_notes.md           # derivations in your own words
│   ├── learning_notes.md       # one short note per phase
│   ├── results.md              # final numbers and discussion
│   └── figures/
├── src/cislunar/
│   ├── constants.py            # GM values, L, T, V, with citations
│   ├── units.py                # nondim <-> km, s, km/s converters
│   ├── cr3bp.py                # EOM, Omega, Jacobi constant, A-matrix
│   ├── integrators.py          # own RK4 + adaptive Dormand-Prince, event detection
│   ├── stm.py                  # 42-dim augmented propagation
│   ├── lagrange.py             # equilibrium points + linear analysis
│   ├── hill.py                 # zero-velocity curves
│   ├── shooting.py             # differential correctors (Newton)
│   ├── orbits.py               # Lyapunov, halo, continuation, stability
│   ├── manifolds.py            # eigenvectors, tube generation
│   ├── poincare.py             # Poincare-section crossings and maps
│   ├── transfer.py             # patch-point search, delta-v, baseline comparison
│   ├── plotting.py             # all figures, consistent style
│   └── bicircular.py           # (extension) Sun-perturbed model
├── tests/
├── scripts/                    # run_phase1_*.py ... run_final_demo.py
└── notebooks/                  # demo.ipynb
```

### 3. Phased build plan
Each phase lists: Goal, Implement, Tests / acceptance, Figures. Do not advance until acceptance passes.

**Phase 0: Setup**
* Create repo, `pyproject.toml`, virtual environment, pytest config, a `Makefile` or `justfile` with `test`, `figures`, `lint`.
* Initialize git; commit after every phase with a descriptive message.
* Create `PROGRESS.md` and `docs/learning_notes.md`.
* Acceptance: `pytest` runs (even with zero tests) and `import cislunar` works.

**Phase 1: Constants, units, dynamics, Jacobi constant**
* Implement: `constants.py` (cite sources), `units.py`, `cr3bp.py` with `equations_of_motion(t, X, mu)`, `pseudo_potential`, `jacobi_constant`, and `a_matrix`.
* Tests:
  * `mu` computed from GM values matches `0.01215058...` to 7 digits.
  * EOM vectorizes or at least handles single states correctly.
  * `a_matrix` matches a finite-difference Jacobian of the EOM to ~1e-6 at several random states.
  * Jacobi constant is invariant under the EOM: `dC/dt` evaluated analytically is ~0 (to 1e-12) at random states.
* Acceptance: all tests pass.

**Phase 2: Integrators and event detection**
* Implement: `integrators.py` containing (a) a fixed-step RK4, (b) an adaptive Dormand-Prince 5(4) with step-size control, (c) event detection with root refinement (bisection or secant on dense output or Hermite interpolation) so crossings like `y = 0` are located to ~1e-12.
* Tests:
  * Convergence order: for a smooth test ODE, RK4 error scales as `h^4` and DOPRI5 error scales as `tol` behaves as expected (plot the log-log slope).
  * Compare your DOPRI5 against `scipy.integrate.solve_ivp(method="DOP853", rtol=1e-13, atol=1e-13)` on a CR3BP trajectory over 10 time units: final-state difference < 1e-8.
  * Jacobi constant drift on a 20-time-unit trajectory with tight tolerance: < 1e-9. Plot drift vs tolerance (this is a great figure).
* Figures: convergence plot, Jacobi drift vs tolerance, a sample trajectory in the rotating frame.

**Phase 3: Lagrange points**
* Implement: `lagrange.py` with a hand-written root finder for L1-L3 and closed forms for L4-L5. Return positions and the Jacobi constant at each point.
* Tests:
  * Residual of the force equation at each point < 1e-13.
  * Cross-check against `scipy.optimize.brentq`.
  * L4/L5 form equilateral triangles with the primaries (all three side lengths equal 1).
  * Routh criterion computed numerically for several `mu` values, including Earth-Moon and Sun-Earth (`mu ≈ 3.0035e-6`), and the Earth-Moon result reported.
* Figures: the five points plotted with Earth and Moon (to scale where possible, with an inset).

**Phase 4: Zero-velocity curves and Hill regions**
* Implement: `hill.py` that, for a given Jacobi constant `C`, computes the forbidden region `2*Omega(x,y,0) < C`.
* Why it matters: this shows geometrically when the "neck" at L1 or L2 opens, which is the condition for a spacecraft to pass between the Earth and Moon regions.
* Figures: contour plots for several `C` values bracketing `C(L1)`, `C(L2)`, `C(L3)`, with the necks visibly opening.
* Acceptance: neck-opening values of `C` match `C(L1)` and `C(L2)` from Phase 3.

**Phase 5: Linear stability at the collinear points**
* Implement: build the 6x6 linearization at L1/L2/L3 and compute eigenvalues with `numpy.linalg.eig`. Separately compute `lambda`, `omega_p`, `nu` from the analytic formulas in 1.6.
* Tests: numeric eigenvalues match the analytic ones to 1e-10. Confirm the saddle x center x center structure (one real pair, two imaginary pairs).
* Deliverable: a printed table for L1, L2, L3 with `lambda`, `omega_p`, `nu`, and the linear periods `2*pi/omega_p`.

**Phase 6: State transition matrix propagation**
* Implement: `stm.py` that integrates the 42-dim system and returns `X(t)` and `Phi(t)`.
* Tests:
  * Compare `Phi(T, 0)` to a finite-difference STM obtained by perturbing each initial component by ~1e-7 and re-propagating (central differences). Agreement to ~1e-6 relative.
  * Symplectic check: `Phi^T J Phi = J` (with `J` the standard symplectic matrix) holds for the rotating-frame system after accounting for the Coriolis term; if this is troublesome, instead verify `det(Phi) = 1` (volume preservation, since `trace(A) = 0`) to 1e-9. Document which check you used and why.
  * Jacobi constant conserved along the augmented propagation.

**Phase 7: Planar Lyapunov orbits**
* Implement: `shooting.py` with the differential corrector from 1.7 using `z0 = 0` (planar case: free `x0`... with `vy0` as the only needed correction, targeting `vx_f = 0`; the 1-variable version is a good warm-up). Seed from the linear solution at L1/L2: `x0 = xL + A_x`, `vy0` from the linear eigenvector, for small `A_x` (e.g., 1e-4 to 1e-3).
* Add continuation: natural-parameter continuation in `x0`, tracking period `T`, Jacobi constant `C`, and stability index versus amplitude.
* Tests:
  * Converged orbit closes: `|X(T) - X(0)| < 1e-10`.
  * Period at small amplitude approaches the linear period `2*pi/omega_p` from Phase 5.
  * Jacobi constant is constant along each orbit.
  * Compare several orbits (period, `C`, initial conditions) against the JPL Three-Body Periodic Orbit Catalog (ssd.jpl.nasa.gov, "Three-Body Periodic Orbits"), noting which entries you used. If you cannot fetch the catalog, ask me to paste a few entries.
* Figures: a family of nested Lyapunov orbits around L1 and L2, colored by `C` or period; stability index vs amplitude.

**Phase 8: Halo orbits**
* Implement in `orbits.py`:
  * Seed options (try in this order): (a) initial conditions from the JPL catalog (best for validation), (b) a Richardson third-order analytic approximation (Richardson 1980), (c) continuation of the planar Lyapunov family: near the amplitude where the vertical and planar frequencies coincide, a halo family bifurcates; step off in `z0` and correct.
  * Differential correction from 1.7 with `z0` fixed, free `(x0, vy0)`.
  * Pseudo-arclength continuation along the halo family (natural continuation will fail near turning points). Track `T`, `C`, `z0`, perilune/closest approach to the Moon, and stability index.
  * Northern and southern families (mirror in `z`).
  * Monodromy matrix, eigenvalues, stability index.
* Tests:
  * Closure `< 1e-10`, symmetry crossing conditions `< 1e-11`.
  * Monodromy eigenvalues: reciprocal pairs to 1e-6, trivial pair `(1,1)` to 1e-6.
  * Match JPL catalog entries for at least 3 halo orbits (period and initial conditions) to ~1e-6 relative, or document the discrepancy and its cause.
* Figures: 3D family of L1 and L2 halos (northern and southern) in the rotating frame; projections onto xy, xz, yz; stability index vs `z0`.
* Stretch within this phase: extend the L2 southern family down to the Near-Rectilinear Halo Orbit (NRHO) regime used by NASA's Gateway (period near 6.5 days, 9:2 resonance with the lunar synodic month, very close perilune). Report period, perilune/apolune altitude from your own computation; compare to published Gateway values (cite the source).

**Phase 9: Invariant manifolds**
* Implement: `manifolds.py`
  * Eigen-decompose the monodromy matrix; select the real eigenvalue pair with `|lambda| > 1` and `< 1`.
  * Transport eigenvectors along the orbit using the stored STM history (1.9).
  * Generate stable and unstable manifold trajectories for both `±` branches, propagating with event stops (e.g., when the trajectory returns within a minimum distance of a primary, escapes beyond a radius, or hits a Poincare section).
  * Parameter study: effect of `eps` on the manifold shape.
* Tests:
  * Each manifold trajectory conserves its Jacobi constant, equal to the orbit's `C` to ~1e-8 (initial displacement slightly changes it; quantify).
  * Stable-manifold trajectories propagated forward from a late point should asymptotically approach the halo orbit (distance to the orbit decreases roughly exponentially at rate `ln |lambda_s|`). Verify numerically and plot.
* Figures: 3D tube plots for L1 and L2 halos (unstable in one color, stable in another, `±` branches distinguished), plus Earth, Moon, and the halo orbit. This is your hero visual.

**Phase 10: Poincare sections**
* Implement: `poincare.py` to record crossings of a chosen surface (e.g., `x = 1 - mu` through the Moon, or `y = 0`) by manifold trajectories, and plot the 2D crossings, e.g., `(y, vy)`.
* Use: intersect the stable manifold of an L2 halo (coming from Earth) with a section and look for a patch point where its crossing is close to a trajectory emanating from Earth orbit.
* Acceptance: section plots reproduce the characteristic closed-curve cross-sections of the manifold tubes.

**Phase 11: Low-energy transfer design**
* Baseline (direct transfer): from a circular LEO parking orbit (altitude 185-300 km, make it a parameter; compute circular speed in km/s from `GM_earth`), compute a patched-conic estimate of the trans-lunar injection delta-v and the lunar-orbit-insertion delta-v. Cite the formulas you use. Clearly label it an estimate.
* Low-energy design (backward-from-halo approach):
  1. Propagate the stable manifold backward until it reaches a radius from Earth equal to the LEO radius (`R_earth + altitude`) or its closest approach to Earth.
  2. At that point, convert to Earth-centered inertial velocity (1.10) and compute the impulsive delta-v from the circular LEO velocity (match flight-path angle by choosing the best point on the manifold; search over the manifold parameters `t_i`, `±`, and time along the trajectory).
  3. Minimize total delta-v subject to arriving on the manifold. Arrival at the halo is ballistic (asymptotic), plus a small insertion maneuver if you terminate at finite distance.
* Optional refinement: multiple shooting to turn the manifold-based first guess into a continuous, converged trajectory; or a Nelder-Mead/SLSQP search over patch-point parameters.
* Outputs: a table comparing, for direct vs low-energy: total delta-v (m/s), time of flight (days), and arrival state. Be honest about the result. Low-energy transfers typically trade longer flight time (weeks to months) for lower delta-v, mostly saved at lunar arrival; if your delta-v savings are small or negative in this model, report that and explain why. Do not tune results to look good.
* Figures: the full transfer in the rotating frame and in the Earth-centered inertial frame, delta-v vs flight time tradeoff curve, delta-v breakdown bar chart.
* Acceptance: the transfer trajectory is propagated end-to-end with the full nonlinear CR3BP, reaches the halo vicinity, and its Jacobi constant bookkeeping is consistent (impulse at departure changes `C`; after the burn it stays constant).

**Phase 12: Visualization, interactivity, packaging**
* Plotly interactive 3D HTML of the halo, manifolds, and final transfer (exportable single file).
* Animation (GIF or MP4) of the transfer in both frames.
* Consistent plotting style (dark and light variants), labeled axes with units (km and days alongside non-dimensional values).
* A single entry point: `python scripts/run_final_demo.py` reproduces all headline figures and the results table from scratch in a few minutes.
* A demo notebook walking through the pipeline.

**Phase 13: Documentation and presentation material**
* `README.md`: motivation, one hero figure, quickstart, results table, validation summary, limitations.
* `docs/math_notes.md`: derive the EOM in the rotating frame, the Jacobi constant, the variational equations, the differential-correction Jacobian (including the crossing-time correction), and the manifold construction, in my own words as much as possible.
* `docs/results.md`: final numbers, plots, and an honest limitations section (see Section 6).
* `docs/qa_prep.md`: answers to the defense questions in Section 7.

### 4. Extension tracks (choose after the core is done; ask me which)
1. Bicircular four-body model: add the Sun as a periodically moving third gravitational source. Re-propagate the transfer and show how the CR3BP solution degrades; then re-correct it with multiple shooting. Strong "real-world" payoff.
2. Ephemeris-level validation: load JPL DE440 via `spiceypy` or `skyfield`, convert your CR3BP transfer into an inertial ephemeris model, and measure the drift. (Requires downloading kernels; ask first.)
3. Station-keeping: quantify how fast a halo orbit departs (via `lambda_u`) and design a simple control law (e.g., target-point or LQR on the monodromy-derived dynamics) with a delta-v-per-year estimate.
4. Sun-Earth L2 (JWST-style): run the entire pipeline with `mu ≈ 3.0035e-6` and compare scale and stability differences with Earth-Moon. Cheap to do once the code is parameterized by `mu`, and a nice generalization demo.
5. Low-thrust transfer: add a constant-acceleration thrust term and compare against the impulsive solution.

### 5. Engineering standards
* All physics code takes `mu` as an explicit argument; no global state. This is what makes Sun-Earth reuse trivial.
* Functions are pure and unit-tested. Numeric tolerances in tests are named constants with a comment justifying each.
* Every numerical algorithm file starts with a docstring stating the equations, references, and assumptions.
* Reproducibility: fixed seeds where randomness appears, pinned dependencies, and `make figures` regenerates everything.
* Performance: profile after correctness. Use numba or vectorization for manifold generation (hundreds of trajectories) only once tests pass.
* Git: one commit per phase minimum, meaningful messages, tag `v0.1-core` after Phase 9 and `v1.0` at completion.
* References to keep in `docs/references.md` (verify each exists and cite what you actually used): Richardson (1980) halo orbit approximation; Howell (1984) halo family; Koon, Lo, Marsden, Ross, Dynamical Systems, the Three-Body Problem and Space Mission Design; Szebehely, Theory of Orbits; Parker and Chua, Practical Numerical Algorithms for Chaotic Systems; NASA/JPL Three-Body Periodic Orbit Catalog; published CAPSTONE and Gateway NRHO mission-design papers.

### 6. Honest-limitations checklist (must appear in results.md)
* CR3BP assumes circular Earth-Moon motion, a massless spacecraft, and no Sun, no solar radiation pressure, no non-spherical gravity. Real missions need an ephemeris model.
* Impulsive-burn assumption.
* Patched-conic baseline is an estimate, not a mission-grade number.
* Manifold results depend on `eps`; state the sensitivity you measured.
* Differences from published values and their likely causes.

### 7. Defense-prep questions
1. Why do we use a rotating frame, and what are the "fictitious" forces in the equations?
2. What does the Jacobi constant physically represent, and why is it the only conserved quantity?
3. Why are the collinear Lagrange points unstable but L4/L5 can be stable?
4. Why does differential correction need the STM, and where does the `(a/vy)` correction come from?
5. Why must the monodromy matrix have a pair of eigenvalues equal to 1?
6. What exactly is an invariant manifold, and why does riding one save fuel?
7. Why does the low-energy transfer take so much longer?
8. How do you know your integrator is accurate? What did the Jacobi drift plot show?
9. How would the result change in a four-body or ephemeris model?
10. What is the biggest source of error or uncertainty in your final delta-v number?

### 8. Final acceptance checklist
* [ ] All phases complete, all tests green (`pytest -q`).
* [ ] Core algorithms self-implemented and cross-checked against independent references.
* [ ] Lagrange points, a Lyapunov family, a halo family (and ideally the NRHO) validated against published or catalog values.
* [ ] Manifold tubes generated and verified (Jacobi conservation, asymptotic approach to the orbit).
* [ ] A complete LEO-to-halo transfer with a delta-v and time-of-flight table versus a direct-transfer baseline, reported honestly.
* [ ] Interactive 3D figure, animation, README, math notes, results, and defense-prep doc.
* [ ] One command reproduces the headline results.
* [ ] `PROGRESS.md` and `docs/learning_notes.md` are complete and current.

When everything above is checked, summarize the whole project in 10 lines: what was built, the headline numbers, the biggest surprise, and the most valuable next extension.
