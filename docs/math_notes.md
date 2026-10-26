# Math notes

A derivation-level walkthrough of every algorithm in the code. It is a draft for you to rewrite in your own words: the spec asks for that, and explaining it yourself is the best defense prep.

## 1. Equations of motion in the rotating frame
Take an inertial frame and a frame rotating at constant angular velocity **ω** = (0, 0, 1) (non-dimensional units). For any position **r**, the transport theorem gives

- velocity: **v**_in = **v**_rot + **ω** × **r**
- acceleration: **a**_in = **a**_rot + 2 **ω** × **v**_rot + **ω** × (**ω** × **r**).

Newton's law in the inertial frame is **a**_in = −∇U, with U = −(1−μ)/r₁ − μ/r₂. Solving for **a**_rot:

**a**_rot = −∇U − 2 **ω** × **v** − **ω** × (**ω** × **r**).

With **ω** = ẑ:
- **Coriolis term:** −2 ẑ × **v** = (2ẏ, −2ẋ, 0).
- **Centrifugal term:** −ẑ × (ẑ × **r**) = (x, y, 0) = ∇(½(x² + y²)).

Collecting the gravity and centrifugal terms into one pseudo-potential gives **Ω** = ½(x² + y²) + (1−μ)/r₁ + μ/r₂ (+ μ(1−μ)/2, a constant we keep by convention). Then

ẍ − 2ẏ = Ω_x,  ÿ + 2ẋ = Ω_y,  z̈ = Ω_z.

## 2. The Jacobi integral
Dot the equations of motion with the velocity (ẋ, ẏ, ż). The Coriolis terms cancel: ẋ(−2ẏ) + ẏ(2ẋ) = 0. What remains is

ẋẍ + ẏÿ + żz̈ = Ω_x ẋ + Ω_y ẏ + Ω_z ż = dΩ/dt.

The left side is d(½v²)/dt, so ½v² − Ω is constant. Define **C = 2Ω − v²**. The code checks this numerically through `jacobi_rate` (< 1e-12).

**Hill regions:** v² ≥ 0 forces 2Ω ≥ C. Where 2Ω < C, motion is impossible.

## 3. Equilibria and their linear dynamics
Equilibria need **v** = 0 and ∇Ω = 0. On the x-axis this reduces to one equation per interval between the primaries, which we solve with Brent's method. L4 and L5 are equilateral by symmetry.

Linearize at L_i with ξ = x − x_L. The Hessian there is diagonal: Ω_xx = 1 + 2c, Ω_yy = 1 − c, Ω_zz = −c, where c = (1−μ)/|x_L+μ|³ + μ/|x_L−1+μ|³. The equations become:
- ξ̈ − 2η̇ = (1+2c)ξ
- η̈ + 2ξ̇ = (1−c)η
- ζ̈ = −cζ

Substituting e^{st} into the planar pair gives s⁴ + (2−c)s² + (1+c−2c²) = 0. This is a quadratic in u = s², with discriminant 9c² − 8c > 0. For c > 1 one root u₊ is positive (giving ±λ, the saddle) and one root u₋ is negative (giving ±iω_p, the center). The vertical motion is a center with ν = √c.

## 4. Variational equations and the State Transition Matrix
For **Ẋ** = f(**X**), a small deviation δ**X** obeys δ**Ẋ** = A(t) δ**X**, with A = ∂f/∂**X** = [[0, I], [Ω_XX, K]]. The STM Φ(t,0) maps δ**X**(0) to δ**X**(t), so

Φ̇ = A Φ,  Φ(0) = I.

**Structure checks:**
- **Volume:** trace A = 0. Liouville's formula det Φ(t) = exp(∫ tr A dt) then gives det Φ = 1.
- **Symplecticity:** with canonical momenta **p** = **v** + **ω** × **r**, the system is Hamiltonian, so Φ_can^T J Φ_can = J. Changing variables with T = ∂(q,p)/∂(q,v) = [[I, 0], [S, I]] turns this into Φ^T J_v Φ = J_v, where J_v = T^T J T = [[W, I], [−I, 0]] and W = S − S^T = [[0, −2, 0], [2, 0, 0], [0, 0, 0]]. Its infinitesimal form, A^T J_v + J_v A = 0, is tested at random states.

## 5. Differential correction with the crossing-time term
Start at **X**₀ = (x₀, 0, z₀, 0, ẏ₀, 0) and integrate to the next y = 0 crossing at time t_f. Perturb a free variable q_j:

- δy_f = Φ_{2j} δq_j + ẏ_f δt = 0  ⇒  δt = −Φ_{2j} δq_j / ẏ_f
- δẋ_f = Φ_{4j} δq_j + ẍ_f δt = (Φ_{4j} − (ẍ_f/ẏ_f) Φ_{2j}) δq_j

and the same for ż. This gives the 2×2 Newton matrix of spec §1.7 (1-based indices).

**Mirror theorem:** a trajectory that crosses the xz-plane perpendicularly twice is periodic. So driving ẋ_f = ż_f = 0 at the half period is enough.

**Fixed period:** add the equation t_f − T/2 = 0, whose row is dt_f/dq_j = −Φ_{2j}/ẏ_f.

**Fixed energy:** add C(**X**₀) − C* = 0, with dC/dx₀ = 2Ω_x and dC/dẏ₀ = −2ẏ₀.

## 6. Continuation
- **Natural continuation:** step x₀ and re-correct. The new guess comes from extrapolating the last two solutions, and the step is halved on failure or a period jump.
- **Pseudo-arclength:** the unknowns are q = (x₀, z₀, ẏ₀) with F(q) = (ẋ_f, ż_f) = 0. The solution set is a curve. Its tangent τ is the null vector of ∂F/∂q, found by SVD. The next point solves [F(q); (q − q_prev)·τ − Δs] = 0. Because the step is measured along the curve, the method passes folds where z₀ (or any single coordinate) turns back.
- **Halo bifurcation:** a second monodromy eigenvalue pair passes through +1 (ν_v = 1). We find it by bisection, then step off in z₀.

## 7. Monodromy matrix
M = Φ(T, 0).
- The flow vector f(**X**₀) satisfies M f = f: shifting along the orbit stays on the orbit.
- Since dC/d**X**₀ = ∇C^T M, conservation of C gives the left eigenvector ∇C^T M = ∇C^T.
- Symplecticity forces reciprocal pairs (λ, 1/λ).
- **Stability index:** ν = (λ + 1/λ)/2.
- For λ ≠ 1, the left/right eigenvector relation gives ∇C · v_λ = 0. So moving along an unstable or stable eigenvector does not change the energy to first order (the change is O(ε²)).

## 8. Manifolds and Poincaré sections
- **Transport:** v(t_i) = Φ(t_i, 0) v(0) is an eigenvector of the monodromy matrix based at t_i, because M_i = Φ(t_i) M Φ(t_i)⁻¹.
- **Initial states:** **X** = γ(t_i) ± ε v(t_i)/‖v_pos‖. Unstable strands are integrated forward and stable strands backward.
- **Role of ε:** one period maps ε to λ_u ε at the same phase. So changing ε by a factor of 10 shifts the departure by T ln10 / ln λ_u. It picks a different strand of the same tube.
- **Time-reversal symmetry:** the CR3BP is invariant under (x, y, z, ẋ, ẏ, ż, t) → (x, −y, z, −ẋ, ẏ, −ż, −t). This maps W^u of a symmetric orbit onto W^s. Our tube cuts are mirror images to the precision of their starting points.

## 9. Transfers
- **Screening conic:** a conic with perigee r₀ (tangential LEO departure) through a patch point at true anomaly ν has e = (r_p − r₀)/(r₀ − r_p cos ν) and p = r₀(1 + e). The burns are:
  - Δv₁ = √(μ/p)(1 + e) − √(μ/r₀)
  - Δv₂ = |**v**_m − (v_r r̂ + v_θ θ̂)|
- **CR3BP correction:** single shooting from LEO with unknowns (**v**_dep, TOF) and equations **r**(TOF) = **r**_patch. The 3×4 system is solved with the minimum-norm Newton step and a backtracking line search. With TOF fixed, Φ_rv was nearly singular (smallest singular value ~1e-3).
- **Frame invariance of Δv:** **v**_in = R(**v**_rot + **ω** × **r**). The **ω** × **r** term is the same before and after an impulse at the same point, so |Δ**v**_in| = |Δ**v**_rot|.

## 10. Predictability
Write the initial error as δ**X** = Σ c_k v_k over the monodromy eigenvectors. With w_u the left eigenvector for λ_u, c_u = (w_u · δ**X**)/(w_u · v_u). After n periods the unstable part dominates, so the spread is ≈ |c_u| λ_u^{t/T}. Time to reach a size d: t = T ln(d/|c_u|)/ln λ_u.

**Trajectory-correction manoeuvre (linear):** with Φ the STM from the correction time t_c to arrival, δ**r**_f = Φ_rr δ**r** + Φ_rv (δ**v** + Δ**v**) = 0 ⇒ Δ**v** = −Φ_rv⁻¹ (Φ_rr δ**r** + Φ_rv δ**v**).
