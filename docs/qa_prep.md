# Defense prep: the ten questions

Short answers in plain language, each backed by a number from this project. Practise saying each in under a minute.

## 1. Why use a rotating frame, and what are the "fictitious" forces?
In a frame that turns with the Moon, Earth and Moon stand still. The gravity field then doesn't change with time, so there is an energy-like quantity that stays constant (the Jacobi constant) and there are fixed equilibrium points. The price is two extra terms.
- **Centrifugal term** (`x, y` in the equations): pushes outward, away from the spin axis.
- **Coriolis term** (`2y'`, `−2x'`): pushes sideways on anything that moves.

Neither is a real force. Both appear only because the camera spins. In the frame turning once per 27.3-day sidereal month, they produce the halo orbits and tubes.

## 2. What does the Jacobi constant mean, and why is it the only conserved quantity?
`C = 2Ω − v²` is minus twice the energy measured in the rotating frame. Lower C means more energy. It is conserved because the field is time-independent in that frame and the Coriolis force does no work: it is always perpendicular to the velocity. Ordinary energy and angular momentum are *not* conserved, because the Moon keeps moving and tugs the spacecraft. In general the restricted three-body problem has no other conserved quantity: Poincaré showed it is non-integrable, which is why its motion can be chaotic. Our integrator holds C to 7e-13 over 87 days.

## 3. Why are L1–L3 unstable while L4/L5 can be stable?
At L1, L2 and L3, the gravity "landscape" (2Ω) is a saddle. In one direction a nudge grows exponentially: at L1, by a factor e every 1.48 days. L4 and L5 sit on hilltops of 2Ω, which you'd also expect to be unstable. But there the Coriolis force bends a drifting spacecraft sideways into a small loop around the point. This works only if the smaller body is light enough: 27μ(1−μ) < 1, i.e. μ < 0.0385. Earth–Moon (27μ(1−μ) = 0.324) passes; we confirmed it from the eigenvalues for seven values of μ.

## 4. Why does differential correction need the STM, and where does the (a/v_y) term come from?
We want a starting velocity that makes the orbit cross the x-axis perpendicularly (vx = 0). The STM tells us how a small change in the start changes the end, so we can solve for the right correction with Newton's method. The (a/v_y) term exists because changing the start also changes *when* we hit y = 0. From y_f = 0: `dt = −Φ₂ⱼ dq / v_y`. In that extra time, vx changes by `a_x · dt`. Leave it out and Newton converges slowly or not at all. With it, our orbits converge in 2–3 iterations to residuals of 1e-13.

## 5. Why must the monodromy matrix have two eigenvalues equal to 1?
- **Sliding along the orbit:** start a little further along it, and after one period you are exactly as far along again. So the flow direction is an eigenvector with eigenvalue 1.
- **Energy:** the gradient of C is a left eigenvector with eigenvalue 1, because energy is conserved.
- **Reciprocal pairs:** the dynamics are symplectic, so eigenvalues come in pairs (λ, 1/λ), and the partner of 1 is also 1.

We verified M f = f and ∇Cᵀ M = ∇Cᵀ to 1e-7. The two individual eigenvalues come out split by up to 1e-5. That is expected: a repeated eigenvalue with only one eigenvector is numerically sensitive, and we explain why in the results.

## 6. What is an invariant manifold, and why does riding one save fuel?
The stable manifold is the set of all paths that approach the halo orbit as time goes on. It forms a tube: start on it and you land on the halo with the engine off. "Invariant" means a path on the tube stays on the tube forever. It saves fuel because the spacecraft arrives already matched to the halo, instead of arriving fast and braking. Our direct transfer brakes with 941 m/s; the manifold transfer pays only 325 m/s to join the tube.

## 7. Why does the low-energy transfer take so much longer?
Near the halo the stable manifold approaches slowly: the distance shrinks only by a factor of about 1,000 per 15-day lap. Before that, the path wanders the Earth–Moon system for months, using many weak gravitational nudges instead of one big burn. You trade time for fuel. Our best design takes 168 days instead of 6 and saves 671 m/s. The Pareto front shows the trade: about 3,660 m/s at 24 days, 3,450 m/s at 143 days.

## 8. How do you know your integrator is accurate?
- **Order of accuracy:** halving the step cuts the error by the right factors (measured orders 4.17 for RK4 and 4.96 for Dormand–Prince).
- **Independent check:** our Dormand–Prince matches scipy's DOP853 to 5e-12.
- **Conserved quantity:** the Jacobi-drift plot shows drift falling in step with the tolerance, from 1.6e-5 at tolerance 1e-6 to 7e-13 at 1e-13.
- **Symmetry:** the integrator is exactly time-reversible. Running it backward from a mirrored state reproduces the forward result bit for bit.

## 9. How would the result change in a four-body or ephemeris model?
- The Moon's orbit is slightly elliptical (e ≈ 0.055), and the Sun's gravity is about half the Moon's tidal effect at these distances. Halo orbits would no longer repeat exactly; they become quasi-periodic, and station-keeping costs a few to tens of m/s per year.
- The biggest change is to transfers. With the Sun, low-energy paths can pass through Sun–Earth manifolds and lose energy far from Earth (the CAPSTONE-style ballistic lunar transfer). That typically saves more arrival Δv than our Earth–Moon-only result.
- Our CR3BP solution would be the first guess for re-correcting in the higher-fidelity model using multiple shooting.

## 10. What is the biggest source of error in your final Δv?
Not numerics: integration error is around 1e-10. The largest uncertainty is the **model**, mainly the missing Sun, which could change the low-energy design substantially. After that:
- **Search:** we found good solutions, not a proven optimum. Nelder–Mead refinement changed the result by about 10 m/s.
- **Ignored details:** the impulsive-burn assumption and the free choice of parking-orbit plane.
- **Navigation:** a 1 m/s TLI error costs a median 6.4 m/s correction one day later, which a real budget must include.
