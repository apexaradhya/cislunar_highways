# Learning notes

One short note per phase: the idea in plain language, an analogy, the key equation, and one question you should be able to answer.

## Phase 1: The rotating frame and the Jacobi constant
**Idea.** We spin our camera at the same rate the Moon goes around Earth, so both stay parked on the x-axis. The price is two "fake" forces. The centrifugal force pushes things outward. The Coriolis force pushes sideways on anything that moves, which is why the equations contain `2y'` and `-2x'`.
**Analogy.** Rolling a marble across a spinning merry-go-round: to someone riding it, the marble's path curves, even though nothing pushes it.
**Key equation.** `C = 2Ω − v²`, where `Ω = (x²+y²)/2 + (1−μ)/r₁ + μ/r₂ (+ μ(1−μ)/2)`. C is a kind of "energy" in the rotating frame and never changes along a trajectory. The Coriolis force always pushes sideways (perpendicular to the motion), so, like a magnet bending a moving charge, it does no work.
**Check yourself.** Why doesn't the Coriolis term show up in dC/dt? (Answer: the Coriolis force is perpendicular to v, so v·(Coriolis) = 2(vx·vy − vy·vx) = 0.)

## Phase 2: Numerical integration
**Idea.** A computer can't solve the CR3BP exactly, so it takes many small steps. RK4 samples the slope four times per step. Dormand–Prince computes a 5th-order and a 4th-order answer at the same time. The difference between them estimates the error, so it can shrink its steps near the Moon and stretch them out in empty space.
**Analogy.** Driving on a winding road at night: slow down in the curves, speed up on the straights, and keep checking the GPS.
**Key equation.** Error ∝ hᵖ. Halving the step cuts RK4's error 16× (p = 4) and Dormand–Prince's 32× (p = 5). We measured slopes of 4.17 and 4.96.
**Check yourself.** How do you know the integrator is right without knowing the true answer? (Answer: C must stay constant. Our drift fell from 1.6×10⁻⁵ to 7×10⁻¹³ as the tolerance tightened, and we agree with scipy's independent DOP853 to 5×10⁻¹².)

## Phase 3: Lagrange points
**Idea.** In the rotating frame, there are five spots where Earth's pull, the Moon's pull and the centrifugal push all cancel out exactly. Put a spacecraft there at rest and, in a perfect world, it stays put. Three of them (L1, L2, L3) lie on the Earth–Moon line. Two (L4, L5) each form an equilateral triangle with Earth and the Moon.
**Analogy.** Points on a hilly landscape where the ground is perfectly flat. L1–L3 are like mountain passes (saddles): flat, but step off the wrong way and you roll away. L4/L5 are hilltops. You'd expect them to be unstable too, but the Coriolis force keeps a small drift curving around the point, as long as the Moon is light enough: 27μ(1−μ) < 1.
**Key equation.** `x − (1−μ)(x+μ)/|x+μ|³ − μ(x−1+μ)/|x−1+μ|³ = 0`, solved with Brent's method. L1 is 58,019 km from the Moon and L2 is 64,515 km.
**Check yourself.** Why does each of the three intervals contain exactly one collinear point? (Answer: f goes from −∞ to +∞ across each interval and is always increasing, since f′ = 1 + 2c > 0.)

## Phase 4: Hill regions (zero-velocity curves)
**Idea.** Since C = 2Ω − v² and v² can't be negative, a spacecraft with a given C can only go where 2Ω ≥ C. Everywhere else is a "forbidden zone". As you add energy (lower C), the walls retreat, and "necks" open, first at L1 (Earth ↔ Moon), then L2 (out past the Moon), then L3.
**Analogy.** A ball rolling in a bowl with a dent in the rim. With too little energy it can't get over the dent. With just a bit more, there's a narrow pass. L1 is that pass, and it's exactly where low-energy transfers squeeze through.
**Key equation.** Forbidden region: `2Ω(x, y, 0) < C`. The neck at Lᵢ opens when C < C(Lᵢ).
**Check yourself.** We found where the necks open using only geometry (flood-filling the allowed region on a grid). Why is it convincing that the results match C(L1), C(L2), C(L3) from Phase 3 to within 10⁻⁵? (Answer: the two methods are independent, one root-finding and one picture-based, so agreement is strong evidence both are right.)

## Phase 5: Linear stability at L1, L2, L3
**Idea.** Zoom in close enough to a Lagrange point and the curved gravity field looks like a simple "tilted saddle". The motion then splits into three independent pieces:
- one direction where small errors grow exponentially (the saddle, rate λ)
- an in-plane oscillation (frequency ω_p)
- an up-and-down oscillation (frequency ν).

That's "saddle × center × center".
**Analogy.** A marble on a horse saddle: nudge it front-to-back and it wobbles back; nudge it side-to-side and it falls off. The two "center" directions are where halo and Lyapunov orbits live. The saddle direction is where the manifold "highways" come from.
**Key equation.** `s⁴ + (2 − c)s² + (1 + c − 2c²) = 0`, giving `λ² = (c − 2 + √(9c² − 8c))/2`. At L1, λ = 2.93, so errors grow by e every 1.48 days.
**Check yourself.** Why do halo orbits near L1 need regular station-keeping burns, while JWST-style orbits around Sun–Earth L2 need them only every few weeks? (Hint: λ in physical units depends on the system's time unit: 4.34 days for Earth–Moon, ~58 days for Sun–Earth.)

## Phase 6: The State Transition Matrix (STM)
**Idea.** The STM Φ is a 6×6 "sensitivity table". Entry Φᵢⱼ tells you how much final coordinate i changes when you nudge initial coordinate j. It answers "if I aim 1 km to the left, where do I end up?" without re-flying the trajectory. We get it by integrating 36 extra equations, `dΦ/dt = A(t) Φ`, alongside the trajectory.
**Analogy.** A ripple map. Drop a pebble (small error) at the start, and Φ tells you how big the ripple is, and in which direction it points, at the end.
**Key equation.** `δX(t) ≈ Φ(t,0) δX(0)`. Two built-in sanity checks: det Φ = 1 (phase-space volume is never squashed, because trace A = 0) and `ΦᵀJΦ = J` (the dynamics are symplectic, a "Hamiltonian fingerprint").
**Check yourself.** Why do we check Φ against finite differences, and why don't they agree perfectly? (Answer: finite differences are an independent, brute-force estimate. They carry their own truncation and round-off error, about 10⁻⁹ relative here.)

## Phase 7: Planar Lyapunov orbits and differential correction
**Idea.** A periodic orbit is a path that closes on itself. We guess a starting point on the x-axis, fly until we cross the x-axis again (half an orbit), and check whether we crossed it perpendicularly (vx = 0). If we didn't, the STM tells us how to tweak the starting speed vy₀ to fix it. Repeat (Newton's method). By the mirror symmetry of the CR3BP, a perpendicular crossing guarantees the orbit closes.
**Analogy.** Throwing a ball at a wall so it bounces straight back. Miss the angle a bit, see how far off you were, adjust the throw, and repeat. Two or three throws are enough.
**Key equation.** `δvx_f = [Φ₄₅ − (a_x/v_y) Φ₂₅] δvy₀`. The `(a_x/v_y)` term exists because changing vy₀ also changes *when* you cross the axis.
**Check yourself.** Why can't we just take a big orbit's starting point from the linear formula? (Answer: the linear solution is only accurate for small orbits. For big ones, Newton's method starts too far away and diverges. That's why we "walk" out step by step, which is called continuation.)
