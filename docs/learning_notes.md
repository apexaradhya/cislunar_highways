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
