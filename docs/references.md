# References

What each source was used for. These were cited from prior knowledge. This environment could not reach publisher sites, NTRS or JPL to re-check them, so verify bibliographic details before submission. Items marked *unverified* were seen only in a web-search summary.

## Models and dynamics
- Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies.* Academic Press. CR3BP equations, Jacobi constant, Lagrange points, Routh criterion, zero-velocity curves.
- Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2011). *Dynamical Systems, the Three-Body Problem and Space Mission Design.* Marsden Books. Linear dynamics at L1/L2, Hill regions, tubes, Poincaré cuts, the literature values for the Earth–Moon Lagrange points.
- Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2000). Heteroclinic connections between periodic orbits and resonance transitions in celestial mechanics. *Chaos* 10(2), 427–469. Tube cuts and heteroclinic connections (Phase 10).
- Gómez, G., Koon, W. S., Lo, M. W., Marsden, J. E., Masdemont, J., & Ross, S. D. (2004). Connecting orbits and invariant manifolds in the spatial restricted three-body problem. *Nonlinearity* 17, 1571–1606. Manifold construction from eigenvectors (Phase 9).
- Howell, K. C. (1984). Three-dimensional, periodic, "halo" orbits. *Celestial Mechanics* 32, 53–71. Symmetric differential correction; halo families.
- Grebow, D. J. (2006). *Generating Periodic Orbits in the Circular Restricted Three-Body Problem with Applications to Lunar South Pole Coverage.* M.S. thesis, Purdue University. Halo bifurcation from the Lyapunov family; NRHO regime.
- Parker, J. S., & Anderson, R. L. (2014). *Low-Energy Lunar Trajectory Design.* JPL Deep Space Communications and Navigation Series. Context on low-energy and ballistic lunar transfers.
- Parker, T. S., & Chua, L. O. (1989). *Practical Numerical Algorithms for Chaotic Systems.* Springer. Shooting and continuation; sensitivity and predictability.
- Richardson, D. L. (1980). Analytic construction of periodic orbits about the collinear points. *Celestial Mechanics* 22, 241–253. Listed in the spec but **not used**: halos were seeded from the Lyapunov bifurcation instead.

## Numerical methods
- Dormand, J. R., & Prince, P. J. (1980). A family of embedded Runge–Kutta formulae. *J. Comput. Appl. Math.* 6(1), 19–26. DOPRI5 tableau.
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary Differential Equations I* (2nd ed.). Springer. Step-size control, initial step, PI controller, dense-output coefficients (code `dopri5.f`).
- Shampine, L. F. (1986). Some practical Runge–Kutta formulas. *Mathematics of Computation* 46, 135–150. Continuous extension for DOPRI5.
- Brent, R. P. (1973). *Algorithms for Minimization without Derivatives.* Prentice-Hall; and Forsythe, Malcolm & Moler (1977), *Computer Methods for Mathematical Computations*, `zeroin`. Root finder.
- Keller, H. B. (1977). Numerical solution of bifurcation and nonlinear eigenvalue problems. In *Applications of Bifurcation Theory*, Academic Press. Pseudo-arclength continuation.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra.* SIAM, Lecture 25. Sensitivity of defective eigenvalues (trivial monodromy pair).

## Astrodynamics and constants
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications* (4th ed.). Conic geometry and Kepler's equation for the screening conics.
- Curtis, H. D. (2014). *Orbital Mechanics for Engineering Students* (3rd ed.), ch. 8; Bate, R. R., Mueller, D. D., & White, J. E. (1971). *Fundamentals of Astrodynamics*, ch. 7. Patched-conic baseline.
- Petit, G., & Luzum, B. (eds.) (2010). *IERS Conventions (2010)*, IERS Technical Note 36. GM of Earth, Earth radius.
- NASA Moon Fact Sheet (NSSDCA). Mean Earth–Moon distance, lunar radius, synodic month.
- Davis, D. C., et al. (2020). AAS/AIAA conference paper on the Gateway NRHO and its osculating parameters (**exact title to confirm**). Gateway NRHO perilune and apolune ranges. **Unverified:** the numbers came from a search summary; the PDF (ntrs.nasa.gov) was unreachable. Confirm the exact title and values.
- NASA/JPL Solar System Dynamics, Three-Body Periodic Orbits catalog (ssd.jpl.nasa.gov/tools/periodic_orbits.html). **Not accessed:** the host was blocked by this environment's network policy. Catalog-comparison tests are written and skip until entries are provided.
