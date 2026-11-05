# Cislunar Highways

designed by Aradhya Pasikanti

Low energy E2M transfers in the circular restricted three body problem (earth, moon, and spacecraft/satellite). Built from scratch and checked against independent references (NASA JPL)

### Overview
Spacecraft don't have to fly straight to the Moon. Between the Earth and Moon there are gravitational paths (cislunar highways) where gravity can carry spacecraft into orbits with the Moon (using 0 fuel consumption). This project finds cislunar highways mathematically, then heuristically plans a trip from Earth orbit to a lunar orbit. The highway route is currently down to **16 % less fuel** than flying direct but takes **168 days instead of 6**. All the physics and numerical methods are written from scratch and tested against independent references as well.

Logic Flow:
1. Finds the five Lagrange points (L1-L5) of the Earth–Moon system
2. Computes the halos around L1 and L2 + included the Gateway's near rectilinear orbit
3. Traces invariant manifolds which are the cislunar highways that lead orbits even with engine off
4. Uses manifolds to design an efficient fuel saving transfer from low Earth orbit

The integrators, root finder, differential corrector, continuation and manifold generation are hand written. SciPy is used as an independent cross check, Nelder–Mead optimizer, and labelling connected regions.

## Results

| Design | TLI | Second burn | **Total Δv** | Flight time |
|---|---|---|---|---|
| Conic → 100 km lunar orbit (textbook estimate) | 3,135 m/s | 824 m/s | **3,959 m/s** | 5 d |
| Direct → L2 halo | 3,161 m/s | 941 m/s | **4,103 m/s** | 6 d |
| Low energy, riding the stable manifold → L2 halo | 3,107 m/s | 325 m/s | **3,432 m/s** | 168 d |


Lagrange points are printed through the L2 halo orbit which are used to determine low energy transfers with manifold tubes.


