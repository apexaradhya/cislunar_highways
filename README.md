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

Low earth orbits at 185 km to an L2 halo computed in the full CR3B problem:

| Design | TLI | Second burn | **Total Δv** | Flight time |
|---|---|---|---|---|
| Conic → 100 km lunar orbit (textbook estimate) | 3,135 m/s | 824 m/s | **3,959 m/s** | 5 d |
| Direct → L2 halo | 3,161 m/s | 941 m/s | **4,103 m/s** | 6 d |
| Low energy, riding the stable manifold → L2 halo | 3,107 m/s | 325 m/s | **3,432 m/s** | 168 d |


- **Saving:** the manifold route saves 671 m/s (16 %) but takes 27× longer. Sweeping 546 transfers across 7 halos found savings of 450–660 m/s for L2 halos but only 40–150 m/s for L1 halos
- **Connections:** verified zero-Δv path (through NASA JPL) from an L1 orbit to an L2 orbit takes around 48.5 days
- **Gateway orbit:**  9:2 approximately rectilinear halo orbit repeats roughly every 6.562 days with perilune radius 3249 km and apolune radius 71222 km. Gateway values are about 3200–3560 km and 70000–71850 km.
- **Predictions:** linear theory forecast how fast 500 very very slightly perturbed spacecraft spread apart from about 10 km to 10000 km within half a day.


Lagrange points are printed through the L2 halo orbit which I used to determine low energy transfers with its manifold tubes.


Please note that this project was designed and built by myself (Aradhya Pasikanti), but claude was used in the development of the following technical aspects (all qualitative design done by me): claude implemented the halo-orbit solvers and the invariant-manifold with Poincaré-section tools. In addition, it also facilated in compiling and cross checking the data of the 106 tests with NASA JPL's Gateway data.
