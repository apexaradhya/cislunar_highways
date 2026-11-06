"""Physical constants for the Earth-Moon CR3BP, with sources.

Everything derived (mu, length/time/velocity units) is *computed* from the
primary constants below, never typed in by hand.

Sources
-------
GM_EARTH_KM3_S2 = 398600.4418 km^3/s^2
    IERS Conventions (2010), IERS Technical Note 36, Table 1.1
    (GM_earth = 3.986004418e14 m^3/s^2, TT-compatible). Same value as WGS-84/EGM-96.
GM_MOON_KM3_S2 = 4902.8000 km^3/s^2
    Value given in PROJECT_SPEC.md; consistent with the JPL planetary ephemerides
    (DE430/DE440 lunar GM ~4902.80 km^3/s^2) to the 5 digits quoted.
EARTH_MOON_DISTANCE_KM = 384400 km
    Conventional mean Earth-Moon distance (NASA Moon Fact Sheet, "semimajor axis
    384.4 x 10^6 m"). The CR3BP idealizes this as a constant circular separation.
R_EARTH_KM = 6378.1363 km   (IERS Conventions 2010, Table 1.1, equatorial radius)
R_MOON_KM = 1737.4 km       (NASA Moon Fact Sheet, volumetric mean radius)
SECONDS_PER_DAY = 86400 s   (SI day)

Note on mu: these GM values give mu = 0.01215058392, while the JPL Three-Body
Periodic Orbit Catalog uses mu = 0.0121505856 (from a different GM dataset).
The two agree to 7 significant digits. All physics functions take mu as an
argument, so a comparison against the catalog uses the catalog's own mu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GM_EARTH_KM3_S2: float = 398600.4418
GM_MOON_KM3_S2: float = 4902.8000
EARTH_MOON_DISTANCE_KM: float = 384400.0
R_EARTH_KM: float = 6378.1363
R_MOON_KM: float = 1737.4
SECONDS_PER_DAY: float = 86400.0


@dataclass(frozen=True)
class SystemUnits:
    """Characteristic units of a CR3BP system.

    mu = GM2 / (GM1 + GM2)
    L  = primary separation [km]
    T  = 1/n with n = sqrt((GM1 + GM2) / L^3) [s]   (so the frame rotates at rate 1)
    V  = L / T [km/s]
    """

    mu: float
    length_km: float
    time_s: float
    gm1_km3_s2: float
    gm2_km3_s2: float

    @property
    def velocity_km_s(self) -> float:
        return self.length_km / self.time_s

    @property
    def time_days(self) -> float:
        return self.time_s / SECONDS_PER_DAY


def make_system(gm1_km3_s2: float, gm2_km3_s2: float, length_km: float) -> SystemUnits:
    """Build the unit system for two primaries with gravitational parameters GM1 >= GM2."""
    gm_total = gm1_km3_s2 + gm2_km3_s2
    mean_motion = math.sqrt(gm_total / length_km**3)  # rad/s, Kepler's third law
    return SystemUnits(
        mu=gm2_km3_s2 / gm_total,
        length_km=length_km,
        time_s=1.0 / mean_motion,
        gm1_km3_s2=gm1_km3_s2,
        gm2_km3_s2=gm2_km3_s2,
    )


EARTH_MOON: SystemUnits = make_system(GM_EARTH_KM3_S2, GM_MOON_KM3_S2, EARTH_MOON_DISTANCE_KM)

# Mean synodic month (new moon to new moon). Explanatory Supplement to the
# Astronomical Almanac (3rd ed., 2013), ch. 3; also NASA Moon Fact Sheet.
SYNODIC_MONTH_DAYS: float = 29.530589
