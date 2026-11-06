"""Conversions between non-dimensional CR3BP units and km, s, km/s, days."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cislunar.constants import EARTH_MOON, SECONDS_PER_DAY, SystemUnits


def length_to_km(x: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float) * units.length_km


def km_to_length(x_km: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(x_km, dtype=float) / units.length_km


def velocity_to_km_s(v: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(v, dtype=float) * units.velocity_km_s


def km_s_to_velocity(v_km_s: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(v_km_s, dtype=float) / units.velocity_km_s


def time_to_days(t: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(t, dtype=float) * units.time_s / SECONDS_PER_DAY


def days_to_time(t_days: ArrayLike, units: SystemUnits = EARTH_MOON) -> NDArray[np.float64]:
    return np.asarray(t_days, dtype=float) * SECONDS_PER_DAY / units.time_s
