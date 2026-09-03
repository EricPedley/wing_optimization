"""Propeller mass/inertia scaling laws, the propeller-side counterpart to
motor_scaling.py.

Same purpose as that module: let the optimizer propose a diameter and blade
count and get a realistic mass and rotational inertia back, instead of being
limited to SimITL's five catalogue propellers. Calibration data lives in
data/prop_datasheets.csv. The fits here are visibly noisier than the motor
ones -- see mass_g_per_blade's docstring for why, and for a concrete case
where the trend is not even monotonic.
"""

import csv
from pathlib import Path

import jax.numpy as jnp

DATA_DIR = Path(__file__).parent / "data"

MM_PER_INCH = 25.4


def _read_prop_datasheets():
    with open(DATA_DIR / "prop_datasheets.csv", newline="") as f:
        return list(csv.DictReader(f))


def diameter_mm(diameter_in):
    """Convenience: most props are specified in inches, this module in mm."""
    return diameter_in * MM_PER_INCH


# --- Mass vs. diameter and blade count ---------------------------------------
#
# Seven datasheet points across five distinct diameters, two blade counts.
# Fit as mass_per_blade = c * diameter^p, then multiplied by blade count --
# i.e. each blade is assumed to weigh about the same regardless of how many
# other blades share the hub, which is the natural first assumption and is
# roughly borne out by the two same-diameter, different-blade-count pairs
# below (76.2mm: 0.425 vs 0.433 g/blade, ~2% apart; 45mm: 0.143 vs 0.165
# g/blade, ~13% apart) -- close enough that blade count is not the dominant
# source of scatter here.
#
# Diameter is: not everything else about a propeller is scattered -- 45mm and
# 60.96mm are both 3-blade and only 16mm apart, but the 45mm one is *lighter
# per blade* (0.143 g) than the 50.8mm one (0.233 g), which in turn is
# heavier per blade than the 60.96mm one (0.21 g). That last pair is not just
# noisy, it is non-monotonic: a strictly bigger propeller (60.96mm > 50.8mm)
# weighing less. Real propellers vary in chord, thickness, and hub/rib design
# independent of diameter -- a fit against diameter alone cannot see any of
# that, and this data shows it. The power-law fit below has R^2 ~ 0.77 in
# log-log space and the single worst point misses by ~40%. Treat any one
# prediction from this fit as good to a factor of ~1.4, not as a precise
# number -- fine for picking a rough design point, not for picking between
# two similar-sized props.
_PROP_ROWS = _read_prop_datasheets()
_CAL_DIAMETER_MM = jnp.array([float(row["diameter_mm"]) for row in _PROP_ROWS])
_CAL_BLADE_COUNT = jnp.array([float(row["blade_count"]) for row in _PROP_ROWS])
_CAL_MASS_G = jnp.array([float(row["mass_g"]) for row in _PROP_ROWS])
_CAL_MASS_PER_BLADE_G = _CAL_MASS_G / _CAL_BLADE_COUNT

_MASS_EXPONENT, _MASS_LOG_COEFFICIENT = jnp.polyfit(
    jnp.log(_CAL_DIAMETER_MM), jnp.log(_CAL_MASS_PER_BLADE_G), 1)
_MASS_COEFFICIENT_G = jnp.exp(_MASS_LOG_COEFFICIENT)


def prop_mass_kg(diameter_mm_, blade_count):
    """Propeller mass, kg, from the diameter/blade-count power-law fit above.

    See the calibration comment for how much scatter to expect: good to
    roughly a factor of 1.4 on any single prediction, not a precise number.
    """
    per_blade_g = _MASS_COEFFICIENT_G * diameter_mm_ ** _MASS_EXPONENT
    return per_blade_g * blade_count * 1e-3


# --- Rotational inertia, from mass and diameter -------------------------------
#
# Not fit from an independent inertia dataset -- propeller inertia is almost
# never published -- but derived geometrically as I = k * mass * radius^2,
# with the shape factor k (effectively (radius of gyration / radius)^2)
# calibrated against SimITL's own five catalogue propellers, which do carry
# both a mass and a propInertia field:
#
#   5.1x3.5x3   k = 0.483
#   4x3x3       k = 0.339
#   3x4x3       k = 0.685   (an outlier: unusually high pitch for its
#                             diameter, so more blade mass sits further out)
#   3.5x2.5x3   k = 0.474
#   5.1x3.7x3R  k = 0.483
#
# mean 0.493, +-22% (one standard deviation). A rigid-body mass distributed
# entirely at the radius would give k = 1; a point mass at the hub would give
# k = 0; 0.5 is what a lamina tapering roughly linearly from root to tip
# comes out to, so this is a physically sane middle-of-the-road number, not
# just an empirical curve fit. Reuse the SAME uncertainty caveat as the mass
# fit above: a single prediction is good to worse than +-20%.
_INERTIA_SHAPE_FACTOR = 0.493


def prop_inertia_kg_m2(mass_kg, diameter_mm_):
    """Rotational inertia about the shaft, kg.m^2, from mass and diameter."""
    radius_m = 0.5 * diameter_mm_ * 1e-3
    return _INERTIA_SHAPE_FACTOR * mass_kg * radius_m ** 2
