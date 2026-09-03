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
# 34 datasheet points across the 40-76mm diameter range, most from real
# vendor SKUs (Gemfan, HQProp) -- widened from an original 7 hand-picked
# points. Fit as mass_per_blade = c * diameter^p, then multiplied by blade
# count -- i.e. each blade is assumed to weigh about the same regardless of
# how many other blades share the hub, which is the natural first assumption
# and is roughly borne out where same-diameter/different-blade-count pairs
# exist in the data (e.g. the 45mm Gemfan pair: 0.165 vs 0.143 g/blade, ~13%
# apart) -- close enough that blade count is not the dominant source of
# scatter here.
#
# Diameter is: not everything else about a propeller is scattered -- material
# (ultralight polycarbonate vs. standard), rib/hub design, and freestyle vs.
# whoop-class geometry all vary independent of diameter, and this data shows
# it (e.g. HQProp's "Ultralight" line runs consistently 20-35% lighter per
# blade than a same-diameter standard prop). The worst single calibration
# point misses the fit by ~1.6x. Treat any one prediction from this fit as
# good to roughly that factor, not as a precise number -- fine for picking a
# rough design point, not for picking between two similar-sized props. This
# is exactly the situation nearest_catalogue_prop below is for: when a real
# part's own mass matters more than the fit's extrapolation.
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


# --- Nearest real, buyable propeller ------------------------------------------
#
# The fits above answer "what would a prop at this (diameter, blade_count) be
# like" for a continuous optimizer; this answers "what actual part should I
# order" -- the closest prop in data/prop_datasheets.csv by normalized
# distance in (diameter, pitch, blade_count), the propeller-side counterpart
# to motor_scaling.nearest_catalogue_motor.
#
# Only rows with a recorded pitch are eligible: the original 7 calibration
# points (kept for the mass fit above) predate this catalogue and have no
# pitch on record, so they cannot be meaningfully compared against a
# (diameter, pitch, blade_count) design point.
#
# This lookup is also the tool for catching the failure mode this catalogue
# was specifically built to check: prop_aero_model.py's BEMT fit has no
# built-in penalty for an unrealistic pitch/diameter ratio (unlike
# quad_model.MAX_TIP_MACH, which exists precisely because the model was once
# caught extrapolating a similar way -- see that constant's docstring
# history). Real open (non-ducted) props at this diameter range top out
# around P/D ~1.1; ducted cinewhoop props reach ~1.2. A "nearest" match with
# a large distance, or a design point with P/D well above ~1.2, is a signal
# the continuous optimizer has wandered into a region with no real part to
# back it up, not a legitimately efficient design.
_DIAMETER_SCALE_MM = 15.0
_PITCH_SCALE_MM = 15.0
_BLADE_SCALE = 1.0


def pitch_to_diameter_ratio(diameter_mm_, pitch_mm_):
    """P/D -- see the module comment above for why this ratio matters: it is
    the single number most predictive of whether a design point corresponds
    to anything real vendors sell in this size class."""
    return pitch_mm_ / jnp.maximum(diameter_mm_, 1e-9)


def nearest_catalogue_prop(diameter_mm_, pitch_mm_, blade_count, n=3):
    """The n closest real props in data/prop_datasheets.csv, nearest first.

    Distance is normalized Euclidean in (diameter, pitch, blade_count) --
    _DIAMETER_SCALE_MM/_PITCH_SCALE_MM/_BLADE_SCALE are rough "how much of a
    difference matters" scales, not a fit, so treat the ranking as
    approximate near ties -- same caveat as
    motor_scaling.nearest_catalogue_motor.
    """
    diameter_mm_ = float(diameter_mm_)
    pitch_mm_ = float(pitch_mm_)
    blade_count = float(blade_count)

    scored = []
    for row in _PROP_ROWS:
        if not row["pitch_mm"]:
            continue
        row_diameter = float(row["diameter_mm"])
        row_pitch = float(row["pitch_mm"])
        row_blades = float(row["blade_count"])
        dist = (
            ((diameter_mm_ - row_diameter) / _DIAMETER_SCALE_MM) ** 2
            + ((pitch_mm_ - row_pitch) / _PITCH_SCALE_MM) ** 2
            + ((blade_count - row_blades) / _BLADE_SCALE) ** 2
        )
        scored.append({
            "name": row["name"],
            "vendor": row["vendor"],
            "diameter_mm": row_diameter,
            "pitch_mm": row_pitch,
            "blade_count": row_blades,
            "mass_g": float(row["mass_g"]),
            "hub_bore_mm": float(row["hub_bore_mm"]) if row["hub_bore_mm"] else None,
            "pitch_to_diameter": row_pitch / row_diameter,
            "source_url": row["source_url"],
            "notes": row["notes"],
            "distance": dist ** 0.5,
        })
    scored.sort(key=lambda s: s["distance"])
    return scored[:n]
