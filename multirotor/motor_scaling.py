"""Scaling laws that stand in for testing discrete catalogue motors.

motor_model.py evaluates a motor given (kV, R, I0, Rth, mass); this module is
what lets the optimizer propose a stator size and kV and get realistic values
for the rest, instead of being limited to the handful of motors SimITL ships.
Everything here is calibrated from vendor datasheet points in
data/motor_datasheets.csv -- see each fit's docstring for how many points and
distinct stator sizes back it, and what to fix first.
"""

import csv
from pathlib import Path

import jax.numpy as jnp

import multirotor.motor_model as mm

DATA_DIR = Path(__file__).parent / "data"


def _read_motor_datasheets():
    with open(DATA_DIR / "motor_datasheets.csv", newline="") as f:
        return list(csv.DictReader(f))


def stator_volume_mm3(diameter_mm, height_mm):
    """Stator volume from the size code FPV motor part numbers already encode.

    A motor named "1202.5" is a 12mm-diameter, 2.5mm-tall stator (vendors
    write the half-mm digit after a decimal, e.g. "1202.5" not "120205"); a
    "1002" is 10mm x 2mm. This is what makes stator volume close to a free
    variable -- most catalogue motors already tell you theirs in the name.
    """
    return diameter_mm ** 2 * height_mm


# --- Mass vs. stator volume --------------------------------------------------
#
# Calibrated from every datasheet row in data/motor_datasheets.csv that has a
# mass (most rows do; R and I0 are frequently missing from vendor listings,
# but weight almost always is published). As of this writing that is 19
# points across 8 distinct stator sizes (0802, 1002, 1102, 1103, 1104,
# 1202.5, 1203, 1204) -- a real improvement on the original two-point line
# (1002 and 1202.5 only), though still a simple linear fit with no claim that
# the relationship truly is linear across this whole size range.
#
# The intercept is a genuine least-squares fit now rather than forced through
# two points, so unlike the original two-point version it is not
# automatically ~0 -- see the module docstring history in git for that prior
# caveat, which more data was expected to resolve.
_MOTOR_ROWS = _read_motor_datasheets()
_CAL_VOLUME_MM3 = jnp.array([
    stator_volume_mm3(float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
    for row in _MOTOR_ROWS if row["mass_g"]
])
_CAL_MASS_G = jnp.array([float(row["mass_g"]) for row in _MOTOR_ROWS if row["mass_g"]])

_MASS_SLOPE_G_PER_MM3, _MASS_INTERCEPT_G = jnp.polyfit(_CAL_VOLUME_MM3, _CAL_MASS_G, 1)


def motor_mass_kg(volume_mm3):
    """Linear mass(volume) fit, in kg, from the calibration data above."""
    grams = _MASS_SLOPE_G_PER_MM3 * volume_mm3 + _MASS_INTERCEPT_G
    return grams * 1e-3


# --- Motor constant (Km) vs. stator volume -----------------------------------
#
# Km = Kt / sqrt(R) is, to first order, a property of the physical stator
# rather than the winding: rewinding a motor with more turns scales
# Kt ~ 1/turns and R ~ turns^2, so their ratio cancels the turns dependence.
# Fitting Km against volume rather than fitting R directly is what lets kV
# and stator size be independent free variables in the optimizer while R
# still comes out physically consistent for whichever kV is chosen -- see
# motor_resistance_ohm below.
#
# Calibrated from every datasheet row in data/motor_datasheets.csv that has
# both a kV and a resistance -- as of this writing 12 points across 6 distinct
# stator sizes (0802, 1002, 1103, 1104, 1203, 1204), up from the original 6
# points across 2 sizes (1002, 1203 only). More sizes is exactly the gap the
# original calibration comment called out as the priority fix, since two
# sizes cannot distinguish a real power-law exponent from a line through two
# points. Within-size agreement (Km roughly constant across kV at fixed
# volume) should be re-checked whenever this list grows -- see
# test_km_is_roughly_constant_within_a_stator_size.
_CAL_KV_RPM_PER_V = jnp.array([
    float(row["kv_rpm_per_v"]) for row in _MOTOR_ROWS if row["resistance_ohm"]
])
_CAL_R_OHM = jnp.array([
    float(row["resistance_ohm"]) for row in _MOTOR_ROWS if row["resistance_ohm"]
])
_CAL_VOLUME_MM3_KM = jnp.array([
    stator_volume_mm3(float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
    for row in _MOTOR_ROWS if row["resistance_ohm"]
])

_CAL_KT = mm.KT_NUMERATOR / _CAL_KV_RPM_PER_V
_CAL_KM = _CAL_KT / jnp.sqrt(_CAL_R_OHM)

# Power-law fit Km = c * volume^a, done as a linear regression in log-log
# space across all calibration rows above (not per-size means -- sizes with
# more windings on record get proportionally more weight in the fit).
_KM_EXPONENT, _KM_LOG_COEFFICIENT = jnp.polyfit(
    jnp.log(_CAL_VOLUME_MM3_KM), jnp.log(_CAL_KM), 1)
_KM_COEFFICIENT = jnp.exp(_KM_LOG_COEFFICIENT)


def motor_constant_km(volume_mm3):
    """Km = Kt / sqrt(R), N.m/sqrt(W), from the volume power-law fit above."""
    return _KM_COEFFICIENT * volume_mm3 ** _KM_EXPONENT


def motor_resistance_ohm(kv, volume_mm3):
    """Phase resistance implied by a chosen kV on a given stator volume.

    Inverts Km = Kt / sqrt(R): R = (Kt(kv) / Km(volume))^2. This is what lets
    the optimizer treat kV and stator volume as independent free variables
    while R comes out consistent with both, instead of R being its own free
    (and physically ungrounded) variable.
    """
    kt = mm.motor_constant(kv)
    km = motor_constant_km(volume_mm3)
    return (kt / jnp.maximum(km, 1e-12)) ** 2


# --- Nearest real, buyable motor ---------------------------------------------
#
# The fits above answer "what would a motor at this (kV, stator volume) be
# like" for an optimizer searching a continuous box; this answers the
# different question "what actual part should I order" -- the closest motor
# in data/motor_datasheets.csv by normalized distance in (kV, stator volume),
# with its real datasheet mass/R/I0 rather than the fitted estimate, so a
# reported TWR/current/etc. can be re-checked against a buildable part
# instead of an interpolated one.

_KV_SCALE_RPM_PER_V = 10000.0  # normalizes kV and volume onto comparable scales
_VOLUME_SCALE_MM3 = 300.0      # for a nearest-neighbor distance in 2D


def nearest_catalogue_motor(kv, volume_mm3, n=3):
    """The n closest real motors in data/motor_datasheets.csv, nearest first.

    Distance is normalized Euclidean in (kV, stator volume) -- KV_SCALE and
    VOLUME_SCALE are rough "how much of a difference matters" scales, not a
    fit, so treat the ranking as approximate near ties. Only rows with a
    known kV and stator size are considered; rows missing R/I0/mass are still
    returned (with those fields as None) since even a mechanical-only match
    (e.g. confirming a stator size exists) is useful, but callers wanting a
    fully specified part should filter on "resistance_ohm" being present.
    """
    kv = float(kv)
    volume_mm3 = float(volume_mm3)

    def _float_or_none(s):
        return float(s) if s else None

    scored = []
    for row in _MOTOR_ROWS:
        row_kv = float(row["kv_rpm_per_v"])
        row_volume = stator_volume_mm3(
            float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
        dist = ((kv - row_kv) / _KV_SCALE_RPM_PER_V) ** 2 + (
            (volume_mm3 - row_volume) / _VOLUME_SCALE_MM3) ** 2
        scored.append({
            "name": row["name"],
            "vendor": row["vendor"],
            "stator_diameter_mm": float(row["stator_diameter_mm"]),
            "stator_height_mm": float(row["stator_height_mm"]),
            "volume_mm3": row_volume,
            "kv_rpm_per_v": row_kv,
            "mass_g": _float_or_none(row["mass_g"]),
            "resistance_ohm": _float_or_none(row["resistance_ohm"]),
            "i0_a": _float_or_none(row["i0_a"]),
            "max_current_a": _float_or_none(row["max_current_a"]),
            "source_url": row["source_url"],
            "distance": dist ** 0.5,
        })
    scored.sort(key=lambda s: s["distance"])
    return scored[:n]
