"""Scaling laws that stand in for testing discrete catalogue motors.

motor_model.py evaluates a motor given (kV, R, I0, Rth, mass); this module is
what lets the optimizer propose a stator size and kV and get realistic values
for the rest, instead of being limited to the handful of motors SimITL ships.
Everything here is a placeholder calibrated from as few datasheet points as
it took to get the optimizer running end to end -- see each fit's docstring
for exactly how thin the evidence is and what to fix first.
"""

import jax.numpy as jnp

import multirotor.motor_model as mm


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
# Calibrated from exactly two datasheet points: a 1002 (10mm x 2mm stator,
# 200 mm^3) at 2.5 g, and a 1202.5 (12mm x 2.5mm, 360 mm^3) at 4.5 g. Two
# points determine a line completely, so this is not evidence the
# relationship is linear or that the intercept is meaningful -- it is just
# enough to unblock the optimizer.
#
# The intercept below comes out to ~0 g/mm^3, which is almost certainly
# wrong: shaft, wires, and PCB are a fixed mass overhead that does not shrink
# with stator volume, so a real fit (more points, spanning a wider size
# range) should land on a positive intercept. With only two points forcing a
# line through both, there is no way to tell a real intercept from
# coincidence -- more data is what actually fixes this, not a manual nudge.
_CAL_VOLUME_MM3 = jnp.array([
    stator_volume_mm3(10.0, 2.0),   # 1002
    stator_volume_mm3(12.0, 2.5),   # 1202.5
])
_CAL_MASS_G = jnp.array([2.5, 4.5])

_MASS_SLOPE_G_PER_MM3 = (
    (_CAL_MASS_G[1] - _CAL_MASS_G[0]) / (_CAL_VOLUME_MM3[1] - _CAL_VOLUME_MM3[0])
)
_MASS_INTERCEPT_G = _CAL_MASS_G[0] - _MASS_SLOPE_G_PER_MM3 * _CAL_VOLUME_MM3[0]


def motor_mass_kg(volume_mm3):
    """Linear mass(volume) fit, in kg, from the two-point calibration above."""
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
# Calibrated from six datasheet points across two stator sizes: three
# different windings each on a 1002 (10mm x 2mm, 200 mm^3) and a 1203 (12mm x
# 3mm, 432 mm^3). Three windings per size is enough to see the premise hold
# up -- Km really does stay roughly constant across kV at a fixed size, the
# three 1002 values landing within about +-6% of their mean and the three
# 1203 values within about +-5% of theirs. But two stator *sizes* is nowhere
# near enough to trust the power-law exponent between them; that number is a
# line through two points dressed up as a fit; the exponent
# (KM_EXPONENT below, currently ~0.69) should be treated as a rough starting
# guess. What to fix first: more sizes, not more windings per size -- the
# within-size agreement is already about as good as it is going to get.
_CAL_KV_RPM_PER_V = jnp.array([
    14000.0, 19000.0, 22000.0,   # 1002
    6000.0, 8000.0, 11500.0,     # 1203
])
_CAL_R_OHM = jnp.array([
    0.175, 0.089, 0.075,   # 1002
    0.320, 0.167, 0.100,   # 1203
])
_CAL_VOLUME_MM3_KM = jnp.array([
    stator_volume_mm3(10.0, 2.0), stator_volume_mm3(10.0, 2.0),
    stator_volume_mm3(10.0, 2.0),
    stator_volume_mm3(12.0, 3.0), stator_volume_mm3(12.0, 3.0),
    stator_volume_mm3(12.0, 3.0),
])

_CAL_KT = mm.KT_NUMERATOR / _CAL_KV_RPM_PER_V
_CAL_KM = _CAL_KT / jnp.sqrt(_CAL_R_OHM)

# Power-law fit Km = c * volume^a, done as a linear regression in log-log
# space. With only two distinct volumes in the calibration set this reduces
# exactly to the line through the two per-size mean Km values.
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
