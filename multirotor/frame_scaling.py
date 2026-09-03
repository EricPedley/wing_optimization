"""Rough geometric model of frame mass as a function of propeller diameter.

Not calibrated from any datasheet, unlike motor_scaling.py/prop_scaling.py --
there is no catalogue of "frame mass vs prop size" to fit. This is instead a
first-principles CAD-style estimate: guess a plausible geometry (a plated
center rectangle plus four arms, sized so the prop tips clear the frame and
each other), compute its volume, and multiply by material density. Good for
"does a bigger prop cost 3g or 30g of frame", not for predicting a real
frame's mass to better than maybe +-30%.

Assumptions, all fixed constants below rather than fit: 2mm carbon fiber
plate for the center (top and bottom) and arms, 4 aluminum standoffs sized
20mm x 60mm x 15mm tall joining the two plates, and 5mm radial clearance
between prop tip and the nearest frame edge. Arm width is not derived from
anything physical -- see ARM_WIDTH_MM.
"""

import jax.numpy as jnp

# --- Material properties ------------------------------------------------------

CARBON_FIBER_DENSITY_KG_M3 = 1600.0  # typical CF plate, 1.5-1.8 g/cm^3
ALUMINUM_DENSITY_KG_M3 = 2700.0

PLATE_THICKNESS_M = 2.0e-3

# --- Center plate -------------------------------------------------------------
#
# Approximated as a rectangle just big enough to carry the 4 standoffs
# (20mm x 60mm footprint) plus a little edge margin around each standoff so
# it isn't sitting right at the plate boundary. Two plates (top + bottom)
# sandwich the standoffs, hence "double plates" in the ask.
STANDOFF_FOOTPRINT_X_M = 20.0e-3
STANDOFF_FOOTPRINT_Y_M = 60.0e-3
CENTER_PLATE_MARGIN_M = 5.0e-3  # edge margin around the standoff footprint

CENTER_PLATE_LENGTH_M = STANDOFF_FOOTPRINT_Y_M + 2.0 * CENTER_PLATE_MARGIN_M
CENTER_PLATE_WIDTH_M = STANDOFF_FOOTPRINT_X_M + 2.0 * CENTER_PLATE_MARGIN_M
N_CENTER_PLATES = 2  # top + bottom

# --- Standoffs ------------------------------------------------------------
#
# 4 aluminum standoffs, 15mm tall, modeled as solid cylinders of a made-up
# 5mm diameter (a plausible M3-ish standoff size) -- the 20mm x 60mm figure
# in the ask is the footprint they're arranged in on the plate, not their own
# cross-section, so it sizes CENTER_PLATE_* above rather than the standoffs
# themselves.
N_STANDOFFS = 4
STANDOFF_HEIGHT_M = 15.0e-3
STANDOFF_DIAMETER_M = 5.0e-3


def standoff_volume_m3():
    radius = 0.5 * STANDOFF_DIAMETER_M
    return jnp.pi * radius ** 2 * STANDOFF_HEIGHT_M


# --- Arms -------------------------------------------------------------------
#
# Each arm is a flat carbon plate running from the center plate's edge out to
# where the prop needs to be: far enough that the prop tip clears the frame
# (PROP_TIP_CLEARANCE_M) and, implicitly, clears the adjacent props (a
# standard X/+ quad layout puts each motor at 45 degrees off the body's
# X/Y axes, so this model does not need a separate prop-prop clearance check
# -- the frame-clearance one dominates for any reasonable arm layout).
#
# Arm width is not given by the ask ("IDK how wide the arms should be") and
# is not derivable from anything else here, so it's a flat guess: wide enough
# to be a plausible single-piece carbon arm, scaled up slowly with prop size
# since a bigger prop implies more thrust/torque and wants a stiffer arm.
# Treat this as the single biggest source of error in the whole model.
ARM_WIDTH_BASE_M = 8.0e-3
ARM_WIDTH_PER_PROP_DIAMETER = 0.08  # arm width grows at 8% of prop diameter
PROP_TIP_CLEARANCE_M = 5.0e-3
N_ARMS = 4


def arm_width_m(prop_diameter_m):
    return ARM_WIDTH_BASE_M + ARM_WIDTH_PER_PROP_DIAMETER * prop_diameter_m


def arm_length_m(prop_diameter_m):
    """Center-plate corner to prop tip, along the arm's own direction.

    Motors sit on the plate's diagonal (standard X-quad layout), so the
    relevant plate dimension is its half-diagonal, and the arm need only
    span the straight-line gap from that corner out to where the tip
    clearance requires the prop center to be. Floored at 0: for a small
    enough prop the clearance point can fall inside the plate's own
    footprint, at which point no arm length is needed at all (the motor
    would mount straight to the plate corner).
    """
    prop_radius_m = 0.5 * prop_diameter_m
    tip_reach_m = prop_radius_m + PROP_TIP_CLEARANCE_M
    center_plate_half_diagonal_m = 0.5 * jnp.sqrt(
        CENTER_PLATE_LENGTH_M ** 2 + CENTER_PLATE_WIDTH_M ** 2)
    return jnp.maximum(tip_reach_m - center_plate_half_diagonal_m, 0.0)


def frame_mass_kg(prop_diameter_m):
    """Total frame mass: center plates + standoffs + 4 arms, in kg.

    Rough ballpark only -- see module docstring. Arm width in particular is a
    guess, not a derived quantity.
    """
    center_plate_volume_m3 = (
        N_CENTER_PLATES * CENTER_PLATE_LENGTH_M * CENTER_PLATE_WIDTH_M
        * PLATE_THICKNESS_M)

    arm_volume_m3 = (
        N_ARMS * arm_length_m(prop_diameter_m) * arm_width_m(prop_diameter_m)
        * PLATE_THICKNESS_M)

    carbon_volume_m3 = center_plate_volume_m3 + arm_volume_m3
    carbon_mass_kg = carbon_volume_m3 * CARBON_FIBER_DENSITY_KG_M3

    standoff_mass_kg = (
        N_STANDOFFS * standoff_volume_m3() * ALUMINUM_DENSITY_KG_M3)

    return carbon_mass_kg + standoff_mass_kg
