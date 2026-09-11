"""Rigid-body inertia tensor estimate for the whoop build, for sim/config use.

SimITL wants `invInertia` per body axis in the quad JSON (see
simitl-playback/config.h applyTo). This module computes the diagonal of the
inertia tensor from a lumped-mass model of the actual build:

  - motors + propellers as point masses at their mount positions
    (the dominant term -- they sit far from the CG),
  - the two center plates as boxes (geometry from frame_scaling.py:
    24mm x 90mm x 2.5mm CF, sandwiching 15mm standoffs),
  - four arms as boxes from the plate edge out to each motor,
  - battery and electronics stack as centered boxes (they sit near the CG,
    so their exact dimensions barely matter).

Axes follow the SimITL/pr0p quad-config convention: x lateral, y vertical,
z longitudinal -- matching the motor{1..4}Pos entries in
hq-51mm-whoop.json. I_yy is the yaw axis.

Not calibrated to a measured build; motors and the 90mm plates dominate, so
expect ~+-20% on each diagonal term.
"""

import numpy as np

# --- Build parameters (the realized HQ 51mm 1S whoop) -------------------------

# Motor mount positions, meters, same as the quad config motor{1..4}Pos.
MOTOR_POSITIONS_M = np.array([
    [ 0.0425, 0.005, -0.028],  # motor1
    [ 0.0425, 0.005,  0.028],  # motor2
    [-0.0425, 0.005, -0.028],  # motor3
    [-0.0425, 0.005,  0.028],  # motor4
])

MOTOR_MASS_KG = 0.0045        # GTS-V3 1203
PROP_MASS_KG = 0.00036        # HQProp 51mmx2
BATTERY_MASS_KG = 0.0162      # LAVA II 1S 680mAh
FRAME_ELECTRONICS_MASS_KG = 0.0424  # "mass" field in the quad config

# Frame decomposition (must sum to FRAME_ELECTRONICS_MASS_KG; electronics is
# the remainder, modeled as a centered box).
CF_DENSITY = 1600.0
PLATE_THICKNESS_M = 2.5e-3
PLATE_X_M = 24e-3
PLATE_Z_M = 90e-3
PLATE_Y = (0.0, 15e-3)        # bottom / top plate center heights
STANDOFF_MASS_KG = 4 * np.pi * (2.5e-3)**2 * 15e-3 * 2700.0
ARM_WIDTH_M = 6e-3
ARM_PLATE_EDGE_X_M = 12e-3    # arms run out in +-x from the plate edge

BATTERY_BOX_M = (0.055, 0.012, 0.017)   # x, y, z extents
BATTERY_Y_M = -8e-3                      # slung under the bottom plate
ELECTRONICS_BOX_M = (0.020, 0.013, 0.020)
ELECTRONICS_Y_M = 7.5e-3                 # between the plates


def _box_inertia_diag(mass, size_m):
    """Diagonal inertia of a solid box about its own CG, (Ixx, Iyy, Izz)."""
    x, y, z = size_m
    return np.array([y*y + z*z, x*x + z*z, x*x + y*y]) * mass / 12.0


def _add(I, mass, own_inertia_diag, pos):
    """Parallel-axis accumulate: I += own + m*(r^2*I - rr^T) diagonal terms."""
    r = np.asarray(pos)
    return I + own_inertia_diag + mass * np.array(
        [r[1]**2 + r[2]**2, r[0]**2 + r[2]**2, r[0]**2 + r[1]**2])


def quad_inertia(motor_positions=MOTOR_POSITIONS_M, motor_mass=MOTOR_MASS_KG,
                 prop_mass=PROP_MASS_KG, battery_mass=BATTERY_MASS_KG,
                 frame_electronics_mass=FRAME_ELECTRONICS_MASS_KG):
    """Diagonal inertia (Ixx, Iyy, Izz), kg.m^2, about the quad CG.

    The CG is assumed at the origin laterally and vertically midway through
    the stacked structure; vertical offsets are small (~mm) and contribute
    negligibly, so the plate/battery y-positions above are treated as exact.
    """
    I = np.zeros(3)

    # Motors + props as point masses at the mounts.
    for p in motor_positions:
        I = _add(I, motor_mass + prop_mass, np.zeros(3), p)

    # Center plates.
    plate_mass = PLATE_X_M * PLATE_Z_M * PLATE_THICKNESS_M * CF_DENSITY
    for y in PLATE_Y:
        I = _add(I, plate_mass,
                 _box_inertia_diag(plate_mass, (PLATE_X_M, PLATE_THICKNESS_M, PLATE_Z_M)),
                 (0.0, y, 0.0))

    # Standoffs at the plate corners-ish: point masses at (+-10, 7.5mm, +-30).
    for sx in (-10e-3, 10e-3):
        for sz in (-30e-3, 30e-3):
            I = _add(I, STANDOFF_MASS_KG / 4.0, np.zeros(3),
                     (sx, 7.5e-3, sz))

    # Arms: boxes from the plate edge to each motor, at motor z, plate height.
    arms_mass = 0.0
    for p in motor_positions:
        sign = np.sign(p[0])
        length = abs(p[0]) - ARM_PLATE_EDGE_X_M
        mid = (sign * (ARM_PLATE_EDGE_X_M + length / 2.0), 0.0, p[2])
        m = length * ARM_WIDTH_M * PLATE_THICKNESS_M * CF_DENSITY
        arms_mass += m
        I = _add(I, m, _box_inertia_diag(m, (length, PLATE_THICKNESS_M, ARM_WIDTH_M)), mid)

    # Battery and electronics: centered boxes.
    I = _add(I, battery_mass, _box_inertia_diag(battery_mass, BATTERY_BOX_M),
             (0.0, BATTERY_Y_M, 0.0))
    electronics_mass = (frame_electronics_mass - 2.0 * plate_mass
                        - STANDOFF_MASS_KG - arms_mass)
    I = _add(I, electronics_mass,
             _box_inertia_diag(electronics_mass, ELECTRONICS_BOX_M),
             (0.0, ELECTRONICS_Y_M, 0.0))

    return dict(I=np.array(I), plate_mass=plate_mass, arms_mass=arms_mass,
                electronics_mass=electronics_mass)


def main():
    r = quad_inertia()
    I = r["I"]
    names = ["x (roll-axis, lateral)", "y (yaw-axis, vertical)",
             "z (pitch-axis, longitudinal)"]
    total_mass = (FRAME_ELECTRONICS_MASS_KG + BATTERY_MASS_KG
                  + 4 * (MOTOR_MASS_KG + PROP_MASS_KG))
    print(f"total mass: {total_mass*1e3:.1f} g "
          f"(frame+elec {FRAME_ELECTRONICS_MASS_KG*1e3:.1f}, "
          f"bat {BATTERY_MASS_KG*1e3:.1f}, "
          f"motors+props {4*(MOTOR_MASS_KG+PROP_MASS_KG)*1e3:.1f})")
    print(f"frame split: plates {2*r['plate_mass']*1e3:.1f} g, "
          f"arms {r['arms_mass']*1e3:.1f} g, "
          f"standoffs {STANDOFF_MASS_KG*1e3:.1f} g, "
          f"electronics {r['electronics_mass']*1e3:.1f} g")
    for name, i in zip(names, I):
        print(f"I{name}: {i:.3e} kg.m^2  ->  invInertia {1.0/i:.0f}")


if __name__ == "__main__":
    main()
