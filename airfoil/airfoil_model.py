"""JAX model of a tailsitter flying wing: elevon sizing, trim, and stall.

The aircraft is a 3D-printed twin-motor tailsitter VTOL.  It starts in hover,
transitions, and cruises, so the control surfaces must work in three regimes
that stress them in completely different ways.  The elevons are sized by the
worst of the three, which is usually *not* cruise, so all three are modelled.

Two things make this different from a textbook wing-sizing problem:

Reynolds number.  A 60 mm chord at 10 m/s gives Re ~ 40,000.  Below Re ~ 70,000
the boundary layer separates while still laminar and does not reattach, so the
usual attached-flow machinery (panel methods, integral boundary layers, the
pressure-gradient separation criteria built on them) is solving the wrong
problem.  What survives at this Reynolds number is inviscid theory for lift and
moment -- those are set by the shape turning the flow, which happens whether the
layer is laminar or not -- plus empirical correlations for anything viscous.  So
Cl, Cm, and elevon effectiveness come from thin-airfoil theory (exact, closed
form, differentiable), and Cl_max comes from a correlation.  The correlation is
openly approximate; that is preferable to a detailed model that would be
confidently wrong here.

Propeller slipstream.  Control power scales with local dynamic pressure, and the
props blow over part of the elevon span.  In hover that is the *only* reason the
aircraft is controllable at all: freestream is zero, so an unwashed elevon does
nothing.  In forward flight the washed strip is roughly 1.5x as effective per
degree as the unwashed strip, which makes the elevon a non-uniform control
surface whose authority depends on throttle.  Motor spanwise position therefore
sets both yaw authority (differential thrust moment arm) and how much of the
elevon is usefully blown, which is why it is a design variable rather than a
layout decision made up front.

Sign conventions follow the classic thin-airfoil development: x measured from
the leading edge as a fraction of chord, flap deflection eta positive downward,
pitching moment positive nose-up.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- Physical constants -------------------------------------------------------

RHO = 1.225          # kg/m^3, sea level
MU = 1.81e-5         # Pa.s, dynamic viscosity of air
G = 9.81             # m/s^2

# --- Airframe constants -------------------------------------------------------
#
# Measured or specified.  Thrust is the least certain of these: it was specced
# for TWR > 1 from motor/prop data (1002 21000kv, 2x0.9in 3-blade) but has not
# been measured, and it sets whether the aircraft can hover at all.  Everything
# downstream of THRUST_PER_MOTOR should be treated as provisional until it is.

SPAN = 0.256                 # m, capped by the 3D printer bed
AVIONICS_MASS = 0.025        # kg, FC + RX + battery + servos
PROPULSION_MASS = 0.012      # kg, 2x (motor + ESC + prop), estimated
THRUST_PER_MOTOR = 0.035     # kgf, ~35 g.  UNMEASURED -- see note above.
PROP_DIAMETER = 0.051        # m, 2 inch

BATTERY_LENGTH = 0.065       # m, drives the minimum root chord
BATTERY_THICKNESS = 0.011    # m, drives the minimum root thickness
SERVO_THICKNESS = 0.008      # m, servo body depth at the hinge line

# Section thickness at the hinge line as a fraction of maximum thickness.  A
# typical section has its maximum thickness near 30% chord and has thinned to
# roughly half of it by 75% chord, where the elevon hinge sits.  This factor is
# what makes the servo rather than the battery the binding volume constraint.
HINGE_THICKNESS_FRACTION = 0.5

# Foaming PLA printed as a hollow shell.  Areal density is wall thickness times
# foamed density; 0.4 mm walls of LW-PLA at ~0.6 g/cm^3 gives ~0.24 kg/m^2.
SKIN_AREAL_DENSITY = 0.24    # kg/m^2 of wetted area

# --- Model constants ----------------------------------------------------------

# Thin-airfoil theory overpredicts flap effectiveness because it ignores the
# boundary layer thickening over the deflected surface.  The usual correction is
# 0.7-0.85 at normal Reynolds numbers; at Re ~ 40,000 the layer is thicker
# relative to the chord and the loss is worse.  This is the single largest
# source of error in the control-authority numbers, so it is deliberately
# pessimistic: an elevon sized against it will be adequate, one sized against
# inviscid theory may not be.
FLAP_EFFECTIVENESS_FACTOR = 0.65

# Past roughly 15-20 degrees the flow separates over the deflected surface and
# further deflection buys much less than theory says.  Authority must be
# evaluated at a realistic deflection rather than extrapolated linearly, or the
# optimizer will "solve" an authority shortfall by demanding 40 degrees.
FLAP_STALL_DEG = 15.0
FLAP_SATURATION_DEG = 30.0

# Momentum theory gives the *fully developed* slipstream velocity, reached a few
# diameters downstream.  The elevon sits close behind the disk where only about
# half the contraction has happened.
SLIPSTREAM_DEVELOPMENT = 0.5

# The slipstream contracts slightly; immediately behind the disk its width is
# close to the prop diameter.
SLIPSTREAM_WIDTH_FACTOR = 1.0

ARCCOS_EPS = 1e-9


# --- Thin-airfoil flap theory -------------------------------------------------
#
# Closed form, so the derivatives with respect to hinge position are exact.  The
# hinge position is a design variable and these are the equations that pick it,
# so that matters.


def hinge_angle(x_hinge):
    """Angular hinge location theta_f = arccos(1 - 2 x_f).

    x_hinge is the hinge station as a fraction of chord from the leading edge,
    so a 25% chord elevon is x_hinge = 0.75.
    """
    # arccos has infinite derivative at +/-1, which a hinge at the leading or
    # trailing edge would hit.  Neither is a real design, but the optimizer may
    # probe the bounds, and a NaN there poisons the whole gradient.
    c = jnp.clip(1.0 - 2.0 * x_hinge, -1.0 + ARCCOS_EPS, 1.0 - ARCCOS_EPS)
    return jnp.arccos(c)


def flap_lift_slope(x_hinge):
    """dCl/deta, per radian of flap deflection.  Inviscid."""
    theta_f = hinge_angle(x_hinge)
    return 2.0 * (jnp.pi - theta_f + jnp.sin(theta_f))


def flap_moment_slope(x_hinge):
    """dCm/deta about the quarter chord, per radian.  Inviscid.

    Negative: a downward deflection pitches the section nose-down.
    """
    theta_f = hinge_angle(x_hinge)
    return 0.25 * (jnp.sin(2.0 * theta_f) - 2.0 * jnp.sin(theta_f))


def flap_effectiveness_ratio(x_hinge):
    """Flap deflection worth this many degrees of angle of attack.

    The classic result that a degree of flap is worth roughly half a degree of
    alpha for a typical 25% chord surface.  Useful as a sanity check and as the
    quantity that actually matters when trading hinge position against
    deflection range.
    """
    return flap_lift_slope(x_hinge) / (2.0 * jnp.pi)


def flap_deflection_efficiency(deflection_deg):
    """Fraction of theoretical effectiveness retained at a given deflection.

    Unity below FLAP_STALL_DEG, then rolls off smoothly rather than switching so
    the gradient stays useful.  ``exp`` rather than a tanh knee: tanh(0) is 0,
    which would leave the sub-stall region at half effectiveness instead of
    full.  The decay reaches about 1/e at the saturation angle, matching the
    observed behaviour that large deflections keep helping, just much less per
    degree.
    """
    excess = jnp.maximum(jnp.abs(deflection_deg) - FLAP_STALL_DEG, 0.0)
    return jnp.exp(-excess / (FLAP_SATURATION_DEG - FLAP_STALL_DEG))


def effective_flap_lift_slope(x_hinge, deflection_deg):
    """dCl/deta corrected for viscosity and for deflection-angle stall."""
    return (flap_lift_slope(x_hinge)
            * FLAP_EFFECTIVENESS_FACTOR
            * flap_deflection_efficiency(deflection_deg))


def effective_flap_moment_slope(x_hinge, deflection_deg):
    """dCm/deta corrected for viscosity and for deflection-angle stall."""
    return (flap_moment_slope(x_hinge)
            * FLAP_EFFECTIVENESS_FACTOR
            * flap_deflection_efficiency(deflection_deg))


# --- Propeller slipstream -----------------------------------------------------


def induced_velocity(thrust_n, v_inf, disk_area):
    """Slipstream velocity increment from momentum theory.

    Solves the general (non-static) momentum-theory relation

        T = 2 rho A (v_inf + w) w

    for the induced velocity w, which is a quadratic with the positive root

        w = -v_inf/2 + sqrt((v_inf/2)^2 + T / (2 rho A))

    Static hover is the v_inf -> 0 limit, w = sqrt(T / (2 rho A)), so the same
    expression covers hover, transition, and cruise without a special case.
    That matters here because transition is exactly the regime where neither
    limit is a good approximation.
    """
    half_v = 0.5 * v_inf
    return -half_v + jnp.sqrt(half_v ** 2 + thrust_n / (2.0 * RHO * disk_area))


def slipstream_velocity(thrust_n, v_inf, disk_area):
    """Local flow speed over an elevon inside the slipstream."""
    w = induced_velocity(thrust_n, v_inf, disk_area)
    return v_inf + SLIPSTREAM_DEVELOPMENT * 2.0 * w


def washed_span_fraction(motor_y, elevon_inboard_y, elevon_outboard_y):
    """Fraction of the elevon span that sits inside the slipstream.

    Computed as the overlap of two intervals along the span: the slipstream
    footprint centred on the motor, and the elevon.  Written with clipped
    differences so it stays differentiable as the motor slides across the elevon
    edge, which the optimizer will do.
    """
    half_width = 0.5 * SLIPSTREAM_WIDTH_FACTOR * PROP_DIAMETER
    wash_lo = motor_y - half_width
    wash_hi = motor_y + half_width

    overlap = (jnp.minimum(wash_hi, elevon_outboard_y)
               - jnp.maximum(wash_lo, elevon_inboard_y))
    elevon_span = jnp.maximum(elevon_outboard_y - elevon_inboard_y, 1e-9)
    return jnp.clip(overlap / elevon_span, 0.0, 1.0)


def dynamic_pressure_ratio(thrust_n, v_inf, motor_y,
                           elevon_inboard_y, elevon_outboard_y):
    """Span-averaged q over the elevon, relative to freestream q.

    This is the multiplier on control authority from prop wash.  In hover the
    freestream contributes nothing and the ratio is formally infinite, so
    callers in hover should use :func:`control_authority` with the absolute
    dynamic pressure instead of this ratio.
    """
    disk_area = 0.25 * jnp.pi * PROP_DIAMETER ** 2
    v_wash = slipstream_velocity(thrust_n, v_inf, disk_area)
    f = washed_span_fraction(motor_y, elevon_inboard_y, elevon_outboard_y)

    q_inf = jnp.maximum(v_inf ** 2, 1e-9)
    return f * (v_wash ** 2) / q_inf + (1.0 - f)


def elevon_dynamic_pressure(thrust_n, v_inf, motor_y,
                            elevon_inboard_y, elevon_outboard_y):
    """Span-averaged absolute dynamic pressure over the elevon, in Pa.

    Valid in every regime including hover, where the freestream term vanishes
    and the whole contribution comes from the washed fraction.
    """
    disk_area = 0.25 * jnp.pi * PROP_DIAMETER ** 2
    v_wash = slipstream_velocity(thrust_n, v_inf, disk_area)
    f = washed_span_fraction(motor_y, elevon_inboard_y, elevon_outboard_y)

    q_washed = 0.5 * RHO * v_wash ** 2
    q_clean = 0.5 * RHO * v_inf ** 2
    return f * q_washed + (1.0 - f) * q_clean


# --- Control authority --------------------------------------------------------


def pitch_moment(deflection_deg, x_hinge, thrust_n, v_inf, motor_y,
                 elevon_inboard_y, elevon_outboard_y, area, chord):
    """Pitching moment from symmetric elevon deflection, in N.m.

    Both elevons deflect together, so the moment is the section moment carried
    over the elevon span at the local dynamic pressure.
    """
    q = elevon_dynamic_pressure(thrust_n, v_inf, motor_y,
                                elevon_inboard_y, elevon_outboard_y)
    dcm = effective_flap_moment_slope(x_hinge, deflection_deg)
    return q * area * chord * dcm * jnp.radians(deflection_deg)


def roll_moment(deflection_deg, x_hinge, thrust_n, v_inf, motor_y,
                elevon_inboard_y, elevon_outboard_y, chord):
    """Rolling moment from differential elevon deflection, in N.m.

    The lift increment acts at the elevon's spanwise centroid, and both sides
    contribute, hence the factor of two.  Outboard elevon area is worth more
    for roll than inboard area because of the longer moment arm, which is part
    of why elevon spanwise extent is a design variable and not just a fraction.
    """
    q = elevon_dynamic_pressure(thrust_n, v_inf, motor_y,
                                elevon_inboard_y, elevon_outboard_y)
    dcl = effective_flap_lift_slope(x_hinge, deflection_deg)

    elevon_span = elevon_outboard_y - elevon_inboard_y
    centroid_y = 0.5 * (elevon_inboard_y + elevon_outboard_y)
    elevon_area = elevon_span * chord

    delta_lift = q * elevon_area * dcl * jnp.radians(deflection_deg)
    return 2.0 * delta_lift * centroid_y


def yaw_moment(differential_thrust_n, motor_y):
    """Yaw moment from differential thrust, in N.m.

    The only yaw control on the aircraft -- there is no rudder -- so the motor
    spanwise position is what sets yaw authority, and it trades directly against
    the wash coverage and roll inertia that the same variable controls.
    """
    return differential_thrust_n * motor_y


def hinge_moment(deflection_deg, x_hinge, thrust_n, v_inf, motor_y,
                 elevon_inboard_y, elevon_outboard_y, chord, alpha_deg=0.0):
    """Aerodynamic torque about the hinge line, in N.m.

    This is what the servo fights, and it is the number the linkage model
    consumes.  Note the elevon chord enters squared while authority grows
    roughly linearly with it, which is the real cost of an oversized elevon.

    Thin-airfoil theory predicts hinge moment poorly because it is dominated by
    the pressure distribution right at the hinge, where the thin approximation
    is weakest.  The coefficients below are representative empirical values for
    a plain flap rather than anything derived, so size the servo with margin.
    """
    q = elevon_dynamic_pressure(thrust_n, v_inf, motor_y,
                                elevon_inboard_y, elevon_outboard_y)
    elevon_chord = (1.0 - x_hinge) * chord
    elevon_span = elevon_outboard_y - elevon_inboard_y

    # Both negative: the aerodynamic load tends to centre the surface.
    ch_alpha = -0.55
    ch_delta = -0.85
    ch = ch_alpha * jnp.radians(alpha_deg) + ch_delta * jnp.radians(deflection_deg)

    return q * (elevon_chord ** 2) * elevon_span * jnp.abs(ch)


# --- Geometry and mass --------------------------------------------------------


def planform(root_chord, tip_chord):
    """Wing area and mean aerodynamic chord for a straight-tapered wing."""
    area = 0.5 * (root_chord + tip_chord) * SPAN
    taper = tip_chord / jnp.maximum(root_chord, 1e-9)
    mac = (2.0 / 3.0) * root_chord * (
        (1.0 + taper + taper ** 2) / jnp.maximum(1.0 + taper, 1e-9)
    )
    return area, mac


def reynolds(v_inf, chord):
    """Chord-based Reynolds number.

    Worth computing at the tip as well as the mean: the tip chord is smaller, so
    the tip runs at a lower Re and stalls earlier, compounding the tip-stall
    tendency that sweep already creates.
    """
    return RHO * v_inf * chord / MU


def wing_mass(root_chord, tip_chord, root_thickness, tip_thickness):
    """Structural mass of a hollow printed shell, in kg.

    Wetted area is approximated as the planform area times a perimeter factor
    that accounts for the airfoil's curved upper and lower surfaces being longer
    than the chord.  For thin sections the factor is close to 2; thickness adds
    a little.  Both surfaces, hence the 2.
    """
    area, _ = planform(root_chord, tip_chord)

    mean_thickness_ratio = 0.5 * (
        root_thickness / jnp.maximum(root_chord, 1e-9)
        + tip_thickness / jnp.maximum(tip_chord, 1e-9)
    )
    perimeter_factor = 2.0 * (1.0 + 1.5 * mean_thickness_ratio ** 2)
    return area * perimeter_factor * SKIN_AREAL_DENSITY


def total_mass(root_chord, tip_chord, root_thickness, tip_thickness):
    """All-up mass, in kg."""
    return (wing_mass(root_chord, tip_chord, root_thickness, tip_thickness)
            + AVIONICS_MASS + PROPULSION_MASS)


def volume_slack(root_chord, root_thickness):
    """How much room the root section has beyond what the avionics need, in m.

    Positive means the battery and servos fit.  The battery is the binding item
    on thickness and its length is what sets the minimum root chord; the servo
    needs its own depth at the hinge line, which sits further aft where the
    section is thinner, so it is checked separately.
    """
    chord_slack = root_chord - BATTERY_LENGTH
    thickness_slack = root_thickness - BATTERY_THICKNESS

    # The servo sits at the hinge line, well aft of maximum thickness, where the
    # section has thinned considerably.  This is what actually binds: the
    # battery fits in 12 mm of root thickness, but the servo needs 8 mm at the
    # hinge, which for a section thinned to about half its maximum by that
    # station demands 16 mm at the root.  Every millimetre of servo depth
    # therefore costs two millimetres of root thickness.
    hinge_thickness = root_thickness * HINGE_THICKNESS_FRACTION
    servo_slack = hinge_thickness - SERVO_THICKNESS

    return jnp.minimum(jnp.minimum(chord_slack, thickness_slack), servo_slack)


# --- Flight conditions --------------------------------------------------------


def stall_speed(mass_kg, area, cl_max):
    """Level-flight stall speed, in m/s."""
    weight = mass_kg * G
    return jnp.sqrt(2.0 * weight / (RHO * area * jnp.maximum(cl_max, 1e-6)))


def thrust_to_weight(mass_kg):
    """Static thrust-to-weight ratio.  Must exceed 1 to hover at all."""
    return 2.0 * THRUST_PER_MOTOR / jnp.maximum(mass_kg, 1e-9)


def cruise_lift_coefficient(mass_kg, area, v_inf):
    """Cl required for level flight at a given speed."""
    weight = mass_kg * G
    return 2.0 * weight / (RHO * jnp.maximum(v_inf ** 2, 1e-9) * area)


# --- Design point evaluation --------------------------------------------------


DESIGN_VARS = ["root_chord", "tip_chord", "root_thickness", "tip_thickness",
               "x_hinge", "elevon_inboard_frac", "motor_frac"]


def unpack(p):
    """Design vector to named geometry, with span fractions turned into metres."""
    root_chord, tip_chord, root_t, tip_t, x_hinge, elevon_in_f, motor_f = p
    semi = 0.5 * SPAN
    return {
        "root_chord": root_chord,
        "tip_chord": tip_chord,
        "root_thickness": root_t,
        "tip_thickness": tip_t,
        "x_hinge": x_hinge,
        "elevon_inboard_y": elevon_in_f * semi,
        "elevon_outboard_y": semi,
        "motor_y": motor_f * semi,
    }


def evaluate(p, v_cruise=12.0, cl_max=0.8, deflection_deg=10.0):
    """Everything the elevon sizing decision needs, at one design point.

    Authority is reported at four conditions because the binding one is not
    obvious in advance and is usually not cruise.  Hover and transition are
    fed almost entirely by prop wash; idle descent is the case where the wash
    goes away while the aircraft still needs to be controllable, which is the
    condition most likely to be missed.
    """
    g = unpack(p)
    area, mac = planform(g["root_chord"], g["tip_chord"])
    mass = total_mass(g["root_chord"], g["tip_chord"],
                      g["root_thickness"], g["tip_thickness"])

    thrust_hover = THRUST_PER_MOTOR * G
    v_stall = stall_speed(mass, area, cl_max)

    # Thrust required in cruise is a small fraction of hover thrust, so the
    # wash contribution there is correspondingly weaker.
    conditions = {
        "hover": (0.0, thrust_hover),
        "transition": (0.5 * v_stall, thrust_hover),
        "cruise": (v_cruise, 0.3 * thrust_hover),
        "idle_descent": (v_cruise, 0.05 * thrust_hover),
    }

    authority = {}
    for name, (v, thrust) in conditions.items():
        authority[name] = {
            "q": elevon_dynamic_pressure(
                thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"]),
            "pitch": pitch_moment(
                deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"], area, mac),
            "roll": roll_moment(
                deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"], mac),
            "hinge": hinge_moment(
                deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"], mac),
        }

    return {
        "area": area,
        "mac": mac,
        "aspect_ratio": SPAN ** 2 / jnp.maximum(area, 1e-9),
        "mass": mass,
        "twr": thrust_to_weight(mass),
        "wing_loading": mass * G / jnp.maximum(area, 1e-9),
        "v_stall": v_stall,
        "re_mac": reynolds(v_stall, mac),
        "re_tip": reynolds(v_stall, g["tip_chord"]),
        "root_tc": g["root_thickness"] / jnp.maximum(g["root_chord"], 1e-9),
        "tip_tc": g["tip_thickness"] / jnp.maximum(g["tip_chord"], 1e-9),
        "volume_slack": volume_slack(g["root_chord"], g["root_thickness"]),
        "yaw_moment": yaw_moment(0.5 * thrust_hover, g["motor_y"]),
        "wash_fraction": washed_span_fraction(
            g["motor_y"], g["elevon_inboard_y"], g["elevon_outboard_y"]),
        "flap_effectiveness": flap_effectiveness_ratio(g["x_hinge"]),
        "authority": authority,
    }


evaluate_jit = jax.jit(evaluate, static_argnames=())
