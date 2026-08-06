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
FIXED_MASS = 0.025           # kg, FC + RX + battery + servos + motors + ESC + props
THRUST_PER_MOTOR = 0.035     # kgf, ~35 g.  UNMEASURED -- see note above.
PROP_DIAMETER = 0.051        # m, 2 inch

BATTERY_LENGTH = 0.065       # m, drives the minimum root chord
MIN_ROOT_THICKNESS = 0.015   # m, drives the minimum root thickness

# Chord fraction where the battery's forward face sits.  Kept well forward
# both because that is where the section is deepest and because the battery is
# the densest single item, so its position dominates the centre of gravity --
# and a tailsitter flying wing needs its CG forward to be stable in cruise.
BATTERY_STATION = 0.10

# The servo is a box, and which of its dimensions binds depends on how it is
# mounted, so all three are named separately rather than collapsed into one
# "servo thickness".  Dimensions below are from CAD.  SERVO_LENGTH runs along
# the chord when the servo is mounted conventionally; SERVO_DEPTH is the
# dimension that has to fit inside the section thickness.
#
# At 15 mm deep this servo is nearly as thick as the whole root section, so it
# is the dominant packaging constraint and the reason servo_station is a design
# variable: it only fits near maximum thickness, and it may not fit at all
# without either a deeper root or mounting it on its side.
SERVO_LENGTH = 0.021         # m, longest dimension, lies along the chord
SERVO_DEPTH = 0.008          # m, the dimension that fights section thickness
SERVO_WIDTH = 0.015          # m, spanwise

# Section thickness profile.  A typical section reaches maximum thickness near
# 30% chord and thins toward the trailing edge roughly as a quadratic.  This is
# what makes servo *chordwise position* matter: the same servo needs far less
# root thickness at 45% chord than at the hinge line, which is the whole reason
# to consider moving it forward.
MAX_THICKNESS_STATION = 0.30

# Printed as a hollow shell, so skin mass is wall thickness times material
# density.  Set FILAMENT_DENSITY to the *as-printed* density, not the spool
# figure: LW-PLA foams to roughly 0.4-0.6 g/cm^3 when printed hot enough to
# activate, and stays near 1.2 g/cm^3 when it does not.  The difference is a
# factor of two or three in wing mass, which propagates straight into TWR.
WALL_THICKNESS = 0.4e-3      # m
FILAMENT_DENSITY = 900.0     # kg/m^3, as printed
SKIN_AREAL_DENSITY = WALL_THICKNESS * FILAMENT_DENSITY  # kg/m^2 of wetted area

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


def available_differential_thrust(throttle_fraction):
    """Differential thrust available at a given throttle setting, in N.

    Yaw authority is *not* worst in hover, which is the natural assumption.  In
    hover the motors sit near half throttle with room to push one up and pull
    the other down, and the aircraft is fighting only its own inertia.  In
    forward flight at cruise throttle there is less headroom above the current
    setting, and yaw additionally fights the wing's weathercock stability.  So
    the binding yaw case is forward flight, and it has to be checked there.

    The differential is limited by whichever headroom is smaller: how far the
    up-motor can rise toward full, or how far the down-motor can fall toward
    zero.  Symmetric about half throttle, hence the min.
    """
    t = jnp.clip(throttle_fraction, 0.0, 1.0)
    headroom = jnp.minimum(1.0 - t, t)
    return 2.0 * headroom * THRUST_PER_MOTOR * G


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
            + FIXED_MASS)


def thickness_at(x, max_thickness):
    """Section thickness at chord fraction ``x``, given the maximum thickness.

    A crude but adequate profile: thickness rises from zero at the leading edge
    to its maximum at MAX_THICKNESS_STATION, then falls roughly quadratically to
    near zero at the trailing edge.  Only the aft branch matters for packaging,
    since everything competing for space sits behind the leading edge.

    This is what makes servo chordwise position a real design variable rather
    than a detail: at 45% chord a section is still near full thickness, while at
    the 75% hinge line it has thinned by half.
    """
    fore = x / MAX_THICKNESS_STATION
    aft = 1.0 - ((x - MAX_THICKNESS_STATION) / (1.0 - MAX_THICKNESS_STATION)) ** 2
    shape = jnp.where(x < MAX_THICKNESS_STATION, fore, aft)
    return max_thickness * jnp.clip(shape, 0.0, 1.0)


def local_geometry(root_chord, tip_chord, root_thickness, tip_thickness,
                   span_fraction):
    """Chord and maximum thickness at a fraction of the semi-span.

    Straight-tapered wing with both chord and thickness lofted linearly between
    the root and tip sections.
    """
    chord = root_chord + (tip_chord - root_chord) * span_fraction
    thickness = root_thickness + (tip_thickness - root_thickness) * span_fraction
    return chord, thickness


def servo_slack(chord, thickness, servo_station, x_hinge):
    """Room around the servo at its chordwise station, in m.

    ``servo_station`` is the chord fraction where the servo body is centred, and
    ``chord``/``thickness`` are the section it sits in -- which is its own
    spanwise station, not the root.  Positive return means it fits.

    Two things are checked: the section is deep enough for the servo where it
    actually sits, and the servo body fits between that station and the hinge.

    Moving the servo forward relaxes the depth constraint quickly, because
    section thickness aft of maximum falls off quadratically.  It does not come
    free -- the pushrod gets longer and drives the horn increasingly off-axis,
    which shows up as varying mechanical advantage over the stroke -- but that
    cost belongs to the linkage model, not here.  What this function reports is
    the packaging half of the trade.
    """
    depth_slack = thickness_at(servo_station, thickness) - SERVO_DEPTH

    # The servo body occupies chord centred on its station; the hinge must be
    # far enough aft that the body does not run into it.
    body_aft_edge = servo_station + 0.5 * SERVO_LENGTH / chord
    clearance_slack = (x_hinge - body_aft_edge) * chord

    return jnp.minimum(depth_slack, clearance_slack)


def pushrod_length(chord, servo_station, x_hinge):
    """Chordwise distance from the servo output to the hinge line, in m.

    The quantity the linkage model needs: a longer run means the horn is driven
    further off-axis, so mechanical advantage varies more across the stroke.
    """
    return (x_hinge - servo_station) * chord


def volume_slack(root_chord, tip_chord, root_thickness, tip_thickness,
                 servo_station, servo_span_fraction, x_hinge):
    """How much room the wing has beyond what it must hold, in m.

    Positive means everything fits.  The battery and the servos are checked at
    different spanwise stations because that is where they actually live: the
    battery occupies the centreline, and the servos sit outboard near the
    elevons they drive.  Checking both at the root would have them fighting for
    the same chord, which is a constraint that does not exist -- and one the
    model previously invented, making the design look infeasible when it was
    only badly drawn.
    """
    chord_slack = root_chord - BATTERY_LENGTH
    thickness_slack = root_thickness - MIN_ROOT_THICKNESS

    chord, thickness = local_geometry(root_chord, tip_chord, root_thickness,
                                      tip_thickness, servo_span_fraction)
    servo = servo_slack(chord, thickness, servo_station, x_hinge)

    # The servo must sit outboard of the battery, which occupies the centre
    # section out to roughly half its own width either side of the centreline.
    span_slack = servo_span_fraction * 0.5 * SPAN - 0.5 * SERVO_WIDTH

    return jnp.minimum(jnp.minimum(chord_slack, thickness_slack),
                       jnp.minimum(servo, span_slack))


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


# --- Inertia and angular acceleration -----------------------------------------
#
# A control moment in N.m says nothing on its own about whether the aircraft
# responds usefully; what matters is the angular acceleration it produces, which
# is the moment divided by the relevant inertia.  Expressing authority that way
# also makes the three axes comparable, which they are not in raw moment terms.


def inertia(root_chord, tip_chord, root_thickness, tip_thickness):
    """Roll, pitch, and yaw moments of inertia about the CG, in kg.m^2.

    The wing skin is treated as a lamina with mass spread over the planform, and
    the fixed mass as a point at the centre.  Both are crude, but the ratios
    between axes -- which is what sets the relative difficulty of each -- come
    out about right for a flying wing, and roll inertia in particular is
    dominated by the span term, which is modelled honestly.

    Roll uses the span, pitch the chord, and yaw both, which is why a long-span
    low-chord wing is sluggish in roll and quick in pitch.
    """
    w_mass = wing_mass(root_chord, tip_chord, root_thickness, tip_thickness)
    _, mac = planform(root_chord, tip_chord)

    # Lamina about its own centroid: b^2/12 for roll, c^2/12 for pitch.  The
    # taper concentrates mass inboard, which the uniform assumption overstates
    # slightly; at this taper the error is a few percent.
    i_roll = w_mass * SPAN ** 2 / 12.0
    i_pitch = w_mass * mac ** 2 / 12.0

    # The fixed mass sits near the centreline, so it adds little to roll but
    # does add to pitch, spread over roughly the battery length.
    i_pitch = i_pitch + FIXED_MASS * (BATTERY_LENGTH ** 2) / 12.0

    # Perpendicular axis theorem for a lamina: yaw is the sum of the other two.
    i_yaw = i_roll + i_pitch
    return i_roll, i_pitch, i_yaw


def angular_acceleration(moment, inertia_value):
    """Angular acceleration in rad/s^2, the meaningful measure of authority."""
    return moment / jnp.maximum(inertia_value, 1e-12)


# --- Design point evaluation --------------------------------------------------


DESIGN_VARS = ["root_chord", "tip_chord", "root_thickness", "tip_thickness",
               "x_hinge", "elevon_inboard_frac", "motor_frac", "servo_station",
               "servo_span_frac"]


def unpack(p):
    """Design vector to named geometry, with span fractions turned into metres."""
    (root_chord, tip_chord, root_t, tip_t, x_hinge, elevon_in_f, motor_f,
     servo_station, servo_span_f) = p
    semi = 0.5 * SPAN
    servo_chord, servo_thickness = local_geometry(
        root_chord, tip_chord, root_t, tip_t, servo_span_f)
    return {
        "root_chord": root_chord,
        "tip_chord": tip_chord,
        "root_thickness": root_t,
        "tip_thickness": tip_t,
        "x_hinge": x_hinge,
        "elevon_inboard_y": elevon_in_f * semi,
        "elevon_outboard_y": semi,
        "motor_y": motor_f * semi,
        "servo_station": servo_station,
        "servo_span_frac": servo_span_f,
        "servo_y": servo_span_f * semi,
        "servo_chord": servo_chord,
        "servo_thickness": servo_thickness,
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

    i_roll, i_pitch, i_yaw = inertia(
        g["root_chord"], g["tip_chord"],
        g["root_thickness"], g["tip_thickness"])

    # Throttle needed in each regime, which is what limits differential thrust
    # and so yaw authority.  Hover sits near the thrust required to hold the
    # aircraft up; cruise needs far less.
    throttles = {"hover": 1.0 / jnp.maximum(thrust_to_weight(mass), 1e-9),
                 "transition": 0.8, "cruise": 0.3, "idle_descent": 0.05}

    authority = {}
    for name, (v, thrust) in conditions.items():
        m_pitch = pitch_moment(
            deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
            g["elevon_inboard_y"], g["elevon_outboard_y"], area, mac)
        m_roll = roll_moment(
            deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
            g["elevon_inboard_y"], g["elevon_outboard_y"], mac)
        m_yaw = yaw_moment(
            available_differential_thrust(throttles[name]), g["motor_y"])
        authority[name] = {
            "q": elevon_dynamic_pressure(
                thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"]),
            "pitch": m_pitch,
            "roll": m_roll,
            "yaw": m_yaw,
            "hinge": hinge_moment(
                deflection_deg, g["x_hinge"], thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"], mac),
            # Angular accelerations, which is what "enough authority" means.
            "alpha_pitch": angular_acceleration(m_pitch, i_pitch),
            "alpha_roll": angular_acceleration(m_roll, i_roll),
            "alpha_yaw": angular_acceleration(m_yaw, i_yaw),
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
        "volume_slack": volume_slack(
            g["root_chord"], g["tip_chord"],
            g["root_thickness"], g["tip_thickness"],
            g["servo_station"], g["servo_span_frac"], g["x_hinge"]),
        "servo_depth_available": thickness_at(
            g["servo_station"], g["servo_thickness"]),
        "pushrod_length": pushrod_length(
            g["servo_chord"], g["servo_station"], g["x_hinge"]),
        "yaw_moment": yaw_moment(0.5 * thrust_hover, g["motor_y"]),
        "wash_fraction": washed_span_fraction(
            g["motor_y"], g["elevon_inboard_y"], g["elevon_outboard_y"]),
        "flap_effectiveness": flap_effectiveness_ratio(g["x_hinge"]),
        "i_roll": i_roll,
        "i_pitch": i_pitch,
        "i_yaw": i_yaw,
        "authority": authority,
    }


evaluate_jit = jax.jit(evaluate, static_argnames=())


# --- Objective ----------------------------------------------------------------
#
# Minimize stall speed subject to floors on hover control authority and on
# thrust-to-weight, with the packaging constraints enforced throughout.
#
# Stall speed is the right thing to minimize because it is what makes the
# aircraft launchable, landable, and forgiving in transition, and because it is
# a single honest scalar rather than a weighted blend of incommensurable terms.
# Everything else enters as a constraint, which is where the real design
# knowledge lives: a constraint says "this must be true", which is a statement
# you can defend, whereas a weight says "this is worth 0.3 of that", which
# usually is not.
#
# The authority floors are set in hover for pitch and roll, because with no
# freestream those axes have only prop wash to work with.  Yaw is floored in
# cruise instead: differential thrust is limited by throttle headroom, and at
# cruise throttle there is less of it than in hover, so hover is not the binding
# case for yaw the way it is for the other two axes.

# Minimum angular accelerations, rad/s^2.  These decide the whole design, so
# they are derived rather than guessed: each is the acceleration that swings the
# aircraft 30 degrees in a stated time, from theta = a t^2 / 2, which is the
# form a pilot or a rate controller actually cares about.
#
#     30 deg in 150 ms  ->  47 rad/s^2      crisp, quad-like
#     30 deg in 250 ms  ->  17 rad/s^2      adequate for attitude hold
#     30 deg in 400 ms  ->   6.5 rad/s^2    sluggish but flyable
#
# Pitch and roll are floored for crisp response in hover, where a tailsitter is
# balancing and has to reject gusts. Yaw gets the sluggish floor: it is the weak
# axis on a twin without a rudder, it matters least for stability, and holding
# it to the same standard would drive the motors outboard for no real gain.
#
# Worth knowing that at the current geometry every one of these is satisfied
# with an order of magnitude to spare, because a 42 g aircraft has very little
# inertia.  They are floors that keep a shrinking design honest, not targets.
MIN_ALPHA_PITCH_HOVER = 47.0
MIN_ALPHA_ROLL_HOVER = 47.0
MIN_ALPHA_YAW_CRUISE = 6.5
MIN_TWR = 1.3

# Reynolds number below which the section data underpinning this model stops
# meaning much.  Not a hard physical limit, but a statement that the model
# should not be trusted to rank designs past it.
MIN_RE_TIP = 25000.0

# Chord is capped by the printer bed the same way span is.  Without this the
# optimizer drives the chord up without limit, because in this model area is
# almost free: it lowers stall speed and the only thing pushing back is skin
# mass.  That is a real gap -- a very low aspect ratio wing has poor lift curve
# slope, high induced drag, and in a tailsitter presents a large sail area to
# gusts in hover -- but the printer bound is the honest constraint to state
# here, rather than inventing an aerodynamic penalty the model cannot compute.
MAX_CHORD = 0.256

# Aspect ratio floor.  Below roughly 2.5 the lifting-line and thin-airfoil
# assumptions behind every lift number in this model break down: a low aspect
# ratio wing carries much of its lift through nonlinear vortex effects that
# nothing here represents, and Cl_max in particular would be badly overstated.
# So this is a validity bound on the model, not a claim about what flies well.
MIN_ASPECT_RATIO = 2.5

# Large enough that a violated constraint always costs more than the stall speed
# it could buy.  Stall speed is order 5 m/s and a 1% shortfall squares to 1e-4,
# so the weight has to be big for the penalty to bite at all near the boundary;
# at 5000 a 1% violation costs 0.5 m/s of equivalent stall speed, which is more
# than the optimizer can usually gain by cheating.
CONSTRAINT_WEIGHT = 5000.0


def _shortfall(value, floor):
    """Fractional shortfall below a floor, zero when satisfied.

    Normalized by the floor so constraints in different units contribute
    comparably, and squared by the caller so the penalty is smooth at the
    boundary rather than kinked.
    """
    return jnp.maximum(0.0, floor - value) / jnp.maximum(jnp.abs(floor), 1e-9)


def constraints(p, cl_max=0.8, deflection_deg=10.0):
    """Each constraint's fractional shortfall.  All zero means feasible."""
    r = evaluate(p, cl_max=cl_max, deflection_deg=deflection_deg)
    hover = r["authority"]["hover"]
    cruise = r["authority"]["cruise"]

    return {
        "alpha_pitch_hover": _shortfall(
            jnp.abs(hover["alpha_pitch"]), MIN_ALPHA_PITCH_HOVER),
        "alpha_roll_hover": _shortfall(
            jnp.abs(hover["alpha_roll"]), MIN_ALPHA_ROLL_HOVER),
        "alpha_yaw_cruise": _shortfall(
            jnp.abs(cruise["alpha_yaw"]), MIN_ALPHA_YAW_CRUISE),
        "twr": _shortfall(r["twr"], MIN_TWR),
        "re_tip": _shortfall(r["re_tip"], MIN_RE_TIP),
        "aspect_ratio": _shortfall(r["aspect_ratio"], MIN_ASPECT_RATIO),
        # Packaging: volume_slack is already a signed distance in metres, so a
        # floor of zero with a millimetre-scale normalization keeps it on the
        # same footing as the others.
        "packaging": jnp.maximum(0.0, -r["volume_slack"]) / 0.001,
    }


def cost(p, cl_max=0.8, deflection_deg=10.0):
    """Stall speed plus penalties for violated constraints.

    A penalty method rather than a projection: the constraints couple through
    the geometry (thickness feeds mass feeds stall speed feeds Reynolds number),
    so there is no cheap feasible set to project onto, and squared shortfalls
    keep the whole thing differentiable for the same reason the linkage model
    smooths its dead-point penalty.
    """
    r = evaluate(p, cl_max=cl_max, deflection_deg=deflection_deg)
    violations = constraints(p, cl_max=cl_max, deflection_deg=deflection_deg)
    penalty = sum(v ** 2 for v in violations.values())
    return r["v_stall"] + CONSTRAINT_WEIGHT * penalty


cost_jit = jax.jit(cost)
constraints_jit = jax.jit(constraints)


# Current best guess at the configuration, in DESIGN_VARS order.  Root chord is
# above the 65 mm the battery alone would need, mostly to keep the thickness
# ratio sane at this Reynolds number once the section is deep enough to hold
# everything.
BASELINE = jnp.array([
    0.105,   # root_chord, m
    0.074,   # tip_chord, m  (taper 0.70; less taper keeps tip Re up)
    0.016,   # root_thickness, m
    0.006,   # tip_thickness, m
    0.75,    # x_hinge  (25% chord elevon, where dCm/deta peaks)
    0.30,    # elevon_inboard_frac
    0.47,    # motor_frac
    0.45,    # servo_station, chord fraction where the servo body sits
    0.40,    # servo_span_frac, outboard of the battery, near its elevon
])


def report(p=None, v_cruise=12.0, cl_max=0.8, deflection_deg=10.0):
    """Print a readable summary of one design point."""
    p = BASELINE if p is None else p
    g = unpack(p)
    r = evaluate(p, v_cruise=v_cruise, cl_max=cl_max,
                 deflection_deg=deflection_deg)

    print("=== Geometry ===")
    print(f"  root chord      {float(g['root_chord']) * 1e3:8.1f} mm")
    print(f"  tip chord       {float(g['tip_chord']) * 1e3:8.1f} mm"
          f"   (taper {float(g['tip_chord'] / g['root_chord']):.2f})")
    print(f"  root t/c        {float(r['root_tc']) * 100:8.1f} %"
          f"   ({float(g['root_thickness']) * 1e3:.1f} mm)")
    print(f"  tip t/c         {float(r['tip_tc']) * 100:8.1f} %"
          f"   ({float(g['tip_thickness']) * 1e3:.1f} mm)")
    print(f"  area            {float(r['area']) * 1e4:8.1f} cm^2")
    print(f"  aspect ratio    {float(r['aspect_ratio']):8.2f}")
    print(f"  MAC             {float(r['mac']) * 1e3:8.1f} mm")

    print("\n=== Mass and performance ===")
    print(f"  all-up mass     {float(r['mass']) * 1e3:8.1f} g")
    print(f"  thrust/weight   {float(r['twr']):8.2f}"
          f"   {'OK' if float(r['twr']) > 1.0 else 'CANNOT HOVER'}")
    print(f"  wing loading    {float(r['wing_loading']):8.1f} N/m^2")
    print(f"  stall speed     {float(r['v_stall']):8.1f} m/s"
          f"   (at Cl_max {cl_max})")
    print(f"  Re at MAC       {float(r['re_mac']):8,.0f}")
    print(f"  Re at tip       {float(r['re_tip']):8,.0f}"
          f"   {'-- very low' if float(r['re_tip']) < 30000 else ''}")

    print("\n=== Packaging ===")
    slack = float(r['volume_slack']) * 1e3
    print(f"  volume slack    {slack:+8.1f} mm"
          f"   {'FITS' if slack >= 0 else 'DOES NOT FIT'}")

    # Report every packaging constraint with its own slack, so the binding one
    # is visible rather than hidden inside a single minimum.  Which one binds
    # moves around as the servo station and root chord change, and knowing
    # which is what tells you what to go fix.
    rc = float(g["root_chord"])
    rt = float(g["root_thickness"])
    sc = float(g["servo_chord"])
    terms = {
        "battery chord": rc - BATTERY_LENGTH,
        "min thickness": rt - MIN_ROOT_THICKNESS,
        "servo depth": float(r["servo_depth_available"]) - SERVO_DEPTH,
        "servo/hinge clearance": (
            float(g["x_hinge"]) - float(g["servo_station"])
            - 0.5 * SERVO_LENGTH / sc) * sc,
        "servo outboard of battery": (
            float(g["servo_y"]) - 0.5 * SERVO_WIDTH),
    }
    binding = min(terms, key=terms.get)
    for name, value in terms.items():
        mark = "  <-- binding" if name == binding else ""
        print(f"    {name:<26}{value * 1e3:+7.1f} mm{mark}")
    print(f"    servo at {float(g['servo_station']) * 100:.0f}% chord,"
          f" {float(g['servo_span_frac']) * 100:.0f}% semi-span"
          f" ({sc * 1e3:.0f} mm local chord),"
          f" pushrod {float(r['pushrod_length']) * 1e3:.1f} mm")

    print("\n=== Elevon and motors ===")
    print(f"  hinge at        {float(g['x_hinge']) * 100:8.0f} % chord"
          f"   ({(1 - float(g['x_hinge'])) * 100:.0f}% chord elevon)")
    print(f"  effectiveness   {float(r['flap_effectiveness']):8.3f}"
          f"   deg alpha per deg elevon")
    print(f"  elevon span     {float(g['elevon_inboard_y']) * 1e3:.0f}"
          f" to {float(g['elevon_outboard_y']) * 1e3:.0f} mm from centreline")
    print(f"  motor at        {float(g['motor_y']) * 1e3:8.1f} mm"
          f"   ({float(g['motor_y']) / (0.5 * SPAN) * 100:.0f}% semi-span)")
    print(f"  wash fraction   {float(r['wash_fraction']):8.2f}"
          f"   of elevon span in the slipstream")
    print(f"  yaw moment      {float(r['yaw_moment']) * 1e3:8.2f} mN.m"
          f"   at 50% differential thrust")

    print(f"\n=== Control authority at {deflection_deg:.0f} deg deflection ===")
    print(f"  {'condition':<14}{'q [Pa]':>9}{'pitch':>10}{'roll':>10}"
          f"{'hinge':>10}   (mN.m)")
    for name, a in r["authority"].items():
        print(f"  {name:<14}{float(a['q']):9.1f}{float(a['pitch']) * 1e3:10.3f}"
              f"{float(a['roll']) * 1e3:10.3f}{float(a['hinge']) * 1e3:10.3f}")

    h_max = max(float(a["hinge"]) for a in r["authority"].values())
    print(f"\n  peak hinge moment {h_max * 1e3:.3f} mN.m"
          f" = {h_max * 1e4 / G:.2f} g.cm"
          f"  ({3 * h_max * 1e4 / G:.2f} g.cm with 3x margin)")

    print("\n=== Deflection roll-off ===")
    for d in (5, 10, 15, 20, 25, 30):
        eff = float(flap_deflection_efficiency(float(d)))
        print(f"  {d:2d} deg   efficiency {eff:.3f}"
              f"   worth {d * eff:5.1f} deg of ideal deflection")

    return r


if __name__ == "__main__":
    report()
