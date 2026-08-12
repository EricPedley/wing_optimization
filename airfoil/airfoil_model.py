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

import functools

import jax
import jax.numpy as jnp

from airfoil import linkage_coupling

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
THRUST_PER_MOTOR = 0.035     # kgf, ~35 g.  UNMEASURED -- see note above.
PROP_DIAMETER = 0.051        # m, 2 inch

# --- Mass breakdown -----------------------------------------------------------
#
# These used to be a single lumped FIXED_MASS of 25 g, which was enough while
# nothing asked *where* the mass sat.  The centre of gravity does ask, and the
# answer decides whether the aircraft is stable in pitch at all, so the lump is
# split into items with chordwise stations.  The sum is still 25 g -- the split
# reapportions the same mass rather than changing it, so thrust-to-weight and
# stall speed are untouched by this refactor and any movement in them is a real
# consequence of the geometry rather than an accounting change.
#
# The split is estimated, not weighed.  Motors and props dominate the forward
# group and the battery the aft, which is the part that matters for the CG; the
# avionics are small and near the middle where their exact station matters least.
BATTERY_MASS = 0.008         # kg, the 1S cell
MOTOR_MASS = 0.0075          # kg, both motors with their props and mounts
AVIONICS_MASS = 0.0055       # kg, FC + RX + ESC + wiring, near the centre
SERVO_MASS = 0.004           # kg, both servos

FIXED_MASS = BATTERY_MASS + MOTOR_MASS + AVIONICS_MASS + SERVO_MASS  # kg

# Chord fraction where the avionics stack sits.  Small and centrally placed, so
# this is a weak lever on the CG compared with the battery and the motors; it is
# named rather than buried so the CG sum reads as a complete accounting.
AVIONICS_STATION = 0.35

# --- Motor mounting geometry --------------------------------------------------
#
# The motor station is derived rather than chosen, because it is not free: the
# prop has to clear the leading edge, and that clearance plus the motor's own
# length is the whole of it.
#
# Working aft from the prop disk:
#
#   * the prop tip sweeps a circle of radius PROP_DIAMETER/2 about the shaft, so
#     the leading edge must sit at least that far aft of the disk plus a margin,
#     or the blade strikes the wing;
#   * the motor body occupies MOTOR_BODY_LENGTH between the disk and its mount
#     face;
#   * the mount face itself has to stand proud of the leading edge by
#     MOTOR_MOUNT_STANDOFF so it is a flat pad rather than a knife edge.
#
# The motor's mass centre sits about halfway along its body, which is what the
# CG calculation wants, and that lands *ahead* of the leading edge -- a negative
# chord station.  That is the point: the motors are the only significant mass
# forward of the wing, and they are what makes a tailless layout balance.
MOTOR_BODY_LENGTH = 0.004    # m, disk plane to mount face, 4 mm
MOTOR_MOUNT_STANDOFF = 0.003  # m, mount pad proud of the leading edge
# Radial gap between the prop tip circle and the leading edge.  Not zero: the
# blade flexes under load and the wing is not a perfect surface.
PROP_TIP_CLEARANCE = 0.004   # m

BATTERY_LENGTH = 0.065       # m, drives the minimum root chord
BATTERY_THICKNESS = 0.011    # m, depth the section must provide where it sits
MIN_ROOT_THICKNESS = 0.015   # m, floor regardless of what the battery needs

# Chord fraction where the battery's forward face sits.  A design variable now
# rather than a constant, because it trades against root thickness in a way that
# is not obvious: forward is better for the centre of gravity, but the section
# is still thin near the nose, so a battery pushed forward forces a thicker root
# to keep the same depth under it.  The starting value is nose-ward; the
# optimizer moves it.
BATTERY_STATION_DEFAULT = 0.05

# The servo is a box, and which of its dimensions binds depends on how it is
# mounted, so all three are named separately rather than collapsed into one
# "servo thickness".  Dimensions below are from CAD.  SERVO_LENGTH runs along
# the chord when the servo is mounted conventionally; SERVO_DEPTH is the
# dimension that has to fit inside the section thickness.
#
# At 15 mm deep this servo is nearly as thick as the whole root section, so it
# is the dominant packaging constraint and the reason servo_chord_frac is a
# design variable: it only fits near maximum thickness, and it may not fit at
# all without either a deeper root or mounting it on its side.
SERVO_LENGTH = 0.021         # m, longest dimension, lies along the chord
SERVO_DEPTH = 0.008          # m, the dimension that fights section thickness
SERVO_WIDTH = 0.015          # m, spanwise

# Linear servo output force at stall.
#
# PLACEHOLDER -- not from the datasheet.  Replace with the real stall force,
# derated for the duty cycle the elevon actually sees.  Everything downstream of
# this number is only as good as it is, and the elevon size the optimizer
# returns moves roughly as its square root.
SERVO_MAX_FORCE = 1.0        # N

# The linkage's mechanical advantage used to be a constant here, measured from a
# separate optimization of the linkage alone.  It is now a design output --
# the linkage geometry is part of the same design vector as the wing, and the
# advantage falls out of it.  See airfoil.linkage_coupling.

# Safety factor on the required servo force.
#
# The hinge moment model is the weakest part of this file -- see hinge_moment,
# whose coefficients are representative plain-flap values rather than anything
# derived or measured -- so the constraint is applied against a derated servo
# rather than pretending the prediction is tight.
SERVO_FORCE_MARGIN = 1.5

# Wiring reach.  Both are hard limits set by the harness that already exists,
# not by aerodynamics, and both bind against things the optimizer wants: it
# pushes the motors outboard for yaw authority and the servos outboard to sit
# in thinner, shorter-chord section.  Lengthening either harness would buy real
# performance, so these are worth revisiting rather than treating as fixed.
MAX_MOTOR_Y = 0.055          # m from the centreline
MAX_SERVO_Y = 0.060          # m from the centreline

# Elevon planform.  With a constant hinge *fraction* the hinge line converges on
# the trailing edge as the chord tapers, so the elevon shrinks outboard -- on
# this wing from 18 mm at the root to 9 mm at the tip, which is impractically
# small to hinge and horn, and puts the least surface where the roll moment arm
# is longest.  A constant elevon *chord* instead keeps the hinge line parallel
# to the trailing edge, which is what flying wings are normally built with and
# what is far easier to print and hinge.
#
# The cost is that the hinge fraction then varies along the span (0.14 at the
# root to 0.28 at the tip here), so the section flap derivatives vary too.  They
# are evaluated at the mean aerodynamic chord, which is the standard
# approximation and good to a few percent over this range.
CONSTANT_CHORD_ELEVON = False

# Minimum elevon chord that can actually be built: below this there is no room
# for a hinge, a horn, and a pushrod attachment.
MIN_ELEVON_CHORD = 0.012     # m

# Fixed wing left outboard of the elevon, so the hinge has structure to anchor
# into at its outer end rather than terminating in mid-air at the wingtip.
ELEVON_TIP_MARGIN = 0.010    # m

# Width of the constant-chord centre section, tip to tip across the centreline,
# so half of it sits either side.  The root section is held at root_chord over
# this strip and only then starts tapering toward the tip, which is what gives
# the centreline hardware -- the battery in particular -- a parallel-sided bay
# to sit in instead of a wedge that starts losing chord immediately.
#
# Constant rather than a design variable for now: it trades printability and
# packaging room against a little area and mass, and there is nothing in the
# objective that can price that trade honestly yet.
ROOT_SECTION_WIDTH = 0.010   # m, full width across the centreline

# Thickness ratio band, from published low-Reynolds-number section data rather
# than from anything this model computes -- nothing here predicts Cl_max against
# thickness, so this is an imported bound of the same character as the aspect
# ratio floor.  Between roughly Re 30,000 and 70,000 the useful range is 6-9%:
# below about 4% the leading edge is sharp enough that stall becomes abrupt and
# very sensitive to angle of attack, and above about 12% the laminar boundary
# layer separates over the aft upper surface and Cl_max falls.
#
# Without this the optimizer drives the tip to whatever minimum thickness is
# allowed, because thinner means less skin area means less mass means lower
# stall speed, and nothing else pushes back.  That trades a badly behaved tip
# section for a fraction of a gram.
MIN_THICKNESS_RATIO = 0.06
MAX_THICKNESS_RATIO = 0.12

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

# Cl_max correlation.  See max_lift_coefficient for what these mean and how far
# they should be trusted, which is not very far: the base value is the flat-plate
# figure this model used as a constant before camber existed, and the gains are
# representative of published low-Reynolds-number section data rather than fitted
# to any particular one.
#
# CL_MAX_CAMBER_GAIN is the important one.  At 2% camber it adds about 0.16 to
# Cl_max, which is roughly a 10% reduction in stall speed -- enough that camber
# is worth having, not so much that it overwhelms the trim cost that pays for it.
# If this number is badly wrong, the balance between camber and reflex is wrong
# with it, which is the first thing to check if the optimum looks strange.
CL_MAX_BASE = 0.80
CL_MAX_CAMBER_GAIN = 8.0     # per unit camber/chord
CL_MAX_TC_GAIN = 1.5         # per unit thickness ratio, about the reference
CL_MAX_TC_REF = 0.085        # t/c where the band is happiest
CL_MAX_RE_GAIN = 0.12        # per e-fold of Reynolds number
CL_MAX_RE_REF = 40000.0      # the Reynolds number this wing actually flies at
# Nothing at this Reynolds number reaches a Cl_max like this; the ceiling exists
# so an optimizer probing absurd camber cannot buy unlimited stall speed.
CL_MAX_CEILING = 1.35


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


# --- Camber -------------------------------------------------------------------
#
# The camber line is parameterized by its Birnbaum-Glauert coefficients A1 and
# A2 rather than by a geometric shape, which is worth explaining because it is
# the unusual choice.
#
# Thin-airfoil theory only ever sees the camber line through the first three
# coefficients of the Fourier series for its slope:
#
#     dz/dx(theta) = A0' + A1 cos(theta) + A2 cos(2 theta) + ...
#     Cl      = 2 pi (alpha - alpha_L0)
#     Cm_c/4  = (pi / 4) (A2 - A1)
#
# Everything past A2 changes the shape without changing either the lift or the
# moment, so a parameterization carrying more terms would hand the optimizer
# variables the objective cannot see -- exactly the failure documented at
# yaw_moment, where a flat cost returns whatever the random start held.
#
# Parameterizing by A1 and A2 directly has a second payoff.  Reflex -- the
# trailing-edge upsweep a tailless wing uses to trim itself -- is the condition
# Cm_c/4 = 0, which in these coordinates is exactly A2 = A1.  As a geometric
# shape, say a NACA 5-digit reflexed section, the same condition is a root-solve
# against fitted table constants that are not differentiable in the design
# variable.  Here it is algebra, and the gradient is exact.
#
# The cost is that the design variables are no longer directly meaningful to
# look at, which is what camber_line and camber_max exist to undo.


def camber_slope(x, a1, a2):
    """dz/dx of the camber line at chord fraction ``x``.

    The A0' term is omitted deliberately: it is a uniform incidence of the whole
    camber line, which is indistinguishable from flying at a different angle of
    attack and would double-count against alpha.  What is left is the shape.
    """
    theta = hinge_angle(x)
    return a1 * jnp.cos(theta) + a2 * jnp.cos(2.0 * theta)


def camber_line(x, a1, a2):
    """Camber line height z/c at chord fraction ``x``.

    Integrated from the slope in the theta variable, where the integral is
    closed form.  With x = (1 - cos theta)/2, dx = (sin theta / 2) dtheta,

        z(theta) = integral from 0 to theta of (A1 cos u + A2 cos 2u)(sin u / 2) du
                 = A1 sin^2(theta) / 4
                 + A2 (-cos^3(theta)/3 + cos(theta)/2 - 1/6)

    The A1 term closes on its own -- it is zero at both ends -- but the A2 term
    is not: it reaches -A2/3 at the trailing edge.  That is not an algebra slip
    but a real property of the series, and it means the raw integral describes a
    line whose endpoints do not both lie on the chord line.

    Subtracting the linear ramp that joins those endpoints is what makes this a
    camber line *measured from the chord*, which is the definition every other
    quantity here assumes.  Geometrically the ramp is a uniform rotation of the
    section, indistinguishable from flying at a different angle of attack, so
    removing it here and letting alpha carry it is the same bookkeeping choice
    that drops the A0' term in camber_slope.  Neither Cl nor Cm_c/4 is affected.
    """
    theta = hinge_angle(x)
    term_a1 = a1 * jnp.sin(theta) ** 2 / 4.0
    cos_t = jnp.cos(theta)
    term_a2 = a2 * (-cos_t ** 3 / 3.0 + cos_t / 2.0 - 1.0 / 6.0)
    # Re-reference to the straight line from the leading to the trailing edge.
    # Only the A2 term has anything to remove; x is the fraction along it.
    return term_a1 + term_a2 + a2 * x / 3.0


def alpha_zero_lift(a1, a2):
    """Zero-lift angle of attack, in radians.

    Negative for a conventionally cambered section: it already makes lift at
    zero incidence, so it reaches zero lift only when pitched nose-down.  This
    is why a cambered wing cruises at a lower angle of attack than a symmetric
    one of the same area, which shows up downstream as a slightly smaller hinge
    moment.
    """
    return -0.5 * a1


def cm_quarter_chord(a1, a2):
    """Section pitching moment about the quarter chord.  Inviscid.

    Independent of angle of attack -- that is what makes the quarter chord the
    aerodynamic centre -- so this is the constant the wing has to trim against.
    Negative (nose-down) for ordinary camber, zero when A2 = A1, and positive
    for reflex, which is the whole reason reflex exists.
    """
    return 0.25 * jnp.pi * (a2 - a1)


def camber_max(a1, a2, n=41):
    """Largest |z|/c on the camber line, and the station where it occurs.

    Sampled rather than solved: the extremum of a two-term series has a closed
    form, but it involves a case split on which term dominates, and a split is a
    discontinuity the optimizer would feel.  A fixed grid is smooth in the
    coefficients, which matters more here than the last fraction of a percent of
    accuracy, since this feeds an already-approximate Cl_max correlation.
    """
    x = jnp.linspace(0.0, 1.0, n)
    z = camber_line(x, a1, a2)
    k = jnp.argmax(jnp.abs(z))
    return jnp.abs(z[k]), x[k]


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

    Note the objective cannot currently see any of that.  Every authority floor
    is met with a wide margin at every reachable motor position, and stall speed
    does not depend on where the motors sit, so the cost is flat in motor_frac
    and the optimizer leaves it wherever it started.  Motor position is chosen
    by :data:`MOTOR_FRAC_PREFERENCE` below rather than optimized; the sweep in
    ``sweep.py`` is what shows the tradeoff.
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


def servo_force(hinge_moment_nm, advantage):
    """Servo force needed to hold a hinge moment, in N.

    Virtual work through the linkage: the servo pushes F along its rail while
    the elevon absorbs tau at the hinge, so F = tau / (dx/dtheta) and the
    denominator is exactly the mechanical advantage the linkage model reports.

    ``advantage`` is a design output rather than a constant -- it comes from the
    linkage geometry, which the optimizer chooses -- so it is passed in.  See
    airfoil.linkage_coupling for the units and for why it is no longer pasted
    in from a separate optimization.

    This is the term that punishes an oversized elevon.  Authority grows about
    linearly with elevon chord while hinge moment grows with its square, so past
    some size the servo saturates and further chord buys deflection the servo
    cannot hold -- which is to say, no authority at all.
    """
    return hinge_moment_nm / jnp.maximum(advantage, 1e-9)


# --- Geometry and mass --------------------------------------------------------


def root_section_fraction():
    """Semi-span fraction occupied by the constant-chord centre section."""
    return jnp.clip(0.5 * ROOT_SECTION_WIDTH / (0.5 * SPAN), 0.0, 1.0)


def taper_fraction(span_fraction):
    """How far into the taper a station sits, 0 at the root, 1 at the tip.

    Zero across the whole constant-chord centre section, then rising linearly
    over what is left of the semi-span.  Everything that lofts between the root
    and tip sections -- chord, thickness, and the leading edge -- goes through
    this, so the parallel-sided centre strip exists once rather than being
    re-derived in every consumer.
    """
    f0 = root_section_fraction()
    remaining = jnp.maximum(1.0 - f0, 1e-9)
    return jnp.clip((span_fraction - f0) / remaining, 0.0, 1.0)


def planform(root_chord, tip_chord):
    """Wing area and mean aerodynamic chord.

    The wing is a constant-chord centre section of width ROOT_SECTION_WIDTH
    joined to a straight-tapered outer panel, so both quantities are the
    span-weighted combination of a rectangle and a trapezoid rather than the
    single trapezoid a pure taper would give.

    MAC is the standard integral (2/S) * integral of c^2 over the semi-span,
    which for these two pieces is closed form: the rectangle contributes
    c_root^2 over its span, the trapezoid the usual (2/3) c_r (1+L+L^2)/(1+L).
    """
    f0 = root_section_fraction()
    semi = 0.5 * SPAN
    span_root = f0 * semi          # per side, constant-chord strip
    span_taper = (1.0 - f0) * semi  # per side, tapered panel

    area_root = root_chord * span_root
    area_taper = 0.5 * (root_chord + tip_chord) * span_taper
    area = 2.0 * (area_root + area_taper)

    # Integral of c^2 over each piece.
    int_c2_root = root_chord ** 2 * span_root
    int_c2_taper = (span_taper * (root_chord ** 2 + root_chord * tip_chord
                                  + tip_chord ** 2) / 3.0)
    mac = 2.0 * (int_c2_root + int_c2_taper) / jnp.maximum(area, 1e-12)
    return area, mac


def reynolds(v_inf, chord):
    """Chord-based Reynolds number.

    Worth computing at the tip as well as the mean: the tip chord is smaller, so
    the tip runs at a lower Re and stalls earlier, compounding the tip-stall
    tendency that sweep already creates.
    """
    return RHO * v_inf * chord / MU


# --- Sweep --------------------------------------------------------------------
#
# Area and mean aerodynamic chord depend only on the chord distribution, not on
# where the sections sit fore and aft, so nothing above needs sweep.  What sweep
# changes is where the lift acts, which is what a stability model would use --
# and there is not one here yet.  These functions exist so the geometry is
# stated explicitly rather than implied by whatever the plots happen to draw,
# and so the quarter-chord sweep is visible: with a straight leading edge and
# taper, the quarter-chord line sweeps *forward*, which is the wrong direction
# for a tailless aircraft.


def leading_edge_x(span_fraction, le_sweep_deg):
    """Leading edge position aft of the root leading edge, in m.

    The centre section is unswept as well as untapered -- it is a straight
    extrusion of the root -- so the sweep starts at the outboard edge of that
    strip.  ``le_sweep_deg`` is the sweep of the outer panel, and the tip ends
    up slightly less far aft than a wing swept from the centreline would.
    """
    f0 = root_section_fraction()
    swept = jnp.maximum(span_fraction - f0, 0.0) * 0.5 * SPAN
    return jnp.tan(jnp.radians(le_sweep_deg)) * swept


def quarter_chord_sweep_deg(root_chord, tip_chord, le_sweep_deg):
    """Sweep of the quarter-chord line, degrees, positive aft.

    The aerodynamically meaningful sweep, since section lift acts near the
    quarter chord.  Taper drags it forward of the leading-edge sweep: each
    section's quarter chord sits a quarter of its own chord aft of its leading
    edge, and outboard chords are shorter, so the quarter-chord line leans
    forward relative to the leading edge by an amount that grows with taper.
    """
    semi = 0.5 * SPAN
    dx = (leading_edge_x(1.0, le_sweep_deg) + 0.25 * tip_chord) - 0.25 * root_chord
    return jnp.degrees(jnp.arctan2(dx, semi))


def trailing_edge_sweep_deg(root_chord, tip_chord, le_sweep_deg):
    """Sweep of the trailing edge, degrees, positive aft."""
    semi = 0.5 * SPAN
    dx = (leading_edge_x(1.0, le_sweep_deg) + tip_chord) - root_chord
    return jnp.degrees(jnp.arctan2(dx, semi))


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


# --- Balance ------------------------------------------------------------------
#
# Everything in this section measures chordwise position aft of the *root*
# leading edge, in metres.  That origin is arbitrary but it has to be stated,
# because the centre of gravity and the aerodynamic centre are only meaningful
# relative to each other and a sign error between them inverts the stability
# conclusion.  Aft is positive throughout.


def motor_station():
    """Chordwise position of the motor mass centre, in m aft of the root LE.

    Derived from the prop rather than chosen.  Working forward from the leading
    edge: the mount pad stands proud by MOTOR_MOUNT_STANDOFF, the motor body
    occupies MOTOR_BODY_LENGTH ahead of that, and the prop disk sits at its
    front.  The mass centre is taken at the middle of the body.

    The result is negative -- the motors sit ahead of the wing -- which is the
    entire reason a tailless aircraft with the battery inside the wing can
    balance at all.

    Prop tip clearance is checked separately in :func:`prop_clearance_slack`
    rather than folded in here, because it constrains how far *outboard* the
    motor must sit for the blade to miss the leading edge, which is a different
    question from where the motor's mass acts.
    """
    return -(MOTOR_MOUNT_STANDOFF + 0.5 * MOTOR_BODY_LENGTH)


def prop_clearance_slack(le_sweep_deg, motor_y):
    """Gap between the prop tip circle and the leading edge, in m.

    The prop is a disk of radius PROP_DIAMETER/2 standing ahead of the wing on
    a mount MOTOR_MOUNT_STANDOFF proud of the leading edge.  Its blade sweeps
    inboard as well as outboard, so the tip passes closest to the wing on the
    inboard side, where the leading edge is further forward than at the motor's
    own station because of sweep.

    Positive means the blade clears.  This is what stops the optimizer from
    sliding the motors inboard for packaging reasons until the props would
    strike the wing -- a constraint that did not exist while the motor station
    was not modelled at all.
    """
    semi = 0.5 * SPAN
    tip_radius = 0.5 * PROP_DIAMETER

    # Where the blade tip passes on the inboard side, and how far aft the
    # leading edge has fallen back by the motor's own station.  With aft sweep
    # the inboard leading edge is *forward* of the motor's, so the gap closes.
    inboard_y = jnp.maximum(motor_y - tip_radius, 0.0)
    le_at_motor = leading_edge_x(motor_y / semi, le_sweep_deg)
    le_at_inboard = leading_edge_x(inboard_y / semi, le_sweep_deg)

    # The mount pad is at the leading edge of the motor's own station, so the
    # disk plane sits that far forward; the blade tip sweeps back to the
    # inboard leading edge, which is further forward by this much.
    sweep_encroachment = le_at_motor - le_at_inboard
    return MOTOR_MOUNT_STANDOFF + sweep_encroachment - PROP_TIP_CLEARANCE


def wing_centroid_x(root_chord, tip_chord, le_sweep_deg):
    """Chordwise centroid of the wing skin, in m aft of the root LE.

    The skin is a lamina of uniform areal density, so its centroid is the
    area-weighted mean of each station's own mid-chord.  Split into the
    constant-chord centre strip and the tapered panel, matching :func:`planform`
    so the two cannot disagree about what the wing is.

    Sweep enters here and nowhere else in the mass accounting, which is what
    makes le_sweep_deg finally cost something: sweeping the wing aft drags the
    skin centroid aft with it, moving the CG toward the aerodynamic centre and
    eating the static margin that the same sweep was meant to buy.
    """
    f0 = root_section_fraction()
    semi = 0.5 * SPAN
    span_root = f0 * semi
    span_taper = (1.0 - f0) * semi

    # Centre strip: a rectangle, unswept, so its centroid is at half the root
    # chord and its area is chord times span.
    area_root = root_chord * span_root
    x_root = 0.5 * root_chord

    # Tapered panel: integrate c(y) * (le(y) + c(y)/2) over the panel and divide
    # by its area.  Both c and le are linear in y there, so a three-point
    # Simpson rule is exact for the quadratic integrand and avoids writing out
    # an algebraic expression that would have to be re-derived if the loft
    # changed.
    def panel_integrand(t):
        # t is the taper fraction, 0 at the strip edge and 1 at the tip.
        span_fraction = f0 + t * (1.0 - f0)
        chord = root_chord + (tip_chord - root_chord) * t
        le = leading_edge_x(span_fraction, le_sweep_deg)
        return chord, chord * (le + 0.5 * chord)

    c0, m0 = panel_integrand(0.0)
    c1, m1 = panel_integrand(0.5)
    c2, m2 = panel_integrand(1.0)
    simpson = lambda a, b, c: (a + 4.0 * b + c) / 6.0
    area_taper = simpson(c0, c1, c2) * span_taper
    moment_taper = simpson(m0, m1, m2) * span_taper

    total_area = area_root + area_taper
    moment = area_root * x_root + moment_taper
    return moment / jnp.maximum(total_area, 1e-12)


def cg_station(root_chord, tip_chord, root_thickness, tip_thickness,
               le_sweep_deg, battery_station, servo_chord_frac,
               servo_span_frac):
    """Centre of gravity, in m aft of the root leading edge.

    A mass-weighted sum over every item the aircraft is made of.  The wing skin
    and the battery dominate and pull aft; the motors are the only thing pulling
    forward, and they do it with a short arm and a small mass, which is why the
    battery station ends up being the strongest lever the optimizer has on
    balance.

    The servos are placed at their own spanwise station's leading edge plus
    their chordwise fraction, so moving them fore and aft moves the CG the way
    it really would.  It is a small term -- 4 g against 42 -- but it is free to
    include and it stops servo_chord_frac from being decided purely by
    packaging.
    """
    semi = 0.5 * SPAN

    w_mass = wing_mass(root_chord, tip_chord, root_thickness, tip_thickness)
    x_wing = wing_centroid_x(root_chord, tip_chord, le_sweep_deg)

    # Battery: a box on the centreline whose forward face is at its station, so
    # its mass acts half its own length aft of that.
    x_battery = battery_station * root_chord + 0.5 * BATTERY_LENGTH

    # Servos: at their own section, which is swept aft of the root.
    servo_chord, _ = local_geometry(root_chord, tip_chord, root_thickness,
                                    tip_thickness, servo_span_frac)
    x_servo = (leading_edge_x(servo_span_frac, le_sweep_deg)
               + servo_chord_frac * servo_chord)

    # Avionics: near the centreline at a fixed fraction of the root chord.
    x_avionics = AVIONICS_STATION * root_chord

    # Motors: ahead of the leading edge, hence negative.  Their own station is
    # swept aft with the wing, so the sweep does not buy as much forward arm as
    # the bare standoff suggests.
    x_motor = leading_edge_x(0.0, le_sweep_deg) + motor_station()

    mass = w_mass + FIXED_MASS
    moment = (w_mass * x_wing
              + BATTERY_MASS * x_battery
              + SERVO_MASS * x_servo
              + AVIONICS_MASS * x_avionics
              + MOTOR_MASS * x_motor)
    return moment / jnp.maximum(mass, 1e-12)


def aerodynamic_centre_x(root_chord, tip_chord, le_sweep_deg):
    """Wing aerodynamic centre, in m aft of the root leading edge.

    The area-weighted mean of each station's own quarter chord, which is where
    thin-airfoil theory puts the section aerodynamic centre.  For a straight
    tapered wing this is the standard construction and is good to a few percent
    against lifting-line at this aspect ratio.

    What it does *not* include is the aft shift with Mach number (irrelevant
    here) or the small forward shift from a very low aspect ratio.  The latter
    is real at AR 2.5 but it is smaller than the uncertainty already carried by
    the Cl_max correlation, and including it would imply a precision the rest of
    the model does not have.
    """
    f0 = root_section_fraction()
    span_root = f0 * 0.5 * SPAN
    span_taper = (1.0 - f0) * 0.5 * SPAN

    area_root = root_chord * span_root
    x_root = 0.25 * root_chord

    def panel_integrand(t):
        span_fraction = f0 + t * (1.0 - f0)
        chord = root_chord + (tip_chord - root_chord) * t
        le = leading_edge_x(span_fraction, le_sweep_deg)
        return chord, chord * (le + 0.25 * chord)

    c0, m0 = panel_integrand(0.0)
    c1, m1 = panel_integrand(0.5)
    c2, m2 = panel_integrand(1.0)
    simpson = lambda a, b, c: (a + 4.0 * b + c) / 6.0
    area_taper = simpson(c0, c1, c2) * span_taper
    moment_taper = simpson(m0, m1, m2) * span_taper

    total_area = area_root + area_taper
    moment = area_root * x_root + moment_taper
    return moment / jnp.maximum(total_area, 1e-12)


def static_margin(root_chord, tip_chord, root_thickness, tip_thickness,
                  le_sweep_deg, battery_station, servo_chord_frac,
                  servo_span_frac, mac):
    """Static margin, as a fraction of the mean aerodynamic chord.

    Positive means the centre of gravity is ahead of the aerodynamic centre,
    which is the condition for pitch stability: a gust that raises the nose
    raises the lift, and lift acting behind the CG pitches the nose back down.
    Negative means the aircraft diverges in pitch and has to be flown by a
    controller fast enough to catch it, which is not what this airframe is.

    This is the number that finally prices le_sweep_deg.  Until now nothing in
    the objective could see sweep at all -- it was documented as a variable the
    optimizer would leave wherever it started.
    """
    x_cg = cg_station(root_chord, tip_chord, root_thickness, tip_thickness,
                      le_sweep_deg, battery_station, servo_chord_frac,
                      servo_span_frac)
    x_ac = aerodynamic_centre_x(root_chord, tip_chord, le_sweep_deg)
    return (x_ac - x_cg) / jnp.maximum(mac, 1e-9)


def thickness_at(x, max_thickness):
    """Section thickness at chord fraction ``x``, given the maximum thickness.

    The NACA four-digit thickness distribution, a placeholder standing in until
    a section is actually chosen.  It is used rather than something simpler
    because the leading-edge shape decides how much room there is at the front
    of the section, which is where the battery wants to sit.  A profile that
    rises linearly from zero understates that room badly: the real nose goes as
    sqrt(x), so a section is already at half its maximum thickness by 5% chord,
    not by 15%.

    Normalized so the maximum equals ``max_thickness`` exactly, since the raw
    polynomial peaks at 0.1 for a nominal t/c of 0.2, and everything downstream
    treats the argument as the true maximum.

    Note this puts maximum thickness at 30% chord, which is where the four-digit
    family puts it and close to MAX_THICKNESS_STATION.
    """
    xc = jnp.clip(x, 0.0, 1.0)
    shape = (0.2969 * jnp.sqrt(jnp.maximum(xc, 1e-12))
             - 0.1260 * xc
             - 0.3516 * xc ** 2
             + 0.2843 * xc ** 3
             - 0.1015 * xc ** 4)
    # The polynomial's own peak, so the returned maximum is exactly the value
    # asked for regardless of the coefficients above.
    return max_thickness * shape / 0.1000


def local_camber(a1, a2, tip_a1, tip_a2, span_fraction):
    """Camber coefficients at a fraction of the semi-span.

    Lofted on the same schedule as chord and thickness, so the constant-chord
    centre strip is a true prismatic extrusion of the root section rather than
    one that starts twisting immediately.
    """
    t = taper_fraction(span_fraction)
    return a1 + (tip_a1 - a1) * t, a2 + (tip_a2 - a2) * t


def local_geometry(root_chord, tip_chord, root_thickness, tip_thickness,
                   span_fraction):
    """Chord and maximum thickness at a fraction of the semi-span.

    Constant across the centre section, then lofted linearly between the root
    and tip sections over the tapered panel.  Thickness follows the same
    schedule as chord so the centre strip is a true prismatic extrusion of the
    root section rather than a chord-constant but thinning one.
    """
    t = taper_fraction(span_fraction)
    chord = root_chord + (tip_chord - root_chord) * t
    thickness = root_thickness + (tip_thickness - root_thickness) * t
    return chord, thickness


def elevon_chord_at(chord, x_hinge, mac):
    """Elevon chord at a station whose local chord is ``chord``, in m.

    With a constant-chord elevon the hinge line runs parallel to the trailing
    edge, so every station gets the same elevon chord: the one that ``x_hinge``
    implies at the mean aerodynamic chord.  Otherwise the elevon is a fixed
    fraction of the local chord and tapers with it.
    """
    if CONSTANT_CHORD_ELEVON:
        return jnp.broadcast_to((1.0 - x_hinge) * mac, jnp.shape(chord))
    return (1.0 - x_hinge) * chord


def hinge_fraction_at(chord, x_hinge, mac):
    """Hinge station as a fraction of the local chord.

    Constant for a tapering elevon, but varying for a constant-chord one, which
    is why the section flap derivatives are evaluated at the mean aerodynamic
    chord rather than pretending one fraction holds everywhere.
    """
    return 1.0 - elevon_chord_at(chord, x_hinge, mac) / jnp.maximum(chord, 1e-9)


def servo_slack(chord, thickness, servo_chord_frac, x_hinge, a1=0.0, a2=0.0):
    """Room around the servo at its chordwise station, in m.

    ``servo_chord_frac`` is the chord fraction where the servo body is centred, and
    ``chord``/``thickness`` are the section it sits in -- which is its own
    spanwise station, not the root.  Positive return means it fits.

    Two things are checked: the section is deep enough for the servo where it
    actually sits, and the servo body fits between that station and the hinge.

    Depth is the straight-slot depth over the body's whole footprint, for the
    same reason as the battery: the servo is a rigid box, it cannot follow the
    camber line, and it is long enough relative to this chord that the section
    changes noticeably across it.  Checking only the centre station -- which is
    what this did before there was any camber to worry about -- overstates the
    room at both ends.

    Moving the servo forward relaxes the depth constraint quickly, because
    section thickness aft of maximum falls off quadratically.  It does not come
    free -- the pushrod gets longer and drives the horn increasingly off-axis,
    which shows up as varying mechanical advantage over the stroke -- but that
    cost belongs to the linkage model, not here.  What this function reports is
    the packaging half of the trade.
    """
    half_len = 0.5 * SERVO_LENGTH / jnp.maximum(chord, 1e-9)
    depth_slack = box_depth(chord, thickness,
                            servo_chord_frac - half_len,
                            servo_chord_frac + half_len,
                            a1, a2) - SERVO_DEPTH

    # The servo body occupies chord centred on its station; the hinge must be
    # far enough aft that the body does not run into it.
    body_aft_edge = servo_chord_frac + half_len
    clearance_slack = (x_hinge - body_aft_edge) * chord

    return jnp.minimum(depth_slack, clearance_slack)


def pushrod_length(chord, servo_chord_frac, x_hinge):
    """Chordwise distance from the servo output to the hinge line, in m.

    The quantity the linkage model needs: a longer run means the horn is driven
    further off-axis, so mechanical advantage varies more across the stroke.
    """
    return (x_hinge - servo_chord_frac) * chord


def box_depth(chord, thickness, x0, x1, a1, a2, n=9):
    """Depth of the largest straight slot between ``x0`` and ``x1``, in m.

    The quantity a rigid box actually needs, which is *not* the section's
    thickness.  Thickness is measured perpendicular to the camber line and each
    station has its own; a battery or a servo is a straight box that has to fit
    between one flat plane above and another below, across its whole length.
    So the usable depth is the overlap of the surfaces over the box's footprint:
    the lowest upper surface minus the highest lower surface.

    On a symmetric section the two are the same thing, which is why this did not
    exist before camber did.  Once the section is cambered they diverge, and the
    difference is not small: at the optimum the section reports 11.0 mm of
    thickness under the battery while the straight slot is 10.8 mm, which is the
    difference between the battery fitting and not.

    Sampled over the footprint rather than evaluated at the two ends.  With
    camber the extremes of the two surfaces need not be at the ends -- the upper
    surface peaks in the middle while the lower one is still falling -- so the
    two-point check that was correct for a symmetric section is not correct here.
    """
    x = jnp.linspace(x0, x1, n)
    half = 0.5 * thickness_at(x, thickness)
    camber = camber_line(x, a1, a2) * chord
    upper = camber + half
    lower = camber - half
    return jnp.min(upper) - jnp.max(lower)


def battery_slack(root_chord, root_thickness, battery_station, a1=0.0, a2=0.0):
    """Room around the battery at its chordwise station, in m.

    Two checks.  The battery must fit between its station and the trailing edge,
    which wants it forward; and the section must be deep enough for it over its
    whole length, which wants it centred on the thickest part.

    Depth is the straight-slot depth over the battery's whole footprint, not the
    section thickness at its faces -- see :func:`box_depth`.  A battery is a
    rigid box and cannot follow the camber line, so on a cambered section the
    slot is always shallower than the thickness, and checking thickness alone
    reports a battery fitting when it does not.
    """
    length_slack = (1.0 - battery_station) * root_chord - BATTERY_LENGTH
    aft_station = battery_station + BATTERY_LENGTH / jnp.maximum(root_chord, 1e-9)
    depth_slack = box_depth(root_chord, root_thickness, battery_station,
                            aft_station, a1, a2) - BATTERY_THICKNESS
    return jnp.minimum(length_slack, depth_slack)


def volume_slack(root_chord, tip_chord, root_thickness, tip_thickness,
                 servo_chord_frac, servo_span_fraction, x_hinge, battery_station,
                 a1=0.0, a2=0.0):
    """How much room the wing has beyond what it must hold, in m.

    Positive means everything fits.  The battery and the servos are checked at
    different spanwise stations because that is where they actually live: the
    battery occupies the centreline, and the servos sit outboard near the
    elevons they drive.  Checking both at the root would have them fighting for
    the same chord, which is a constraint that does not exist -- and one the
    model previously invented, making the design look infeasible when it was
    only badly drawn.
    """
    battery = battery_slack(root_chord, root_thickness, battery_station, a1, a2)
    thickness_slack = root_thickness - MIN_ROOT_THICKNESS

    chord, thickness = local_geometry(root_chord, tip_chord, root_thickness,
                                      tip_thickness, servo_span_fraction)
    # The servo has to clear the hinge in *its own* section, where the hinge
    # fraction is not x_hinge once the elevon is constant-chord.
    _, mac = planform(root_chord, tip_chord)
    x_hinge_local = hinge_fraction_at(chord, x_hinge, mac)
    servo = servo_slack(chord, thickness, servo_chord_frac, x_hinge_local, a1, a2)

    # The servo must sit outboard of the battery, which occupies the centre
    # section out to roughly half its own width either side of the centreline.
    span_slack = servo_span_fraction * 0.5 * SPAN - 0.5 * SERVO_WIDTH

    return jnp.minimum(jnp.minimum(battery, thickness_slack),
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


def max_lift_coefficient(camber, thickness_ratio, re):
    """Cl_max of the section, from an empirical correlation.

    This used to be a constant passed in as 0.8, which was defensible while
    nothing in the design could change it.  Camber can, and camber's entire
    payoff is here -- a cambered section reaches a higher Cl before it stalls,
    which is what lets the wing fly slower.  Leaving Cl_max constant while
    adding camber as a design variable would have made camber pure cost with no
    benefit, and the optimizer would have driven it to zero for the wrong
    reason.

    The correlation is deliberately crude and its form matters more than its
    coefficients:

      * camber helps, strongly, and it is the dominant term at this scale;
      * thickness helps a little near 8-9% and the band is already constrained
        elsewhere, so a linear term about a reference is enough;
      * Reynolds number helps logarithmically, which is the standard way the
        low-Re literature reports it.

    At Re ~ 40,000 none of this is better than +/- 0.15 absolute, and it is the
    single largest uncertainty in the stall speed the optimizer minimizes.  It
    is an imported bound of the same character as MIN_THICKNESS_RATIO: a
    statement about what published section data shows, not something this model
    derives.  Do not read the second decimal place.
    """
    cl = (CL_MAX_BASE
          + CL_MAX_CAMBER_GAIN * camber
          + CL_MAX_TC_GAIN * (thickness_ratio - CL_MAX_TC_REF)
          + CL_MAX_RE_GAIN * jnp.log(jnp.maximum(re, 1e3) / CL_MAX_RE_REF))
    # Floored well below anything the correlation should produce, purely so a
    # design the optimizer probes on its way somewhere else cannot return a
    # negative Cl_max and a NaN stall speed.
    return jnp.clip(cl, 0.2, CL_MAX_CEILING)


# --- Trim ---------------------------------------------------------------------
#
# A tailless aircraft has no separate surface to trim with, so the elevons do it
# and whatever they spend on trim is no longer available for control.  That is
# the coupling this section exists to make visible: before it, camber and sweep
# were free, and the elevon's whole throw was assumed available for manoeuvring.


def trim_deflection_deg(cm_c4, cl_cruise, margin, x_hinge, iterations=2):
    """Elevon deflection needed to trim in cruise, in degrees.

    Moment balance about the centre of gravity:

        Cm_cg = Cm_c/4 - Cl * static_margin + dCm/deta * eta = 0

    The middle term is the moment from carrying the lift behind the centre of
    gravity, which is what a positive static margin means.  It is *negative* --
    nose-down -- because lift acting aft of the CG pitches the nose down, and
    that sign is the whole reason a stable tailless wing needs up elevon to trim
    rather than down.  Getting it backwards inverts the trim direction, makes a
    stable aircraft appear to need down elevon, and rewards exactly the wrong
    camber; it is the single easiest sign to get wrong in this file.

    Strictly this is implicit, because dCm/deta depends on the deflection
    through the stall roll-off.  Rather than call a solver, it is iterated from
    zero a fixed number of times: below FLAP_STALL_DEG the roll-off is unity and
    the first pass is already exact, and above it the map is a strong
    contraction, so two passes are well converged.  A fixed number of passes
    also keeps the function trivially differentiable, which a while-loop would
    not -- and this quantity is differentiated twice over in ``sensitivity``.

    A positive result means trailing edge down.  A stable, cambered wing needs
    *up* elevon to trim, so the usual answer here is negative.
    """
    eta = 0.0
    for _ in range(iterations):
        dcm = effective_flap_moment_slope(x_hinge, eta)
        eta = -(cm_c4 - cl_cruise * margin) / jnp.where(
            jnp.abs(dcm) < 1e-9, 1e-9, dcm)
        eta = jnp.degrees(eta)
        # The balance is written per radian but reported in degrees, so convert
        # once here and feed the degree value back in, which is what the
        # roll-off expects.
        eta = jnp.clip(eta, -FLAP_SATURATION_DEG, FLAP_SATURATION_DEG)
    return eta


def trim_alpha_deg(cl_cruise, a1, a2):
    """Angle of attack in trimmed cruise, in degrees.

    Thin-airfoil lift-curve slope from the zero-lift angle the camber sets.  A
    cambered wing flies at a lower alpha for the same Cl, which slightly reduces
    the hinge moment -- a real second-order benefit of camber that the model
    would otherwise miss, since ``hinge_moment`` takes alpha as an argument and
    was previously always given zero.

    No aspect-ratio correction to the lift-curve slope.  At AR 2.8 the real
    slope is meaningfully below 2 pi, so this understates alpha; it is left
    simple because alpha enters only the hinge-moment correction, where the
    coefficient it multiplies is itself a representative value rather than a
    measured one.
    """
    return jnp.degrees(cl_cruise / (2.0 * jnp.pi) + alpha_zero_lift(a1, a2))


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
               "x_hinge", "elevon_inboard_frac", "motor_frac", "servo_chord_frac",
               "servo_span_frac", "le_sweep_deg", "battery_station",
               # Camber, as Birnbaum-Glauert coefficients rather than as a
               # geometric shape.  See the camber section for why: reflex is
               # exactly A2 = A1 in these coordinates, and everything past A2
               # would be invisible to both lift and moment.
               # Root camber, and the tip's as an increment on it.  Stated as a
               # delta rather than as an independent pair so that "the same
               # section everywhere" is the origin rather than a coincidence the
               # optimizer has to find, and so washout can be bounded directly:
               # the useful range is a small twist, not an arbitrary second
               # section.
               "camber_a1", "camber_a2", "tip_camber_da1", "tip_camber_da2",
               # Linkage geometry, in millimetres.  See airfoil.linkage_coupling
               # for why these live in the same vector as the wing: the servo's
               # stroke is a fixed budget spent on torque or throw, and only the
               # coupled problem knows which side to be on.
               #
               # Two things are absent deliberately.  servo_x is the pushrod run,
               # which the wing geometry already determines.  The control rod
               # length is solved for rather than chosen, because it is what sets
               # the throw symmetric about zero -- see linkage_model._solve_rod_length.
               "servo_height", "servo_rod_dy", "servo_travel_mm",
               "flap_x_mm", "flap_y_mm"]


def unpack(p):
    """Design vector to named geometry, with span fractions turned into metres.

    Indexed by name rather than destructured positionally, so that adding a
    design variable is a one-line change to DESIGN_VARS instead of a silent
    mis-assignment of every variable after the insertion point.
    """
    v = {name: p[i] for i, name in enumerate(DESIGN_VARS)}
    semi = 0.5 * SPAN
    servo_span_f = v["servo_span_frac"]
    servo_chord, servo_thickness = local_geometry(
        v["root_chord"], v["tip_chord"], v["root_thickness"],
        v["tip_thickness"], servo_span_f)
    return {
        "root_chord": v["root_chord"],
        "tip_chord": v["tip_chord"],
        "root_thickness": v["root_thickness"],
        "tip_thickness": v["tip_thickness"],
        "x_hinge": v["x_hinge"],
        "elevon_inboard_y": v["elevon_inboard_frac"] * semi,
        # The elevon stops short of the tip so the hinge has structure to anchor
        # into at its outboard end instead of ending in mid-air.
        "elevon_outboard_y": semi - ELEVON_TIP_MARGIN,
        "motor_y": v["motor_frac"] * semi,
        "servo_chord_frac": v["servo_chord_frac"],
        "battery_station": v["battery_station"],
        "servo_span_frac": servo_span_f,
        "servo_y": servo_span_f * semi,
        "servo_chord": servo_chord,
        "servo_thickness": servo_thickness,
        "le_sweep_deg": v["le_sweep_deg"],
        "camber_a1": v["camber_a1"],
        "camber_a2": v["camber_a2"],
        # Tip section, as root plus the increment.  Aerodynamic washout: a tip
        # with less camber than the root reaches its stall angle later, so the
        # root stalls first and the aircraft drops its nose instead of a wing.
        # On a swept tailless wing that is the difference between a stall and a
        # departure, which is why the tip is allowed its own shape at all.
        "tip_camber_a1": v["camber_a1"] + v["tip_camber_da1"],
        "tip_camber_a2": v["camber_a2"] + v["tip_camber_da2"],
        "tip_camber_da1": v["tip_camber_da1"],
        "tip_camber_da2": v["tip_camber_da2"],
        # Linkage, millimetres, passed through untouched.  servo_height is the
        # rail's offset perpendicular to the hinge axis, which is a different
        # axis from servo_y above -- see airfoil.linkage_coupling.
        "servo_height": v["servo_height"],
        "servo_rod_dy": v["servo_rod_dy"],
        # Where the control rod actually attaches on the servo end, which is the
        # linkage model's servo_y.  The servo body sits at servo_height; the rod
        # picks up at an offset from it, so the two are separate quantities tied
        # together rather than one number doing both jobs.
        "servo_rod_y": v["servo_height"] + v["servo_rod_dy"],
        "servo_travel_mm": v["servo_travel_mm"],
        "flap_x_mm": v["flap_x_mm"],
        "flap_y_mm": v["flap_y_mm"],
    }


def required_deflection_deg(unit_alpha, floor):
    """Deflection needed to reach an angular acceleration floor, in degrees.

    A division rather than a root find, because authority is *exactly* linear in
    deflection below FLAP_STALL_DEG: flap_deflection_efficiency is identically
    one there, so the moment is proportional to the deflection and
    ``unit_alpha`` -- the acceleration at one degree -- is the whole slope.

    That linearity is load-bearing.  If flap_deflection_efficiency is ever
    changed to roll off gradually from zero deflection instead of staying flat
    to FLAP_STALL_DEG, this inversion silently under-predicts and has to become
    a real solve.  There is a test pinning the flat region for that reason.

    Above the peak the curve turns over -- more deflection buys *less*
    authority -- so a result past FLAP_STALL_DEG does not mean "deflect harder",
    it means the surface is too small at any deflection.  Returned unclamped so
    the deflection_reachable constraint can report that honestly.
    """
    return floor / jnp.maximum(jnp.abs(unit_alpha), 1e-9)


def evaluate(p, v_cruise=12.0, cl_max=None, deflection_deg=None):
    """Everything the elevon sizing decision needs, at one design point.

    Authority is reported at four conditions because the binding one is not
    obvious in advance and is usually not cruise.  Hover and transition are
    fed almost entirely by prop wash; idle descent is the case where the wash
    goes away while the aircraft still needs to be controllable, which is the
    condition most likely to be missed.

    ``deflection_deg`` of None -- the default -- means derive it, which is the
    design intent: the deflection the aircraft flies at is a consequence of the
    authority floors and of what the linkage can deliver, not a number chosen up
    front.  An explicit value is still honoured, because the plots and the
    roll-off table need to ask what would happen at a deflection this design
    does not actually use.

    ``cl_max`` of None means the same thing for the same reason: it is derived
    from the camber, thickness, and Reynolds number the design actually has.  It
    used to be a constant 0.8 passed in from outside, which was the only honest
    choice while nothing in the design could move it, and is now the override
    rather than the default.
    """
    g = unpack(p)
    area, mac = planform(g["root_chord"], g["tip_chord"])
    mass = total_mass(g["root_chord"], g["tip_chord"],
                      g["root_thickness"], g["tip_thickness"])

    camber, camber_x = camber_max(g["camber_a1"], g["camber_a2"])
    tip_camber, tip_camber_x = camber_max(g["tip_camber_a1"], g["tip_camber_a2"])

    # Pitching moment of the wing, not of the root section.  With the tip free
    # to carry a different camber the two sections no longer agree, and it is
    # the area-weighted average that the aircraft actually trims against --
    # weighting by chord because a section's moment scales with its own chord
    # squared while its area scales with the chord.  Sampled across the
    # semi-span rather than averaged between root and tip, since chord and
    # camber both vary and their product is not linear.
    _spans = jnp.linspace(0.0, 1.0, 9)
    _chords = jnp.stack([local_geometry(g["root_chord"], g["tip_chord"],
                                        g["root_thickness"], g["tip_thickness"],
                                        s)[0] for s in _spans])
    _cms = jnp.stack([
        cm_quarter_chord(*local_camber(g["camber_a1"], g["camber_a2"],
                                       g["tip_camber_a1"], g["tip_camber_a2"], s))
        for s in _spans])
    cm_c4 = (jnp.sum(_cms * _chords ** 2) / jnp.maximum(
        jnp.sum(_chords ** 2), 1e-12))

    # Cl_max, and with it the stall speed, now depends on the design.  There is
    # a circularity to break: Cl_max wants the Reynolds number, which wants the
    # stall speed, which wants Cl_max.  Broken by evaluating Reynolds at the
    # stall speed the *base* Cl_max implies, which is within a few percent and
    # feeds a correlation whose logarithmic Reynolds term is far less sensitive
    # than that.
    if cl_max is None:
        v_stall_seed = stall_speed(mass, area, CL_MAX_BASE)
        tc_mean = 0.5 * (g["root_thickness"] / jnp.maximum(g["root_chord"], 1e-9)
                         + g["tip_thickness"] / jnp.maximum(g["tip_chord"], 1e-9))
        # At the mean aerodynamic chord and at the tip, taking the worse.  The
        # tip runs at a lower Reynolds number and stalls first, and tip stall on
        # a swept tailless wing is the departure mode that actually hurts.
        # The tip is evaluated with its *own* camber, which is what makes
        # washout cost something: a tip decambered to stall late also makes less
        # lift, so the wing's usable Cl_max falls with it.  Without this the
        # optimizer would decamber the tip for free.
        cl_max = jnp.minimum(
            max_lift_coefficient(camber, tc_mean, reynolds(v_stall_seed, mac)),
            max_lift_coefficient(tip_camber, g["tip_thickness"]
                                 / jnp.maximum(g["tip_chord"], 1e-9),
                                 reynolds(v_stall_seed, g["tip_chord"])))

    thrust_hover = THRUST_PER_MOTOR * G
    v_stall = stall_speed(mass, area, cl_max)

    # Balance, and the trim the balance demands.  Cruise is the trim condition:
    # it is where the aircraft spends its time and where the freestream, rather
    # than the prop wash, is what the elevon has to work against.
    margin = static_margin(
        g["root_chord"], g["tip_chord"], g["root_thickness"],
        g["tip_thickness"], g["le_sweep_deg"], g["battery_station"],
        g["servo_chord_frac"], g["servo_span_frac"], mac)
    cl_cruise = cruise_lift_coefficient(mass, area, v_cruise)
    eta_trim = trim_deflection_deg(cm_c4, cl_cruise, margin, g["x_hinge"])
    alpha_trim = trim_alpha_deg(cl_cruise, g["camber_a1"], g["camber_a2"])

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

    # The pushrod run, which is the linkage model's servo_x.  Measured against
    # the hinge fraction at the servo's *own* section rather than at the MAC,
    # because that is the section the pushrod physically lies in.
    pushrod = pushrod_length(
        g["servo_chord"], g["servo_chord_frac"],
        hinge_fraction_at(g["servo_chord"], g["x_hinge"], mac))

    lk = linkage_coupling.linkage_metrics(
        pushrod, g["servo_rod_y"], g["servo_travel_mm"],
        g["flap_x_mm"], g["flap_y_mm"])
    advantage = lk["advantage"]

    # Deflection and incidence the elevon already carries before any control
    # input, which is zero everywhere except in forward flight.  In hover there
    # is no freestream to trim against and the wing is not making lift, so the
    # elevon starts from neutral; in cruise and idle descent it starts from the
    # trim deflection and the section sits at the trim angle of attack.
    trim_bias = {"hover": 0.0, "transition": 0.5 * eta_trim,
                 "cruise": eta_trim, "idle_descent": eta_trim}
    alpha_bias = {"hover": 0.0, "transition": 0.5 * alpha_trim,
                  "cruise": alpha_trim, "idle_descent": alpha_trim}

    def moments(cond, thrust, v, delta_deg):
        """Control moments at one condition and deflection.

        The hinge moment is taken at the *total* deflection -- trim plus control
        -- because that is what the servo physically holds, while the control
        moments are taken at the control deflection alone, because trim by
        definition produces no net moment.  Conflating the two would have the
        servo sized for control alone and stalling the moment it is asked to
        trim as well, which is the failure this whole section exists to catch.
        """
        total_deg = delta_deg + trim_bias[cond]
        return (
            pitch_moment(delta_deg, g["x_hinge"], thrust, v, g["motor_y"],
                         g["elevon_inboard_y"], g["elevon_outboard_y"],
                         area, mac),
            roll_moment(delta_deg, g["x_hinge"], thrust, v, g["motor_y"],
                        g["elevon_inboard_y"], g["elevon_outboard_y"], mac),
            yaw_moment(available_differential_thrust(throttles[cond]),
                       g["motor_y"]),
            hinge_moment(total_deg, g["x_hinge"], thrust, v, g["motor_y"],
                         g["elevon_inboard_y"], g["elevon_outboard_y"], mac,
                         alpha_deg=alpha_bias[cond]),
        )

    # --- Derive the deflection, unless one was asked for explicitly.
    #
    # Hover sets the requirement: it is the condition with the least dynamic
    # pressure that still has to hold the aircraft up.  Evaluating at one degree
    # gives the slope directly, which is all the inversion needs.
    v_hov, thrust_hov = conditions["hover"]
    unit_pitch, unit_roll, _, unit_hinge = moments("hover", thrust_hov, v_hov, 1.0)
    delta_req = jnp.maximum(
        required_deflection_deg(angular_acceleration(unit_pitch, i_pitch),
                                MIN_ALPHA_PITCH_HOVER),
        required_deflection_deg(angular_acceleration(unit_roll, i_roll),
                                MIN_ALPHA_ROLL_HOVER))

    # What the mechanism can actually deliver, whichever of the two limits is
    # tighter.  Both are divisions for the same reason delta_req is: the hinge
    # moment is linear in deflection, so the deflection at which the servo
    # stalls is just the force limit scaled by the force at one degree.
    force_limit = SERVO_MAX_FORCE / SERVO_FORCE_MARGIN
    unit_force = servo_force(unit_hinge, advantage)
    # Worst case across conditions, not hover: hover has no freestream, so it is
    # not where the hinge moment peaks.  Scale by the ratio of peak dynamic
    # pressure to hover's, which is exactly how the hinge moment scales.
    q_hover = elevon_dynamic_pressure(
        thrust_hov, v_hov, g["motor_y"],
        g["elevon_inboard_y"], g["elevon_outboard_y"])
    q_peak = functools.reduce(jnp.maximum, [
        elevon_dynamic_pressure(thrust, v, g["motor_y"],
                                g["elevon_inboard_y"], g["elevon_outboard_y"])
        for v, thrust in conditions.values()])
    unit_force_peak = unit_force * q_peak / jnp.maximum(q_hover, 1e-12)
    delta_force = force_limit / jnp.maximum(unit_force_peak, 1e-12)
    # What the mechanism can give, less what trim has already spent.  A surface
    # held at its trim deflection can only travel the rest of the way to its
    # stop before it runs out, so the throw available for control is the
    # geometric throw minus the trim bias -- which is exactly how a tailless
    # aircraft's control authority degrades as it is asked to trim harder.
    delta_geom_net = jnp.maximum(lk["delta_max_deg"] - jnp.abs(eta_trim), 0.0)
    delta_max = jnp.minimum(delta_geom_net, delta_force)

    if deflection_deg is None:
        deflection_deg = delta_req

    authority = {}
    for name, (v, thrust) in conditions.items():
        m_pitch, m_roll, m_yaw, m_hinge = moments(name, thrust, v, deflection_deg)
        authority[name] = {
            "q": elevon_dynamic_pressure(
                thrust, v, g["motor_y"],
                g["elevon_inboard_y"], g["elevon_outboard_y"]),
            "pitch": m_pitch,
            "roll": m_roll,
            "yaw": m_yaw,
            "hinge": m_hinge,
            "servo_force": servo_force(m_hinge, advantage),
            # Angular accelerations, which is what "enough authority" means.
            "alpha_pitch": angular_acceleration(m_pitch, i_pitch),
            "alpha_roll": angular_acceleration(m_roll, i_roll),
            "alpha_yaw": angular_acceleration(m_yaw, i_yaw),
        }

    # The servo has to cope with the worst condition, not an average one, and
    # which condition that is moves with the design: hover has the most prop
    # wash but no freestream, cruise the reverse.  Reduced here rather than in
    # constraints so the reporting and the constraint see the same number.
    peak_servo_force = functools.reduce(
        jnp.maximum, [a["servo_force"] for a in authority.values()])

    return {
        "area": area,
        "mac": mac,
        "aspect_ratio": SPAN ** 2 / jnp.maximum(area, 1e-9),
        "peak_servo_force": peak_servo_force,
        "servo_force_limit": force_limit,
        # Deflection, derived.  delta_req is what the authority floors demand,
        # delta_max what the mechanism can give, and the gap between them is the
        # margin the design is carrying.
        "deflection_deg": deflection_deg,
        "delta_req": delta_req,
        "delta_max": delta_max,
        "delta_geom": lk["delta_max_deg"],
        # Balance and trim.  eta_trim is what cruise costs before any control
        # input; delta_geom_net is what is left of the mechanism's throw after
        # paying it.
        "cl_max": cl_max,
        "camber": camber,
        "camber_station": camber_x,
        "tip_camber": tip_camber,
        "tip_camber_station": tip_camber_x,
        # Aerodynamic washout, as the camber the tip gives up relative to the
        # root.  Positive means the tip is decambered, which is what makes the
        # root stall first.  This is the quantity the tip-stall constraint acts
        # on, and it is reported because it is the shape difference between the
        # two sections stated as one number.
        "washout": camber - tip_camber,
        "cm_c4": cm_c4,
        "cm_c4_root": cm_quarter_chord(g["camber_a1"], g["camber_a2"]),
        "cm_c4_tip": cm_quarter_chord(g["tip_camber_a1"], g["tip_camber_a2"]),
        "static_margin": margin,
        "cg_station": cg_station(
            g["root_chord"], g["tip_chord"], g["root_thickness"],
            g["tip_thickness"], g["le_sweep_deg"], g["battery_station"],
            g["servo_chord_frac"], g["servo_span_frac"]),
        "ac_station": aerodynamic_centre_x(
            g["root_chord"], g["tip_chord"], g["le_sweep_deg"]),
        "cl_cruise": cl_cruise,
        "eta_trim": eta_trim,
        "alpha_trim": alpha_trim,
        "delta_geom_net": delta_geom_net,
        "prop_clearance": prop_clearance_slack(
            g["le_sweep_deg"], g["motor_y"]),
        "delta_force": delta_force,
        "delta_req_pitch": required_deflection_deg(
            angular_acceleration(unit_pitch, i_pitch), MIN_ALPHA_PITCH_HOVER),
        "delta_req_roll": required_deflection_deg(
            angular_acceleration(unit_roll, i_roll), MIN_ALPHA_ROLL_HOVER),
        # Linkage.
        "linkage_advantage": advantage,
        "linkage_rod_length": lk["rod_length"],
        "linkage_violation": lk["violation"],
        "linkage_valid_fraction": lk["valid_fraction"],
        "linkage_deadness": lk["deadness"],
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
            g["servo_chord_frac"], g["servo_span_frac"], g["x_hinge"],
            g["battery_station"], g["camber_a1"], g["camber_a2"]),
        # Straight-slot depths, not section thicknesses: both items are rigid
        # boxes and neither can follow the camber line.  See box_depth.
        "servo_depth_available": box_depth(
            g["servo_chord"], g["servo_thickness"],
            g["servo_chord_frac"] - 0.5 * SERVO_LENGTH
            / jnp.maximum(g["servo_chord"], 1e-9),
            g["servo_chord_frac"] + 0.5 * SERVO_LENGTH
            / jnp.maximum(g["servo_chord"], 1e-9),
            g["camber_a1"], g["camber_a2"]),
        "battery_depth_available": box_depth(
            g["root_chord"], g["root_thickness"], g["battery_station"],
            g["battery_station"]
            + BATTERY_LENGTH / jnp.maximum(g["root_chord"], 1e-9),
            g["camber_a1"], g["camber_a2"]),
        "battery_station": g["battery_station"],
        # Measured against the hinge fraction in the servo's own section, not
        # the design variable: with a constant-chord elevon x_hinge is the
        # fraction at the mean aerodynamic chord only, and the servo sits
        # outboard of that in shorter chord where the hinge lies a larger
        # fraction aft.  Using x_hinge there overstates the run.
        "pushrod_length": pushrod_length(
            g["servo_chord"], g["servo_chord_frac"],
            hinge_fraction_at(g["servo_chord"], g["x_hinge"], mac)),
        "yaw_moment": yaw_moment(0.5 * thrust_hover, g["motor_y"]),
        "wash_fraction": washed_span_fraction(
            g["motor_y"], g["elevon_inboard_y"], g["elevon_outboard_y"]),
        "flap_effectiveness": flap_effectiveness_ratio(g["x_hinge"]),
        "le_sweep_deg": g["le_sweep_deg"],
        "c4_sweep_deg": quarter_chord_sweep_deg(
            g["root_chord"], g["tip_chord"], g["le_sweep_deg"]),
        "te_sweep_deg": trailing_edge_sweep_deg(
            g["root_chord"], g["tip_chord"], g["le_sweep_deg"]),
        # Thickness ratio at both ends.  Both are needed because chord and
        # thickness taper at different rates, so the extremes of t/c are not
        # necessarily at the extremes of either one.
        "min_tc": jnp.minimum(
            g["root_thickness"] / jnp.maximum(g["root_chord"], 1e-9),
            g["tip_thickness"] / jnp.maximum(g["tip_chord"], 1e-9)),
        "max_tc": jnp.maximum(
            g["root_thickness"] / jnp.maximum(g["root_chord"], 1e-9),
            g["tip_thickness"] / jnp.maximum(g["tip_chord"], 1e-9)),
        # Elevon chord at the inboard and outboard ends of the surface, which is
        # what decides whether it can be hinged and horned at all.  Equal when
        # the elevon is constant-chord; the outboard one is the small one when
        # it is not.
        "elevon_chord_inboard": elevon_chord_at(
            local_geometry(g["root_chord"], g["tip_chord"],
                           g["root_thickness"], g["tip_thickness"],
                           g["elevon_inboard_y"] / (0.5 * SPAN))[0],
            g["x_hinge"], mac),
        "elevon_chord_outboard": elevon_chord_at(
            local_geometry(g["root_chord"], g["tip_chord"],
                           g["root_thickness"], g["tip_thickness"],
                           g["elevon_outboard_y"] / (0.5 * SPAN))[0],
            g["x_hinge"], mac),
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

# Static margin floor, as a fraction of the mean aerodynamic chord.
#
# 5% is the low end of what is normally built.  A tailless aircraft is usually
# flown between 5% and 15%: below about 5% the pitch damping is weak enough that
# a gust response takes a long time to settle, and above about 15% the elevon
# spends so much of its throw trimming that little is left to manoeuvre with.
# The floor rather than a target, because the optimizer has every incentive to
# sit right at it -- margin costs trim deflection, which costs authority.
#
# Worth stating plainly: at the geometry this model started from, the static
# margin is slightly *negative*.  This constraint is what fixes that, and it
# does so mostly by sweeping the wing further aft.
MIN_STATIC_MARGIN = 0.05

# Largest trim deflection considered acceptable, in degrees.
#
# Trim eats the elevon twice over: once directly, since a surface held at trim
# has less travel left before its stop, and once through drag, since a
# permanently deflected surface is a permanently separated one at this Reynolds
# number.  5 degrees is about where a flying wing stops being efficient.
MAX_TRIM_DEFLECTION = 5.0

# Minimum aerodynamic washout, as camber given up between root and tip.
#
# A swept wing stalls at the tip first: the tip runs at a lower Reynolds number
# because its chord is shorter, and sweep drives the boundary layer outboard.
# On a tailless aircraft the elevons are at the tip, so a tip stall takes the
# pitch control with it at the moment it is most needed, and because the tip is
# behind the centre of gravity the nose pitches *up* as it goes -- the classic
# swept-wing departure.
#
# Taking camber out of the tip makes the root reach its stall angle first, so
# the aircraft drops its nose and recovers instead.  0.5% of chord is a modest
# amount, chosen to be clearly better than nothing without demanding a tip so
# decambered that it stops carrying its share of the lift.
#
# This constraint is what gives the tip section a reason to differ from the root
# at all.  Without it the two would be identical, since matching them costs
# nothing and washout costs Cl_max.
MIN_WASHOUT = 0.005

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

# Motor position is invisible to the objective: stall speed does not depend on
# it, and every authority floor is met several times over at every reachable
# position, so the cost is exactly flat in it.  Left alone the optimizer returns
# whatever the random start happened to hold, which reads like a recommendation
# and is not one.
#
# So it gets a tie-break instead: a tiny pull toward the outboard limit, on the
# grounds that yaw is the weakest axis and its authority grows linearly with the
# moment arm, while wash coverage saturates once the slipstream sits inside the
# elevon.  The weight is small enough that it can never trade against a real
# constraint -- it only decides between positions the objective rates equally.
MOTOR_FRAC_PREFERENCE = 1.0
TIEBREAK_WEIGHT = 1e-4

# Sweep is the same story one step removed, and it is worth being explicit about
# because the balance model makes it look solved when it is not.
#
# Adding the static margin constraint gave sweep something to push against for
# the first time -- before it, sweep was documented as a variable the optimizer
# would leave wherever it started.  But the constraint is a *floor*, not a
# gradient: once the margin clears MIN_STATIC_MARGIN, stall speed does not care
# how much sweep produced it, so d(stall)/d(sweep) is still exactly zero and any
# sweep from about 30 degrees upward is rated identically.  The first solve
# returned 43.9 degrees for that reason, which is not a recommendation, it is the
# upper end of a flat region.
#
# So sweep gets a tie-break toward the *least* that does the job.  Sweep is not
# free in reality -- it aggravates tip stall, which on a tailless aircraft is a
# departure rather than a nuisance, and it costs span efficiency -- but none of
# that is in this model, so the preference is stated as a preference rather than
# dressed up as an optimum.  Anything past the floor is buying margin the model
# cannot price with drag it cannot see.
#
# The weight is larger than the motor tie-break's because it has more to do.
# Motor position is flat over a narrow box, while sweep is flat from about 30
# degrees to the 45-degree bound -- a genuinely wide plateau across which the
# static margin runs from 5% to 16% and the trim deflection from -4.1 to -1.4
# degrees, which are meaningfully different aircraft that the cost rates
# identically to four decimal places.  At 1e-3 the pull is still three orders of
# magnitude below the ~5.0 in cost units that the nearest active constraint
# carries, so it cannot trade against anything real; it only picks a point out of
# the plateau.
LE_SWEEP_PREFERENCE = 0.0
SWEEP_TIEBREAK_WEIGHT = 1e-3


def _shortfall(value, floor):
    """Fractional shortfall below a floor, zero when satisfied.

    Normalized by the floor so constraints in different units contribute
    comparably, and squared by the caller so the penalty is smooth at the
    boundary rather than kinked.
    """
    return jnp.maximum(0.0, floor - value) / jnp.maximum(jnp.abs(floor), 1e-9)


def _excess(value, ceiling):
    """Fractional overshoot above a ceiling, zero when satisfied."""
    return jnp.maximum(0.0, value - ceiling) / jnp.maximum(jnp.abs(ceiling), 1e-9)


def constraints(p, cl_max=None, deflection_deg=None):
    """Each constraint's fractional shortfall.  All zero means feasible."""
    g = unpack(p)
    r = evaluate(p, cl_max=cl_max, deflection_deg=deflection_deg)
    cruise = r["authority"]["cruise"]

    return {
        # Hover pitch and roll authority are not checked at an assumed
        # deflection any more.  The floors are inverted for the deflection they
        # need and the mechanism is asked whether it can reach it, which is the
        # same requirement stated in terms of something the design controls.
        "deflection_authority": _shortfall(r["delta_max"], r["delta_req"]),
        # The required deflection has to land below the authority peak.  Past
        # FLAP_STALL_DEG a larger deflection buys *less* authority, so a
        # requirement beyond it cannot be met by deflecting harder -- the
        # surface itself is too small.
        "deflection_reachable": _excess(r["delta_req"], FLAP_STALL_DEG),
        # Yaw comes from differential thrust alone and does not depend on
        # deflection at all, so it stays a plain floor.
        "alpha_yaw_cruise": _shortfall(
            jnp.abs(cruise["alpha_yaw"]), MIN_ALPHA_YAW_CRUISE),
        # The linkage has to close, and has to keep moving through its whole
        # stroke.  Both matter because an unclosable or dead linkage reports a
        # *larger* mechanical advantage, so without these the optimizer is
        # actively rewarded for choosing mechanisms that cannot be built.
        "linkage_closes": r["linkage_violation"],
        "linkage_valid": _shortfall(r["linkage_valid_fraction"], 1.0),
        "linkage_dead": r["linkage_deadness"],
        # The servo rail sits inside the wing, so its offset from the hinge axis
        # is capped by the section depth there.  The horn is deliberately not
        # checked against this: it is a control horn, and standing proud of the
        # surface is what control horns do.
        "servo_rail_depth": _excess(
            g["servo_height"] / linkage_coupling.MM_PER_M,
            0.5 * r["servo_depth_available"]),
        # Pitch stability.  The centre of gravity has to sit ahead of the
        # aerodynamic centre, or the aircraft diverges in pitch.  This is the
        # constraint that finally gives le_sweep_deg something to push against:
        # until the balance was modelled, sweep was invisible to the objective
        # and the optimizer left it wherever the random start put it.
        "static_margin": _shortfall(r["static_margin"], MIN_STATIC_MARGIN),
        # Trim is not free.  A design that needs a large trim deflection has
        # spent its elevon before any control input arrives, and it also flies
        # with a permanently deflected surface, which is draggy and stalls
        # earlier than the section data assumes.
        "trim_deflection": _excess(jnp.abs(r["eta_trim"]), MAX_TRIM_DEFLECTION),
        # Tip stall protection.  The tip has to be less cambered than the root
        # so the root stalls first -- see MIN_WASHOUT.  Normalized by the floor
        # like the other fractional shortfalls.
        "washout": _shortfall(r["washout"], MIN_WASHOUT),
        # The prop has to miss the wing.  Only binds once the motors are pulled
        # inboard, which the wiring reach and the packaging both encourage.
        "prop_clearance": jnp.maximum(0.0, -r["prop_clearance"]) / 0.001,
        "twr": _shortfall(r["twr"], MIN_TWR),
        "re_tip": _shortfall(r["re_tip"], MIN_RE_TIP),
        "aspect_ratio": _shortfall(r["aspect_ratio"], MIN_ASPECT_RATIO),
        # Packaging: volume_slack is already a signed distance in metres, so a
        # floor of zero with a millimetre-scale normalization keeps it on the
        # same footing as the others.
        "packaging": jnp.maximum(0.0, -r["volume_slack"]) / 0.001,
        # Harness reach.  Both push against what the optimizer wants, so
        # without them it happily places hardware the wiring cannot reach.
        "motor_reach": _excess(g["motor_y"], MAX_MOTOR_Y),
        "servo_reach": _excess(g["servo_y"], MAX_SERVO_Y),
        # The servo must sit within the span of the elevon it drives, or the
        # pushrod would have to run diagonally across the wing to reach a horn
        # it does not line up with.  Both edges of the servo body are checked,
        # not just its centre, so the whole box lands inside the elevon.
        "servo_inboard_of_elevon": _shortfall(
            g["servo_y"] - 0.5 * SERVO_WIDTH, g["elevon_inboard_y"]),
        "servo_outboard_of_elevon": _excess(
            g["servo_y"] + 0.5 * SERVO_WIDTH, g["elevon_outboard_y"]),
        # The narrow end of the elevon still has to fit a hinge, a horn, and a
        # pushrod attachment.  Binding mainly when the elevon tapers, since a
        # constant-chord one is the same width everywhere.
        "elevon_chord": _shortfall(
            jnp.minimum(r["elevon_chord_inboard"], r["elevon_chord_outboard"]),
            MIN_ELEVON_CHORD),
        # The servo has to actually be able to hold the deflection the
        # authority constraints above assume it can reach.  Without this the
        # optimizer sees an elevon as pure upside: authority grows with its
        # chord and nothing charges for the hinge moment, which grows with the
        # square of it.  This is the term that makes elevon size a trade.
        "servo_force": _excess(r["peak_servo_force"], r["servo_force_limit"]),
        # Thickness ratio band.  The lower bound is what keeps the optimizer
        # from thinning the tip into an abrupt-stalling section to save a
        # fraction of a gram of skin.
        "min_thickness_ratio": _shortfall(r["min_tc"], MIN_THICKNESS_RATIO),
        "max_thickness_ratio": _excess(r["max_tc"], MAX_THICKNESS_RATIO),
    }


def cost(p, cl_max=None, deflection_deg=None):
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

    # Tie-break only.  See MOTOR_FRAC_PREFERENCE: the objective is exactly flat
    # in motor position, so without this the answer is whichever random start
    # survived, which is noise dressed up as a result.
    g = unpack(p)
    tiebreak = (MOTOR_FRAC_PREFERENCE - g["motor_y"] / (0.5 * SPAN)) ** 2

    # The same for sweep, which the static margin constraint bounds from below
    # but nothing bounds from above.  See LE_SWEEP_PREFERENCE: normalized by the
    # bound's own scale so the weight means the same thing regardless of the box.
    sweep_tiebreak = ((g["le_sweep_deg"] - LE_SWEEP_PREFERENCE) / 45.0) ** 2

    return (r["v_stall"] + CONSTRAINT_WEIGHT * penalty
            + TIEBREAK_WEIGHT * tiebreak
            + SWEEP_TIEBREAK_WEIGHT * sweep_tiebreak)


cost_jit = jax.jit(cost)
constraints_jit = jax.jit(constraints)


def stall_only(p, cl_max=None):
    """The objective without any penalty, for sensitivity reporting."""
    return evaluate(p, cl_max=cl_max)["v_stall"]


def active_constraints(p, cl_max=None, deflection_deg=None, tol=1e-3):
    """Constraints sitting at their limit rather than comfortably satisfied.

    A penalty method never drives a shortfall to exactly zero -- it settles
    where the penalty gradient balances the objective's -- so "active" means
    within a small tolerance of the boundary, including slightly past it.
    """
    return [name
            for name, value in constraints(
                p, cl_max=cl_max, deflection_deg=deflection_deg).items()
            if float(value) > tol * tol]


def sensitivity(p, bounds, cl_max=None, deflection_deg=None):
    """Per-variable gradients and bound status at a design point.

    Two gradients are reported because they answer different questions.
    d(stall)/dx is what the variable is actually worth: how much stall speed it
    would buy if nothing else pushed back.  d(cost)/dx is what the optimizer
    felt, penalties included, and near a boundary it is dominated by whichever
    constraint is active rather than by the objective.

    A variable with a near-zero d(stall)/dx is invisible to the objective, and
    whatever value it holds was decided by a constraint, a tie-break, or the
    starting scatter -- not by optimization.  That distinction is the whole
    point of the table.
    """
    p = jnp.asarray(p)
    d_stall = jax.grad(lambda q: stall_only(q, cl_max=cl_max))(p)
    d_cost = jax.grad(lambda q: cost(q, cl_max=cl_max,
                                     deflection_deg=deflection_deg))(p)

    # Which constraints are active, i.e. sitting at zero slack rather than
    # comfortably satisfied.  A variable pinned by one of these is just as
    # constrained as one sitting on a box bound, but nothing about its value
    # shows that, so the two are reported together.
    active = active_constraints(p, cl_max=cl_max,
                                deflection_deg=deflection_deg)

    # Which active constraint each variable actually moves.  Taking the
    # gradient of every constraint with respect to every variable is what
    # distinguishes "this variable is pinned by the tip Reynolds floor" from
    # "some unrelated constraint happens to be active".
    def constraint_vector(q):
        return jnp.stack([v for v in constraints(
            q, cl_max=cl_max, deflection_deg=deflection_deg).values()])

    names = list(constraints(p, cl_max=cl_max,
                             deflection_deg=deflection_deg).keys())
    jac = jax.jacobian(constraint_vector)(p)

    rows = []
    for i, name in enumerate(DESIGN_VARS):
        lo, hi = bounds[name]
        value = float(p[i])
        span = max(hi - lo, 1e-12)
        # Proximity as a fraction of the box, so "at a bound" means the same
        # thing for a chord in metres and a sweep angle in degrees.
        at_lower = (value - lo) / span < 1e-3
        at_upper = (hi - value) / span < 1e-3

        pinned = [names[k] for k in range(len(names))
                  if names[k] in active and abs(float(jac[k, i])) > 1e-9]

        rows.append({
            "name": name,
            "value": value,
            "lower": lo,
            "upper": hi,
            "d_stall": float(d_stall[i]),
            "d_cost": float(d_cost[i]),
            "at_bound": "lower" if at_lower else ("upper" if at_upper else ""),
            "pinned_by": pinned,
        })
    return rows


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
    0.45,    # servo_chord_frac, chord fraction where the servo body sits
    0.40,    # servo_span_frac, outboard of the battery, near its elevon
    20.0,    # le_sweep_deg, aft.  Enough to pull the quarter chord back to
             # roughly neutral against this taper; there is no stability model
             # here to choose it properly.
    BATTERY_STATION_DEFAULT,   # battery_station, chord fraction of its nose
    # Camber, as Glauert coefficients.  Started at A2 = A1, which is exactly
    # zero pitching moment: a neutral point that presupposes neither camber for
    # lift nor reflex for trim, and lets the optimizer decide which it wants.
    0.08,    # camber_a1
    0.08,    # camber_a2
    # Tip camber as an increment on the root.  Started at zero -- the same
    # section everywhere -- so that any washout in the answer was chosen rather
    # than inherited from the start.
    0.0,     # tip_camber_da1
    0.0,     # tip_camber_da2
    # Linkage, mm.  Taken from the sliders the interactive linkage tool was
    # left on, except servo_x, which is now derived from the wing geometry.
    3.55,    # servo_height, servo body depth off the hinge axis
    0.0,     # servo_rod_dy, rod pickup offset from the body
    9.0,     # servo_travel_mm, the servo's stroke
    0.0,     # flap_x_mm, horn offset along the flap
    10.0,    # flap_y_mm, horn radius.  8.19 mm advantage at +/-26.7 deg.
])


def report(p=None, v_cruise=12.0, cl_max=None, deflection_deg=None):
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

    print("\n=== Section and balance ===")
    print(f"  camber          {float(r['camber']) * 100:8.2f} %"
          f"   (at {float(r['camber_station']) * 100:.0f}% chord,"
          f" A1 {float(g['camber_a1']):+.3f} A2 {float(g['camber_a2']):+.3f})")
    print(f"  tip camber      {float(r['tip_camber']) * 100:8.2f} %"
          f"   (washout {float(r['washout']) * 100:+.2f}%,"
          f" floor {MIN_WASHOUT * 100:.1f}%)")
    reflex = "reflexed" if float(g["camber_a2"]) > float(g["camber_a1"]) else "plain"
    print(f"  Cm about c/4    {float(r['cm_c4']):+8.4f}     ({reflex},"
          f" root {float(r['cm_c4_root']):+.4f} tip {float(r['cm_c4_tip']):+.4f})")
    print(f"  Cl_max          {float(r['cl_max']):8.2f}     (was a constant 0.80)")
    print(f"  CG              {float(r['cg_station']) * 1e3:8.1f} mm aft of root LE")
    print(f"  AC              {float(r['ac_station']) * 1e3:8.1f} mm aft of root LE")
    sm = float(r["static_margin"])
    verdict = "STABLE" if sm >= MIN_STATIC_MARGIN else (
        "marginal" if sm > 0 else "DIVERGENT")
    print(f"  static margin   {sm * 100:+8.1f} %      {verdict}")
    print(f"  trim deflection {float(r['eta_trim']):+8.2f} deg"
          f"   (alpha {float(r['alpha_trim']):+.2f} deg at cruise)")
    print(f"  throw after trim{float(r['delta_geom_net']):8.1f} deg"
          f"   (of {float(r['delta_geom']):.1f} deg geometric)")
    print(f"  prop clearance  {float(r['prop_clearance']) * 1e3:+8.1f} mm")

    print("\n=== Mass and performance ===")
    print(f"  all-up mass     {float(r['mass']) * 1e3:8.1f} g")
    print(f"  thrust/weight   {float(r['twr']):8.2f}"
          f"   {'OK' if float(r['twr']) > 1.0 else 'CANNOT HOVER'}")
    print(f"  wing loading    {float(r['wing_loading']):8.1f} N/m^2")
    print(f"  stall speed     {float(r['v_stall']):8.1f} m/s"
          f"   (at Cl_max {float(r['cl_max']):.2f})")
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
    bs = float(g["battery_station"])
    terms = {
        "battery length": (1.0 - bs) * rc - BATTERY_LENGTH,
        "battery depth": float(r["battery_depth_available"]) - BATTERY_THICKNESS,
        "min thickness": rt - MIN_ROOT_THICKNESS,
        "servo depth": float(r["servo_depth_available"]) - SERVO_DEPTH,
        # Local hinge fraction, matching what volume_slack enforces.  Reporting
        # this against the design variable instead would print a slack the
        # constraint does not agree with, which is worse than not printing it.
        "servo/hinge clearance": (
            float(hinge_fraction_at(sc, float(g["x_hinge"]), float(r["mac"])))
            - float(g["servo_chord_frac"]) - 0.5 * SERVO_LENGTH / sc) * sc,
        "servo outboard of battery": (
            float(g["servo_y"]) - 0.5 * SERVO_WIDTH),
        "motor wiring reach": MAX_MOTOR_Y - float(g["motor_y"]),
        "servo wiring reach": MAX_SERVO_Y - float(g["servo_y"]),
        "servo inside elevon (in)": (
            float(g["servo_y"]) - 0.5 * SERVO_WIDTH
            - float(g["elevon_inboard_y"])),
        "servo inside elevon (out)": (
            float(g["elevon_outboard_y"])
            - float(g["servo_y"]) - 0.5 * SERVO_WIDTH),
        "elevon tip anchor": 0.5 * SPAN - float(g["elevon_outboard_y"]),
    }
    binding = min(terms, key=terms.get)
    for name, value in terms.items():
        mark = "  <-- binding" if name == binding else ""
        print(f"    {name:<26}{value * 1e3:+7.1f} mm{mark}")
    print(f"    servo at {float(g['servo_chord_frac']) * 100:.0f}% chord,"
          f" {float(g['servo_span_frac']) * 100:.0f}% semi-span"
          f" ({sc * 1e3:.0f} mm local chord),"
          f" pushrod {float(r['pushrod_length']) * 1e3:.1f} mm")

    print("\n=== Sweep ===")
    c4 = float(r["c4_sweep_deg"])
    print(f"  leading edge    {float(r['le_sweep_deg']):+8.1f} deg")
    print(f"  quarter chord   {c4:+8.1f} deg"
          f"   {'aft' if c4 > 0 else 'FORWARD -- destabilising for a tailless wing'}")
    print(f"  trailing edge   {float(r['te_sweep_deg']):+8.1f} deg")
    print("  (no stability model: sweep is reported, not chosen)")

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

    delta = float(r["deflection_deg"])
    d_req, d_max = float(r["delta_req"]), float(r["delta_max"])
    d_geom, d_force = float(r["delta_geom"]), float(r["delta_force"])

    print("\n=== Linkage and deflection ===")
    print(f"  pushrod run     {float(r['pushrod_length']) * 1e3:8.1f} mm"
          f"   (the linkage model's servo_x, derived from the wing)")
    print(f"  horn radius     {float(g['flap_y_mm']):8.2f} mm"
          f"   offset {float(g['flap_x_mm']):+.2f} mm")
    print(f"  servo body      {float(g['servo_height']):8.2f} mm"
          f"   off the hinge axis, stroke {float(g['servo_travel_mm']):.1f} mm")
    print(f"  rod pickup      {float(g['servo_rod_y']):8.2f} mm"
          f"   ({float(g['servo_rod_dy']):+.2f} mm from the body)")
    print(f"  rod length      {float(r['linkage_rod_length']) * 1e3:8.1f} mm")
    print(f"  advantage       {float(r['linkage_advantage']) * 1e3:8.2f} mm"
          f"   worst case over the stroke")
    print(f"  validity        violation {float(r['linkage_violation']):.3g},"
          f" valid {float(r['linkage_valid_fraction']):.3f},"
          f" deadness {float(r['linkage_deadness']):.3g}")

    binds = "geometry" if d_geom <= d_force else "servo force"
    print(f"\n  throw          +/-{d_geom:6.1f} deg   geometric")
    print(f"  stall at        {d_force:8.1f} deg   where the servo gives out")
    print(f"  delta_max       {d_max:8.1f} deg   <-- {binds} binds")

    dp, dr = float(r["delta_req_pitch"]), float(r["delta_req_roll"])
    print("\n  required deflection")
    print(f"    pitch hover   {dp:8.2f} deg"
          f"{'   <-- binding' if dp >= dr else ''}")
    print(f"    roll hover    {dr:8.2f} deg"
          f"{'   <-- binding' if dr > dp else ''}")
    print(f"  delta_req       {d_req:8.2f} deg")
    print(f"  margin          {d_max / max(d_req, 1e-9):8.1f}x"
          f"   {'OK' if d_max >= d_req else 'CANNOT REACH'}")
    print(f"  headroom        {FLAP_STALL_DEG - d_req:8.2f} deg"
          f"   before the authority peak at {FLAP_STALL_DEG:.0f} deg")

    print(f"\n=== Control authority at {delta:.2f} deg deflection (derived) ===")
    print(f"  {'condition':<14}{'q [Pa]':>9}{'pitch':>10}{'roll':>10}"
          f"{'hinge':>10}   (mN.m)")
    for name, a in r["authority"].items():
        print(f"  {name:<14}{float(a['q']):9.1f}{float(a['pitch']) * 1e3:10.3f}"
              f"{float(a['roll']) * 1e3:10.3f}{float(a['hinge']) * 1e3:10.3f}")

    h_max = max(float(a["hinge"]) for a in r["authority"].values())
    print(f"\n  peak hinge moment {h_max * 1e3:.3f} mN.m"
          f" = {h_max * 1e4 / G:.2f} g.cm"
          f"  ({3 * h_max * 1e4 / G:.2f} g.cm with 3x margin)")

    f_peak = float(r["peak_servo_force"])
    f_limit = float(r["servo_force_limit"])
    print(f"  through a {float(r['linkage_advantage']) * 1e3:.2f} mm advantage"
          f" that is {f_peak:.3f} N at the servo,")
    print(f"  against {f_limit:.2f} N available"
          f" ({SERVO_MAX_FORCE:.1f} N stall / {SERVO_FORCE_MARGIN:.1f} margin)"
          f"   {'OK' if f_peak <= f_limit else 'SERVO STALLS'}"
          f"   [{f_limit / max(f_peak, 1e-9):.0f}x margin]")

    print("\n=== Deflection roll-off ===")
    for d in (5, 10, 15, 20, 25, 30):
        eff = float(flap_deflection_efficiency(float(d)))
        print(f"  {d:2d} deg   efficiency {eff:.3f}"
              f"   worth {d * eff:5.1f} deg of ideal deflection")

    return r


if __name__ == "__main__":
    report()
