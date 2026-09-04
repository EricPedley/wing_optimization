"""Parametric propeller aerodynamics: thrust and torque as closed-form
functions of diameter, geometric pitch, and blade count.

motor_model.py mirrors SimITL, whose propeller model is a curve *fit per
catalogue part* (propAFactor, propThrustFactor, ...) -- there is no way to
ask it "what if the propeller were 2mm bigger." This module is what answers
that: a real (if simplified) aerodynamic derivation that takes diameter,
pitch, and blade count as free variables, the same three the optimizer is
meant to search over.

Derivation -- blade-element momentum theory with the "ideal twist"
simplification
------------------------------------------------------------------------------
A real optimized rotor blade is twisted so that induced velocity is uniform
across the disk (this is the condition for minimum induced power at a given
thrust). Deriving that from a general blade shape needs a numerical radial
integral. But a *constant-pitch* propeller -- pitch p meaning the blade would
advance p per revolution through an unslipping medium -- already has almost
exactly this shape: local blade angle theta(r) = atan(p / (2*pi*r)) is
hyperbolic in r, the same shape ideal twist requires. So "pitch" as
propellers are actually specified is already close to an ideally-twisted
blade, and assuming uniform induced velocity w across the disk is a
reasonable approximation rather than an extra assumption on top of the
input.

With that, blade element theory (linear lift slope, small angles, constant
chord) gives local thrust proportional to r, and integrating across the span
gives one closed-form thrust equation; momentum theory (the same
T = 2*rho*A*(v_inf+w)*w relation airfoil_model.py already uses for the wing's
propwash) gives a second. Together they are one quadratic in the total local
inflow velocity u = v_inf + w, solved in closed form below.

What is NOT modeled: chord/planform (fixed at a constant ratio of diameter,
see CHORD_TO_DIAMETER_RATIO), the real airfoil section (a flat lift-curve
slope and profile drag coefficient stand in for one, see CL_ALPHA/CD0), blade
root cutout, tip loss, stall, and compressibility. Good enough to rank design
points against each other; not a substitute for a real BEMT/CFD tool when it
comes time to build one.
"""

import jax
import jax.numpy as jnp

RHO = 1.225  # kg/m^3, sea level

# The optimizer's free variables are diameter, pitch, and blade count only --
# not chord or airfoil section -- so those are fixed assumptions standing in
# for a real blade design.
#
# Only the product blade_count * CHORD_TO_DIAMETER_RATIO * CL_ALPHA (the
# combined "s" term in bemt_thrust_torque) is identifiable from a static
# thrust-vs-rpm curve -- chord and lift slope trade off against each other
# exactly, so a fit can only pin one once the other is fixed. Here
# CHORD_TO_DIAMETER_RATIO is held at its assumed 0.10 and CL_ALPHA is what
# was fit, so CL_ALPHA below is really "whatever lift slope makes a 0.10
# chord ratio match the data" -- it is doing double duty for both, not a
# clean aerodynamic lift-curve-slope measurement on its own.
#
# CL_ALPHA was originally fit jointly against two real bench tests (45mm and
# 50.78mm props); see multirotor/calibrate_prop_aero.py for a much larger
# recalibration attempt against ~300 real bench rows across 18 props (25mm-
# 101.6mm pitch, 31mm-76.5mm diameter -- see data/tmotor_*.csv) that
# superseded that original fit, and produced an important negative result
# worth recording here: NO single CL_ALPHA reconciles this whole dataset,
# and it is not just noise. Per-prop CL_ALPHA fits (each prop fit against
# only its own bench data) range from ~1.4 to >50 (several hit the search
# ceiling, meaning the true per-prop optimum is even more extreme than that)
# -- values well past any physical airfoil lift-curve slope (~2*pi rad^-1 at
# most) for the small end, meaning this model's fixed-chord/fixed-Cl_alpha/
# no-stall/no-Reynolds-effects assumptions genuinely cannot reproduce some
# props' static thrust curve at ANY CL_ALPHA. The clearest pattern: props at
# or below ~51mm diameter are structurally unfittable regardless of
# pitch/diameter ratio (GF1608-3, GF1609-4, GF35mm-3, GF2015-2 all pushed the
# search past CL_ALPHA=50 with no convergence), which points at low-Reynolds-
# number blade behavior this model has no term for, not just the
# chord/Cl_alpha degeneracy the original two-point calibration flagged.
#
# Given no honest single global fit exists, CL_ALPHA/INDUCED_POWER_FACTOR
# below are calibrated against a DELIBERATELY NARROWED subset matching this
# design's actual candidate range rather than the full dataset: 4 props,
# 60-77mm diameter, pitch/diameter ratio 0.55-1.05 (Gemfan Hurricane 3018-2,
# Gemfan GF3028-3, HQProp T3x1.8x3, HQProp T3x2x3 -- 89 bench rows). Within
# that subset the fit is genuinely good (mean error 0-20% per prop, worst
# single-row error under 40%) -- but it is explicitly NOT validated outside
# roughly this diameter/pitch range, and should not be trusted for a design
# point far outside it (e.g. anything near or below ~51mm diameter) without
# rerunning calibrate_prop_aero.py against a subset matching THAT range
# instead.
CHORD_TO_DIAMETER_RATIO = 0.10
CL_ALPHA = 4.03   # per radian, narrowed-subset fit -- see above
CD0 = 0.02       # representative profile drag coefficient; NOT independently
                 # fit -- see INDUCED_POWER_FACTOR below for why the bench
                 # data can't cleanly separate the two, and what actually
                 # carries the non-ideal-loss calibration.

# Real rotors always need more shaft power than ideal (Froude) momentum
# theory predicts for a given thrust -- non-uniform inflow, tip losses, and
# (worse for small props) low-Reynolds-number blade performance all cost
# power that a bare actuator-disk/ideal-twist model has no way to represent.
# Standard rotor-engineering practice folds all of that into one induced
# power correction factor kappa (>= 1, equivalently 1/FigureOfMerit):
# actual induced power = kappa * ideal induced power.
#
# Refit alongside the CL_ALPHA update above, from the SAME narrowed 4-prop/
# 89-row subset (60-77mm diameter, P/D 0.55-1.05) -- see calibrate_prop_aero.py
# and CL_ALPHA's docstring for why a narrowed subset was used instead of the
# full ~300-row dataset (no honest global fit exists across the full size
# range). Backs mechanical torque out of each bench row's current via that
# row's own real motor kV/I0 (data/motor_datasheets.csv; motor_torque's I0
# subtraction only needs kV/I0, not R, so this did not need per-motor R at
# all) and fits induced_power_factor by least squares against this model's
# induced-only torque prediction. Came out at ~0.94 -- notably BELOW 1.0
# (the ideal-momentum-theory floor a real rotor should never beat), which is
# almost certainly this fit absorbing some of CL_ALPHA's own residual error
# in the same direction (the two are not fully separable from static thrust/
# torque data alone -- see CD0's note below) rather than a real physical
# result; do not read "kappa < 1" as this design's rotors beating ideal
# efficiency. Floored implicitly by CD0's fixed value pulling some profile
# drag out separately, but the two remain degenerate here.
INDUCED_POWER_FACTOR = 0.94

# Reference rpm the SimITL-style parameterization (see to_simitl_params) is
# built around. Arbitrary: it cancels out of the resulting parameters
# entirely (the static thrust curve is an exact rpm^2 law here, so SimITL's
# linear "b" coefficient comes out exactly zero regardless of what rpm the
# catalogue-style constants are nominally referenced to).
REFERENCE_RPM = 30000.0


def _omega_rad_s(rpm):
    return 2.0 * jnp.pi * rpm / 60.0


def bemt_thrust_torque(rpm, vel, diameter_m, pitch_m, blade_count,
                        chord_to_diameter_ratio=CHORD_TO_DIAMETER_RATIO,
                        cl_alpha=CL_ALPHA, cd0=CD0,
                        induced_power_factor=INDUCED_POWER_FACTOR):
    """Thrust (N) and torque (N.m) from the closed-form BEMT/momentum model.

    The four calibrated constants (module-level defaults, documented above)
    are also accepted as overrides so a caller -- namely app.py -- can vary
    them without mutating module state, e.g. to let a user explore how
    sensitive a design is to the calibration's real uncertainty bands.

    Solves for the total local inflow velocity u = v_inf + w (w the induced
    velocity) from

        momentum theory:      T = 2*rho*A*u*w
        blade element theory: T = (B*c*Cl_alpha/4) * R^2 * (Omega^2*pitch/(2*pi) - Omega*u)

    which combine into one quadratic in u:

        2*pi*u^2 + (s*Omega - 2*pi*v_inf)*u - s*Omega^2*pitch/(2*pi) = 0,
        s = B*c*Cl_alpha/4

    Torque is INDUCED_POWER_FACTOR times the ideal induced power (T*u, the
    actuator-disk energy relation -- see INDUCED_POWER_FACTOR's docstring
    for why the ideal value is scaled up rather than trusted directly) plus
    blade profile drag torque ((1/8)*rho*Omega^2*B*c*Cd0*R^4, the standard
    blade-element result for a rotor's profile power), both divided by
    Omega.
    """
    radius_m = 0.5 * diameter_m
    chord_m = chord_to_diameter_ratio * diameter_m
    omega = _omega_rad_s(rpm)

    s = blade_count * chord_m * cl_alpha / 4.0
    a_coef = 2.0 * jnp.pi
    b_coef = s * omega - 2.0 * jnp.pi * vel
    c_coef = -s * omega ** 2 * pitch_m / (2.0 * jnp.pi)

    discriminant = jnp.maximum(b_coef ** 2 - 4.0 * a_coef * c_coef, 0.0)
    u = (-b_coef + jnp.sqrt(discriminant)) / (2.0 * a_coef)
    w = u - vel

    thrust = jnp.maximum(2.0 * RHO * jnp.pi * radius_m ** 2 * u * w, 0.0)

    induced_torque = induced_power_factor * thrust * u / jnp.maximum(omega, 1e-6)
    profile_torque = (0.125 * RHO * omega ** 2 * blade_count * chord_m * cd0
                       * radius_m ** 4)
    torque = induced_torque + profile_torque
    return thrust, torque


def to_simitl_params(diameter_m, pitch_m, blade_count,
                      chord_to_diameter_ratio=CHORD_TO_DIAMETER_RATIO,
                      cl_alpha=CL_ALPHA, cd0=CD0,
                      induced_power_factor=INDUCED_POWER_FACTOR):
    """Fit this model into motor_model.py's SimITL-mirrored parameterization.

    motor_model.steady_state_rpm and spin_up_time_s already solve the
    motor/propeller equilibrium and transient in closed form, but they are
    written against SimITL's (propAFactor, propTorqueFactor, propMaxRpm,
    propThrustFactor.x/y/z) constants, not against this module's (rpm, vel)
    -> (thrust, torque) function directly. Re-deriving a separate equilibrium
    solver for this model would duplicate (and risk drifting from) the
    already-validated one, so instead this function calibrates SimITL-style
    constants from the BEMT model, and everything downstream reuses
    motor_model.py unchanged.

    - propAFactor: this model's static (vel=0) thrust curve is an exact
      rpm^2 law (see bemt_thrust_torque's docstring -- the "ideal twist"
      assumption makes the induced velocity, and hence thrust, exactly
      proportional to Omega at v_inf=0), so SimITL's linear coefficient b is
      exactly zero at vel=0 and propAFactor is just thrust(ref_rpm, 0) /
      ref_rpm^2, exact, not a fit.
    - propTorqueFactor: SimITL assumes one fixed torque/thrust ratio across
      the whole envelope; this uses the ratio at (ref_rpm, vel=0) as that
      representative value, the same simplification SimITL itself makes.
    - propThrustFactor.z: thrust at (ref_rpm, vel=0), by construction equal
      to propAFactor * ref_rpm^2 (keeps b at exactly zero for vel=0).
    - propThrustFactor.y: dThrust/dvel at vel=0, exact via autodiff of the
      BEMT model rather than a numerical fit.
    - propThrustFactor.x: set to zero. SimITL's own catalogue values for
      this term are tiny relative to the linear term across the velocities a
      multirotor actually flies at (e.g. an x of -3.5e-5 contributes
      -0.014N at 20 m/s against a linear term's several newtons), so
      dropping it is a small approximation on top of an approximate model,
      not a new source of error.
    """
    ref_rpm = REFERENCE_RPM
    thrust0, torque0 = bemt_thrust_torque(
        ref_rpm, 0.0, diameter_m, pitch_m, blade_count,
        chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)
    prop_a_factor = thrust0 / ref_rpm ** 2
    prop_torque_factor = torque0 / jnp.maximum(thrust0, 1e-9)

    thrust_at_vel = lambda v: bemt_thrust_torque(  # noqa: E731
        ref_rpm, v, diameter_m, pitch_m, blade_count,
        chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)[0]
    thrust_factor_y = jax.grad(thrust_at_vel)(0.0)

    return {
        "prop_a_factor": prop_a_factor,
        "prop_torque_factor": prop_torque_factor,
        "prop_max_rpm": ref_rpm,
        "thrust_factor_x": 0.0,
        "thrust_factor_y": thrust_factor_y,
        "thrust_factor_z": thrust0,
    }
