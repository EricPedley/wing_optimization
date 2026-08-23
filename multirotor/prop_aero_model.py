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
# CL_ALPHA fit jointly against two real bench tests, both static
# (thrust-stand) throttle sweeps with the low-throttle points dropped as ESC
# deadband:
#   prop A: 45mm/3-blade/1.5in pitch, 4.2V, rpm 16,600-43,700
#   prop B: 50.78mm/3-blade/1.9in pitch, 7.4V, rpm 21,600-47,800
#
# A single global CL_ALPHA cannot reconcile both exactly: fit jointly, it
# systematically UNDER-predicts prop A by ~5-10% and OVER-predicts prop B by
# ~4-10% -- opposite signs, not just noise. That is a real result, not a
# fitting failure: it means something this model holds fixed (most likely
# CHORD_TO_DIAMETER_RATIO, since chord and CL_ALPHA are degenerate -- see
# above -- but possibly real blades departing from the ideal-twist
# assumption differently at different pitch/diameter ratios) actually varies
# between these two prop designs. Treat +-10% as this model's honest
# uncertainty band on an unseen prop's thrust until a third, differently-
# proportioned data point either narrows that band or shows it scales with
# pitch/diameter ratio in a way worth modeling explicitly.
CHORD_TO_DIAMETER_RATIO = 0.10
CL_ALPHA = 4.41   # per radian, joint fit -- see above
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
# This was calibrated by backing mechanical torque out of the same two bench
# tests used for CL_ALPHA (torque = (bench_current - I0) * Kt, using each
# prop's fitted R/kV from that session, I0 assumed ~0.3A) and comparing to
# this model's induced-only torque. kappa is NOT constant across the tested
# throttle range for either prop -- it climbs with rpm, and at low throttle
# comes out below 1 (physically impossible), which says the low-throttle
# points are dominated by fit error in R/kV/I0 there, not real physics. At
# the high-throttle end (80-100%, closest to where this design actually
# operates, current-limited near max throttle) it is roughly 1.68-2.19 for
# the 45mm prop and 1.18-1.39 for the 50.78mm prop -- a real and fairly wide
# spread between the two props (not just noise), averaging to about 1.6.
# CD0's contribution is degenerate with kappa here (both scale roughly the
# same way with rpm at fixed geometry, so this data cannot separate a
# profile-drag mechanism from an induced-loss mechanism), so kappa is
# carrying essentially the whole non-ideal correction and CD0 is left at its
# unfit placeholder as a minor secondary term.
#
# Treat this the same way as CL_ALPHA's uncertainty band: +-25% or so on
# power/current predictions until a third prop (ideally with a wider
# throttle range clean of low-throttle deadband) narrows it.
INDUCED_POWER_FACTOR = 1.6

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
