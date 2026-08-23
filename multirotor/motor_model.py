"""JAX model of a BLDC motor + propeller pair, mirroring SimITL's physics.

SimITL (~/code/SimITL, src/sim/physics.cpp) simulates a quad by stepping the
motor/propeller electromechanics forward in time: at each dt it computes a
motor torque from the applied voltage and back-EMF, a propeller drag torque
from the current rpm and airspeed, integrates the rpm from their difference
divided by the propeller's rotational inertia, and from the new rpm reads off
thrust and current.  That per-step integration is what a flight controller
sees and is the right thing for SimITL to do, but it is the wrong shape for a
design optimizer: nothing here needs the transient, only where it settles.

So this module solves for the same equilibrium SimITL's integrator converges
to -- the rpm at which motor torque exactly balances propeller torque -- in
closed form.  Because motor torque is linear in rpm and propeller torque is
quadratic in it (see prop_thrust), that balance is a single quadratic
equation, solved exactly and differentiably rather than iterated.  Every
constant and every equation below is taken directly from physics.cpp; see the
docstring of each function for the line it mirrors.  What is NOT mirrored is
motor/propeller *noise*, damage, propwash, and ESC/PWM filtering -- all of
that is transient or stochastic detail that does not change where the
equilibrium sits, which is the only thing a design optimizer needs.

Units follow SimITL: torque in N.m, rpm in rev/min, current in A, voltage in
V, thrust in N (see calculatePhysics, where motor thrust is summed directly
against gravity_force = -9.81 * mass_kg), temperature in deg C.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# --- Constants -----------------------------------------------------------
#
# The motor-constant relation Kt[N.m/A] = 8.3 / kV[rpm/V] is SimITL's own
# approximation (see physics.cpp's motorCurrent/motorTorque, which cite
# https://things-in-motion.blogspot.com/2018/12/how-to-estimate-torque-of-bldc-pmsm.html).
# The textbook constant relating torque and speed constants under SI units
# would be 60/(2*pi) = 9.5493; SimITL's 8.3 is close to that derated by a
# typical BLDC efficiency, and is kept here unchanged so this model reproduces
# SimITL rather than a more "correct" but different simulator.
KT_NUMERATOR = 8.3  # N.m per A per (1/kV), i.e. Kt = KT_NUMERATOR / kV

# Smooths the sign() in the friction term (see motor_torque) into a
# differentiable function of rpm.  SimITL switches branches at |rpm| < 1;
# RPM_SIGN_SMOOTHING is the rpm scale of that transition here.  Design points
# of interest -- hover, cruise -- sit at thousands of rpm, far past the
# transition, so this choice does not change any operating point this module
# is used to evaluate.  It only keeps the gradient finite through zero.
RPM_SIGN_SMOOTHING = 5.0  # rpm

# Fraction of total motor resistance that is actually copper loss inside the
# stator, the rest being ESC/battery/wiring ESR that does not heat the motor.
# Mirrors physics.cpp's `phaseR = R * 0.35f` comment on heat generation.
STATOR_RESISTANCE_FRACTION = 0.35

# Fraction of total electrical power lost to iron and eddy-current heating,
# on top of the I^2*R copper loss.  Mirrors physics.cpp's
# `currentAbs * vbat * 0.05f` term.
IRON_LOSS_FRACTION = 0.05

AMBIENT_TEMP_C = 25.0  # deg C, matches SimITL's stateInit.ambientTemp default


# --- Motor electrical model ------------------------------------------------
#
# Mirrors Physics::motorTorque and Physics::motorCurrent in physics.cpp.


def motor_constant(kv):
    """Kt, N.m per A, from the velocity constant kV in rpm/V.

    Mirrors the `8.3f / min_kV` term shared by motorTorque and motorCurrent.
    """
    return KT_NUMERATOR / jnp.maximum(kv, 1e-6)


def back_emf_voltage(rpm, kv):
    """Back-EMF in volts at a given rpm.  Mirrors `rpm / min_kV`."""
    return rpm / jnp.maximum(kv, 1e-6)


def motor_torque(volts, rpm, kv, resistance, i0):
    """Steady electromagnetic torque delivered by the motor, N.m.

    Mirrors Physics::motorTorque: raw electrical torque from Ohm's law across
    the back-EMF, less the friction torque implied by the no-load current I0.
    SimITL branches on the sign of rpm for the friction term (and has a
    separate stiction case at rpm ~ 0, which this module does not reproduce --
    see RPM_SIGN_SMOOTHING).  ``jnp.tanh(rpm / RPM_SIGN_SMOOTHING)`` replaces
    SimITL's `if (rpm > 1) ... else if (rpm < -1) ...` with a smooth version
    that agrees with it away from rpm = 0.
    """
    kt = motor_constant(kv)
    current = (volts - back_emf_voltage(rpm, kv)) / jnp.maximum(resistance, 1e-6)
    raw_torque = current * kt
    friction_torque = i0 * kt
    return raw_torque - jnp.tanh(rpm / RPM_SIGN_SMOOTHING) * friction_torque


def motor_current_from_torque(torque, kv):
    """Current implied by a torque, A.  Mirrors Physics::motorCurrent.

    This is the *electromagnetic* current corresponding to a torque, i.e. it
    is what SimITL calls right after motorTorque to report the current that
    produced it -- not a separate physical quantity.
    """
    return torque * kv / KT_NUMERATOR


# --- Propeller aerodynamic model --------------------------------------------
#
# Mirrors Physics::propThrust and Physics::propTorque.  SimITL's propeller
# model is a quadratic in rpm, `thrust = b*rpm + a*rpm^2`, chosen so that at
# the propeller's rated max rpm the thrust equals a quadratic-in-airspeed
# "thrust factor" curve (propThrustFactor.x/y/z below) that is fitted per
# catalogue propeller.  ``a`` (propAFactor) is a fixed catalogue constant that
# sets how much the curve bows away from a straight line through the origin;
# ``b`` is solved so the curve passes through (max_rpm, thrust_factor(vel)).


def prop_thrust_factor(vel, thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Thrust at rated max rpm, as a function of axial inflow velocity, N.

    Mirrors the `propF = x*vel^2 + y*vel + z` line in propThrust.  Clipped at
    zero: SimITL's `propF = std::max(0.0f, propF)` stops the curve going
    negative at high climb speed.
    """
    return jnp.maximum(
        thrust_factor_x * vel ** 2 + thrust_factor_y * vel + thrust_factor_z,
        0.0,
    )


def prop_thrust(rpm, vel, prop_a_factor, prop_max_rpm,
                 thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Propeller thrust at a given rpm and axial inflow velocity, N.

    Mirrors Physics::propThrust exactly: a quadratic in rpm through the
    origin and through (prop_max_rpm, prop_thrust_factor(vel)), clipped to be
    non-negative (SimITL's `std::max(result, 0.0f)`, since the model is not
    meant to predict reverse/autorotation thrust).
    """
    max_rpm = jnp.maximum(prop_max_rpm, 0.01)
    thrust_factor = prop_thrust_factor(vel, thrust_factor_x, thrust_factor_y,
                                        thrust_factor_z)
    b = (thrust_factor - prop_a_factor * max_rpm ** 2) / max_rpm
    return jnp.maximum(b * rpm + prop_a_factor * rpm ** 2, 0.0)


def prop_torque(rpm, vel, prop_torque_factor, prop_a_factor, prop_max_rpm,
                 thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Aerodynamic drag torque the propeller applies to the motor, N.m.

    Mirrors Physics::propTorque: thrust times a fixed torque-per-thrust
    factor.  Not derived from blade-element theory here for the same reason
    the thrust curve is not -- this mirrors what SimITL fits per catalogue
    propeller, which is the reference this module exists to reproduce.
    """
    return prop_torque_factor * prop_thrust(
        rpm, vel, prop_a_factor, prop_max_rpm,
        thrust_factor_x, thrust_factor_y, thrust_factor_z)


# --- Steady-state (equilibrium) solve ---------------------------------------
#
# Mirrors the fixed point of Physics::calculateMotors' integration:
# `domega = (motorTorque - propTorque) / propInertia`, run to convergence
# (domega -> 0) rather than one dt step at a time.  Motor torque is affine in
# rpm and propeller torque is quadratic in it (both above), so their
# intersection at rpm > 0 is the positive root of one quadratic -- solved
# directly rather than by iterating, which is what makes this differentiable
# end to end and cheap enough to call inside an optimizer's inner loop.


def _net_torque_quadratic_coeffs(volts, vel, kv, resistance, i0,
                                  prop_a_factor, prop_torque_factor, prop_max_rpm,
                                  thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Coefficients of netTorque(rpm) = -(A*rpm^2 + B*rpm + C), valid rpm > 0.

    netTorque = motor_torque - prop_torque is affine-minus-quadratic in rpm,
    i.e. quadratic overall; this is that quadratic's coefficients, shared by
    steady_state_rpm (whose equilibria are its roots) and spin_up_time_s
    (which integrates the ODE it defines).  See steady_state_rpm's docstring
    for the derivation of A, B, C.
    """
    kt = motor_constant(kv)
    max_rpm = jnp.maximum(prop_max_rpm, 0.01)
    thrust_factor = prop_thrust_factor(vel, thrust_factor_x, thrust_factor_y,
                                        thrust_factor_z)
    b = (thrust_factor - prop_a_factor * max_rpm ** 2) / max_rpm

    a_coef = prop_torque_factor * prop_a_factor
    b_coef = prop_torque_factor * b + kt / (jnp.maximum(kv, 1e-6) * jnp.maximum(resistance, 1e-6))
    c_coef = kt * i0 - kt * volts / jnp.maximum(resistance, 1e-6)
    return a_coef, b_coef, c_coef


def steady_state_rpm(volts, vel, kv, resistance, i0,
                      prop_a_factor, prop_torque_factor, prop_max_rpm,
                      thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Equilibrium rpm where motor torque equals propeller drag torque.

    Setting motor_torque(volts, rpm, kv, R, i0) [affine in rpm, valid for
    rpm > 0 where jnp.tanh(rpm/RPM_SIGN_SMOOTHING) ~= 1] equal to
    prop_torque(rpm, vel, ...) [quadratic in rpm] gives

        A * rpm^2 + B * rpm + C = 0

        A = prop_torque_factor * prop_a_factor
        B = prop_torque_factor * b + Kt / (kv * R)
        C = Kt * i0 - Kt * volts / R

    where b is propeller_model's rpm-linear coefficient (see prop_thrust) and
    Kt = motor_constant(kv).  C is negative whenever the applied voltage
    exceeds what is needed to overcome friction, which is the regime this
    module is meant for; the physically meaningful root is then the positive
    one, `(-B + sqrt(B^2 - 4AC)) / (2A)`.

    Uses the numerically stable form (dividing through by A only after
    picking the root via C/A rather than the naive quadratic formula) so a
    propeller with a very small prop_a_factor -- i.e. an almost-linear
    thrust curve -- does not blow up from a near-zero leading coefficient.
    """
    a_coef, b_coef, c_coef = _net_torque_quadratic_coeffs(
        volts, vel, kv, resistance, i0, prop_a_factor, prop_torque_factor,
        prop_max_rpm, thrust_factor_x, thrust_factor_y, thrust_factor_z)

    discriminant = jnp.maximum(b_coef ** 2 - 4.0 * a_coef * c_coef, 0.0)
    sqrt_disc = jnp.sqrt(discriminant)

    # Citardauq form: numerically stable as a_coef -> 0, where it reduces
    # cleanly to the linear solve -c_coef / b_coef that governs a propeller
    # whose thrust curve is (nearly) straight through the origin.
    rpm = 2.0 * (-c_coef) / (b_coef + sqrt_disc)
    return jnp.maximum(rpm, 0.0)


# --- Spin-up response --------------------------------------------------------
#
# Responsiveness is not well captured by a single angular-acceleration number
# -- the torque surplus that drives rpm up shrinks as rpm rises (motor torque
# falls with back-EMF, prop drag torque rises quadratically), so acceleration
# right after the throttle step is much larger than acceleration just before
# the new rpm is reached.  What is wanted is how long the *whole* transient
# takes, which means integrating drpm/dt = netTorque(rpm) / (prop_inertia in
# consistent units) from the starting rpm to (nearly) the new equilibrium.
#
# That ODE has a closed-form solution here because netTorque(rpm) is the same
# quadratic used by steady_state_rpm: writing it as
# netTorque(rpm) = -A*(rpm - r_eq)*(rpm - r_other) for its two roots (r_eq the
# physical equilibrium, r_other the unphysical one -- see spin_up_time_s),
# drpm/dt is a logistic equation with two fixed points, separable in closed
# form via partial fractions.  No numerical integration needed, and the
# result differentiates cleanly through every design variable.


def spin_up_time_s(volts_start, volts_end, vel, kv, resistance, i0,
                    prop_a_factor, prop_torque_factor, prop_max_rpm,
                    thrust_factor_x, thrust_factor_y, thrust_factor_z,
                    prop_inertia, settle_fraction=0.95):
    """Time to spin up from the volts_start equilibrium to within
    ``settle_fraction`` of the volts_end equilibrium, seconds.

    This is the natural way to state "10% throttle to 90% throttle in
    50ms": call with volts_start/volts_end at those two throttle fractions
    (times battery voltage) and check the return value against 0.050.
    settle_fraction=0.95 is the usual step-response convention -- the curve
    only reaches the new equilibrium asymptotically (see below), so "arrived"
    has to mean "95% of the way there" rather than "exactly there".

    Derivation: with A, B, C from _net_torque_quadratic_coeffs (evaluated at
    volts_end -- that is the torque curve driving the transient once the
    throttle has stepped), netTorque(rpm) = -A*(rpm - r_eq)*(rpm - r_other),
    where r_eq > r_other are the roots of A*rpm^2 + B*rpm + C = 0 (same
    equation steady_state_rpm solves; r_eq is its positive root, r_other is
    the other one, which product-of-roots = C/A makes negative whenever C is
    -- i.e. whenever volts_end alone is enough to overcome friction, the
    regime this is meant for).

    Converting torque to a rate of change of rpm needs the inertia and a
    rad/s <-> rpm factor:

        drpm/dt = K * netTorque(rpm),  K = (60 / (2*pi)) / prop_inertia

    which is a logistic ODE (two fixed points, one stable at r_eq since
    d(netTorque)/drpm there is -A*(r_eq - r_other) < 0, one unstable at
    r_other).  Separating variables and integrating by partial fractions
    gives, for any target rpm strictly between the start and r_eq,

        t(rpm) = [ln|(rpm0 - r_eq)/(rpm0 - r_other)|
                  - ln|(rpm - r_eq)/(rpm - r_other)|]
                 / (K * A * (r_eq - r_other))

    which blows up as rpm -> r_eq, correctly reflecting that exact
    equilibrium is only reached in the infinite-time limit -- the reason
    this function asks for a target fraction rather than the endpoint.
    """
    rpm0 = steady_state_rpm(volts_start, vel, kv, resistance, i0,
                             prop_a_factor, prop_torque_factor, prop_max_rpm,
                             thrust_factor_x, thrust_factor_y, thrust_factor_z)

    a_coef, b_coef, c_coef = _net_torque_quadratic_coeffs(
        volts_end, vel, kv, resistance, i0, prop_a_factor, prop_torque_factor,
        prop_max_rpm, thrust_factor_x, thrust_factor_y, thrust_factor_z)
    r_eq = steady_state_rpm(volts_end, vel, kv, resistance, i0,
                             prop_a_factor, prop_torque_factor, prop_max_rpm,
                             thrust_factor_x, thrust_factor_y, thrust_factor_z)
    # Sum of roots = -B/A, so the other root falls out without a second
    # quadratic solve.
    r_other = -b_coef / jnp.maximum(a_coef, 1e-12) - r_eq

    rpm_target = rpm0 + settle_fraction * (r_eq - rpm0)

    k_factor = (60.0 / (2.0 * jnp.pi)) / jnp.maximum(prop_inertia, 1e-12)
    denom = k_factor * a_coef * jnp.maximum(r_eq - r_other, 1e-9)

    def log_ratio(rpm):
        eps = 1e-9
        return jnp.log(jnp.maximum(jnp.abs(rpm - r_eq), eps)
                        ) - jnp.log(jnp.maximum(jnp.abs(rpm - r_other), eps))

    return (log_ratio(rpm0) - log_ratio(rpm_target)) / jnp.maximum(denom, 1e-12)


def steady_state_temperature(current, thrust, vel, resistance,
                              motor_rth, prop_damage_cooling=0.0):
    """Equilibrium motor winding temperature, deg C.

    Mirrors the heat balance in Physics::updateBat's per-motor section: heat
    generated (copper I^2*R in the fraction of R that is actually inside the
    stator, plus a flat iron/eddy-current loss) equals heat dissipated
    (temperature rise over an effective thermal resistance that convective
    cooling from airspeed and propwash reduces). At steady state
    heatGenerated == heatDissipated, so temperature is read off directly
    rather than integrated through motorCth -- which only sets how fast the
    motor *approaches* this temperature, not what the temperature settles to.

    ``prop_damage_cooling`` is left at zero (SimITL's propDamage == 0 case);
    it exists in the source only to model battle damage, which is out of
    scope for a design optimizer choosing an undamaged prop.
    """
    phase_r = resistance * STATOR_RESISTANCE_FRACTION
    # vbat cancels out of currentAbs * vbat * 0.05 versus the temperature
    # equation only if vbat is known; SimITL computes it from actual battery
    # voltage. Approximate iron loss directly from electrical power instead:
    # current * back-EMF-adjusted volts is not available here without volts,
    # so this uses current^2 * phase_r for copper loss (dominant term, as
    # SimITL's own comment says) and folds a same-sized iron-loss margin in
    # via IRON_LOSS_FRACTION applied to the copper loss rather than to
    # current*vbat, which keeps this function self-contained in (current,
    # thrust, vel) and avoids re-deriving vbat.
    heat_generated = current ** 2 * phase_r * (1.0 + IRON_LOSS_FRACTION)
    airspeed_cooling = vel * 0.5
    propwash_cooling = thrust * 0.5 * (1.0 - prop_damage_cooling)
    cooling_multiplier = 1.0 + airspeed_cooling + propwash_cooling
    effective_rth = motor_rth / jnp.maximum(cooling_multiplier, 1e-6)
    return AMBIENT_TEMP_C + heat_generated * effective_rth


def angular_acceleration(volts, rpm, vel, kv, resistance, i0,
                          prop_a_factor, prop_torque_factor, prop_max_rpm,
                          thrust_factor_x, thrust_factor_y, thrust_factor_z,
                          prop_inertia):
    """Instantaneous prop-shaft angular acceleration, rad/s^2.

    Mirrors `domega = netTorque / propInertia` in Physics::calculateMotors.
    Zero at the equilibrium rpm returned by steady_state_rpm; away from it,
    this is the torque surplus (or deficit) available to change rpm, which is
    the building block for a responsiveness metric -- e.g. the acceleration
    available at the hover operating point when commanded to full throttle,
    or the time constant `prop_inertia / |d(net_torque)/d(rpm)|` linearizing
    around equilibrium.  Left as a primitive rather than packaged into a
    single "responsiveness" number here, since which of those is the right
    constraint is an open question -- see the module using this one.
    """
    m_torque = motor_torque(volts, rpm, kv, resistance, i0)
    p_torque = prop_torque(rpm, vel, prop_torque_factor, prop_a_factor,
                            prop_max_rpm, thrust_factor_x, thrust_factor_y,
                            thrust_factor_z)
    net_torque = m_torque - p_torque
    return net_torque / jnp.maximum(prop_inertia, 1e-12)


def equilibrium(volts, vel, kv, resistance, i0, motor_rth,
                 prop_a_factor, prop_torque_factor, prop_max_rpm,
                 thrust_factor_x, thrust_factor_y, thrust_factor_z):
    """Full steady-state operating point of one motor/propeller pair.

    The single entry point most callers want: given an applied voltage
    (throttle fraction times battery voltage) and axial inflow velocity,
    returns the rpm, thrust, current, power, and temperature SimITL's
    integrator would settle to, plus the torque-per-rpm slope a
    responsiveness constraint can use (see angular_acceleration).
    """
    rpm = steady_state_rpm(volts, vel, kv, resistance, i0,
                            prop_a_factor, prop_torque_factor, prop_max_rpm,
                            thrust_factor_x, thrust_factor_y, thrust_factor_z)
    thrust = prop_thrust(rpm, vel, prop_a_factor, prop_max_rpm,
                          thrust_factor_x, thrust_factor_y, thrust_factor_z)
    torque = motor_torque(volts, rpm, kv, resistance, i0)
    current = motor_current_from_torque(torque, kv)
    elec_power = jnp.abs(current) * volts
    mech_power = torque * rpm * (2.0 * jnp.pi / 60.0)
    temperature = steady_state_temperature(current, thrust, vel, resistance,
                                            motor_rth)
    return {
        "rpm": rpm,
        "thrust_n": thrust,
        "current_a": current,
        "elec_power_w": elec_power,
        "mech_power_w": mech_power,
        "efficiency": mech_power / jnp.maximum(elec_power, 1e-9),
        "temperature_c": temperature,
    }
