"""Reduced-order propeller model calibrated from the prop_factor_graph.py fit.

The BEMT in prop_aero_model.py has a single global CL_ALPHA / CD0 /
INDUCED_POWER_FACTOR that cannot reconcile the full bench catalogue (the
prop_aero_model docstring itself notes that).  This module builds a
continuous, differentiable propeller representation from the per-prop fitted
kT, kP, p values instead:

- Static thrust at the reference rpm is predicted from prop geometry by a
  log-linear regression on diameter, pitch, and blade count.
- Torque at the reference rpm is predicted by a parallel log-linear model.
- The SimITL-style prop_a_factor / prop_torque_factor / thrust_factor_z are
  then derived from those reference values, so motor_model.py's closed-form
  equilibrium solves can be reused unchanged.
- Forward-flight slope (thrust_factor_y) is kept from prop_aero_model.py's
  BEMT so the prop still has some velocity dependence, but the thrust/torque
  magnitudes are anchored to the bench data.

This is intentionally a drop-in replacement for prop_aero_model in the
optimizer path.  The module exposes the same names so quad_model.py can
import it as `pa`.
"""

import jax.numpy as jnp

import multirotor.motor_model as mm
import multirotor.prop_aero_model as pam

RHO = 1.225

# Reference rpm at which the static-thrust / torque regressions are evaluated.
# This is arbitrary: the to_simitl_params output is scale-consistent by
# construction, and the same value is used for all props.
REFERENCE_RPM = 30000.0

# Placeholders kept for interface compatibility with quad_model.py / app.py.
CHORD_TO_DIAMETER_RATIO = 0.10
CL_ALPHA = 4.03
CD0 = 0.02
INDUCED_POWER_FACTOR = 0.94

# Log-linear regression coefficients fitted to the prop_factor_graph.py
# nominal T_ref and Q_ref for the 18 powered catalogue props at REFERENCE_RPM.
# log T_ref = T_COEFF[0] + T_COEFF[1]*log(d) + T_COEFF[2]*log(P) + T_COEFF[3]*log(B)
# log Q_ref = Q_COEFF[0] + Q_COEFF[1]*log(d) + Q_COEFF[2]*log(P) + Q_COEFF[3]*log(B)
T_COEFF = jnp.array([8.16694988, 2.40186994, 0.59413203, 0.25558947])
Q_COEFF = jnp.array([5.00727218, 2.78297140, 0.84173650, 0.37027720])


def _ref_value(coeff, diameter_m, pitch_m, blade_count):
    """Predict log(T_ref) or log(Q_ref) from log geometry."""
    log_d = jnp.log(diameter_m)
    log_p = jnp.log(pitch_m)
    log_b = jnp.log(blade_count)
    X = jnp.array([1.0, log_d, log_p, log_b])
    return jnp.exp(jnp.dot(coeff, X))


def T_ref(diameter_m, pitch_m, blade_count):
    """Predicted static thrust (N) at REFERENCE_RPM, vel=0."""
    return _ref_value(T_COEFF, diameter_m, pitch_m, blade_count)


def Q_ref(diameter_m, pitch_m, blade_count):
    """Predicted shaft torque (N.m) at REFERENCE_RPM, vel=0."""
    return _ref_value(Q_COEFF, diameter_m, pitch_m, blade_count)


def to_simitl_params(diameter_m, pitch_m, blade_count,
                      chord_to_diameter_ratio=CHORD_TO_DIAMETER_RATIO,
                      cl_alpha=CL_ALPHA, cd0=CD0,
                      induced_power_factor=INDUCED_POWER_FACTOR,
                      max_rpm=REFERENCE_RPM):
    """Return SimITL-style prop parameters using the factor-graph model.

    Keeps thrust_factor_x and thrust_factor_y from prop_aero_model.py's BEMT
    so the prop retains a plausible forward-flight slope; the thrust/torque
    magnitudes are overridden by the empirical T_ref / Q_ref predictions.
    """
    aero = pam.to_simitl_params(
        diameter_m, pitch_m, blade_count,
        chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)

    T0 = T_ref(diameter_m, pitch_m, blade_count)
    Q0 = Q_ref(diameter_m, pitch_m, blade_count)

    aero["prop_a_factor"] = T0 / (max_rpm ** 2)
    aero["prop_torque_factor"] = Q0 / jnp.maximum(T0, 1e-12)
    aero["thrust_factor_z"] = T0
    aero["prop_max_rpm"] = max_rpm

    return aero


def bemt_thrust_torque(rpm, vel, diameter_m, pitch_m, blade_count,
                        chord_to_diameter_ratio=CHORD_TO_DIAMETER_RATIO,
                        cl_alpha=CL_ALPHA, cd0=CD0,
                        induced_power_factor=INDUCED_POWER_FACTOR):
    """Thrust (N) and torque (N.m) from the reduced-order factor-graph model.

    Delegates the SimITL quadratic formulation to motor_model.prop_thrust and
    motor_model.prop_torque using the constants from to_simitl_params.
    """
    aero = to_simitl_params(
        diameter_m, pitch_m, blade_count,
        chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)

    thrust = mm.prop_thrust(
        rpm, vel, aero["prop_a_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    torque = mm.prop_torque(
        rpm, vel, aero["prop_torque_factor"], aero["prop_a_factor"],
        aero["prop_max_rpm"], aero["thrust_factor_x"], aero["thrust_factor_y"],
        aero["thrust_factor_z"])
    return thrust, torque
