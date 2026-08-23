"""Validates the closed-form equilibrium against SimITL's own integration.

motor_model.steady_state_rpm solves in one shot for the rpm SimITL's
Physics::calculateMotors reaches by stepping `domega = netTorque / inertia`
forward in time.  This file re-implements that stepping loop directly from
physics.cpp (same equations, same clamp on drpm) and checks that running it
to convergence lands on what the closed form predicts, for every motor in
SimITL's catalogue crossed with every propeller in it.  If a future edit to
motor_model.py drifts from physics.cpp, this is what catches it.

Run with: uv run --with pytest python -m pytest multirotor/test_motor_model.py -q
"""

import json
import math
from pathlib import Path

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.motor_model as mm  # noqa: E402

SIMITL_CONFIG = Path.home() / "code" / "SimITL" / "tools" / "simitl-playback" / "config"
MOTOR_DIR = SIMITL_CONFIG / "motor"
PROP_DIR = SIMITL_CONFIG / "propeller"

pytestmark = pytest.mark.skipif(
    not MOTOR_DIR.exists() or not PROP_DIR.exists(),
    reason="SimITL catalogue not found at ~/code/SimITL; nothing to cross-check against",
)


def _load_all(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def _integrate_to_equilibrium(volts, vel, kv, resistance, i0,
                               prop_a_factor, prop_torque_factor, prop_max_rpm,
                               tfx, tfy, tfz, prop_inertia,
                               dt=1e-5, steps=400_000):
    """Re-implementation of Physics::calculateMotors' rpm integration.

    Same equations as physics.cpp lines ~604-616: motor torque less prop
    torque, divided by prop inertia gives angular acceleration, integrated
    and converted to a change in rpm, clamped so a single step cannot
    overshoot the back-EMF-limited rpm (`maxdrpm = |volts*kv - rpm|`, SimITL's
    own anti-overshoot clamp -- without it explicit Euler at this stiffness
    oscillates instead of converging).
    """
    kt = mm.KT_NUMERATOR / kv
    rpm = 0.0
    for _ in range(steps):
        current = (volts - rpm / kv) / resistance
        raw_torque = current * kt
        friction_torque = i0 * kt
        m_torque = raw_torque - (friction_torque if rpm > 1.0 else
                                  (-friction_torque if rpm < -1.0 else 0.0))

        max_rpm = max(prop_max_rpm, 0.01)
        thrust_factor = max(tfx * vel ** 2 + tfy * vel + tfz, 0.0)
        b = (thrust_factor - prop_a_factor * max_rpm ** 2) / max_rpm
        thrust = max(b * rpm + prop_a_factor * rpm ** 2, 0.0)
        p_torque = thrust * prop_torque_factor

        net_torque = m_torque - p_torque
        domega = net_torque / prop_inertia
        drpm = domega * dt * 60.0 / (2.0 * math.pi)

        max_drpm = abs(volts * kv - rpm)
        drpm = max(-max_drpm, min(max_drpm, drpm))
        rpm += drpm
    return rpm


@pytest.fixture(scope="module")
def catalogue():
    return _load_all(MOTOR_DIR), _load_all(PROP_DIR)


@pytest.mark.parametrize("throttle", [0.3, 0.6, 0.9])
def test_steady_state_matches_simitl_integration(catalogue, throttle):
    motors, props = catalogue
    vbat = 4.2 * 4  # nominal 4S pack, volts

    for motor in motors:
        for prop in props:
            volts = throttle * vbat
            vel = 0.0

            closed_form = mm.steady_state_rpm(
                volts, vel,
                motor["motorKV"], motor["motorR"], motor["motorI0"],
                prop["propAFactor"], prop["propTorqueFactor"], prop["propMaxRpm"],
                prop["propThrustFactor"]["x"], prop["propThrustFactor"]["y"],
                prop["propThrustFactor"]["z"],
            )

            integrated = _integrate_to_equilibrium(
                volts, vel,
                motor["motorKV"], motor["motorR"], motor["motorI0"],
                prop["propAFactor"], prop["propTorqueFactor"], prop["propMaxRpm"],
                prop["propThrustFactor"]["x"], prop["propThrustFactor"]["y"],
                prop["propThrustFactor"]["z"], prop["propInertia"],
            )

            assert float(closed_form) == pytest.approx(integrated, rel=2e-3), (
                f"{motor['name']} + {prop['name']} @ throttle={throttle}: "
                f"closed form {float(closed_form):.1f} rpm vs "
                f"integrated {integrated:.1f} rpm")


def test_equilibrium_has_zero_net_torque(catalogue):
    """The defining property of the equilibrium, independent of SimITL: motor
    torque must equal propeller torque there, for every catalogue pair."""
    motors, props = catalogue
    volts, vel = 12.0, 0.0

    for motor in motors:
        for prop in props:
            rpm = mm.steady_state_rpm(
                volts, vel,
                motor["motorKV"], motor["motorR"], motor["motorI0"],
                prop["propAFactor"], prop["propTorqueFactor"], prop["propMaxRpm"],
                prop["propThrustFactor"]["x"], prop["propThrustFactor"]["y"],
                prop["propThrustFactor"]["z"],
            )
            alpha = mm.angular_acceleration(
                volts, rpm, vel,
                motor["motorKV"], motor["motorR"], motor["motorI0"],
                prop["propAFactor"], prop["propTorqueFactor"], prop["propMaxRpm"],
                prop["propThrustFactor"]["x"], prop["propThrustFactor"]["y"],
                prop["propThrustFactor"]["z"], prop["propInertia"],
            )
            assert float(alpha) == pytest.approx(0.0, abs=1.0)


def test_gradients_are_finite():
    """The whole point of a JAX model: it must differentiate cleanly through
    every catalogue-scale operating point, not just evaluate."""
    import jax

    def total_thrust(params):
        volts, kv, resistance, i0, a, torque_factor, max_rpm, tfx, tfy, tfz = params
        r = mm.equilibrium(volts, 0.0, kv, resistance, i0, 10.0,
                            a, torque_factor, max_rpm, tfx, tfy, tfz)
        return r["thrust_n"]

    x = jnp.array([12.0, 2800.0, 0.073, 1.1, 1.2e-8, 0.0173, 28000.0,
                    -3.5e-5, -0.13, 16.5])
    grad = jax.grad(total_thrust)(x)
    assert bool(jnp.all(jnp.isfinite(grad)))
