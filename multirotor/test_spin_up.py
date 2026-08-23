"""Validates spin_up_time_s's closed-form ODE solution against direct
numerical integration of the same equations, across SimITL's catalogue.

spin_up_time_s solves drpm/dt = netTorque(rpm) / inertia in closed form via
partial fractions rather than stepping it forward, for the same reason
steady_state_rpm solves its equilibrium in closed form rather than iterating
-- see motor_model.py.  This file checks that closed form against an
explicit Euler integration of the same ODE (small dt, run until the target
rpm is reached) for every motor/propeller pair in the catalogue.

Run with: uv run --with pytest python -m pytest multirotor/test_spin_up.py -q
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


def _integrate_spin_up(volts_end, vel, kv, resistance, i0, prop_a_factor,
                        prop_torque_factor, prop_max_rpm, tfx, tfy, tfz,
                        prop_inertia, rpm0, target_rpm, dt=2e-6, max_steps=2_000_000):
    kt = mm.KT_NUMERATOR / kv
    max_rpm = max(prop_max_rpm, 0.01)
    thrust_factor = max(tfx * vel ** 2 + tfy * vel + tfz, 0.0)
    b = (thrust_factor - prop_a_factor * max_rpm ** 2) / max_rpm

    rpm = rpm0
    t = 0.0
    for _ in range(max_steps):
        if rpm >= target_rpm:
            break
        current = (volts_end - rpm / kv) / resistance
        m_torque = current * kt - i0 * kt
        thrust = max(b * rpm + prop_a_factor * rpm ** 2, 0.0)
        p_torque = thrust * prop_torque_factor
        domega = (m_torque - p_torque) / prop_inertia
        rpm += domega * dt * 60.0 / (2.0 * math.pi)
        t += dt
    return t


@pytest.fixture(scope="module")
def catalogue():
    return _load_all(MOTOR_DIR), _load_all(PROP_DIR)


@pytest.mark.parametrize("throttle_start,throttle_end", [(0.1, 0.9), (0.3, 0.7)])
def test_spin_up_matches_direct_integration(catalogue, throttle_start, throttle_end):
    motors, props = catalogue
    vbat = 4.2 * 4

    for motor in motors:
        for prop in props:
            args = (
                motor["motorKV"], motor["motorR"], motor["motorI0"],
                prop["propAFactor"], prop["propTorqueFactor"], prop["propMaxRpm"],
                prop["propThrustFactor"]["x"], prop["propThrustFactor"]["y"],
                prop["propThrustFactor"]["z"],
            )
            volts_start = throttle_start * vbat
            volts_end = throttle_end * vbat

            closed_form = float(mm.spin_up_time_s(
                volts_start, volts_end, 0.0, *args, prop["propInertia"],
                settle_fraction=0.95))

            rpm0 = float(mm.steady_state_rpm(volts_start, 0.0, *args))
            r_eq = float(mm.steady_state_rpm(volts_end, 0.0, *args))
            target = rpm0 + 0.95 * (r_eq - rpm0)

            integrated = _integrate_spin_up(
                volts_end, 0.0, *args, prop["propInertia"], rpm0, target)

            assert closed_form == pytest.approx(integrated, rel=5e-3, abs=1e-4), (
                f"{motor['name']} + {prop['name']} "
                f"{throttle_start}->{throttle_end}: "
                f"closed form {closed_form*1e3:.2f}ms vs "
                f"integrated {integrated*1e3:.2f}ms")


def test_spin_up_time_increases_with_settle_fraction():
    """Reaching further into the asymptote must take strictly longer -- the
    curve only reaches equilibrium in the infinite-time limit."""
    args = (16.8 * 0.9, 0.0, 2800.0, 0.073, 1.1, 1.2e-8, 0.0173, 28000.0,
            -3.5e-5, -0.13, 16.5, 8.1e-6)
    t50 = float(mm.spin_up_time_s(16.8 * 0.1, *args, settle_fraction=0.50))
    t95 = float(mm.spin_up_time_s(16.8 * 0.1, *args, settle_fraction=0.95))
    t99 = float(mm.spin_up_time_s(16.8 * 0.1, *args, settle_fraction=0.99))
    assert t50 < t95 < t99


def test_spin_up_gradients_are_finite():
    import jax

    def t95(params):
        (volts_start, volts_end, kv, resistance, i0, prop_a, torque_factor,
         max_rpm, tfx, tfy, tfz, inertia) = params
        return mm.spin_up_time_s(volts_start, volts_end, 0.0, kv, resistance,
                                  i0, prop_a, torque_factor, max_rpm, tfx,
                                  tfy, tfz, inertia, settle_fraction=0.95)

    x = jnp.array([1.68, 15.12, 2800.0, 0.073, 1.1, 1.2e-8, 0.0173, 28000.0,
                    -3.5e-5, -0.13, 16.5, 8.1e-6])
    grad = jax.grad(t95)(x)
    assert bool(jnp.all(jnp.isfinite(grad)))
