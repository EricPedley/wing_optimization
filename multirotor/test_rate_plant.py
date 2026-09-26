import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from multirotor import rate_plant as rp


@pytest.fixture
def unit():
    return rp._unit()


def test_realized_parts_replace_sizing_estimates(unit):
    assert unit["kv"] == 11500.0
    assert unit["resistance"] == 0.100
    assert unit["i0"] == 0.845
    assert unit["motor_mass"] == 0.0045
    assert unit["prop_mass"] == pytest.approx(0.00039)
    assert unit["prop_inertia"] == pytest.approx(2.0308518750000001e-07)
    assert unit["prop_a_factor"] == pytest.approx(7.42321001229242e-10)
    assert unit["prop_torque_factor"] == pytest.approx(0.006530149169648785)
    assert unit["battery_capacity_mah"] == 680.0
    assert unit["battery_capacity_charged_mah"] == 680.0
    assert unit["battery_cell_count"] == 1
    assert unit["max_voltage_sag"] == 0.3
    np.testing.assert_allclose(unit["inertia"], 1 / np.array([20300.8, 9838.9, 15887.]))
    np.testing.assert_allclose(unit["motor_positions"],
                               [[.0495, .005, -.035], [.0495, .005, .035],
                                [-.0495, .005, -.035], [-.0495, .005, .035]])


@pytest.mark.parametrize("remaining,ocv", [(680., 4.2), (340., 3.85),
                                         (510., 3.9152173913043478),
                                         (0., 3.49), (-100., .1), (800., 4.4)])
def test_battery_voltage_matches_simitl_equations(unit, remaining, ocv):
    pwm = np.array([0.2, 0.4, 0.6, 0.8])
    depleted = 1 - remaining / 680.
    expected = np.clip(ocv - .3 * .5**2 * (1 + depleted**2) - .006, 0., 100.)
    actual = rp.battery_voltage(remaining, pwm, unit, noise_sample=-.6)
    assert float(actual) == pytest.approx(expected)


def test_battery_uses_rated_and_initial_capacity_separately(unit):
    unit = {**unit, "battery_capacity_charged_mah": 340., "battery_cell_count": 2}
    assert float(rp.battery_voltage(340., jnp.full(4, .5), unit)) == pytest.approx(
        2 * 3.85 - .3 * .25)


def test_battery_sag_uses_mean_command_not_mean_square(unit):
    a = rp.battery_voltage(400., jnp.array([0., 0., 1., 1.]), unit)
    b = rp.battery_voltage(400., jnp.full(4, .5), unit)
    assert float(a) == pytest.approx(float(b))
    assert float(rp.battery_voltage(680., jnp.full(4, -1.), unit)) == pytest.approx(4.2)


def test_throttle_voltage_is_linear_with_simitl_voltage_floor():
    np.testing.assert_allclose(rp.effective_voltage(3.7, jnp.array([0., .4, 1.])),
                               [0., 1.48, 3.7])
    assert float(rp.effective_voltage(.2, .4)) == pytest.approx(.4)


@pytest.mark.parametrize("rpm,volts", [(18000., 2.), (18000., 0.),
                                     (-100., -.2), (0., .02), (0., .2),
                                     (0., -.2), (1., .01), (-1., -.01)])
def test_motor_torque_matches_cpp_including_stiction(unit, rpm, volts):
    kt = 8.3 / unit["kv"]
    raw = (volts - rpm / unit["kv"]) / unit["resistance"] * kt
    friction = unit["i0"] * kt
    if rpm > 1:
        expected = raw - friction
    elif rpm < -1:
        expected = raw + friction
    else:
        expected = raw - np.clip(raw, -friction, friction)
    torque = rp.motor_torque(volts, rpm, unit)
    assert float(torque) == pytest.approx(expected)
    assert float(rp.motor_current(torque, unit)) == pytest.approx(expected * unit["kv"] / 8.3)


def test_battery_step_matches_discharge_and_noise_equations(unit):
    pwm = np.array([.2, .4, .6, .8])
    currents = np.array([-1., 2., 3., 4.])
    result = rp.battery_step(680., pwm, currents, .01, unit,
                             voltage_noise_sample=.5, consumption_noise_sample=-.4)
    current = float(np.sum(np.abs(currents) * pwm))
    assert float(result["vbat"]) == pytest.approx(4.2 - .3 * .5**2 - .005)
    assert float(result["current_a"]) == pytest.approx(current)
    assert float(result["remaining_mah"]) == pytest.approx(680. - current / 3.6 * .01)
    assert float(result["mah_drawn"]) == pytest.approx(current / 3.6 * .01)


def test_battery_step_scalar_inputs_mean_four_identical_motors(unit):
    result = rp.battery_step(680., .5, 2., 1., unit)
    assert float(result["current_a"]) == pytest.approx(4.)
    assert float(result["remaining_mah"]) == pytest.approx(680. - 4. / 3.6)


def test_battery_step_retains_electronics_floor_and_allows_overdischarge(unit):
    result = rp.battery_step(0., jnp.zeros(4), jnp.zeros(4), 1., unit)
    expected_drain = min(.2, .5 / 3.49)
    assert float(result["remaining_mah"]) == pytest.approx(-expected_drain)
    assert float(result["current_a"]) == pytest.approx(3.6 * expected_drain)


def test_battery_and_motor_jacobians_are_jittable(unit):
    def voltage(pwm):
        return rp.effective_voltage(rp.battery_voltage(340., pwm, unit), pwm)

    jac = jax.jit(jax.jacfwd(voltage))(jnp.full(4, .5))
    vbat = 3.85 - .3 * .5**2 * 1.25
    expected = np.eye(4) * vbat - np.full((4, 4), .5 * 2 * .3 * .5 * 1.25 / 4)
    np.testing.assert_allclose(jac, expected, rtol=1e-12, atol=1e-12)
    grad = jax.jit(jax.grad(lambda v: rp.motor_torque(v, 18000., unit)))(2.)
    assert float(grad) == pytest.approx(8.3 / 11500. / .1)
    step = jax.jit(lambda p: rp.battery_step(340., p, jnp.ones(4), .001, unit))
    assert np.isfinite(np.asarray(jax.jacfwd(lambda p: step(p)["remaining_mah"])(jnp.full(4, .5)))).all()


@pytest.mark.parametrize("volts", [.0, .04, .5, 2., 4.2])
def test_equilibrium_matches_independent_cpp_motor_integration(unit, volts):
    rpm = 0.
    dt = 1e-4
    for _ in range(10000):
        raw = (volts - rpm / 11500.) / .1 * (8.3 / 11500.)
        friction = .845 * (8.3 / 11500.)
        torque = raw - (friction if rpm > 1. else
                        -friction if rpm < -1. else np.clip(raw, -friction, friction))
        thrust = max(float(unit["prop_a_factor"]) * rpm**2, 0.)
        net = torque - float(unit["prop_torque_factor"]) * thrust
        drpm = net / float(unit["prop_inertia"]) * dt * 60 / (2 * math.pi)
        bound = abs(volts * 11500. - rpm)
        rpm += np.clip(drpm, -bound, bound)
    assert float(rp.steady_state_rpm(volts, unit)) == pytest.approx(rpm, abs=1e-6)


@pytest.mark.parametrize("remaining", [680., 340., 68.])
def test_hover_is_self_consistent_with_realized_mass_and_sag(unit, remaining):
    hover = rp.hover_point(unit, remaining_mah=remaining)
    assert hover["mass"] == pytest.approx(.08014, abs=1e-4)
    assert 0 < hover["frac"] < 1
    assert hover["thrust_n"] * 4 == pytest.approx(hover["mass"] * 9.81, rel=1e-10)
    assert hover["vbat"] == pytest.approx(float(rp.battery_voltage(
        remaining, jnp.full(4, hover["frac"]), unit)))
    assert hover["volts"] == pytest.approx(hover["frac"] * hover["vbat"])
    assert float(rp.motor_torque(hover["volts"], hover["rpm"], unit)) == pytest.approx(
        unit["prop_torque_factor"] * hover["thrust_n"])
    assert hover["remaining_mah"] == remaining


def test_hover_command_rises_as_battery_discharges(unit):
    assert rp.hover_point(unit, remaining_mah=680.)["frac"] < rp.hover_point(
        unit, remaining_mah=68.)["frac"]


def test_infeasible_hover_raises_instead_of_returning_full_throttle(unit):
    with pytest.raises(ValueError, match="hover"):
        rp.hover_point({**unit, "frame_electronics_mass": 10.})
    with pytest.raises(ValueError, match="hover"):
        rp.hover_point(unit, remaining_mah=-100.)


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_plant_matches_local_cpp_slopes(unit, axis):
    hover = rp.hover_point(unit)
    plant = rp.plant(axis, unit, hover)
    rpm = hover["rpm"]
    kt = 8.3 / 11500.
    dtdw = 2 * unit["prop_a_factor"] * rpm * 60 / (2 * math.pi)
    damping = unit["prop_torque_factor"] * dtdw + kt / (.1 * 11500.) * 60 / (2 * math.pi)
    expected_tau = unit["prop_inertia"] / damping
    expected_dtdcommand = dtdw * kt / .1 * hover["vbat"] / damping
    assert plant["tau"] == pytest.approx(expected_tau)
    assert plant["dv_du"] == pytest.approx(hover["vbat"])
    assert plant["dT_du"] == pytest.approx(expected_dtdcommand)
    axis_index = {"roll": 2, "pitch": 0, "yaw": 1}[axis]
    pos = unit["motor_positions"][0]
    lever = {"roll": abs(float(pos[0])), "pitch": abs(float(pos[2])),
             "yaw": unit["prop_torque_factor"]}[axis]
    assert plant["K"] == pytest.approx(2 * lever * expected_dtdcommand / unit["inertia"][axis_index])


def test_differential_sag_derivative_is_not_collective_derivative(unit):
    hover = rp.hover_point(unit)
    p = rp.plant("roll", unit, hover)
    frac = hover["frac"]
    eps = 1e-6
    directions = np.array([1., 1., -1., -1.])

    def thrust(delta):
        pwm = frac + delta * directions
        vbat = rp.battery_voltage(hover["remaining_mah"], pwm, unit)
        rpm = rp.steady_state_rpm(rp.effective_voltage(vbat, pwm[0]), unit)
        return rp.prop_thrust(rpm, unit)

    fd = (float(thrust(eps)) - float(thrust(-eps))) / (2 * eps)
    assert p["dT_du"] == pytest.approx(fd, rel=1e-7)
    collective = jax.grad(lambda u: rp.effective_voltage(
        rp.battery_voltage(hover["remaining_mah"], jnp.full(4, u), unit), u))(frac)
    assert float(collective) < p["dv_du"]


def test_controls_do_not_delegate_to_optimizer_physics(monkeypatch):
    from multirotor import battery_model as bm, motor_model as mm, quad_model as qm

    def unexpected(*args, **kwargs):
        pytest.fail("Controls must not use the optimizer's fitted motor/battery equations")

    monkeypatch.setattr(qm, "motor_prop_unit", unexpected)
    monkeypatch.setattr(qm, "effective_voltage", unexpected)
    monkeypatch.setattr(mm, "motor_constant", unexpected)
    monkeypatch.setattr(mm, "motor_torque", unexpected)
    monkeypatch.setattr(mm, "steady_state_rpm", unexpected)
    monkeypatch.setattr(bm, "terminal_voltage", unexpected)
    assert rp.plant()["K"] > 0
    assert float(mm.KT_NUMERATOR) == pytest.approx(60 / (2 * math.pi))
    assert qm.DUTY_GAMMA == .789
    assert bm.r_int_ohm("680mAh") == .01026


def test_pid_analysis_entry_points_still_work():
    gains = rp.place_poles("roll", 2 * np.pi * 10)
    assert np.isfinite(list(gains.values())).all()
    gains = rp.design_margins("roll", 60., 70., ki=.1, delay_s=.003)
    w, loop = rp.loop_with_filters("roll", gains["kp"], gains["ki"], gains["kd"])
    assert w.shape == loop.shape
    assert np.isfinite(loop).all()
