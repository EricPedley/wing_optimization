import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from multirotor import rate_plant as rp
from multirotor.four_motor_plant import FourMotorPlant


@pytest.fixture(scope="module")
def plant():
    return FourMotorPlant()


def finite_difference(function, point, epsilon=1e-6):
    columns = []
    for i in range(len(point)):
        perturbation = np.zeros_like(point)
        perturbation[i] = epsilon * max(1., abs(point[i]))
        columns.append((np.asarray(function(point + perturbation))
                        - np.asarray(function(point - perturbation))) / (2 * perturbation[i]))
    return np.column_stack(columns)


def test_hover_equilibrium_and_state_layout(plant):
    assert plant.state_size == 15
    np.testing.assert_array_equal(plant.rate_indices, [0, 1, 2])
    np.testing.assert_allclose(plant.trim_state[11:], plant.trim_command)
    assert np.all((plant.trim_command > 0) & (plant.trim_command < 1))
    np.testing.assert_allclose(plant.trim_command, rp.hover_point()["frac"], rtol=1e-10)
    np.testing.assert_allclose(plant.continuous_dynamics(plant.trim_physical_state, plant.trim_command),
                               0., atol=1e-9)
    np.testing.assert_allclose(plant.step(plant.trim_state, plant.trim_command), plant.trim_state, atol=1e-10)
    assert float(plant.collective_thrust(plant.trim_state)) == pytest.approx(plant.mass * 9.81, rel=1e-10)


def test_variance_trim_and_battery_charge():
    plant = FourMotorPlant(motor_variance=[.97, .98, .965, 1.01], remaining_mah=340.)
    assert np.ptp(plant.trim_command) > .005
    np.testing.assert_allclose(plant.trim_state[3:7], plant.trim_state[3], rtol=1e-9)
    np.testing.assert_allclose(plant.continuous_dynamics(plant.trim_physical_state, plant.trim_command),
                               0., atol=1e-8)
    assert float(plant.collective_thrust(plant.trim_state)) == pytest.approx(plant.mass * 9.81)


def test_continuous_jacobians_match_finite_difference(plant):
    ac, bc = plant.linearize()
    f = jax.jit(plant.continuous_dynamics)
    fd_a = finite_difference(lambda x: f(x, plant.trim_command), plant.trim_physical_state)
    fd_b = finite_difference(lambda u: f(plant.trim_physical_state, u), plant.trim_command)
    np.testing.assert_allclose(ac, fd_a, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(bc, fd_b, rtol=2e-7, atol=2e-7)
    assert ac.shape == (11, 11)
    assert bc.shape == (11, 4)


def test_simitl_signed_inertia_cross_coupling_and_transient_torque(plant):
    state = plant.trim_physical_state.copy()
    state[:3] = [2., -3., 4.]
    state[3:7] *= [1.03, .97, .95, 1.04]
    state[7:11] += [.02, -.04, .03, -.01]
    command = plant.trim_command + [.05, -.02, -.01, .03]
    unit = plant.unit
    rpm = state[3:7] * 60 / (2 * np.pi)
    voltage = (4.2 - unit["max_voltage_sag"] * np.mean(command)**2) * state[7:11]
    torque = ((voltage - rpm / unit["kv"]) / unit["resistance"] - unit["i0"]) * 8.3 / unit["kv"]
    coefficient = (unit["thrust_factor_z"] - unit["prop_a_factor"] * unit["prop_max_rpm"]**2) / unit["prop_max_rpm"]
    thrust = np.maximum(coefficient * rpm + unit["prop_a_factor"] * rpm**2, 0.)
    rates = np.array([state[1], -state[2], -state[0]])
    signed_inertia = unit["inertia"] * [1., -1., 1.]
    force = np.column_stack((np.zeros(4), thrust, np.zeros(4)))
    moment = np.cross(unit["motor_positions"], force).sum(axis=0)
    moment[1] += np.dot([-1., 1., 1., -1.], torque)
    drag = rates * [-1., 1., -1.] * (np.abs(rates) * [1e-5, 2e-6, 1e-5]
                                   * np.asarray(plant.frame_drag_area) + 1e-6)
    acceleration = (moment + np.clip(drag, -2., 2.) - np.cross(rates, signed_inertia * rates)) / signed_inertia
    expected = np.r_[[-acceleration[2], acceleration[0], -acceleration[1]],
                     (torque - unit["prop_torque_factor"] * thrust) / unit["prop_inertia"],
                     2 * np.pi * 100 * (command - state[7:11])]
    np.testing.assert_allclose(plant.continuous_dynamics(state, command), expected, rtol=1e-11, atol=1e-9)
    unsigned_acceleration = (moment + drag - np.cross(rates, unit["inertia"] * rates)) / unit["inertia"]
    assert np.linalg.norm(acceleration - unsigned_acceleration) > 1.


@pytest.mark.parametrize("axis", range(3))
def test_positive_betaflight_mixer_response(plant, axis):
    command = plant.trim_command + .001 * plant.mixer[:, axis]
    state = jnp.asarray(plant.trim_state)
    for _ in range(30):
        state = plant.step(state, command)
    assert state[axis] > 0.
    np.testing.assert_allclose(np.delete(np.asarray(state[:3]), axis), 0., atol=1e-10)


def test_yaw_reacts_to_motor_torque_before_rotors_accelerate(plant):
    state = plant.trim_physical_state.copy()
    state[7:11] += .001 * plant.mixer[:, 2]
    derivative = plant.continuous_dynamics(state, plant.trim_command)
    expected = 4 * .001 * rp.battery_voltage(plant.remaining_mah, plant.trim_command, plant.unit)
    expected *= 8.3 / plant.unit["kv"] / plant.unit["resistance"] / plant.unit["inertia"][1]
    assert float(derivative[2]) == pytest.approx(float(expected), rel=1e-10)
    assert np.ptp(plant.trim_physical_state[3:7]) < 1e-8
    np.testing.assert_allclose(derivative[:2], 0., atol=1e-10)


@pytest.mark.parametrize("delay_steps", [0, 1, 3])
def test_discrete_model_matches_step_jacobian_and_tiny_perturbations(delay_steps):
    plant = FourMotorPlant(esc_delay_s=delay_steps * .002)
    a, b = plant.discrete_model()
    exact_a, exact_b = jax.jacfwd(plant.step, argnums=(0, 1))(
        jnp.asarray(plant.trim_state), jnp.asarray(plant.trim_command))
    np.testing.assert_allclose(a, exact_a, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(b, exact_b, rtol=1e-6, atol=1e-7)
    rng = np.random.default_rng(10)
    dx = rng.normal(size=plant.state_size) * 1e-6
    du = rng.normal(size=4) * 1e-6
    actual = plant.step(plant.trim_state + dx, plant.trim_command + du) - plant.trim_state
    np.testing.assert_allclose(actual, a @ dx + b @ du, rtol=2e-5, atol=1e-9)
    if delay_steps == 0:
        ac, bc = plant.linearize()
        generator = np.zeros((15, 15))
        generator[:11, :11], generator[:11, 11:] = ac, bc
        transition = expm(generator * plant.dt)
        np.testing.assert_allclose(a, transition[:11, :11], atol=1e-12)
        np.testing.assert_allclose(b, transition[:11, 11:], atol=1e-12)


@pytest.mark.parametrize("delay_steps", [1, 3])
def test_delay_queue_is_oldest_first_and_has_exact_latency(delay_steps):
    plant = FourMotorPlant(esc_delay_s=delay_steps * .002)
    command = plant.trim_command + .01 * plant.mixer[:, 0]
    state = plant.trim_state.copy()
    for _ in range(delay_steps):
        state = plant.step(state, command)
        np.testing.assert_allclose(state[:11], plant.trim_physical_state, atol=1e-10)
        np.testing.assert_array_equal(state[-4:], command)
    state = plant.step(state, command)
    alpha = 1 - np.exp(-2 * np.pi * plant.esc_cutoff_hz * plant.dt)
    np.testing.assert_allclose(state[7:11], plant.trim_command + alpha * (command - plant.trim_command), atol=1e-9)
    assert state[0] > 0.


def test_collective_sag_uses_previous_command_not_filtered(plant):
    state = plant.step(plant.trim_state, plant.trim_command + .1)
    np.testing.assert_allclose(state[7:11], plant.trim_command, atol=1e-12)
    np.testing.assert_allclose(state[3:7], plant.trim_state[3:7], atol=1e-10)
    state = plant.step(state, plant.trim_command + .1)
    assert np.all(state[3:7] > plant.trim_state[3:7])
    ac, bc = plant.linearize()
    assert np.all(bc[3:7] < 0.)
    np.testing.assert_allclose(bc[3:7] @ plant.mixer, 0., atol=1e-10)
    assert np.all(np.diag(ac[3:7, 3:7]) < 0.)


def test_command_bounds_and_jittable_scan(plant):
    command = np.array([-2., 3., -.1, 1.1])
    bounded = np.clip(command, 0., 1.)
    np.testing.assert_allclose(plant.step(plant.trim_state, command), plant.step(plant.trim_state, bounded))
    np.testing.assert_allclose(plant.continuous_dynamics(plant.trim_physical_state, command),
                               plant.continuous_dynamics(plant.trim_physical_state, bounded))

    @jax.jit
    def rollout(initial):
        return jax.lax.scan(lambda state, _: (plant.step(state, command), state), initial, None, length=100)

    final, history = rollout(jnp.asarray(plant.trim_state))
    assert np.isfinite(final).all()
    assert np.all(np.asarray(history[:, 3:7]) >= 0.)
    assert np.all((np.asarray(history[:, 7:]) >= 0.) & (np.asarray(history[:, 7:]) <= 1.))


def test_stable_motor_convergence_and_shutdown(plant):
    command = jnp.full(4, .7)
    final = jax.lax.fori_loop(0, 1000, lambda _, x: plant.step(x, command), jnp.asarray(plant.trim_state))
    vbat = rp.battery_voltage(plant.remaining_mah, command, plant.unit)
    expected = rp.steady_state_rpm(rp.effective_voltage(vbat, command), plant.motor_unit) * rp.RPM2RADS
    np.testing.assert_allclose(final[3:7], expected, rtol=1e-10)
    np.testing.assert_allclose(final[7:11], command, atol=1e-12)
    stopped = jax.lax.fori_loop(0, 1000, lambda _, x: plant.step(x, jnp.zeros(4)), final)
    assert np.all(np.asarray(stopped[3:7]) >= 0.)
    assert np.max(stopped[3:7]) < .11
    a, _ = plant.discrete_model()
    assert np.max(np.abs(np.linalg.eigvals(a[3:, 3:]))) < 1.


@pytest.mark.parametrize("kwargs", [
    {"dt": 0.}, {"dt": np.nan}, {"esc_delay_s": -.001},
    {"dt": .003, "esc_delay_s": .002}, {"esc_cutoff_hz": 0.},
    {"motor_variance": [1., 1., 1.]}, {"motor_variance": [1., 1., 0., 1.]},
    {"remaining_mah": np.nan},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        FourMotorPlant(**kwargs)


def test_physical_and_full_state_shapes_are_unambiguous(plant):
    with pytest.raises(ValueError, match="11 physical"):
        plant.continuous_dynamics(plant.trim_state, plant.trim_command)
    with pytest.raises(ValueError, match="full 15-state"):
        plant.step(plant.trim_physical_state, plant.trim_command)
    with pytest.raises(ValueError, match="command"):
        plant.step(plant.trim_state, .5)
    with pytest.raises(ValueError, match="command"):
        plant.continuous_dynamics(plant.trim_physical_state, .5)


def test_infeasible_hover_is_rejected():
    with pytest.raises(ValueError, match="hover"):
        FourMotorPlant(unit={**rp._unit(), "frame_electronics_mass": 10.})
