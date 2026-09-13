import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm
from scipy.optimize import root

from multirotor import rate_plant as rp


class FourMotorPlant:
    physical_state_size = 11
    command_size = 4
    rate_indices = np.arange(3)
    rotor_indices = slice(3, 7)
    filtered_pwm_indices = slice(7, 11)
    queue_indices = slice(11, None)
    sim_to_bf = jnp.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    mixer = np.array([[-1., 1., -1.], [-1., -1., 1.],
                      [1., 1., 1.], [1., -1., -1.]])

    def __init__(self, unit=None, remaining_mah=None, dt=0.002,
                 esc_delay_s=0.002, esc_cutoff_hz=100.0, motor_variance=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(esc_delay_s) or esc_delay_s < 0:
            raise ValueError("esc_delay_s must be finite and nonnegative")
        if not np.isfinite(esc_cutoff_hz) or esc_cutoff_hz <= 0:
            raise ValueError("esc_cutoff_hz must be finite and positive")
        self.dt = float(dt)
        self.esc_delay_s = float(esc_delay_s)
        self.esc_cutoff_hz = float(esc_cutoff_hz)
        self.delay_steps = int(round(esc_delay_s / dt))
        if not np.isclose(esc_delay_s / dt, self.delay_steps, rtol=0., atol=1e-9):
            raise ValueError("esc_delay_s must be an integer multiple of dt")
        self.state_size = self.physical_state_size + 4 * self.delay_steps
        self.n_state = self.state_size
        if unit is None:
            unit = rp._unit()
            quad = json.loads(rp.DEFAULT_QUAD_CONFIG.read_text())
            unit["frame_drag_area"] = np.array([quad["frameDragArea"][a] for a in "xyz"])
            unit["motor_directions"] = np.array([quad[f"motor{i}Dir"] for i in range(1, 5)])
        self.unit = dict(unit)
        inertia = np.asarray(unit["inertia"], dtype=float)
        positions = np.asarray(unit["motor_positions"], dtype=float)
        if inertia.shape != (3,) or not np.all(np.isfinite(inertia) & (inertia > 0)):
            raise ValueError("unit inertia must contain three positive unsigned SimITL moments")
        if positions.shape != (4, 3) or not np.isfinite(positions).all():
            raise ValueError("motor_positions must have shape (4, 3)")
        self.sim_inertia = jnp.asarray(inertia * [1., -1., 1.])
        self.inertia_bf = jnp.asarray(inertia[[2, 0, 1]])
        self.motor_positions = jnp.asarray(positions)
        self.motor_directions = jnp.asarray(unit.get("motor_directions", [-1., 1., 1., -1.]))
        self.frame_drag_area = jnp.asarray(unit.get("frame_drag_area", np.zeros(3)))
        if self.motor_directions.shape != (4,) or not np.all(np.isin(self.motor_directions, [-1., 1.])):
            raise ValueError("motor_directions must contain four signs")
        if self.frame_drag_area.shape != (3,) or not np.all(np.isfinite(self.frame_drag_area) & (self.frame_drag_area >= 0)):
            raise ValueError("frame_drag_area must contain three nonnegative areas")
        variance = np.ones(4) if motor_variance is None else np.asarray(motor_variance, dtype=float)
        if variance.shape != (4,) or not np.all(np.isfinite(variance) & (variance > 0)):
            raise ValueError("motor_variance must contain four positive kV multipliers")
        self.motor_variance = jnp.asarray(variance)
        self.motor_unit = {**self.unit, "kv": jnp.asarray(unit["kv"] * variance)}
        self.remaining_mah = float(unit["battery_capacity_charged_mah"] if remaining_mah is None else remaining_mah)
        if not np.isfinite(self.remaining_mah):
            raise ValueError("remaining_mah must be finite")
        self.prop_inertia = max(float(unit["prop_inertia"]), 1e-8)
        self.mass = float(unit["frame_electronics_mass"] + unit["battery_mass"]
                          + 4 * (unit["motor_mass"] + unit["prop_mass"]))
        self.esc_pole = 2 * np.pi * self.esc_cutoff_hz
        self._solve_trim()
        ac, _ = self.linearize()
        fastest_pole = max(self.esc_pole, float(np.max(np.abs(np.linalg.eigvals(ac)))))
        self.substeps = max(1, int(np.ceil(self.dt / 1e-4)),
                            int(np.ceil(self.dt * fastest_pole / 0.1)))
        self._compiled_step = jax.jit(self._step)

    def _physical_dynamics(self, state, sag_command, filter_command):
        filtered = jnp.clip(state[7:11], 0., 1.)
        rpm = state[3:7] / rp.RPM2RADS
        vbat = rp.battery_voltage(self.remaining_mah, jnp.clip(sag_command, 0., 1.), self.unit)
        motor_torque = rp.motor_torque(rp.effective_voltage(vbat, filtered), rpm, self.motor_unit)
        thrust = rp.prop_thrust(rpm, self.unit)
        rotor_acc = (motor_torque - self.unit["prop_torque_factor"] * thrust) / self.prop_inertia
        rotor_acc = jnp.where((state[3:7] <= 0.) & (rotor_acc < 0.), 0., rotor_acc)
        sim_rates = self.sim_to_bf.T @ state[:3]
        forces = jnp.zeros((4, 3)).at[:, 1].set(thrust)
        moment = jnp.sum(jnp.cross(self.motor_positions, forces), axis=0)
        moment = moment.at[1].add(jnp.dot(self.motor_directions, motor_torque))
        drag = sim_rates * jnp.array([-1., 1., -1.]) * (
            jnp.abs(sim_rates) * jnp.array([1e-5, 2e-6, 1e-5]) * self.frame_drag_area + 1e-6)
        moment = moment + jnp.clip(drag, -2., 2.)
        sim_acc = (moment - jnp.cross(sim_rates, self.sim_inertia * sim_rates)) / self.sim_inertia
        filtered_acc = self.esc_pole * (jnp.clip(filter_command, 0., 1.) - filtered)
        return jnp.concatenate((self.sim_to_bf @ sim_acc, rotor_acc, filtered_acc))

    def continuous_dynamics(self, state, command):
        state, command = jnp.asarray(state), jnp.asarray(command)
        if state.shape != (self.physical_state_size,):
            raise ValueError("continuous_dynamics requires the 11 physical states, not the delay queue")
        if command.shape != (4,):
            raise ValueError("command must have shape (4,)")
        return self._physical_dynamics(state, command, command)

    def collective_thrust(self, state):
        state = jnp.asarray(state)
        if state.shape not in ((self.physical_state_size,), (self.state_size,)):
            raise ValueError("collective_thrust requires a physical or full plant state")
        return jnp.sum(rp.prop_thrust(state[3:7] / rp.RPM2RADS, self.unit))

    def _equilibrium_state(self, command):
        vbat = rp.battery_voltage(self.remaining_mah, command, self.unit)
        rpm = rp.steady_state_rpm(rp.effective_voltage(vbat, command), self.motor_unit)
        return jnp.concatenate((jnp.zeros(3), rpm * rp.RPM2RADS, command))

    def _solve_trim(self):
        hover = rp.hover_point(self.unit, self.remaining_mah)
        scales = jnp.array([self.mass * 9.81 * np.max(np.abs(self.motor_positions[:, 0])),
                            self.mass * 9.81 * np.max(np.abs(self.motor_positions[:, 2])),
                            self.mass * 9.81 * abs(self.unit["prop_torque_factor"])])
        scales = jnp.maximum(scales, 1e-8)

        @jax.jit
        def residual(command):
            state = self._equilibrium_state(command)
            angular_acc = self.continuous_dynamics(state, command)[:3]
            return jnp.concatenate((angular_acc * self.inertia_bf / scales,
                                    jnp.array([self.collective_thrust(state) / (self.mass * 9.81) - 1.])))

        jacobian = jax.jit(jax.jacfwd(residual))
        result = root(lambda u: np.asarray(residual(u)), np.full(4, hover["frac"]),
                      jac=lambda u: np.asarray(jacobian(u)), tol=1e-10)
        if (not np.isfinite(result.x).all() or np.any(result.x <= 0.) or np.any(result.x >= 1.)
                or np.max(np.abs(residual(result.x))) > 1e-8):
            raise ValueError("No feasible zero-torque hover trim for this unit and battery charge")
        self.trim_command = np.asarray(result.x).copy()
        self.trim_physical_state = np.asarray(self._equilibrium_state(self.trim_command)).copy()
        self.trim_state = np.concatenate((self.trim_physical_state,
                                          np.tile(self.trim_command, self.delay_steps)))

    def linearize(self):
        ac, bc = jax.jacfwd(self.continuous_dynamics, argnums=(0, 1))(
            jnp.asarray(self.trim_physical_state), jnp.asarray(self.trim_command))
        return np.asarray(ac), np.asarray(bc)

    def discrete_model(self):
        ac, b_now, b_filter = (np.asarray(x) for x in jax.jacfwd(
            self._physical_dynamics, argnums=(0, 1, 2))(
                jnp.asarray(self.trim_physical_state), jnp.asarray(self.trim_command),
                jnp.asarray(self.trim_command)))
        generator = np.zeros((19, 19))
        generator[:11, :11] = ac
        generator[:11, 11:15] = b_now
        generator[:11, 15:19] = b_filter
        transition = expm(generator * self.dt)
        a, b = np.zeros((self.state_size, self.state_size)), np.zeros((self.state_size, 4))
        a[:11, :11] = transition[:11, :11]
        if self.delay_steps:
            a[:11, 11:15] = transition[:11, 15:19]
            a[:11, -4:] += transition[:11, 11:15]
            a[11:-4, 15:] = np.eye(4 * (self.delay_steps - 1))
            b[-4:] = np.eye(4)
        else:
            b[:11] = transition[:11, 11:15] + transition[:11, 15:19]
        return a, b

    def _step(self, state, command):
        command = jnp.clip(command, 0., 1.)
        delayed = state[11:15] if self.delay_steps else command
        sag = state[-4:] if self.delay_steps else command
        h = self.dt / self.substeps

        def integrate(_, physical):
            k1 = self._physical_dynamics(physical, sag, delayed)
            k2 = self._physical_dynamics(physical + 0.5 * h * k1, sag, delayed)
            k3 = self._physical_dynamics(physical + 0.5 * h * k2, sag, delayed)
            k4 = self._physical_dynamics(physical + h * k3, sag, delayed)
            physical = physical + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
            physical = physical.at[3:7].set(jnp.maximum(physical[3:7], 0.))
            return physical.at[7:11].set(jnp.clip(physical[7:11], 0., 1.))

        physical = jax.lax.fori_loop(0, self.substeps, integrate, state[:11])
        if self.delay_steps:
            return jnp.concatenate((physical, state[15:], command))
        return physical

    def step(self, state, command):
        state, command = jnp.asarray(state), jnp.asarray(command)
        if state.shape != (self.state_size,):
            raise ValueError(f"step requires the full {self.state_size}-state vector, including its delay queue")
        if command.shape != (4,):
            raise ValueError("command must have shape (4,)")
        return self._compiled_step(state, command)
