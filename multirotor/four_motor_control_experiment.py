import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag, lu_factor, lu_solve, solve_discrete_are

from multirotor.control_metrics import tracking_metrics
from multirotor.four_motor_plant import FourMotorPlant
from multirotor.optimal_control import LinearMPC, lqr
from multirotor.rate_control_experiment import save_results


@dataclass(frozen=True)
class FourMotorCase:
    name: str = "small_coupled_step"
    amplitude_dps: tuple = (80., 60., 40.)
    duration_s: float = .8
    noise_dps: float = .25
    waveform: str = "step"
    disturbance_rad_s2: tuple = (0., 0., 0.)
    inertia_scale: tuple = (1., 1., 1.)
    motor_variance: tuple = (1., 1., 1., 1.)
    remaining_mah: float | None = None
    collective_target: float = 1.


def cases():
    return [FourMotorCase(),
            FourMotorCase("large_coupled_step", (300., 200., 100.)),
            FourMotorCase("coupled_reversal", (400., 300., 150.), waveform="reversal"),
            FourMotorCase("angular_disturbance", disturbance_rad_s2=(20., -15., 10.)),
            FourMotorCase("motor_inertia_mismatch", inertia_scale=(1.2, .85, 1.15),
                          motor_variance=(.96, 1.03, .98, 1.02)),
            FourMotorCase("depleted_pack", remaining_mah=68.)]


def reference_signal(t, amplitude_dps, waveform="step"):
    t, amplitude = np.asarray(t), np.asarray(amplitude_dps, dtype=float)
    if amplitude.shape != (3,) or not np.isfinite(amplitude).all():
        raise ValueError("amplitude_dps must contain three finite rates")
    if waveform == "step":
        shape = (t >= .1).astype(float)
    elif waveform == "reversal":
        shape = np.where(t < .1, 0., np.where(t < .35, 1., np.where(t < .6, -1., 0.)))
    else:
        raise ValueError("Unknown waveform")
    return shape[:, None] * np.deg2rad(amplitude)


@lru_cache(maxsize=8)
def _nominal_plant(remaining_mah, dt):
    return FourMotorPlant(remaining_mah=remaining_mah, dt=dt)


@lru_cache(maxsize=8)
def _simulation_plant(remaining_mah, dt, inertia_scale, motor_variance):
    nominal = _nominal_plant(remaining_mah, dt)
    if inertia_scale == (1., 1., 1.) and motor_variance == (1., 1., 1., 1.):
        return nominal
    unit = {**nominal.unit, "inertia": np.asarray(nominal.unit["inertia"]) * inertia_scale}
    return FourMotorPlant(unit=unit, remaining_mah=remaining_mah, dt=dt,
                          motor_variance=motor_variance)


class RateStateObserver:
    def __init__(self, A, B, C, D, dt, noise_dps):
        n = len(A)
        self.A = np.block([[A, D], [np.zeros((3, n)), np.eye(3)]])
        self.B = np.vstack([B, np.zeros((3, 4))])
        self.C = np.c_[C, np.zeros((3, 3))]
        process = np.diag(np.r_[np.full(3, 2e-4), np.full(4, 1e-3),
                               np.full(n - 7, 1e-10), np.full(3, 10.)]) * dt
        measurement = np.eye(3) * max((noise_dps / 100.)**2, 1e-8)
        P = solve_discrete_are(self.A.T, self.C.T, process, measurement)
        self.L = np.linalg.solve(self.C @ P @ self.C.T + measurement, self.C @ P).T
        self.state = np.zeros(n + 3)

    def correct(self, measured_scaled_rates):
        self.state += self.L @ (measured_scaled_rates - self.C @ self.state)
        return self.state[:-3].copy(), self.state[-3:].copy()

    def predict(self, command_deviation):
        self.state = self.A @ self.state + self.B @ command_deviation


class FourMotorController:
    def __init__(self, model, controller="mpc", horizon_ms=40., effort_weight=.1,
                 slew_weight=0., collective_weight=10., noise_dps=.25):
        if controller not in ("lqr", "mpc"):
            raise ValueError("controller must be lqr or mpc")
        weights = np.asarray([effort_weight, slew_weight, collective_weight, noise_dps])
        if not np.isfinite(weights).all() or effort_weight <= 0 or np.any(weights[1:] < 0):
            raise ValueError("Invalid cost weights or measurement noise")
        if controller == "lqr" and slew_weight != 0:
            raise ValueError("LQR baseline requires zero slew weight")
        if not np.isfinite(horizon_ms) or horizon_ms <= 0:
            raise ValueError("horizon_ms must be positive and finite")
        horizon = round(horizon_ms / (model.dt * 1000))
        if horizon < 1 or not np.isclose(horizon * model.dt * 1000, horizon_ms):
            raise ValueError("Horizon must be an integer number of control intervals")
        self.model, self.controller = model, controller
        self.rate_scale = np.deg2rad(100.)
        self.scale = np.r_[np.full(3, self.rate_scale), model.trim_state[3:7],
                           np.ones(model.state_size - 7)]
        A, B = model.discrete_model()
        self.A = A * self.scale[None, :] / self.scale[:, None]
        self.B = B / self.scale[:, None]
        n = len(A)
        self.C = np.eye(3, n)
        self.D = self.C.T * model.dt
        gradient = np.asarray(jax.grad(model.collective_thrust)(jnp.asarray(model.trim_state)))
        self.collective = gradient * self.scale / (model.mass * 9.81)
        Q = model.dt * (self.C.T @ self.C + collective_weight * np.outer(self.collective, self.collective))
        R, S = np.eye(4) * model.dt * effort_weight, np.eye(4) * model.dt * slew_weight
        self.K, P = lqr(self.A, self.B, Q, R)
        target_matrix = np.block([
            [np.eye(n) - self.A, -self.B],
            [self.C, np.zeros((3, 4))],
            [self.collective[None, :], np.zeros((1, 4))],
        ])
        self.target_matrix = target_matrix
        self.target_factor = lu_factor(target_matrix)
        self.observer = RateStateObserver(self.A, self.B, self.C, self.D, model.dt, noise_dps)
        self.mpc = (LinearMPC(self.observer.A, self.observer.B, block_diag(Q, np.zeros((3, 3))),
                             R, horizon, .05 - model.trim_command, 1. - model.trim_command,
                             terminal_cost=block_diag(P, np.zeros((3, 3))), slew_cost=S)
                    if controller == "mpc" else None)
        self.previous = np.zeros(4)

    def tracking_target(self, reference, bias, collective_target=1.):
        rhs = np.r_[self.D @ bias, reference / self.rate_scale, collective_target - 1.]
        target = lu_solve(self.target_factor, rhs)
        residual = float(np.max(np.abs(self.target_matrix @ target - rhs)))
        return target[:-4], target[-4:], residual

    def command(self, measured_rates, reference, collective_target=1., ideal_state=None, ideal_bias=None):
        start = perf_counter()
        estimate, bias = self.observer.correct(np.asarray(measured_rates) / self.rate_scale)
        if ideal_state is not None:
            estimate = (np.asarray(ideal_state) - self.model.trim_state) / self.scale
            bias = np.asarray(ideal_bias) / self.rate_scale
        xr, ur, target_residual = self.tracking_target(np.asarray(reference), bias, collective_target)
        if self.mpc is None:
            raw = ur - self.K @ (estimate - xr)
            deviation = np.clip(raw, .05 - self.model.trim_command, 1. - self.model.trim_command)
            success, status, qp_time, iterations, violation = True, "clipped LQR", 0., 0, 0.
            clipped = bool(np.any(raw != deviation))
        else:
            result = self.mpc.solve(np.r_[estimate, bias], x_ref=np.r_[xr, bias],
                                    u_ref=ur, u_prev=self.previous)
            deviation = result.u
            success, status = result.success, result.status
            qp_time, iterations, violation = result.solve_time_s, result.iterations, result.max_constraint_violation
            clipped = False
        command = self.model.trim_command + deviation
        self.observer.predict(deviation)
        self.previous = deviation.copy()
        elapsed = perf_counter() - start
        diagnostics = dict(success=success, status=status, control_time_s=elapsed, qp_time_s=qp_time,
                           iterations=iterations, constraint_violation=violation, lqr_clipped=clipped,
                           target_residual=target_residual,
                           target_outside_bounds=bool(np.any(ur + self.model.trim_command < .05) or
                                                      np.any(ur + self.model.trim_command > 1.)),
                           estimate=estimate, bias=bias)
        return command, diagnostics


def run_case(case=FourMotorCase(), controller="mpc", horizon_ms=40., effort_weight=.1,
             slew_weight=0., collective_weight=10., dt=.002, observation="estimated", seed=7,
             initial_state=None):
    if observation not in ("estimated", "ideal"):
        raise ValueError("observation must be estimated or ideal")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(case.duration_s) or case.duration_s < dt:
        raise ValueError("dt and duration must be positive with at least one interval")
    if not np.isfinite(case.collective_target) or case.collective_target <= 0:
        raise ValueError("collective_target must be finite and positive")
    disturbance = np.asarray(case.disturbance_rad_s2, dtype=float)
    if disturbance.shape != (3,) or not np.isfinite(disturbance).all():
        raise ValueError("disturbance_rad_s2 must contain three finite accelerations")
    model = _nominal_plant(case.remaining_mah, dt)
    plant = _simulation_plant(case.remaining_mah, dt, tuple(case.inertia_scale), tuple(case.motor_variance))
    policy = FourMotorController(model, controller, horizon_ms, effort_weight,
                                 slew_weight, collective_weight, case.noise_dps)
    physical = plant.trim_state.copy() if initial_state is None else np.asarray(initial_state, dtype=float).copy()
    if physical.shape != plant.trim_state.shape or not np.isfinite(physical).all():
        raise ValueError("initial_state must be a finite full plant state")
    t = np.arange(round(case.duration_s / dt) + 1) * dt
    reference = reference_signal(t, case.amplitude_dps, case.waveform)
    collective_reference = np.where(t >= .1, case.collective_target, 1.)
    noise = np.deg2rad(case.noise_dps) * np.random.default_rng(seed).normal(size=(len(t), 3))
    states, commands, measurements, diagnostics = [physical.copy()], [], [], []
    for k, now in enumerate(t[:-1]):
        applied_disturbance = disturbance if now >= .3 else np.zeros(3)
        measurement = physical[:3] + noise[k]
        command, diagnostic = policy.command(
            measurement, reference[k], collective_reference[k],
            ideal_state=physical if observation == "ideal" else None,
            ideal_bias=applied_disturbance if observation == "ideal" else None)
        physical = np.asarray(plant.step(physical, command)).copy()
        physical[:3] += dt * applied_disturbance
        if not np.isfinite(physical).all():
            raise RuntimeError(f"Nonfinite nonlinear simulation in {case.name} at {now:g}s")
        states.append(physical.copy())
        commands.append(command)
        measurements.append(measurement)
        diagnostics.append(diagnostic)
    states, commands = np.asarray(states), np.asarray(commands)
    rates = states[:, :3]
    collective = np.asarray(jax.jit(jax.vmap(plant.collective_thrust))(jnp.asarray(states))) / (plant.mass * 9.81)
    timing = np.array([d["control_time_s"] for d in diagnostics])
    qp_timing = np.array([d["qp_time_s"] for d in diagnostics])
    successes = np.array([d["success"] for d in diagnostics])
    metrics = tracking_metrics(t, reference, rates, commands, .05, 1., timing, successes, dt)
    metrics["absolute_pwm_effort"] = metrics["command_effort"]
    metrics["command_effort"] = float(np.sum(np.diff(t)[:, None] * (commands - model.trim_command)**2))
    metrics["collective_rms_percent"] = float(100 * np.sqrt(np.trapezoid((collective - collective_reference)**2, t) / t[-1]))
    metrics["tail_rms_error_deg_s"] = float(np.sqrt(np.mean(np.rad2deg(rates[t >= .75 * t[-1]] - reference[t >= .75 * t[-1]])**2)))
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        metrics[f"{name}_rms_error_deg_s"] = float(np.sqrt(np.trapezoid(np.rad2deg(rates[:, axis] - reference[:, axis])**2, t) / t[-1]))
    violations = np.array([d["constraint_violation"] for d in diagnostics])
    metadata = dict(case=case.name, controller=controller, observation=observation,
                    observation_detail="noisy rates only; rotor/filter/queue predicted from known PWM; angular bias estimated"
                    if observation == "estimated" else "oracle full state and applied angular disturbance upper bound",
                    model="nonlinear zero-inflow frozen-battery rate plant; no translation; not full SimITL parity",
                    horizon_ms=horizon_ms if controller == "mpc" else None, dt_s=dt,
                    effort_weight=effort_weight, slew_weight=slew_weight, collective_weight=collective_weight,
                    terminal="physical-state no-slew DARE P; zero terminal weight on fixed disturbance states",
                    reference_preview=False, rate_scale_dps=100., rotor_scale="per-motor nominal hover rad/s",
                    pwm_scale=1., stage_cost_multiplied_by_dt=True, seed=seed, noise_dps=case.noise_dps,
                    remaining_mah=plant.remaining_mah, controller_remaining_mah=model.remaining_mah,
                    effort_definition="integral sum squared PWM deviations from nominal trim",
                    command_trace_definition="absolute four-motor PWM", collective_target=case.collective_target,
                    collective_observed=False, disturbance_model="dt angular-acceleration injection, 3 constant estimated states",
                    disturbance_rad_s2=str(tuple(case.disturbance_rad_s2)),
                    inertia_scale=str(tuple(case.inertia_scale)), motor_variance=str(tuple(case.motor_variance)),
                    timing_scope="observer correction/prediction + equilibrium target + controller; excludes nonlinear simulation and construction",
                    p95_qp_wall_ms=float(np.percentile(qp_timing, 95) * 1000),
                    max_qp_wall_ms=float(np.max(qp_timing) * 1000),
                    solver_statuses="; ".join(sorted(set(d["status"] for d in diagnostics))),
                    max_constraint_violation=float(np.max(violations)) if np.isfinite(violations).all() else None,
                    nonfinite_violation_count=int(np.count_nonzero(~np.isfinite(violations))),
                    lqr_clipped_fraction=float(np.mean([d["lqr_clipped"] for d in diagnostics])),
                    target_outside_bounds_fraction=float(np.mean([d["target_outside_bounds"] for d in diagnostics])),
                    max_target_equilibrium_residual=float(max(d["target_residual"] for d in diagnostics)),
                    target_model="trim-linear equilibrium; nonlinear combined high-rate target may be imperfect")
    label = f"{case.name}: {controller} N={horizon_ms:g}ms R={effort_weight:g} S={slew_weight:g} {observation}"
    trace = dict(label=label, t=t, reference=reference, rates=rates, commands=commands,
                 command_deviations=commands - model.trim_command, full_state=states,
                 measured_rates=np.asarray(measurements), state_estimate=np.array([d["estimate"] for d in diagnostics]),
                 disturbance_estimate=np.array([d["bias"] for d in diagnostics]) * policy.rate_scale,
                 collective=collective, collective_reference=collective_reference,
                 solve_times_s=timing, qp_solve_times_s=qp_timing, solver_success=successes,
                 solver_status=np.array([d["status"] for d in diagnostics]),
                 solver_iterations=np.array([d["iterations"] for d in diagnostics]),
                 constraint_violation=violations)
    return {"label": label, **metadata, **metrics}, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--observation", choices=("estimated", "ideal"), default="estimated")
    parser.add_argument("--include-160ms", action="store_true")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("quad_design_outputs/optimal_control") / f"four_motor_{timestamp}"
    if output.exists():
        raise FileExistsError(output)
    horizons = [10., 40.] if args.quick else [10., 20., 40., 80.]
    if args.include_160ms:
        horizons.append(160.)
    weights = [.1] if args.quick else [.01, .1, 1.]
    slews = [0., .1]
    scenarios = cases()[:2] if args.quick else cases()
    rows, traces = [], []
    for case in scenarios:
        configurations = [("lqr", 40., weight, 0.) for weight in weights]
        configurations += [("mpc", horizon, weight, slew) for horizon in horizons
                           for weight in weights for slew in slews]
        for controller, horizon, weight, slew in configurations:
            row, trace = run_case(case, controller, horizon, weight, slew, observation=args.observation)
            rows.append(row)
            traces.append(trace)
            print(f"{trace['label']}: RMS={row['rms_error_deg_s']:.3f}deg/s "
                  f"collective={row['collective_rms_percent']:.3f}% "
                  f"failures={row['solver_failure_count']} p95={row['p95_solver_wall_ms']:.3f}ms", flush=True)
    save_results(rows, traces, output, "Four-Motor LQR / Constrained MPC Rate Analysis (No Translation)")
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
