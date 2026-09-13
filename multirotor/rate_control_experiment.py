import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import solve_discrete_are

from multirotor import rate_plant as rp
from multirotor.control_metrics import step_metrics, tracking_metrics, write_report
from multirotor.optimal_control import LinearMPC, augment_input_delay, discretize, lqr


@dataclass(frozen=True)
class RateCase:
    name: str = "small_step"
    amplitude_dps: float = 105.0
    duration_s: float = 0.8
    noise_dps: float = 0.25
    gain_scale: float = 1.0
    tau_scale: float = 1.0
    extra_delay_s: float = 0.0
    disturbance_rad_s2: float = 0.0
    waveform: str = "step"


def cases():
    return [RateCase(), RateCase("large_step", 670.),
            RateCase("reversal", 400., waveform="reversal"),
            RateCase("torque_disturbance", disturbance_rad_s2=40.),
            RateCase("gyro_noise", noise_dps=2.),
            RateCase("model_mismatch", gain_scale=.8, tau_scale=1.25,
                     extra_delay_s=.002, disturbance_rad_s2=20.)]


def reference_signal(t, amplitude_dps, waveform="step"):
    t = np.asarray(t)
    if waveform == "step":
        shape = (t >= .1).astype(float)
    elif waveform == "reversal":
        shape = np.where(t < .1, 0., np.where(t < .35, 1., np.where(t < .6, -1., 0.)))
    elif waveform == "ramp":
        shape = np.clip((t - .1) / .1, 0., 1.)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")
    return np.deg2rad(amplitude_dps) * shape


def rate_state_space(K, tau, dt, delay_s=.002, esc_hz=100., gyro_hz=500.):
    if K <= 0 or tau <= 0 or dt <= 0 or delay_s < 0 or esc_hz <= 0 or gyro_hz <= 0:
        raise ValueError("Plant constants and dt must be positive; delay must be nonnegative")
    delay_steps = round(delay_s / dt)
    if not np.isclose(delay_steps * dt, delay_s, atol=1e-10):
        raise ValueError("Delay must be an integer number of controller intervals")
    esc_pole, gyro_pole = 2 * np.pi * esc_hz, 2 * np.pi * gyro_hz
    Ac = np.array([[0., 1., 0., 0.], [0., -1/tau, K/tau, 0.],
                   [0., 0., -esc_pole, 0.], [gyro_pole, 0., 0., -gyro_pole]])
    Bc = np.array([[0.], [0.], [esc_pole], [0.]])
    A0, B0 = discretize(Ac, Bc, dt)
    _, D0 = discretize(Ac, np.array([[1.], [0.], [0.], [0.]]), dt)
    A, B = augment_input_delay(A0, B0, delay_steps)
    D = np.r_[D0[:, 0], np.zeros(A.shape[0] - 4)]
    C = np.zeros((1, A.shape[0]))
    C[0, 3] = 1.
    return A, B, C, D


class RateObserver:
    def __init__(self, A, B, C, D, dt, noise_dps):
        n = len(A)
        self.A = np.block([[A, D[:, None]], [np.zeros((1, n)), np.ones((1, 1))]])
        self.B = np.vstack([B, np.zeros((1, 1))])
        self.C = np.c_[C, 0.]
        process = np.diag(np.r_[1e-5, .5, 1e-5, 1e-5, np.full(n - 4, 1e-9), 25.]) * dt
        measurement = np.array([[max(np.deg2rad(noise_dps)**2, 1e-8)]])
        P = solve_discrete_are(self.A.T, self.C.T, process, measurement)
        self.L = np.linalg.solve(self.C @ P @ self.C.T + measurement, self.C @ P).T
        self.state = np.zeros(n + 1)

    def correct(self, measurement):
        self.state += (self.L @ (np.atleast_1d(measurement) - self.C @ self.state)).ravel()
        return self.state[:-1].copy(), float(self.state[-1])

    def predict(self, command):
        self.state = self.A @ self.state + self.B[:, 0] * command


def tracking_target(reference, disturbance, K, state_size):
    u_ref = -disturbance / K
    x_ref = np.full(state_size, u_ref)
    x_ref[0] = reference
    x_ref[1] = -disturbance
    x_ref[3] = reference
    return x_ref, u_ref


class RatePID:
    def __init__(self, K, tau, dt, limit):
        wn = 2 * np.pi * 10.
        self.kp = tau * wn**2 / K
        self.ki = self.kp * wn / 5.
        self.kd = max(0., (1.4 * wn * tau - 1.) / K)
        self.dt, self.limit = dt, limit
        self.integral = self.previous = self.d1 = self.d2 = 0.

    def command(self, reference, measurement):
        derivative = -(measurement - self.previous) / self.dt
        self.previous = measurement
        self.d1 += -np.expm1(-2*np.pi*75*self.dt) * (derivative - self.d1)
        self.d2 += -np.expm1(-2*np.pi*150*self.dt) * (self.d1 - self.d2)
        error = reference - measurement
        candidate = self.integral + error * self.dt
        raw = self.kp * error + self.ki * candidate + self.kd * self.d2
        bounded = float(np.clip(raw, -self.limit, self.limit))
        if abs(raw) <= self.limit or error * (raw - bounded) < 0:
            self.integral = candidate
        return bounded


def run_case(case, controller="mpc", horizon_ms=40., effort_weight=.1,
             slew_weight=0., dt=.002, observation="estimated", seed=7,
             terminal="lqr", axis="roll"):
    if controller not in ("pid", "lqr", "mpc") or observation not in ("estimated", "ideal"):
        raise ValueError("Unknown controller or observation mode")
    if effort_weight <= 0 or slew_weight < 0 or terminal not in ("lqr", "zero"):
        raise ValueError("Invalid cost weights or terminal option")
    unit = rp._unit()
    hover = rp.hover_point(unit)
    p = rp.plant(axis, unit, hover)
    A, B, C, D = rate_state_space(p["K"], p["tau"], dt)
    At, Bt, Ct, Dt = rate_state_space(p["K"] * case.gain_scale,
                                     p["tau"] * case.tau_scale, dt,
                                     delay_s=.002 + case.extra_delay_s)
    n = len(A)
    limit = 2 * min(hover["frac"] - .05, 1. - hover["frac"])
    rate_scale = np.deg2rad(100.)
    scale = np.r_[rate_scale, p["K"] * limit, limit, rate_scale, np.full(n - 4, limit)]
    An = A * scale[None, :] / scale[:, None]
    Bn = B * limit / scale[:, None]
    Q = np.zeros((n, n))
    Q[0, 0] = dt
    R = np.array([[dt * effort_weight]])
    S = np.array([[dt * slew_weight * (.002 / dt)**2]])
    K, P = lqr(An, Bn, Q, R)
    horizon = round(horizon_ms / (1000 * dt))
    if horizon < 1 or not np.isclose(horizon * dt * 1000, horizon_ms):
        raise ValueError("Horizon must be a positive integer number of intervals")
    if controller == "lqr" and slew_weight:
        raise ValueError("The LQR baseline uses zero slew cost; use MPC for slew sweeps")
    mpc = (LinearMPC(An, Bn, Q, R, horizon, np.array([-1.]), np.array([1.]),
                     terminal_cost=P if terminal == "lqr" else np.zeros_like(P), slew_cost=S)
           if controller == "mpc" else None)
    observer = RateObserver(A, B, C, D, dt, case.noise_dps)
    pid = RatePID(p["K"], p["tau"], dt, limit)
    t = np.arange(round(case.duration_s / dt) + 1) * dt
    reference = reference_signal(t, case.amplitude_dps, case.waveform)
    noise = np.deg2rad(case.noise_dps) * np.random.default_rng(seed).normal(size=len(t))
    physical = np.zeros(len(At))
    rates, commands, timing, qp_timing, successes, estimates = [], [], [], [], [], []
    previous = 0.
    for k, now in enumerate(t):
        disturbance = case.disturbance_rad_s2 if now >= .3 else 0.
        measured = float((Ct @ physical)[0]) + noise[k]
        estimate, bias = observer.correct(measured)
        if observation == "ideal":
            estimate = physical.copy()
            bias = disturbance
        xr, ur = tracking_target(reference[k], bias, p["K"], n)
        started = perf_counter()
        success, qp_time = True, 0.
        if controller == "pid":
            command = pid.command(reference[k], measured)
        elif controller == "lqr":
            command = float(np.clip(ur - limit * (K @ ((estimate - xr) / scale))[0], -limit, limit))
        else:
            result = mpc.solve(estimate / scale, x_ref=xr / scale,
                               u_ref=np.array([ur / limit]), u_prev=np.array([previous / limit]))
            command = float(result.u[0] * limit)
            success, qp_time = result.success, result.solve_time_s
        elapsed = perf_counter() - started
        qp_timing.append(qp_time)
        rates.append(physical[0])
        commands.append(command)
        timing.append(elapsed)
        successes.append(success)
        estimates.append(bias)
        physical = At @ physical + Bt[:, 0] * command + Dt * disturbance
        observer.predict(command)
        previous = command
    rates, commands = np.asarray(rates), np.asarray(commands)
    metrics = tracking_metrics(t, reference, rates, commands[:, None], [-limit], [limit],
                               solve_times_s=np.asarray(timing), solver_success=np.asarray(successes), dt=dt)
    metrics.update(step_metrics(t, reference, rates))
    metadata = dict(case=case.name, controller=controller,
                    horizon_ms=horizon_ms if controller == "mpc" else None,
                    effort_weight=effort_weight if controller != "pid" else None,
                    slew_weight=slew_weight if controller == "mpc" else 0., dt_s=dt,
                    terminal=terminal if controller == "mpc" else None,
                    observation=observation, seed=seed, axis=axis,
                    reference_preview=False, rate_scale_dps=100., command_scale=limit,
                    model="nominal_linear_rate_with_esc_gyro_and_transport_delay",
                    pid_baseline="10Hz pole-placed PID, not Betaflight firmware" if controller == "pid" else None,
                    max_qp_solve_ms=None if controller != "mpc" else float(np.max(qp_timing) * 1000),
                    p95_qp_solve_ms=None if controller != "mpc" else float(np.percentile(qp_timing, 95) * 1000),
                    **{f"plant_{key}": value for key, value in asdict(case).items() if key != "name"})
    label = f"{case.name}: {controller} N={horizon_ms:g}ms R={effort_weight:g} S={slew_weight:g}"
    trace = dict(label=label, t=t, reference=reference, rates=rates, commands=commands[:, None],
                 solve_times_s=np.asarray(timing), qp_solve_times_s=np.asarray(qp_timing),
                 solver_success=np.asarray(successes),
                 disturbance_estimate=np.asarray(estimates))
    return {**metadata, **metrics}, trace


def save_results(rows, traces, output, title):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.json").open("x") as f:
        json.dump(rows, f, indent=2, allow_nan=False)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "metrics.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "traces.npz", **{
        f"run_{i}_{key}": value for i, trace in enumerate(traces)
        for key, value in trace.items() if key != "label"})
    with (output / "trace_labels.json").open("x") as f:
        json.dump([trace["label"] for trace in traces], f)
    selected = traces if len(traces) <= 15 else [traces[i] for i in np.linspace(0, len(traces)-1, 15, dtype=int)]
    write_report(rows, selected, output / "report.html", title=title)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dt-ms", type=float, default=2.)
    parser.add_argument("--observation", choices=["estimated", "ideal"], default="estimated")
    parser.add_argument("--terminal", choices=["lqr", "zero"], default="lqr")
    parser.add_argument("--axis", choices=["roll", "pitch", "yaw"], default="roll")
    args = parser.parse_args()
    horizons = [10., 40., 80.] if args.quick else [10., 20., 40., 80., 160.]
    weights = [.01, .1] if args.quick else [.001, .01, .1, 1., 10.]
    slews = [0., .1] if args.quick else [0., .01, .1, 1.]
    scenarios = cases()[:2] if args.quick else cases()
    rows, traces = [], []
    for case in scenarios:
        configurations = [("pid", 40., .1, 0.)]
        configurations += [("lqr", 40., weight, 0.) for weight in weights]
        configurations += [("mpc", horizon, weight, slew) for horizon in horizons
                           for weight in weights for slew in slews]
        for controller, horizon, weight, slew in configurations:
            row, trace = run_case(case, controller, horizon, weight, slew,
                                  dt=args.dt_ms/1000, observation=args.observation,
                                  terminal=args.terminal, axis=args.axis)
            rows.append(row)
            traces.append(trace)
        print(f"Completed {case.name}: {len(configurations)} configurations", flush=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("quad_design_outputs") / "optimal_control" / f"rate_{timestamp}"
    save_results(rows, traces, output, "Single-Axis LQR / MPC Rate Analysis")
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
