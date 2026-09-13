import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from multirotor.control_metrics import step_metrics, tracking_metrics
from multirotor.four_motor_control_experiment import FourMotorController
from multirotor.four_motor_plant import FourMotorPlant
from multirotor.rate_control_experiment import save_results

ROOT = Path(__file__).resolve().parent.parent
SIMITL = ROOT / "SimITL"
BRIDGE = SIMITL / "build/linux/tools/simitl-playback/simitl-controller-bridge"
QUAD_CONFIG = SIMITL / "tools/simitl-playback/config/quad/hq-51mm-micro.json"
EEPROM = SIMITL / "tools/simitl-playback/test.bin"
TICK_S = 0.000125
HOVER_RC = -0.155


class SimitlBridge:
    def __init__(self, mode, seed=7, quad_config=QUAD_CONFIG, eeprom=EEPROM):
        if mode not in ("bf", "external"):
            raise ValueError("mode must be bf or external")
        for path, name in ((BRIDGE, "bridge binary"), (quad_config, "quad config"),
                           (eeprom, "eeprom")):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
        self.process = subprocess.Popen(
            [str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._request({"op": "init", "mode": mode, "seed": seed,
                       "quad_config": str(quad_config.resolve()),
                       "eeprom_source": str(eeprom.resolve())})
        self.mode = mode

    def _request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Bridge process terminated")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok"):
                raise RuntimeError(f"Bridge error: {response.get('error')}")
            return response

    def step(self, dt_s, motor_pwm=None, rc=None):
        request = {"op": "step", "dt_s": dt_s}
        if motor_pwm is not None:
            request["motor_pwm"] = [float(v) for v in motor_pwm]
        if rc is not None:
            request["rc"] = [float(v) for v in rc]
        return self._request(request)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=10)


def rc_command(roll=0., pitch=0., yaw=0., throttle=HOVER_RC):
    return [roll, pitch, throttle, yaw, 1., 0., 0., -1.]


def rc_waveform(t, axis, value, hover_s=2., step_s=.8):
    rc = rc_command()
    if hover_s <= t < hover_s + step_s:
        rc[axis] = value
    return rc


def run_simitl_case(case_name, axis=0, rc_value=.5, controller="bf",
                    horizon_ms=40., effort_weight=.1, slew_weight=0.,
                    collective_weight=10., dt=.002, duration_s=4., seed=7,
                    observation="estimated"):
    if controller not in ("bf", "lqr", "mpc"):
        raise ValueError("controller must be bf, lqr, or mpc")
    if dt <= 0 or abs(round(dt / TICK_S) * TICK_S - dt) > 1e-12:
        raise ValueError("dt must be a positive multiple of the 0.125ms sim tick")
    steps = round(duration_s / dt)
    mode = "bf" if controller == "bf" else "external"
    bridge = SimitlBridge(mode, seed=seed)
    policy = None
    if controller != "bf":
        policy = FourMotorController(FourMotorPlant(dt=dt), controller, horizon_ms,
                                     effort_weight, slew_weight, collective_weight,
                                     noise_dps=1.)
    t = np.arange(steps + 1) * dt
    gyro = np.zeros((steps + 1, 3))
    setpoint = np.zeros((steps + 1, 3))
    body = np.zeros((steps + 1, 3))
    pwm = np.zeros((steps, 4))
    bf_pwm = np.zeros((steps, 4))
    timing = np.zeros(steps)
    qp_timing = np.zeros(steps)
    successes = np.ones(steps, dtype=bool)
    statuses = []
    clipped = 0
    try:
        obs = bridge.step(dt, motor_pwm=[.05] * 4 if mode == "external" else None,
                          rc=rc_command())
        gyro[0] = np.deg2rad(obs["gyro_deg_s"])
        setpoint[0] = np.deg2rad(obs["bf_setpoint_deg_s"])
        body[0] = np.deg2rad(obs["body_rates_deg_s"])
        for k in range(steps):
            rc = rc_waveform(t[k + 1], axis, rc_value)
            if policy is None:
                obs = bridge.step(dt, rc=rc)
                command = np.asarray(obs["bf_motor_pwm"], dtype=float)
                status, qp_time, was_clipped = "betaflight", 0., False
            else:
                started = perf_counter()
                command, diag = policy.command(
                    np.deg2rad(obs["gyro_deg_s"]),
                    np.deg2rad(obs["bf_setpoint_deg_s"]),
                    ideal_state=None if observation == "estimated" else np.r_[
                        np.deg2rad(obs["body_rates_deg_s"]),
                        np.asarray(obs["motor_rpm"]) * 2 * np.pi / 60,
                        np.asarray(obs["motor_filtered_pwm"]),
                        np.zeros(policy.model.state_size - 11)],
                    ideal_bias=None if observation == "estimated" else np.zeros(3))
                timing[k] = perf_counter() - started
                status, qp_time, was_clipped = diag["status"], diag["qp_time_s"], diag["lqr_clipped"]
                successes[k] = diag["success"]
                obs = bridge.step(dt, motor_pwm=np.clip(command, .05, 1.), rc=rc)
            pwm[k] = np.asarray(obs["motor_pwm"], dtype=float)
            bf_pwm[k] = np.asarray(obs["bf_motor_pwm"], dtype=float)
            gyro[k + 1] = np.deg2rad(obs["gyro_deg_s"])
            setpoint[k + 1] = np.deg2rad(obs["bf_setpoint_deg_s"])
            body[k + 1] = np.deg2rad(obs["body_rates_deg_s"])
            qp_timing[k] = qp_time
            statuses.append(status)
            clipped += was_clipped
    finally:
        bridge.close()
    measured = gyro
    metrics = tracking_metrics(t, setpoint, measured, pwm, .05, 1.,
                               solve_times_s=timing if policy is not None else None,
                               solver_success=successes if policy is not None else None,
                               dt=dt)
    step_end = int(np.flatnonzero(t >= 2.8)[0]) if np.any(t >= 2.8) else len(t)
    metrics.update(step_metrics(t[:step_end], setpoint[:step_end, axis],
                                measured[:step_end, axis]))
    row = dict(case=case_name, controller=controller, mode=mode,
               axis=("roll", "pitch", "yaw")[axis], rc_value=rc_value,
               horizon_ms=horizon_ms if controller == "mpc" else None,
               effort_weight=effort_weight if controller != "bf" else None,
               slew_weight=slew_weight if controller == "mpc" else 0.,
               collective_weight=collective_weight if controller != "bf" else None,
               dt_s=dt, seed=seed, observation=observation,
               model="simitl_closed_loop_physics_bf_sensors" + (
                   "_external_motor_pwm" if mode == "external" else "_bf_pid"),
               reference="bf_setpoint_deg_s from identical rc waveform",
               measurement="gyro_deg_s (BF filtered gyro)",
               clipped_lqr_steps=clipped,
               solver_statuses="; ".join(sorted(set(statuses))),
               p95_qp_ms=float(np.percentile(qp_timing, 95) * 1000) if policy is not None else None,
               max_qp_ms=float(np.max(qp_timing) * 1000) if policy is not None else None,
               **metrics)
    label = f"{case_name}: {controller} {mode} rc{axis}={rc_value}"
    trace = dict(label=label, t=t, reference=setpoint, rates=measured,
                 commands=pwm, bf_commands=bf_pwm, body_rates=body,
                 solve_times_s=timing, qp_solve_times_s=qp_timing,
                 solver_success=successes)
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dt-ms", type=float, default=2.)
    parser.add_argument("--duration-s", type=float, default=4.)
    parser.add_argument("--observation", choices=("estimated", "ideal"), default="estimated")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    runs = [("bf_small", "bf", .5, {}), ("bf_large", "bf", 1., {})]
    if not args.quick:
        runs += [("bf_reversal", "bf", .8, {})]
    runs += [("mpc_small", "mpc", .5, {"horizon_ms": 40.}),
             ("mpc_large", "mpc", 1., {"horizon_ms": 40.}),
             ("lqr_small", "lqr", .5, {}), ("lqr_large", "lqr", 1., {})]
    rows, traces = [], []
    for name, controller, rc_value, kwargs in runs:
        row, trace = run_simitl_case(name, axis=0, rc_value=rc_value,
                                     controller=controller, dt=args.dt_ms / 1000,
                                     duration_s=args.duration_s, seed=args.seed,
                                     observation=args.observation, **kwargs)
        rows.append(row)
        traces.append(trace)
        print(f"{trace['label']}: RMS={row['rms_error_deg_s']:.2f}deg/s "
              f"overshoot={row['overshoot_percent']} sat={row['saturation_fraction']:.2f} "
              f"p95={row['p95_qp_ms']}ms failures={row['solver_failure_count']}",
              flush=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("quad_design_outputs/optimal_control") / f"simitl_{timestamp}"
    save_results(rows, traces, output,
                 "Matched SimITL Comparison: Betaflight PID vs Four-Motor MPC")
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
