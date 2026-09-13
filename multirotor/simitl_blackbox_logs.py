import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from multirotor.four_motor_control_experiment import FourMotorController
from multirotor.four_motor_plant import FourMotorPlant
from multirotor.simitl_mpc_experiment import SimitlBridge, rc_command

DECODE = Path("~/code/blackbox-tools/obj/blackbox_decode").expanduser()


def axes_waveform(t, value, lead_s=1., step_s=.4, settle_s=.4):
    # rc order: [roll, pitch, throttle, yaw, arm, ...] -- step roll, pitch, yaw in turn
    rc = rc_command()
    local = t - lead_s
    for slot, channel in enumerate((0, 1, 3)):
        if 0 <= local - slot * (step_s + settle_s) < step_s:
            rc[channel] = value
    return rc


def run_blackbox_log(controller, output_bbl, rc_value=1., dt=.002, seed=7,
                     step_s=.4, settle_s=.4, lead_s=1., tail_s=.5,
                     horizon_ms=40., effort_weight=.1, slew_weight=0.):
    mode = "bf" if controller == "bf" else "external"
    bridge = SimitlBridge(mode, seed=seed)
    policy = None
    if controller != "bf":
        policy = FourMotorController(FourMotorPlant(dt=dt), controller, horizon_ms,
                                     effort_weight, slew_weight, noise_dps=1.)
    duration = lead_s + 3 * (step_s + settle_s) + tail_s
    steps = round(duration / dt)
    workdir = None
    try:
        obs = bridge.step(dt, motor_pwm=[.05] * 4 if mode == "external" else None,
                          rc=rc_command())
        workdir = Path(obs["workdir"])
        for k in range(steps):
            t = (k + 1) * dt
            rc = axes_waveform(t, rc_value, lead_s, step_s, settle_s)
            if policy is None:
                obs = bridge.step(dt, rc=rc)
            else:
                command, _ = policy.command(np.deg2rad(obs["gyro_deg_s"]),
                                            np.deg2rad(obs["bf_setpoint_deg_s"]))
                obs = bridge.step(dt, motor_pwm=np.clip(command, .05, 1.), rc=rc)
        disarm = rc_command()
        disarm[4] = -1.
        for _ in range(round(.8 / dt)):
            bridge.step(dt, motor_pwm=[0.] * 4 if mode == "external" else None, rc=disarm)
    finally:
        bridge.close()
    log = workdir / "blackbox.bbl"
    if not log.is_file() or log.stat().st_size < 1024:
        raise RuntimeError(f"No blackbox log produced at {log}")
    output_bbl = Path(output_bbl)
    output_bbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(log, output_bbl)
    return output_bbl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("quad_design_outputs/optimal_control/blackbox"))
    parser.add_argument("--rc", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    logs = {}
    for controller, name in (("bf", "pid_axes.bbl"), ("mpc", "mpc_axes.bbl")):
        path = run_blackbox_log(controller, args.output / name, rc_value=args.rc,
                                seed=args.seed)
        logs[controller] = path
        print(f"{controller}: {path} ({path.stat().st_size} bytes)", flush=True)
        if DECODE.is_file():
            subprocess.run([str(DECODE), path.name], cwd=path.parent)
            print(f"  decoded -> {path.parent}/blackbox.01.csv")
        else:
            print(f"  decoder missing: {DECODE}")


if __name__ == "__main__":
    main()
