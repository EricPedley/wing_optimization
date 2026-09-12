"""Step-response driver for SimITL ghost playback.

Two halves:

1. `write_ghost(path, ...)` writes a ghost.json whose rcData holds a hover
   throttle, then steps one rate axis for a fixed window, then releases.
   simitl-playback's ghost mode replays rcData verbatim after the 8s boot +
   0.1s arm sequence, so this is a rate-setpoint step response.

   Run:   simitl-playback g ghost.json --config config/quad/hq-51mm-micro.json -ff -no
   from tools/simitl-playback/, producing quadstate.csv.

2. `plot(csv_path)` reads quadstate.csv and plots commanded rate
   (rc channel -> deg/s via Betaflight default "actual" rates is not known
   here -- we plot stick position) vs measured angular velocity per axis,
   plus motor outputs, so rise time/overshoot can be read off.

Hover throttle for this build (~0.42 of pack voltage) comes from
quad_model's bisection in the session that added this file; a small
mismatch just makes the quad climb/sink slowly during the step, which is
harmless.
"""

import argparse
import json
import sys

import numpy as np


def write_ghost(path, hover_rc=-0.155, axis=0, step_value=0.5,
                hover_sec=2.0, step_sec=0.8, settle_sec=1.2,
                rc_rate_hz=None, ramp_ms=0.0):
    """ghost.json: hover, then step `axis` (rcData index) by step_value.

    rcData channels: 0=roll, 1=pitch, 2=throttle, 3=yaw, 4=arm.
    simitl-playback forces arm-on + throttle-low during its own arming
    window, and replays these rc values verbatim afterwards.

    rc_rate_hz: if set, emit a sample every 1/rate seconds (held between
    samples by the player) -- models a real RC link's discrete packets so
    BF's own rc smoothing/interpolation sees realistic input.
    """
    def rc(roll=0.0, pitch=0.0, yaw=0.0):
        return [roll, pitch, hover_rc, yaw, 1.0, 0.0, 0.0, -1.0]

    def sample(t, rcvals):
        return {
            "time": t,
            "ghostEvent": 0,
            "camAngle": 0.0,
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "position": {"x": 0.0, "y": 2.0, "z": 0.0},
            "rcData": {f"rc{i}": v for i, v in enumerate(rcvals)},
            "motorData": {"m1rpm": 0, "m2rpm": 0, "m3rpm": 0, "m4rpm": 0},
            "propellerData": {"p1dmg": 0, "p2dmg": 0, "p3dmg": 0, "p4dmg": 0},
        }

    rc_step = rc()
    rc_step[axis] = step_value
    ramp_s = ramp_ms / 1e3

    def rc_at(t):
        """rc at time t, with the step edge ramped over ramp_s."""
        v = rc()
        if hover_sec <= t < hover_sec + step_sec:
            k = 1.0 if ramp_s == 0 else min(1.0, (t - hover_sec) / ramp_s)
            v = list(v)
            v[axis] = step_value * k
        elif t >= hover_sec + step_sec and ramp_s > 0:
            # symmetric ramp on the way back down
            k = max(0.0, 1.0 - (t - (hover_sec + step_sec)) / ramp_s)
            v = list(v)
            v[axis] = step_value * k
        return v

    if rc_rate_hz or ramp_ms:
        dt = 1.0 / (rc_rate_hz or 2000.0)
        t_end = hover_sec + step_sec + settle_sec
        samples = [sample(i * dt, rc_at(i * dt))
                   for i in range(int(t_end / dt) + 1)]
        samples.append(sample(t_end, rc()))
    else:
        samples = [
            sample(0.0, rc()),
            sample(hover_sec, rc_step),
            sample(hover_sec + step_sec, rc()),
            sample(hover_sec + step_sec + settle_sec, rc()),
        ]
    with open(path, "w") as f:
        json.dump({"trackId": 0, "quadId": 0, "samples": samples}, f)
    print(f"wrote {path}: axis rc{axis} step {step_value} at t={hover_sec}s "
          f"for {step_sec}s, hover rc2={hover_rc}")


def actual_rates_setpoint_dps(rc):
    """Rate setpoint (deg/s) from stick deflection, ACTUAL rates."""
    # Betaflight ACTUAL rates as configured in SimITL's eeprom (verified via
    # MSP_RC_TUNING: center sensitivity 200 dps, max 670 dps, expo 0):
    #   rate = rc*center + (max - center) * rc*|rc|
    rc = np.asarray(rc, dtype=float)
    return rc * 200.0 + 470.0 * rc * np.abs(rc)


def plot(csv_path):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "setpoint[0]" in df.columns:
        return plot_blackbox(df, csv_path)

    # ghost-mode csv: frame,time,rc0,rc1,rc2,rc3(mislabeled rc4),
    #                 pos,quat,linvel,angvel,motor1..4
    # angvel is body rates (rad/s): x=pitch, y=yaw, z=roll.
    print(df.columns.tolist())
    t = df["frame"] * 1e-3 - 8.0  # real elapsed time; csv time column is
                                   # the ghost *sample* time, not per-frame
    # SimITL body rates (rad/s, left-handed) -> BF gyro convention (deg/s):
    # bf.cpp updateGyroAcc: roll=-gyro[2], pitch=+gyro[0], yaw=-gyro[1].
    ang = pd.DataFrame({
        "pitch": df.iloc[:, 16],
        "yaw": -df.iloc[:, 17],
        "roll": -df.iloc[:, 18],
    }) * 180.0 / np.pi
    motors = df.iloc[:, 19:23]
    rc = df.iloc[:, 2:6]
    rc.columns = ["roll", "pitch", "throttle", "yaw"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("angular rate (deg/s)",
                                        "motor outputs"))
    for name in ["roll", "pitch", "yaw"]:
        fig.add_trace(go.Scatter(x=t, y=ang[name], name=name), row=1, col=1)
    for name in ["roll", "pitch", "yaw"]:
        fig.add_trace(go.Scatter(x=t, y=actual_rates_setpoint_dps(rc[name]),
                                 name="rc " + name + " setpoint",
                                 line_dash="dash"), row=1, col=1)
    for i in range(4):
        fig.add_trace(go.Scatter(x=t, y=motors.iloc[:, i],
                                 name=f"motor{i+1}"), row=2, col=1)
    out = csv_path.replace(".csv", "_step.html")
    fig.write_html(out)
    print(f"wrote {out}")


def plot_blackbox(df, csv_path):
    """Plot a decoded Betaflight blackbox csv: setpoint vs gyro per axis,
    plus P/I/D/F terms and motor outputs -- all in BF's own numbers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = (df["time (us)"] - df["time (us)"].iloc[0]) / 1e6
    axis_names = ["roll", "pitch", "yaw"]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("rate: setpoint vs gyro (deg/s)",
                                        "PID terms (pidSum units)",
                                        "motor outputs"))
    for i, name in enumerate(axis_names):
        fig.add_trace(go.Scatter(x=t, y=df[f"setpoint[{i}]"], name=f"{name} setpoint",
                                 line_dash="dash"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=df[f"gyroADC[{i}]"], name=f"{name} gyro"),
                      row=1, col=1)
        for term in ["axisP", "axisI", "axisD", "axisF"]:
            col = f"{term}[{i}]"
            if col in df.columns:
                fig.add_trace(go.Scatter(x=t, y=df[col], name=f"{name} {term[4:]}"),
                              row=2, col=1)
    for i in range(4):
        fig.add_trace(go.Scatter(x=t, y=df[f"motor[{i}]"], name=f"motor{i}"),
                      row=3, col=1)
    out = csv_path.replace(".csv", "_step.html")
    fig.write_html(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ghost", "plot"])
    ap.add_argument("path")
    ap.add_argument("--axis", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--hover-rc", type=float, default=-0.155)
    a = ap.parse_args()
    if a.cmd == "ghost":
        write_ghost(a.path, a.hover_rc, a.axis, a.step)
    else:
        plot(a.path)
