"""Apply pole-placed PID gains to SimITL's Betaflight via MSP and measure.

Pipeline per target (wn rad/s, zeta):
  1. place_poles -> kp/ki/kd in throttle-fraction units (rad/s domain)
  2. convert to BF P/I/D using scales regressed from a blackbox log
  3. idle sim -> MSP_SET_PID (roll only) + MSP_EEPROM_WRITE -> kill
  4. ghost full-stick roll step -> decode blackbox.bbl -> metrics

Scales (regressed in run 22:20, defaults P=45 I=80 D=30):
  axisP = P * 0.0320 * e_dps            pidSum units
  axisD = D * 2.05e-4 * (-dgyro_dps/dt) pidSum units
  axisI rate = I * 3.62e-3 * e_dps      pidSum/s
  motor_us = pidSum * 0.667             (us)
"""

import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

PB = os.path.expanduser("~/code/SimITL/build/linux/install/bin/simitl-playback")
PB_DIR = os.path.expanduser("~/code/SimITL/tools/simitl-playback")
DECODE = os.path.expanduser("~/code/blackbox-tools/obj/blackbox_decode")

# empirically fit scales (see module docstring)
SP = 0.0320   # axisP per (P * deg/s)
SD = 2.05e-4  # axisD per (D * deg/s^2)
SI = 3.62e-3  # axisI rate per (I * deg/s)
SU = 6.67e-4  # throttle fraction per pidSum unit

RAD2DEG = 180.0 / np.pi


def pid_to_bf(kp, ki, kd):
    """kp/kd: throttle-units per rad/s (per rad/s^2 for kd).
    ki: throttle-units per rad (i.e. per rad/s of error integrated over s).
    Returns BF P,I,D u8 values."""
    P = kp / (SU * SP * RAD2DEG)
    D = kd / (SU * SD * RAD2DEG)
    I = ki / (SU * SI * RAD2DEG)
    return P, I, D


def msp(ws, cmd, payload=b""):
    f = b"$M<" + bytes([len(payload), cmd]) + payload
    c = 0
    for b in f[3:]:
        c ^= b
    ws.send(f + bytes([c]))
    time.sleep(0.4)
    return ws.recv()


def set_roll_pid(P, I, D):
    """Start idle sim, set roll PID via MSP_SET_PID (items order
    roll,pitch,yaw,level,mag as triplets), save eeprom, kill sim."""
    import websocket
    proc = subprocess.Popen([PB, "i", "-no", "-ns"], cwd=PB_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        ws = websocket.create_connection("ws://localhost:5761", timeout=5)
        ws.settimeout(3)
        # current: roll 45/80/30, pitch 47/84/34, yaw 45/80/0, 50/75/75, 40/0/0
        payload = bytes([P, I, D, 47, 84, 34, 45, 80, 0, 50, 75, 75, 40, 0, 0])
        print("  set_pid:", msp(ws, 202, payload).hex())
        print("  eep:", msp(ws, 250).hex())
        ws.close()
    finally:
        proc.kill()
        proc.wait()
        time.sleep(1)


def set_roll_ff(F_roll, pid=(45, 80, 30)):
    """Start idle sim, restore roll PID, set roll feedforward via
    MSP_SET_PID_ADVANCED, save eeprom, kill sim.

    SET_PID_ADVANCED layout (bytes consumed before each field):
      u16 u16 u16 (pid limits) | u8 | u8 (vbatPidComp)
      | u8 ff_transition | u8 | u8 u8 u8 | u16 rateAccel | u16 yawRateAccel
      | u8 angle_limit u8 | u16 u16 anti_gravity | u16
      | u8 iterm_rot u8 u8 iterm_relax u8 | u8 abs u8 throttle_boost
        u8 acro | u16 F_roll u16 F_pitch u16 F_yaw | u8
    """
    import websocket
    proc = subprocess.Popen([PB, "i", "-no", "-ns"], cwd=PB_DIR,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        ws = websocket.create_connection("ws://localhost:5761", timeout=5)
        ws.settimeout(3)
        # restore P/I/D (roll default 45/80/30 etc.)
        pid_payload = bytes([*pid, 47, 84, 34, 45, 80, 0, 50, 75, 75,
                             40, 0, 0])
        print("  set_pid:", msp(ws, 202, pid_payload).hex())
        adv = struct.pack(
            "<3HBB"      # pid limits + reserved + vbatPidComp
            "B"          # feedforward_transition
            "BBBB"       # dtermSetpointWeight + 3 reserved
            "HH"         # rateAccelLimit, yawRateAccelLimit
            "BB"         # angle_limit, reserved
            "HH"         # reserved, anti_gravity_gain
            "H"          # reserved (dtermSetpointWeight)
            "B"          # iterm_rotation
            "B"          # was smart_feedforward
            "BB"         # iterm_relax, iterm_relax_type
            "B"          # abs_control_gain
            "B"          # throttle_boost
            "B"          # acro_trainer_angle_limit
            "HHH"        # F roll, F pitch, F yaw
            "B",         # was antiGravityMode
            0, 0, 0, 0, 0,
            0,           # ff transition
            0, 0, 0, 0,
            0, 0,        # accel limits
            55, 0,       # angle_limit
            0, 80,       # anti_gravity_gain=80
            0,           # dtermSetpointWeight
            1,           # iterm_rotation on
            0,           # smart ff
            3, 0,        # iterm_relax RP(3), type setpoint(0)
            10,          # abs_control_gain
            0,           # throttle_boost
            27,          # acro angle limit
            F_roll, 125, 0,  # F: roll target, pitch default, yaw 0
            0)           # antiGravityMode
        print("  set_adv:", msp(ws, 95, adv).hex())
        print("  eep:", msp(ws, 250).hex())
        ws.close()
    finally:
        proc.kill()
        proc.wait()
        time.sleep(1)


GHOST = os.environ.get("GHOST", "/tmp/ghost_full.json")


def run_step(tag):
    for f in ("blackbox.bbl", "blackbox.01.csv"):
        p = os.path.join(PB_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    subprocess.run([PB, "g", GHOST,
                    "--config", "config/quad/hq-51mm-whoop.json",
                    "-ff", "-no"],
                   cwd=PB_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([DECODE, "blackbox.bbl"], cwd=PB_DIR,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    src = os.path.join(PB_DIR, "blackbox.01.csv")
    dst = os.path.expanduser(
        f"~/code/wing_optimization/quad_design_outputs/step_response/{tag}.csv")
    subprocess.run(["cp", src, dst])
    return dst


def metrics(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    t = (df["time (us)"] - df["time (us)"].iloc[0]) / 1e6
    sp = df["setpoint[0]"].to_numpy()
    g = df["gyroADC[0]"].to_numpy()
    i = np.argmax(np.abs(sp) > 50)
    t0 = t.iloc[i]
    win = (t.to_numpy() > t0) & (t.to_numpy() < t0 + 0.7)
    spv = np.median(sp[(t.to_numpy() > t0 + 0.3) & (t.to_numpy() < t0 + 0.6)])
    w = g[win]
    tw = t.to_numpy()[win] - t0
    rise = tw[np.argmax(w >= 0.9 * spv)]
    peak = w.max()
    ss = np.median(w[tw > 0.3])
    ov = (peak - spv) / spv * 100
    return dict(rise_ms=rise * 1e3, overshoot_pct=ov, steady=ss,
                setpoint=spv)


def main():
    for F in (0, 120, 200):
        print(f"feedforward F={F} (default P45/I80/D30)")
        set_roll_ff(F)
        csv = run_step(f"step_ff{F}")
        m = metrics(csv)
        print(f"  measured: rise={m['rise_ms']:.0f}ms "
              f"overshoot={m['overshoot_pct']:.0f}% "
              f"steady={m['steady']:.0f}/{m['setpoint']:.0f} dps")


if __name__ == "__main__":
    main()
