"""Replay a Blackbox flight log's sticks through the sim and compare.

Usage (from repo root):
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python sim/validate_flight.py \
        --csv /tmp/bbl_wing/wing_gliderport_flights.02.csv

Steps:
  1. Extract rcCommand[0..3] (Betaflight semantic order ROLL,PITCH,YAW,THROTTLE,
     in us) and convert to normalized sim channel order (0=roll 1=pitch
     2=throttle 3=yaw 4=arm) -- the aircraft's AERT channel map means the
     semantic slots are swapped vs raw.
  2. Feed to `simitl-wing replay` (level cruise start at 200 m, 12 m/s).
  3. Compare FC-frame gyro rates (deg/s) between the log and the sim and
     report per-axis stats, plus elevon travel.
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIMITL_WING = REPO / "SimITL/build/linux/tools/simitl-wing/simitl-wing"
AS_BUILT = REPO / "sim/as_built.json"
EEPROM = REPO / "SimITL/build/linux/tools/simitl-wing/wing.bin"

US_CENTER = 1500.0
US_HALF = 500.0


def load_bbl(csv_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]

    def col(name):
        return hdr.index(name)

    ti = col("time (us)")
    rc = [col(f"rcCommand[{i}]") for i in range(4)]
    # gyroADC stays raw LSB even with --unit-rotation; gyroUnfilt is deg/s
    gy = [col(f"gyroUnfilt[{i}]") for i in range(3)]
    mo = [col(f"motor[{i}]") for i in range(2)]
    sv = [col(f"servo[{i}]") for i in range(4)]

    out = []
    for r in rows[1:]:
        try:
            t = float(r[ti]) / 1e6
            out.append(dict(
                t=t,
                rc=[float(r[i]) for i in rc],
                gyro=[float(r[i]) for i in gy],
                motor=[float(r[i]) for i in mo],
                servo=[float(r[i]) for i in sv],
            ))
        except (ValueError, IndexError):
            continue
    return out


def to_sticks(rec, t0, t1):
    """semantic rcCommand us -> normalized sim rcData. rcCommand order is
    {ROLL,PITCH,YAW,THROTTLE}; sim channel order is AETR
    {roll,pitch,throttle,yaw}."""
    rows = []
    for r in rec:
        if not (t0 <= r["t"] <= t1):
            continue
        roll, pitch, yaw, thr = r["rc"]
        cl = lambda v: max(-1.0, min(1.0, v))
        rows.append([
            r["t"] - t0,
            cl((roll - US_CENTER) / US_HALF),      # ch0 roll
            cl((pitch - US_CENTER) / US_HALF),     # ch1 pitch
            cl((thr - US_CENTER) / US_HALF),       # ch2 throttle
            cl((yaw - US_CENTER) / US_HALF),       # ch3 yaw
            1.0, 0.0, 0.0, 0.0,                    # arm + aux
        ])
    return rows


def load_sim(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hi = next(i for i, r in enumerate(rows) if r and r[0] == "t")
    hdr = rows[hi]
    ix = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in rows[hi + 1:]:
        try:
            out.append(dict(
                t=float(r[ix["t"]]) - 5.5,        # shift to stick time
                gyro=[float(r[ix[f"gyro{i}"]]) for i in range(3)],
                elev=[float(r[ix["elevL"]]), float(r[ix["elevR"]])],
                m=[float(r[ix["m0"]]), float(r[ix["m1"]])],
                pos=[float(r[ix["posx"]]), float(r[ix["posy"]]),
                     float(r[ix["posz"]])],
                armed=int(r[ix["armed"]]),
            ))
        except (ValueError, IndexError):
            continue
    return out


def interp(series, t):
    """series = list of (t, [v..]); linear interp."""
    if t <= series[0][0]:
        return series[0][1]
    if t >= series[-1][0]:
        return series[-1][1]
    lo = 0
    while lo + 1 < len(series) and series[lo + 1][0] < t:
        lo += 1
    hi = min(lo + 1, len(series) - 1)
    span = series[hi][0] - series[lo][0]
    w = 1.0 if span < 1e-9 else (t - series[lo][0]) / span
    w = max(0.0, min(1.0, w))
    return [a + (b - a) * w for a, b in zip(series[lo][1], series[hi][1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="decoded blackbox csv")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds of replay to skip before comparing")
    ap.add_argument("--outdir", default="/tmp/wing_val")
    ap.add_argument("--no-run", action="store_true",
                    help="skip the sim run, compare existing out csv")
    args = ap.parse_args()

    rec = load_bbl(args.csv)
    print(f"log: {len(rec)} frames, {rec[0]['t']:.1f}..{rec[-1]['t']:.1f}s")

    active = [r["t"] for r in rec if r["motor"][0] > 500 or r["motor"][1] > 500]
    t_lo, t_hi = min(active), max(active)
    print(f"powered span: {t_lo:.1f}..{t_hi:.1f}s")
    t0 = args.t0 if args.t0 is not None else t_lo + 2.0
    t1 = args.t1 if args.t1 is not None else min(t_hi - 3.0, t_lo + 40.0)

    sticks = to_sticks(rec, t0, t1)
    print(f"replay window: {t0:.1f}..{t1:.1f}s ({len(sticks)} samples)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sticks_path = outdir / "sticks.csv"
    sim_csv = outdir / "sim_replay.csv"

    with open(sticks_path, "w") as f:
        f.write("t,rc0,rc1,rc2,rc3,rc4,rc5,rc6,rc7\n")
        for r in sticks:
            f.write(",".join(f"{v:.4f}" for v in r) + "\n")

    if not args.no_run:
        print("running sim replay...")
        res = subprocess.run(
            [str(SIMITL_WING), "replay",
             "--wing", str(AS_BUILT), "--eeprom", str(EEPROM),
             "--sticks", str(sticks_path)],
            capture_output=True, text=True, timeout=600)
        sim_csv.write_text(res.stdout)
        if res.returncode != 0:
            print(res.stderr[:2000], file=sys.stderr)
            sys.exit(1)

    sim = load_sim(sim_csv)
    if not sim:
        print("empty sim output", file=sys.stderr)
        sys.exit(1)

    # --- compare on the log's time grid -----------------------------------
    sim_gyro = [(s["t"], s["gyro"]) for s in sim]
    sim_elev = [(s["t"], s["elev"]) for s in sim]

    axes = ["roll", "pitch", "yaw"]
    names = ["gyro0", "gyro1", "gyro2"]
    print("\naxis     logRMS  simRMS  errRMS   corr")
    for a in range(3):
        errs, logs, sims = [], [], []
        for r in rec:
            t = r["t"] - t0
            if t < args.settle or t > t1 - t0:
                continue
            sg = interp(sim_gyro, t)[a]
            lg = r["gyro"][a]
            errs.append(sg - lg)
            logs.append(lg)
            sims.append(sg)
        if not errs:
            continue
        n = len(errs)
        rms = lambda v: (sum(x * x for x in v) / n) ** 0.5
        ml = sum(logs) / n
        ms = sum(sims) / n
        var_l = max(0.0, sum(x * x for x in logs) - ml * ml * n)
        var_s = max(0.0, sum(x * x for x in sims) - ms * ms * n)
        cov = sum((a - ml) * (b - ms) for a, b in zip(logs, sims))
        corr = cov / max(1e-9, (var_l * var_s) ** 0.5)
        print(f"{axes[a]:6s}  {rms(logs):7.1f} {rms(sims):7.1f} "
              f"{rms(errs):7.1f}  {corr:6.3f}   (n={n})")

    # control power: correlation between stick deflection and resulting gyro
    # rate inside each dataset (robust to different trajectories)
    def power(sticks_list, gyros_list):
        # sticks_list/gyros_list: list of [r,p,y] triples, equal length
        out = []
        for a in range(3):
            st = [s[a] for s in sticks_list]
            gy = [g[a] for g in gyros_list]
            n = len(st)
            mu_s = sum(st) / n
            mu_g = sum(gy) / n
            vs = max(0.0, sum(x * x for x in st) - mu_s * mu_s * n)
            vg = max(0.0, sum(x * x for x in gy) - mu_g * mu_g * n)
            cov = sum((s - mu_s) * (g - mu_g) for s, g in zip(st, gy))
            out.append(cov / max(1e-9, (vs * vg) ** 0.5))
        return out

    # per-axis pair: roll cmd -> gyro0, pitch -> gyro1, yaw -> gyro2.
    ls_, lg_ = [], []
    for r in rec:
        t = r["t"] - t0
        if t < args.settle or t > t1 - t0:
            continue
        ls_.append([(r["rc"][0] - 1500) / 500.0,    # roll cmd
                    (r["rc"][1] - 1500) / 500.0,    # pitch cmd
                    (r["rc"][2] - 1500) / 500.0])   # yaw cmd (semantic)
        lg_.append(r["gyro"])

    # sim sticks: reload the replay file, sample sim gyro at the same times
    sim_g = [(s["t"], s["gyro"]) for s in sim]
    ss_, sg_ = [], []
    with open(sticks_path) as f:
        for row in csv.reader(f):
            if not row or row[0] == "t":
                continue
            tt = float(row[0])
            if tt < args.settle or tt > t1 - t0 - 0.5:
                continue
            ss_.append([float(row[1]), float(row[2]), float(row[4])])
            sg_.append(interp(sim_g, tt))

    pl = power(ls_, lg_) if ls_ else [0] * 3
    ps = power(ss_, sg_) if ss_ else [0] * 3
    print("\ncontrol power corr(stick, rate):")
    for a in range(3):
        print(f"  {axes[a]:6s} log {pl[a]:+.3f}   sim {ps[a]:+.3f}")

    # elevon travel comparison (log servo us -> deg with the same scale)
    sv_deg = [(r["t"] - t0, [
        (r["servo"][3] - 1500.0) * 0.0406,     # servo3 = +pitch-ish side
        (r["servo"][2] - 1500.0) * 0.0406])    # servo2
        for r in rec if t0 <= r["t"] <= t1]
    es, ls_ = [], []
    for t, sd in sv_deg:
        if t < args.settle:
            continue
        ed = interp(sim_elev, t)
        es.append(ed[1])   # right elevon vs servo3
        ls_.append(sd[0])
    if es:
        n = len(es)
        rms = (sum((a - b) ** 2 for a, b in zip(es, ls_)) / n) ** 0.5
        print(f"\nelevon R: log-mean {sum(ls_)/n:+.1f}deg "
              f"sim-mean {sum(es)/n:+.1f}deg  err-RMS {rms:.1f}deg")

    print(f"\noutputs in {outdir}: sticks.csv sim_replay.csv")


if __name__ == "__main__":
    main()
