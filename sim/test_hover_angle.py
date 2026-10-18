"""Hover ANGLE mode with the patched Betaflight (angle_tailsitter = ON).

* Betaflight's attitude must be a proper rotation of the sim's: the nose-up
  hover pose reads pitch -90 (Betaflight's nose-down-positive convention).
* In ANGLE (AUX2 high) the hands-off hover stays near level, and a rate kick
  about each hover axis is rejected.  With the real config's zero I-gains a
  small steady lean is expected (the untrimmed pitching moment), so the bound
  is loose.

Needs the built SimITL tool and sim/wing.bin; skips otherwise.
"""

import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_DIR = os.path.join(ROOT, "SimITL", "build", "linux", "tools", "simitl-wing")
TOOL = os.path.join(TOOL_DIR, "simitl-wing")
EEPROM_SRC = os.path.join(ROOT, "sim", "wing.bin")
AS_BUILT = os.path.join(ROOT, "sim", "as_built.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TOOL) and os.path.exists(EEPROM_SRC)),
    reason="SimITL simitl-wing tool/eeprom not built")

HOVER_THROTTLE = "-0.1"          # roughly holds altitude in the sim
COL_T, COL_ATT_PITCH, COL_TILT = 0, 33, 44


def run(*args, dur):
    eeprom = os.path.join(tempfile.mkdtemp(), "wing.bin")
    shutil.copyfile(EEPROM_SRC, eeprom)
    out = subprocess.run(
        [TOOL, "idle", "--wing", AS_BUILT, "--eeprom", eeprom, "--dur",
         str(dur), "-v", *args],
        capture_output=True, text=True, cwd=TOOL_DIR, check=True).stdout
    rows = [l.split(",") for l in out.splitlines()
            if l[:1].isdigit() and l.count(",") == 44]
    t = np.array([float(r[COL_T]) for r in rows])
    return t, rows


def test_betaflight_reads_hover_pose_as_nose_up():
    t, rows = run("--frozen", "--nopulses", "--noarm", "--thr", "-1", dur=2)
    assert float(rows[-1][COL_ATT_PITCH]) == pytest.approx(-90, abs=1)


@pytest.mark.parametrize("kick", [None, "0,0,4", "4,0,0", "0,4,0"],
                         ids=["hands-off", "hover-roll", "hover-pitch", "hover-yaw"])
def test_angle_mode_holds_hover(kick):
    extra = ["--kick", kick] if kick else []
    t, rows = run("--hold", "7.0", "--nopulses", "--thr", HOVER_THROTTLE,
                  "--aux2", "1", *extra, dur=10.0)
    tilt = np.array([float(r[COL_TILT]) for r in rows])
    after = t >= 7.0
    assert tilt[after].max() < 20.0, "hover was not held"
    assert tilt[t >= 9.5].max() < 10.0, "did not settle back near level"


# scripted +0.5 pulses in `simitl-wing idle`: roll 7.5-8.5 s, pitch 9.0-10.0 s,
# yaw 10.5-11.5 s.  Betaflight body rates are x nose, y left, z up.
COL_GYRO = {"roll": 22, "pitch": 23, "yaw": 24}


def test_angle_mode_sticks_keep_3d_plane_meaning():
    """Roll stick spins about the vertical (no lean), pitch stick leans the
    nose forward (toward the belly), rudder leans the nose the way it points
    -- the acro-hover meaning of each stick, now self-levelling."""
    t, rows = run("--hold", "7.45", "--thr", HOVER_THROTTLE, "--aux2", "1",
                  dur=11.6)
    tilt = np.array([float(r[COL_TILT]) for r in rows])

    def window(t0, t1, col):
        m = (t >= t0) & (t < t1)
        return np.array([float(r[col]) for r, k in zip(rows, m) if k])

    roll = window(7.5, 8.5, COL_GYRO["roll"])
    assert roll.mean() > 60, "roll stick should spin (body roll right)"
    assert tilt[(t >= 7.5) & (t < 8.5)].max() < 12, "roll stick should not lean"

    assert window(9.0, 10.0, COL_GYRO["pitch"]).mean() > 5, "pitch fwd = nose down"
    assert tilt[(t >= 9.5) & (t < 10.0)].min() > 15, "pitch stick should hold a lean"

    assert window(10.5, 11.5, COL_GYRO["yaw"]).mean() < -5, "right rudder = nose right"
    assert tilt[(t >= 11.0) & (t < 11.5)].min() > 10, "rudder should hold a lean"
