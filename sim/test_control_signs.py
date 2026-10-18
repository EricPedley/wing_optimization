"""Control-direction checks for the Betaflight-in-the-loop tailsitter.

Two convention-free requirements fix the gyro frame and the servo/motor output
mapping, which the CLI dump does not record:

* Feedback: with the airframe frozen in the hover pose, an injected body rate
  must make Betaflight command a moment that opposes it.  A wrong sign on one
  side of the loop (sensor or actuator) fails this.
* Sticks: right roll / forward pitch / right yaw stick must command roll
  right / nose down / nose right.  A *pair* of compensating errors (sensor and
  actuator both flipped) is stable, so only this catches it.

Moments come from sim/wing_model.py (aero frame: x fwd, y right, z down, so
+M = roll right, nose up, nose right) applied to the elevon deflections and
motor outputs the SITL reports.  Thrust is taken as proportional to motor
output, which is enough for a sign.

Needs the built SimITL tool and its wing eeprom; skips otherwise.
"""

import os
import subprocess

import numpy as np
import pytest

from sim import wing_model as wm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_DIR = os.path.join(ROOT, "SimITL", "build", "linux", "tools", "simitl-wing")
TOOL = os.path.join(TOOL_DIR, "simitl-wing")
# the aircraft's verified Betaflight config (sim/load_cli.py); run on a copy
# because Betaflight may write its eeprom
EEPROM_SRC = os.path.join(ROOT, "sim", "wing.bin")
AS_BUILT = os.path.join(ROOT, "sim", "as_built.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TOOL) and os.path.exists(EEPROM_SRC)),
    reason="SimITL simitl-wing tool/eeprom not built")

# Hover pose used by `simitl-wing idle`: nose = world +y, port = world +x,
# dorsal = world -z.  So aero body axes in world: x(nose) = +y, y(right) = -x,
# z(down) = +z.
WORLD_RATE = {0: (0, 1, 0), 1: (-1, 0, 0), 2: (0, 0, 1)}   # p, q, r = +1 rad/s
AXIS = ("roll", "pitch", "yaw")

# CSV columns of `simitl-wing idle`
COL_T, COL_M0, COL_M1, COL_EL, COL_ER = 0, 7, 8, 13, 14


def run_tool(*extra, dur):
    import shutil
    import tempfile
    eeprom = os.path.join(tempfile.mkdtemp(), "wing.bin")
    shutil.copyfile(EEPROM_SRC, eeprom)
    out = subprocess.run(
        [TOOL, "idle", "--wing", AS_BUILT, "--eeprom", eeprom, "--frozen",
         "--thr", "0.3", "--dur", str(dur), *extra],
        capture_output=True, text=True, cwd=TOOL_DIR, check=True).stdout
    rows = [l.split(",") for l in out.splitlines() if l[:1].isdigit()]
    return {round(float(r[COL_T]), 2): r for r in rows}


@pytest.fixture(scope="module")
def model():
    return wm.load_as_built(AS_BUILT)


def moment(model, row):
    p, strips = model
    # motor output columns are per side, {right, left}
    thrust = tuple(0.4 * max(float(row[c]), 0.0) for c in (COL_M0, COL_M1))
    return wm.forces_moments(
        p, strips, np.zeros(3), np.zeros(3),
        float(row[COL_EL]), float(row[COL_ER]), thrust)[1]


@pytest.mark.parametrize("axis", [0, 1, 2], ids=AXIS)
def test_feedback_opposes_injected_rate(model, axis):
    base = run_tool("--nopulses", dur=7.6)
    hit = run_tool("--nopulses", "--inj", ",".join(map(str, WORLD_RATE[axis])),
                   dur=7.6)
    t = max(hit)
    dM = moment(model, hit[t]) - moment(model, base[max(base)])
    assert dM[axis] < 0, f"{AXIS[axis]}: BF response aids the rate ({dM})"


# scripted +0.5 stick pulses in `simitl-wing idle`: roll 7.5-8.5 s,
# pitch 9.0-10.0 s, yaw 10.5-11.5 s; expected moment sign per axis
PULSE = {0: (8.0, +1), 1: (9.5, -1), 2: (11.0, +1)}


@pytest.mark.parametrize("axis", [0, 1, 2], ids=AXIS)
def test_stick_direction(model, axis):
    rows = run_tool(dur=11.6)
    t, want = PULSE[axis]
    dM = moment(model, rows[t]) - moment(model, rows[7.0])
    assert np.sign(dM[axis]) == want, \
        f"{AXIS[axis]} stick commands the wrong way ({dM})"


def test_passthru_drives_differential_thrust():
    """PASSTHRU (AUX2 low) maps sticks straight to the actuators.  Stock
    Betaflight zeroes the PIDs there and the motor mix only reads PIDs, so
    rudder did nothing; patch 0002 feeds the motor mix the sticks too.
    Rudder must split the motors the same way acro does."""
    acro = run_tool("--aux2", "0", dur=11.6)
    passthru = run_tool("--aux2", "-1", dur=11.6)
    split = lambda rows: float(rows[11.0][COL_M0]) - float(rows[11.0][COL_M1])
    assert abs(split(passthru)) > 0.3, "rudder does not split the motors"
    assert np.sign(split(passthru)) == np.sign(split(acro))
    # roll stick moves the elevons, and they hold (no PID decay to zero)
    assert abs(float(passthru[8.0][COL_EL])) > 5.0
    assert float(passthru[8.0][COL_EL]) == pytest.approx(float(passthru[8.4][COL_EL]), abs=0.2)
