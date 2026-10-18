"""Generate sim/as_built.json: the as-built aircraft description consumed by
the SimITL tailsitter physics (tools/simitl-wing/wingcfg.h) and by the JAX
reference model (sim/wing_model.py).

Sources:
  * airfoil/optimum.json          - optimizer design vector (as-flown wing)
  * airfoil/airfoil_model.py      - geometry/camber/inertia helpers
  * airfoil/linkage_coupling.py   - servo linkage throw/advantage

Mass: measured all-up weight 80 g (Onshape mass properties are NOT used --
they were never calibrated, see FLIGHT_SIM_PLAN).  Inertia uses the model's
distribution scaled by measured/model mass.  Thrust is exposed via
thrustScale because static thrust was not measured.

JSON shape: no arrays -- the C++ tools/simitl-wing/wingcfg.h parser only
understands objects/numbers/strings, so vectors are {x,y,z} objects and
two-element lists become named fields.

Run:  uv run python -m sim.as_built
"""

import json
import os
import sys

import numpy as np

from airfoil import airfoil_model as am
from airfoil import linkage_coupling

MEASURED_MASS_KG = 0.080          # user-measured all-up weight
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMUM = os.path.join(ROOT, "airfoil", "optimum.json")
OUT = os.path.join(ROOT, "sim", "as_built.json")


def build():
    with open(OPTIMUM) as f:
        opt = json.load(f)
    p = np.array(opt["x"], dtype=float)
    g = am.unpack(p)

    semi = 0.5 * g["span"]
    cg = float(am.cg_station(
        g["span"], g["root_chord"], g["tip_chord"], g["root_thickness"],
        g["tip_thickness"], g["le_sweep_deg"], g["battery_station"],
        g["servo_chord_frac"], g["servo_span_frac"], g["motor_standoff"]))

    # --- mass + inertia -----------------------------------------------------
    model_mass = float(am.total_mass(
        g["span"], g["root_chord"], g["tip_chord"], g["root_thickness"],
        g["tip_thickness"], g["motor_standoff"]))
    i_roll, i_pitch, i_yaw = (float(v) for v in am.inertia(
        g["span"], g["root_chord"], g["tip_chord"], g["root_thickness"],
        g["tip_thickness"], g["motor_standoff"]))
    scale = MEASURED_MASS_KG / model_mass

    # --- linkage / elevon throw ---------------------------------------------
    _, mac = am.planform(g["span"], g["root_chord"], g["tip_chord"])
    pushrod = float(am.pushrod_length(
        g["servo_chord"], g["servo_chord_frac"],
        am.hinge_fraction_at(g["servo_chord"], g["x_hinge"], float(mac))))
    lk = linkage_coupling.linkage_metrics(
        pushrod, g["servo_rod_y"], g["servo_travel_mm"],
        g["flap_x_mm"], g["flap_y_mm"])
    elevon_max_deg = float(lk["delta_max_deg"])

    # --- motor / prop disk position -----------------------------------------
    motor_y = g["motor_y"]
    le_motor = float(am.leading_edge_x(
        g["span"], abs(motor_y) / semi, g["le_sweep_deg"]))
    # prop disk sits at the front face of the motor body:
    # motor_station is the mass centre (-(standoff + body/2)), so the disk is
    # another half body-length forward of that.
    motor_x_aft = le_motor - (g["motor_standoff"] + am.MOTOR_BODY_LENGTH)

    # --- hinge line, aerospace frame (x fwd, y right, z down) ---------------
    # hinge station x, m aft of root LE, at each end of the elevon
    def hinge_x(y_abs):
        le = float(am.leading_edge_x(g["span"], y_abs / semi, g["le_sweep_deg"]))
        c, _ = am.local_geometry(g["span"], g["root_chord"], g["tip_chord"],
                                 g["root_thickness"], g["tip_thickness"],
                                 y_abs / semi)
        return le + float(am.hinge_fraction_at(float(c), g["x_hinge"],
                                               float(mac))) * float(c)

    xb = hinge_x(g["elevon_inboard_y"])
    xe = hinge_x(g["elevon_outboard_y"])
    yb, ye = g["elevon_inboard_y"], g["elevon_outboard_y"]
    # hinge line direction on the right wing, expressed in aerospace coords
    # (aero x = -(station - cg), aero y = +spanwise)
    dx = (xb - xe)          # aft-positive difference -> negate for aero x
    dy = ye - yb
    axis_r = np.array([-dx, dy, 0.0])
    axis_r /= np.linalg.norm(axis_r)
    # mid-station hinge point, aerospace frame
    ym = 0.5 * (yb + ye)
    xm_aft = hinge_x(ym)
    hinge_pt_r = [cg - xm_aft, ym, 0.0]
    hinge_pt_l = [cg - xm_aft, -ym, 0.0]
    axis_l = [-axis_r[0], axis_r[1], 0.0]  # mirror in span

    doc = {
        "vehicleType": 1,
        "mass": MEASURED_MASS_KG,
        "cgX": cg,
        "inertia": {
            "xx": i_roll * scale, "yy": i_pitch * scale, "zz": i_yaw * scale,
            "xy": 0.0, "xz": 0.0, "yz": 0.0,
        },
        "span": g["span"],
        "rootChord": g["root_chord"],
        "tipChord": g["tip_chord"],
        "rootHalfWidth": am.ROOT_SECTION_WIDTH / 2.0,
        "leSweepDeg": g["le_sweep_deg"],
        "thicknessRatioRoot": g["root_thickness"] / g["root_chord"],
        "thicknessRatioTip": g["tip_thickness"] / g["tip_chord"],
        "camber": {
            "a1Root": g["camber_a1"], "a2Root": g["camber_a2"],
            "a1Tip": g["tip_camber_a1"], "a2Tip": g["tip_camber_a2"],
        },
        "xHinge": g["x_hinge"],
        "elevonInboardY": g["elevon_inboard_y"],
        "elevonOutboardY": g["elevon_outboard_y"],
        "elevonMaxDeg": elevon_max_deg,
        "hinge": {
            "pointR": {"x": hinge_pt_r[0], "y": hinge_pt_r[1],
                       "z": hinge_pt_r[2]},
            "axisR":  {"x": float(axis_r[0]), "y": float(axis_r[1]),
                       "z": float(axis_r[2])},
            "pointL": {"x": hinge_pt_l[0], "y": hinge_pt_l[1],
                       "z": hinge_pt_l[2]},
            "axisL":  {"x": float(axis_l[0]), "y": float(axis_l[1]),
                       "z": float(axis_l[2])},
        },
        "motorY": motor_y,
        "motorXAft": motor_x_aft,
        "motorZ": 0.0,
        "motorSpinDir": {"right": 1, "left": -1},
        "thrustScale": 1.0,
        "propDiameter": am.PROP_DIAMETER,

        # section / aero constants, all from airfoil_model.py so the JSON
        # is self-contained (C++ and python defaults mirror these)
        "aero": {
            "clMaxBase": am.CL_MAX_BASE,
            "clMaxCamberGain": am.CL_MAX_CAMBER_GAIN,
            "clMaxTcGain": am.CL_MAX_TC_GAIN,
            "clMaxTcRef": am.CL_MAX_TC_REF,
            "clMaxReGain": am.CL_MAX_RE_GAIN,
            "clMaxReRef": am.CL_MAX_RE_REF,
            "clMaxCeiling": am.CL_MAX_CEILING,
            "flapEffectiveness": am.FLAP_EFFECTIVENESS_FACTOR,
            "flapStallDeg": am.FLAP_STALL_DEG,
            "flapSaturationDeg": am.FLAP_SATURATION_DEG,
            "slipstreamDevelopment": 0.5,
            "slipstreamWidthFactor": 1.0,
            "cd0": 0.02,
            "cdInduced": 0.0,
            "cdmaxViterna": 0.0,
            "numStrips": 33,
        },

        # typical micro-servo defaults (user: "typical micro-servo defaults")
        # smix targets 2,3 are written to physical servo outputs 0,1 for
        # CUSTOMAIRPLANE (SERVO_PLANE_INDEX_MIN=2 -> first physical output).
        # Headless sign check (frozen pose, armed, stick pulses):
        #   roll+  -> pwm0- pwm1-    (smix: t2=-r-p, t3=-r+p)
        #   pitch+ -> pwm0- pwm1+
        # For roll+ = right roll and pitch+ = nose up (both elevons TE up),
        # output 0 must drive the RIGHT elevon (signR=+1 -> pwm down = TE up)
        # and output 1 the LEFT (signL=-1 -> pwm up = TE up).  TE-down positive.
        "servo": {
            "indexL": 1, "indexR": 0,
            "centerUs": 1500.0,
            # map BF servo output range to full elevon throw; the actual
            # usable pwm range is +/- ~450us around centre
            "degPerUs": elevon_max_deg / 450.0,
            "signL": -1.0, "signR": 1.0,
            "tauSec": 0.02,
            "rateDps": 600.0,
            "minUs": 500.0, "maxUs": 2500.0,
        },
        "fcYawDeg": 180.0,

        # propulsion: no bench data for the as-built motor; closest fitted
        # motor/prop constants from the SimITL factor-graph catalogue.
        # thrustScale is the plan's exposed trim for the unmeasured static
        # thrust.
        "motor": {"kv": 21000.0, "r": 0.07, "i0": 0.845,
                  "rth": 35.0, "cth": 2.5, "maxT": 180.0},
        "propeller": {
            "bladeCount": 3, "maxRpm": 30000.0,
            "aFactor": 5.2746e-10, "torqueFactor": 6.5823e-3,
            "inertia": 1.154e-7,
            "thrustFactor": {"x": 0.0, "y": -9.626e-3, "z": 0.4747},
        },
        # 1S 480 mAh BT2.0 cell; sag tuned for ~9.8 mOhm internal R at hover
        # current
        "battery": {"maxVoltageSag": 0.35, "cellCount": 1,
                    "capacityCharged": 480.0, "capacity": 480.0},
        "ambientTemp": 25.0,
        "eepromName": "wing.bin",
    }

    return doc, dict(model_mass=model_mass, cg=cg, mac=float(mac),
                     elevon_max_deg=elevon_max_deg,
                     pushrod_m=pushrod,
                     advantage=float(lk["advantage"]),
                     motor_x_aft=motor_x_aft)


def main():
    doc, derived = build()
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {OUT}")
    for k, v in derived.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
