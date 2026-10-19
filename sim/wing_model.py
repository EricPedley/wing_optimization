"""Python reference implementation of the TailsitterPhysics aero model.

Mirrors SimITL/src/sim/tailsitter.cpp equation-for-equation so the
`test_wing_equiv` equivalence test can diff the two implementations on a
state grid.  Everything runs in the aerospace frame (x fwd, y right, z
down); the C++<->sim-body axis remap is exercised separately in the C++
`forces` mode which accepts aero-frame inputs directly.

Run:  uv run python -m sim.wing_model   (prints a small sanity table)
"""

import json
import os

import numpy as np

RHO = 1.225
MU = 1.81e-5
G = 9.81
DEG2RAD = np.pi / 180.0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS_BUILT = os.path.join(ROOT, "sim", "as_built.json")


class Params:
    def __init__(self, doc):
        self.mass = doc["mass"]
        self.cgX = doc["cgX"]
        I = doc["inertia"]
        self.inertia = np.array(
            [[I["xx"], -I["xy"], -I["xz"]],
             [-I["xy"], I["yy"], -I["yz"]],
             [-I["xz"], -I["yz"], I["zz"]]])
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.span = doc["span"]
        self.root_chord = doc["rootChord"]
        self.tip_chord = doc["tipChord"]
        self.root_half_width = doc["rootHalfWidth"]
        self.le_sweep_deg = doc["leSweepDeg"]
        self.sweep_c4_deg = doc.get("quarterChordSweepDeg", 0.0)
        self.hinge_sweep_deg = doc.get("hingeSweepDeg", 0.0)
        self.tc_root = doc["thicknessRatioRoot"]
        self.tc_tip = doc["thicknessRatioTip"]

        c = doc["camber"]
        self.a1_root, self.a2_root = c["a1Root"], c["a2Root"]
        self.a1_tip, self.a2_tip = c["a1Tip"], c["a2Tip"]

        self.x_hinge = doc["xHinge"]
        self.elevon_in = doc["elevonInboardY"]
        self.elevon_out = doc["elevonOutboardY"]
        self.elevon_max_deg = doc["elevonMaxDeg"]

        self.motor_y = doc["motorY"]
        self.motor_x_aft = doc["motorXAft"]
        self.motor_z = doc["motorZ"]
        self.spin_dir = np.array([doc["motorSpinDir"]["right"],
                                  doc["motorSpinDir"]["left"]])
        self.thrust_scale = doc["thrustScale"]
        self.prop_diameter = doc["propDiameter"]

        # defaults mirror packets.h TailsitterInit
        a = doc.get("aero", {})
        self.cl_max_base = a.get("clMaxBase", 0.80)
        self.cl_max_camber_gain = a.get("clMaxCamberGain", 8.0)
        self.cl_max_tc_gain = a.get("clMaxTcGain", 1.5)
        self.cl_max_tc_ref = a.get("clMaxTcRef", 0.085)
        self.cl_max_re_gain = a.get("clMaxReGain", 0.12)
        self.cl_max_re_ref = a.get("clMaxReRef", 40000.0)
        self.cl_max_ceiling = a.get("clMaxCeiling", 1.35)
        self.flap_effectiveness = a.get("flapEffectiveness", 0.65)
        self.flap_stall_deg = a.get("flapStallDeg", 15.0)
        self.flap_sat_deg = a.get("flapSaturationDeg", 30.0)
        self.slipstream_dev = a.get("slipstreamDevelopment", 0.5)
        self.slipstream_width = a.get("slipstreamWidthFactor", 1.0)
        self.cd0 = a.get("cd0", 0.02)
        self.cd_induced = a.get("cdInduced", 0.0)
        self.cdmax_viterna = a.get("cdmaxViterna", 0.0)
        self.num_strips = int(a.get("numStrips", 33))

        # derived
        semi = 0.5 * self.span
        f0 = np.clip(self.root_half_width / semi, 0.0, 1.0)
        area = 2.0 * (self.root_chord * f0 * semi
                      + 0.5 * (self.root_chord + self.tip_chord)
                      * (semi - f0 * semi))
        self.wing_area = area
        self.aspect = self.span ** 2 / area
        self.cdmax = self.cdmax_viterna or min(1.11 + 0.018 * self.aspect, 2.0)


def build_strips(p):
    """Same strip layout as TailsitterPhysics::buildStrips."""
    n = max(3, p.num_strips | 1)
    semi = 0.5 * p.span
    f0 = np.clip(p.root_half_width / semi, 0.0, 1.0)
    sweep = np.tan(p.le_sweep_deg * DEG2RAD)
    dy = p.span / n
    rows = []
    for i in range(n):
        y = -semi + (i + 0.5) * dy
        f = abs(y) / semi
        t = np.clip((f - f0) / max(1.0 - f0, 1e-9), 0.0, 1.0)
        chord = p.root_chord + (p.tip_chord - p.root_chord) * t
        tc = p.tc_root + (p.tc_tip - p.tc_root) * t
        le_x = sweep * max(abs(y) - p.root_half_width, 0.0)
        a1 = p.a1_root + (p.a1_tip - p.a1_root) * t
        a2 = p.a2_root + (p.a2_tip - p.a2_root) * t
        ay = abs(y)
        flap_side = -1
        if p.elevon_in <= ay <= p.elevon_out:
            flap_side = 0 if y < 0 else 1     # 0 = left (y<0), 1 = right
        rows.append(dict(y=y, dy=dy, chord=chord, tc=tc, leX=le_x,
                         a1=a1, a2=a2, flap_side=flap_side))
    return rows


def camber_max(a1, a2):
    """41-point scan of the camber line, same as clMaxFor."""
    x = np.linspace(0.0, 1.0, 41)
    th = np.arccos(np.clip(1.0 - 2.0 * x, -1.0 + 1e-9, 1.0 - 1e-9))
    ct = np.cos(th)
    z = (a1 * np.sin(th) ** 2 / 4.0
         + a2 * (-ct ** 3 / 3.0 + ct / 2.0 - 1.0 / 6.0)
         + a2 * x / 3.0)
    return np.max(np.abs(z))


def cl_max_for(p, strip, re):
    cl = (p.cl_max_base
          + p.cl_max_camber_gain * camber_max(strip["a1"], strip["a2"])
          + p.cl_max_tc_gain * (strip["tc"] - p.cl_max_tc_ref)
          + p.cl_max_re_gain * np.log(max(re, 1e3) / p.cl_max_re_ref))
    return float(np.clip(cl, 0.2, p.cl_max_ceiling))


def section_cl_cd_cm(p, cl_max, alpha_eff):
    """Attached-slope + flat-plate post-stall model (sectionClCdCm)."""
    slope = 2.0 * np.pi
    a_s = cl_max / slope
    sgn = 1.0 if alpha_eff >= 0.0 else -1.0
    a_abs = abs(alpha_eff)
    if a_abs <= a_s:
        cl = slope * alpha_eff
        k_ind = p.cd_induced or 1.0 / (np.pi * p.aspect * 0.85)
        cd = p.cd0 + k_ind * cl * cl
    else:
        cl_s = cl_max
        cd_s = p.cd0 + 1.0 / (np.pi * p.aspect * 0.85) * cl_s * cl_s
        cl = sgn * cl_s * np.sin(2.0 * a_abs) / np.sin(2.0 * a_s)
        s2 = np.sin(a_abs) ** 2
        s2s = np.sin(a_s) ** 2
        cd = cd_s + (p.cdmax - cd_s) * (s2 - s2s) / max(1.0 - s2s, 1e-6)
        cd = max(cd, 0.0)
    return cl, cd


def induced_velocity(thrust_n, v_inf, disk_area):
    half_v = 0.5 * v_inf
    return -half_v + np.sqrt(half_v ** 2 + max(thrust_n, 0.0)
                             / (2.0 * RHO * disk_area))


def forces_moments(p, strips, vel_aero, omega_aero, elevon_l_deg,
                   elevon_r_deg, thrust_rl):
    """Body-force/moment in the aerospace frame.  thrust_rl is (right,left)."""
    disk_area = 0.25 * np.pi * p.prop_diameter ** 2
    half_wash = 0.5 * p.slipstream_width * p.prop_diameter
    elevon = (elevon_l_deg, elevon_r_deg)

    F = np.zeros(3)
    M = np.zeros(3)
    for s in strips:
        r = np.array([p.cgX - (s["leX"] + 0.25 * s["chord"]), s["y"], 0.0])
        v = vel_aero + np.cross(omega_aero, r)

        ay = abs(s["y"])
        if abs(ay - p.motor_y) <= half_wash:
            mi = 0 if s["y"] > 0 else 1     # motor 0 = right (y>0)
            w = induced_velocity(thrust_rl[mi], v[0], disk_area)
            v[0] += p.slipstream_dev * 2.0 * w

        # simple sweep theory + reversed flow: same as
        # TailsitterPhysics::computeForces (see the comments there)
        lam = p.sweep_c4_deg * DEG2RAD if ay > p.root_half_width else 0.0
        side = 1.0 if s["y"] > 0 else -1.0
        c_hat = np.array([np.cos(lam), side * np.sin(lam), 0.0])
        s_hat = np.array([-np.sin(lam), side * np.cos(lam), 0.0])
        z_hat = np.array([0.0, 0.0, 1.0])
        un, wn, vs = float(v @ c_hat), float(v[2]), float(v @ s_hat)
        v2 = un * un + wn * wn
        reversed_flow = un < 0.0
        alpha = np.arctan2(wn, -un) if reversed_flow else np.arctan2(wn, un)

        delta_rad = 0.0
        flap_lift_slope = 0.0
        flap_moment_slope = 0.0
        if s["flap_side"] >= 0:
            deg = elevon[s["flap_side"]]
            excess = max(abs(deg) - p.flap_stall_deg, 0.0)
            eff = p.flap_effectiveness * np.exp(
                -excess / (p.flap_sat_deg - p.flap_stall_deg))
            delta_rad = deg * DEG2RAD
            if not reversed_flow:
                th = np.arccos(np.clip(1.0 - 2.0 * p.x_hinge, -1.0 + 1e-9, 1.0 - 1e-9))
                flap_lift_slope = 2.0 * (np.pi - th + np.sin(th)) * eff
                flap_moment_slope = 0.25 * (np.sin(2.0 * th) - 2.0 * np.sin(th)) * eff
            else:
                th = np.arccos(np.clip(1.0 - 2.0 * (1.0 - p.x_hinge), -1.0 + 1e-9, 1.0 - 1e-9))
                flap_lift_slope = -2.0 * (th - np.sin(th)) * eff

        if s["flap_side"] >= 0 and lam > 0.0:
            ch = np.cos(p.hinge_sweep_deg * DEG2RAD) / np.cos(lam)
            flap_lift_slope *= ch * ch
            flap_moment_slope *= ch * ch

        alpha_l0 = 0.0 if reversed_flow else -0.5 * s["a1"]
        alpha_eff = alpha - alpha_l0 + flap_lift_slope * delta_rad / (2.0 * np.pi)

        re = RHO * np.sqrt(v2) * s["chord"] / MU
        cl_max = cl_max_for(p, s, re)
        cl, cd = section_cl_cd_cm(p, cl_max, alpha_eff)
        cm = 0.0 if reversed_flow else (
            0.25 * np.pi * (s["a2"] - s["a1"]) + flap_moment_slope * delta_rad)

        q = 0.5 * RHO * v2
        dA = s["chord"] * s["dy"]
        sa, ca = np.sin(alpha), np.cos(alpha)
        xs = -1.0 if reversed_flow else 1.0
        f_dir = c_hat * (-sa * xs) + z_hat * (-ca)
        d_dir = c_hat * (-ca * xs) + z_hat * (-sa)
        f_span = s_hat * (-0.5 * RHO * abs(vs) * vs * 0.5 * p.cd0 * dA)
        f = f_dir * (q * dA * cl) + d_dir * (q * dA * cd) + f_span
        xa = 0.75 if reversed_flow else 0.25
        rf = np.array([p.cgX - (s["leX"] + xa * s["chord"]), s["y"], 0.0])
        F += f
        M += np.cross(rf, f)
        M[1] += q * s["chord"] ** 2 * s["dy"] * cm

    # motor thrust along +x and its moment arms
    motor_x_aero = p.cgX - p.motor_x_aft
    arms = [np.array([motor_x_aero, p.motor_y, p.motor_z]),
            np.array([motor_x_aero, -p.motor_y, p.motor_z])]
    for i in range(2):
        ft = np.array([thrust_rl[i], 0.0, 0.0])
        F += ft
        M += np.cross(arms[i], ft)

    return F, M


def load_as_built(path=AS_BUILT):
    with open(path) as f:
        doc = json.load(f)
    p = Params(doc)
    return p, build_strips(p)


def main():
    p, strips = load_as_built()
    print(f"span {p.span:.3f} m  area {p.wing_area:.4f} m^2  "
          f"AR {p.aspect:.2f}  mass {p.mass*1000:.0f} g")
    print("alpha_deg   Cl_eff      Fz (N)   Mx")
    for a_deg in (-150, -90, -30, -5, 0, 5, 15, 45, 90, 150):
        a = a_deg * DEG2RAD
        v = 6.0 * np.array([np.cos(a), 0.0, np.sin(a)])
        F, M = forces_moments(p, strips, v, np.zeros(3), 0.0, 0.0,
                              (0.0, 0.0))
        print(f"{a_deg:8d}  {float(-F[2]) / (0.5*RHO*36*p.wing_area):8.3f}"
              f"  {F[2]:8.3f}  {M[0]:8.4f}")


if __name__ == "__main__":
    main()
