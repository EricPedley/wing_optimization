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
    p.downwash = downwash_matrix(p, rows)
    return rows


def _segment_w(px, py, ax, ay, bx, by):
    """z-velocity at (px, py) from a unit vortex segment a->b in the z=0
    plane (Biot-Savart)."""
    r0x, r0y = bx - ax, by - ay
    r1x, r1y = px - ax, py - ay
    r2x, r2y = px - bx, py - by
    cr = r1x * r2y - r1y * r2x
    n1, n2 = np.hypot(r1x, r1y), np.hypot(r2x, r2y)
    if cr * cr < 1e-12 or n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    dot = r0x * (r1x / n1 - r2x / n2) + r0y * (r1y / n1 - r2y / n2)
    return cr / (cr * cr) * dot / (4.0 * np.pi)


def downwash_matrix(p, strips):
    """Discrete Prandtl lifting line: K[i][j] = downwash (aero +z, m/s) on
    the lifting line at strip i per unit circulation (m^2/s) on strip j's
    horseshoe vortex (bound along the span, trailing legs streaming aft).
    Evaluated on the bound line, so only the trailing legs contribute (the
    collinear bound segments induce nothing there) -- the 2D section curve
    already supplies the bound vortex's own effect.

    The vortices are laid out on the *unswept* planform (same span and
    chords, bound vortices on one straight line): the sections already apply
    sweep theory, so sweeping the vortices too would count sweep twice.  This
    supplies only the finite-span (low aspect ratio) downwash.  Same as
    TailsitterPhysics::buildDownwash."""
    n = len(strips)
    far = 1000.0                                  # m: trailing legs end here
    K = np.zeros((n, n))
    for i, si in enumerate(strips):
        px, py = 0.0, si["y"]                     # on the lifting line
        for j, sj in enumerate(strips):
            ay, by = sj["y"] - 0.5 * sj["dy"], sj["y"] + 0.5 * sj["dy"]
            K[i, j] = (_segment_w(px, py, -far, ay, 0.0, ay)
                       + _segment_w(px, py, 0.0, ay, 0.0, by)
                       + _segment_w(px, py, 0.0, by, -far, by))
    # positive circulation (lift) must wash its own control point down
    if np.trace(K) < 0:
        K = -K
    return K


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
    # no Cl^2/(pi A e) term: induced drag comes from the lifting line
    # (each strip's lift is tilted back by its induced angle)
    if a_abs <= a_s:
        cl = slope * alpha_eff
        cd = p.cd0
    else:
        cl_s = cl_max
        cd_s = p.cd0
        # Viterna-Corrigan: flat-plate 1/2 Cdmax sin 2a plus a term that
        # matches cl_s at the stall angle and decays (cl used to be
        # cl_s sin 2a / sin 2a_s, peaking ~3x cl_s at 45 deg)
        a2 = (cl_s - p.cdmax * np.sin(a_s) * np.cos(a_s)) * np.sin(a_s) / np.cos(a_s) ** 2
        cl = sgn * (0.5 * p.cdmax * np.sin(2.0 * a_abs)
                    + a2 * np.cos(a_abs) ** 2 / max(np.sin(a_abs), 1e-6))
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
    """Body-force/moment in the aerospace frame.  thrust_rl is (right,left).
    Same algorithm as TailsitterPhysics::computeForces (see comments there)."""
    disk_area = 0.25 * np.pi * p.prop_diameter ** 2
    half_wash = 0.5 * p.slipstream_width * p.prop_diameter
    elevon = (elevon_l_deg, elevon_r_deg)
    n = len(strips)

    # --- pass 1: per-strip flow and section parameters ----------------------
    st = []
    for s in strips:
        r = np.array([p.cgX - (s["leX"] + 0.25 * s["chord"]), s["y"], 0.0])
        v = vel_aero + np.cross(omega_aero, r)
        ay = abs(s["y"])
        if abs(ay - p.motor_y) <= half_wash:
            mi = 0 if s["y"] > 0 else 1     # motor 0 = right (y>0)
            w = induced_velocity(thrust_rl[mi], v[0], disk_area)
            v[0] += p.slipstream_dev * 2.0 * w

        lam = p.sweep_c4_deg * DEG2RAD if ay > p.root_half_width else 0.0
        side = 1.0 if s["y"] > 0 else -1.0
        c_hat = np.array([np.cos(lam), side * np.sin(lam), 0.0])
        s_hat = np.array([-np.sin(lam), side * np.cos(lam), 0.0])
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
        re = RHO * np.sqrt(v2) * s["chord"] / MU
        st.append(dict(c_hat=c_hat, s_hat=s_hat, vs=vs, v2=v2, vn=np.sqrt(v2),
                       reversed=reversed_flow, alpha=alpha,
                       shift=flap_lift_slope * delta_rad / (2.0 * np.pi) - alpha_l0,
                       flap_m=flap_moment_slope * delta_rad,
                       cl_max=cl_max_for(p, s, re)))

    # --- pass 2: lifting line --------------------------------------------------
    # One linear solve for the circulation: gamma_i = 1/2 c_i 2pi (vfl_i
    # alpha_c,i - (K gamma)_i), with each strip's angle clamped to its stall
    # angle.  Exact lifting line in attached flow; in stall the downwash is
    # that of stall-level lift (a nonlinear post-stall lifting line has
    # multiple solutions and Newton wanders between them).  Reversed-flow
    # strips carry no circulation.  vfl = max(Vn, 0.5).
    vfl = np.array([max(q["vn"], 0.5) for q in st])
    A = np.eye(n)
    rhs = np.zeros(n)
    for i, (s, q) in enumerate(zip(strips, st)):
        if q["reversed"]:
            continue
        a_s = q["cl_max"] / (2.0 * np.pi)
        a_c = float(np.clip(q["alpha"] + q["shift"], -a_s, a_s))
        g = 0.5 * s["chord"] * 2.0 * np.pi
        A[i] += g * p.downwash[i]
        rhs[i] = g * vfl[i] * a_c
    gamma = np.linalg.solve(A, rhs)
    a_ind = np.array([0.0 if q["reversed"] else w / vfl[i]
                      for i, (q, w) in enumerate(zip(st, p.downwash @ gamma))])

    # --- pass 3: forces, lift perpendicular to the downwashed inflow --------
    F = np.zeros(3)
    M = np.zeros(3)
    z_hat = np.array([0.0, 0.0, 1.0])
    for i, (s, q) in enumerate(zip(strips, st)):
        a_flow = q["alpha"] - a_ind[i]
        cl, cd = section_cl_cd_cm(p, q["cl_max"], a_flow + q["shift"])
        cm = 0.0 if q["reversed"] else 0.25 * np.pi * (s["a2"] - s["a1"]) + q["flap_m"]
        qd = 0.5 * RHO * q["v2"]
        dA = s["chord"] * s["dy"]
        sa, ca = np.sin(a_flow), np.cos(a_flow)
        xs = -1.0 if q["reversed"] else 1.0
        # lift is perpendicular to the relative wind (-cos a, -sin a in the
        # section plane): (+sin a, -cos a), mirrored for reversed flow.  (The
        # model used to have -sin a here, tilting lift back by 2 alpha.)
        f_dir = q["c_hat"] * (sa * xs) + z_hat * (-ca)
        d_dir = q["c_hat"] * (-ca * xs) + z_hat * (-sa)
        f_span = q["s_hat"] * (-0.5 * RHO * abs(q["vs"]) * q["vs"] * 0.5 * p.cd0 * dA)
        f = f_dir * (qd * dA * cl) + d_dir * (qd * dA * cd) + f_span
        xa = 0.75 if q["reversed"] else 0.25
        rf = np.array([p.cgX - (s["leX"] + xa * s["chord"]), s["y"], 0.0])
        F += f
        M += np.cross(rf, f)
        M[1] += qd * s["chord"] ** 2 * s["dy"] * cm

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
