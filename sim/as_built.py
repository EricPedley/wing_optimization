"""Generate sim/as_built.json: the as-built aircraft description consumed by
the SimITL tailsitter physics (tools/simitl-wing/wingcfg.h) and by the JAX
reference model (sim/wing_model.py).

Sources:
  * airfoil/optimum.json          - optimizer design vector: the airfoil body
                                    (planform, thickness, camber, sweep)
  * Onshape `Final Assembly` mates - the elevons and props as built, which do
                                    NOT follow the optimizer.  Cached in
                                    sim/assets/onshape_asbuilt.json together
                                    with the render meshes; refresh with
                                    `--fetch`.
  * airfoil/airfoil_model.py      - geometry/camber/inertia helpers
  * airfoil/linkage_coupling.py   - servo linkage throw/advantage

CAD frame: Onshape part-studio coordinates of the `Flying Wing` studio, metres,
x forward, y left, z up, origin at the root leading edge.  Only the right half
(y < 0) is modelled; the left half is its mirror.  Aero frame here: x forward
from the CG, y right, z down, so aero = (cgX + x, -y, -z).

Mass: measured all-up weight 80 g (Onshape mass properties are NOT used --
they were never calibrated, see FLIGHT_SIM_PLAN).  Inertia uses the model's
distribution scaled by measured/model mass.  Thrust is exposed via
thrustScale because static thrust was not measured.

JSON shape: no arrays -- the C++ tools/simitl-wing/wingcfg.h parser only
understands objects/numbers/strings, so vectors are {x,y,z} objects and
two-element lists become named fields.

Run:  uv run python -m sim.as_built            # from cached CAD
      uv run python -m sim.as_built --fetch    # refresh CAD from Onshape
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
ASSETS = os.path.join(ROOT, "sim", "assets")
CAD_CACHE = os.path.join(ASSETS, "onshape_asbuilt.json")

# Onshape elements in the airfoil.onshape document
ASSEMBLY_ID = "8e364e94f22d49c4a15a2f6a"      # Final Assembly
PART_STUDIO_ID = "b893633847e2c346141c4f25"   # Flying Wing
ELEVON_PART = "control surface"


# --- Onshape extraction ---------------------------------------------------------


def _mate_frames(asm):
    """Revolute mates of the assembly as {name: [(occurrence names, cs)]}.

    Each mated entity's coordinate system is in its own part's frame.  For the
    parts used here that frame *is* the part-studio frame the STLs are in, so
    no occurrence transform is applied -- the elevon mesh and its hinge must
    share a frame, and the assembly solve nudges the occurrence by ~0.3 mm.
    """
    ra = asm["rootAssembly"]
    names = {i["id"]: i["name"] for i in ra["instances"]}
    mates = {}
    for f in ra["features"]:
        fd = f["featureData"]
        if f["featureType"] != "mate" or fd.get("mateType") != "REVOLUTE":
            continue
        mates[fd["name"]] = [
            ([names[x] for x in e["matedOccurrence"]], e["matedCS"])
            for e in fd["matedEntities"]]
    return mates


def _entity(mate, name_prefix):
    for occ, cs in mate:
        if any(n.startswith(name_prefix) for n in occ):
            return cs
    raise KeyError(f"no mated entity on '{name_prefix}'")


def fetch_cad():
    """Pull hinge/prop mates and render meshes from Onshape into sim/assets."""
    import base64
    import urllib.parse
    import urllib.request

    from airfoil import onshape as osh

    doc, ws = osh.DOCUMENT_ID, osh.WORKSPACE_ID
    asm = osh._request(
        "GET", f"/assemblies/d/{doc}/w/{ws}/e/{ASSEMBLY_ID}",
        query={"includeMateFeatures": "true"})
    mates = _mate_frames(asm)

    hinge = prop = None
    for mate in mates.values():
        occ_names = [n for occ, _ in mate for n in occ]
        if any(n.startswith(ELEVON_PART) for n in occ_names):
            hinge = _entity(mate, ELEVON_PART)
        elif any(n.startswith("motor mount") for n in occ_names):
            prop = _entity(mate, "motor mount")
    if hinge is None or prop is None:
        raise RuntimeError(f"expected elevon and prop revolute mates, got "
                           f"{list(mates)}")

    parts = osh._request("GET", f"/parts/d/{doc}/w/{ws}/e/{PART_STUDIO_ID}")
    by_name = {p["name"]: p["partId"] for p in parts}
    elevon_id = by_name[ELEVON_PART]
    body_ids = [pid for n, pid in by_name.items() if n != ELEVON_PART]
    bb = osh._request(
        "GET", f"/parts/d/{doc}/w/{ws}/e/{PART_STUDIO_ID}/partid/{elevon_id}"
        "/boundingboxes")

    access, secret = osh.credentials()
    token = base64.b64encode(f"{access}:{secret}".encode()).decode()

    def stl(part_ids, out_name):
        query = urllib.parse.urlencode({
            "partIds": ",".join(part_ids), "units": "meter",
            "mode": "binary", "grouping": "true"})
        url = (f"{osh.BASE_URL}/api/{osh.API_VERSION}/partstudios/d/{doc}"
               f"/w/{ws}/e/{PART_STUDIO_ID}/stl?{query}")
        req = urllib.request.Request(url, headers={
            "Authorization": f"Basic {token}", "Accept": "*/*"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        with open(os.path.join(ASSETS, out_name), "wb") as f:
            f.write(data)

    os.makedirs(ASSETS, exist_ok=True)
    stl(body_ids, "body_right.stl")
    stl([elevon_id], "elevon_right.stl")

    cache = {
        "_source": f"onshape d/{doc} w/{ws} assembly {ASSEMBLY_ID}",
        "frame": "Flying Wing part studio: m, x fwd, y left, z up, "
                 "origin root LE; right half only",
        "hinge": {"origin": hinge["origin"], "axis": hinge["zAxis"]},
        "prop": {"origin": prop["origin"], "axis": prop["zAxis"]},
        "elevonBox": {k: bb[k] for k in
                      ("lowX", "highX", "lowY", "highY", "lowZ", "highZ")},
        "meshes": {"body": "body_right.stl", "elevon": "elevon_right.stl"},
    }
    with open(CAD_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"wrote {CAD_CACHE} and meshes in {ASSETS}")


def load_cad():
    with open(CAD_CACHE) as f:
        return json.load(f)


def cad_to_aero(v, cg):
    """CAD point (x fwd, y left, z up, root-LE origin) -> aero frame."""
    return np.array([cg + v[0], -v[1], -v[2]])


def hinge_aero(cad, cg):
    """Right/left hinge point and axis in the aero frame.

    The right axis points outboard (+y); with the right-hand rule a positive
    rotation then moves its trailing edge down, the physics' positive
    deflection.  The left axis is the mirror of the right one as an axial
    vector (reflect y, then negate), so it also points to the aircraft's
    right -- inboard on the left wing -- and +deg is TE down there too.
    """
    pr = cad_to_aero(cad["hinge"]["origin"], cg)
    ar = np.array(cad["hinge"]["axis"], dtype=float)
    ar = np.array([ar[0], -ar[1], -ar[2]])          # direction, aero frame
    if ar[1] < 0:                                     # outboard on the right
        ar = -ar
    ar /= np.linalg.norm(ar)
    pl = pr * np.array([1, -1, 1])
    al = -(ar * np.array([1, -1, 1]))
    return pr, ar, pl, al


def hinge_chord_fraction(cad, g, y_abs):
    """Chord fraction of the as-built hinge line at spanwise station y_abs."""
    o = np.array(cad["hinge"]["origin"], dtype=float)
    d = np.array(cad["hinge"]["axis"], dtype=float)
    t = (-y_abs - o[1]) / d[1]                        # right half: y_cad = -y
    station_aft = -(o[0] + t * d[0])
    semi = 0.5 * g["span"]
    le = float(am.leading_edge_x(g["span"], y_abs / semi, g["le_sweep_deg"]))
    c, _ = am.local_geometry(g["span"], g["root_chord"], g["tip_chord"],
                             g["root_thickness"], g["tip_thickness"],
                             y_abs / semi)
    return (station_aft - le) / float(c)


def build():
    with open(OPTIMUM) as f:
        opt = json.load(f)
    p = np.array(opt["x"], dtype=float)
    g = am.unpack(p)
    cad = load_cad()

    semi = 0.5 * g["span"]
    cg = float(am.cg_station(
        g["span"], g["root_chord"], g["tip_chord"], g["root_thickness"],
        g["tip_thickness"], g["le_sweep_deg"], g["battery_station"],
        g["servo_chord_frac"], g["servo_span_frac"], g["motor_standoff"],
        g["motor_y"]))

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

    # --- motor / prop disk position (as built: Onshape prop mate) -----------
    # The mate sits on the motor shaft at the prop; the optimizer's motor_y
    # (35.8 mm) is not what was built.
    prop_cad = np.array(cad["prop"]["origin"], dtype=float)
    motor_y = float(-prop_cad[1])
    motor_x_aft = float(-prop_cad[0])
    motor_z = float(-prop_cad[2])

    # --- elevon + hinge line (as built: Onshape hinge mate, elevon part) ----
    box = cad["elevonBox"]
    elevon_in = float(-box["highY"])
    elevon_out = float(-box["lowY"])
    # the physics takes one hinge chord fraction; use its mean over the span
    ys = np.linspace(elevon_in, elevon_out, 41)
    x_hinge = float(np.mean([hinge_chord_fraction(cad, g, y) for y in ys]))
    hinge_pt_r, axis_r, hinge_pt_l, axis_l = hinge_aero(cad, cg)

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
        # quarter-chord sweep for sweep theory in the strip aero (sideslip)
        "quarterChordSweepDeg": float(am.quarter_chord_sweep_deg(
            g["span"], g["root_chord"], g["tip_chord"], g["le_sweep_deg"])),
        # elevon hinge sweep from the Onshape hinge mate (flap effectiveness)
        "hingeSweepDeg": float(np.degrees(np.arctan2(-axis_r[0], axis_r[1]))),
        "thicknessRatioRoot": g["root_thickness"] / g["root_chord"],
        "thicknessRatioTip": g["tip_thickness"] / g["tip_chord"],
        "camber": {
            "a1Root": g["camber_a1"], "a2Root": g["camber_a2"],
            "a1Tip": g["tip_camber_a1"], "a2Tip": g["tip_camber_a2"],
        },
        "xHinge": x_hinge,
        "elevonInboardY": elevon_in,
        "elevonOutboardY": elevon_out,
        "elevonMaxDeg": elevon_max_deg,
        "hinge": {
            "pointR": {"x": float(hinge_pt_r[0]), "y": float(hinge_pt_r[1]),
                       "z": float(hinge_pt_r[2])},
            "axisR":  {"x": float(axis_r[0]), "y": float(axis_r[1]),
                       "z": float(axis_r[2])},
            "pointL": {"x": hinge_pt_l[0], "y": hinge_pt_l[1],
                       "z": hinge_pt_l[2]},
            "axisL":  {"x": float(axis_l[0]), "y": float(axis_l[1]),
                       "z": float(axis_l[2])},
        },
        "motorY": motor_y,
        "motorXAft": motor_x_aft,
        "motorZ": motor_z,
        "motorSpinDir": {"right": 1, "left": -1},
        "motorIndex": {"right": 1, "left": 0},
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

        # Output mapping.  Which physical servo/motor sits on which BF output
        # is not in the CLI dump, so it is fixed by two convention-free
        # requirements, checked by sim/test_control_signs.py:
        #   * every axis is negative feedback (an injected body rate makes BF
        #     command a moment that opposes it), and
        #   * stick right/forward/right gives roll right/nose down/nose right.
        # smix: servo output 0 = -roll -pitch, output 1 = -roll +pitch; the
        # right elevon is on output 1 and the left on output 0 (TE-down +).
        # mmix: motor 0 = thr + yaw, motor 1 = thr - yaw; BF motor 1 (index 0)
        # is the left prop.
        "servo": {
            "indexL": 0, "indexR": 1,
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
                     motor_x_aft=motor_x_aft, motor_y=motor_y,
                     x_hinge=x_hinge, elevon_in=elevon_in,
                     elevon_out=elevon_out)


def main():
    if "--fetch" in sys.argv[1:]:
        fetch_cad()
    doc, derived = build()
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {OUT}")
    for k, v in derived.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
