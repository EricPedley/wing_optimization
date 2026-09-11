"""Generate hq-51mm-micro.glb -- a small FPV quadcopter model for pr0p modding.

Conventions:
  - Units: millimeters (the GLB is exported with mm-scale geometry; scale to
    meters at import if needed, i.e. 0.001).
  - Y up, -Z forward (Unity convention). Camera faces -Z.
  - Motor centers at (+-42.5, ~5, +-28) mm matching
    ~/.config/unity3d/sigsegowl/pr0p/config/quad/hq-51mm-micro.json
    (squashed-X: ~102mm diagonal motor-to-motor).

The propeller is extracted from pr0p's Unity assets (vtx-slayer-3
"Propeller_Baked" mesh in sharedassets0.assets): one blade sector plus the
hub is kept, mirrored 180 degrees for a 2-blade prop, and scaled to 51mm.
Requires UnityPy (`uv run --with UnityPy --with trimesh python
generate_micro_glb.py`). If the assets or UnityPy are unavailable, a
procedural 2-blade prop is generated instead.

Run from anywhere:
    uv run --with trimesh --with UnityPy python model/generate_micro_glb.py
"""

import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_GLB = os.path.join(HERE, "hq-51mm-micro.glb")
PR0P_DATA = os.path.expanduser("~/programs/pr0p/pr0p_Data")

# --- Dimensions (mm) ---------------------------------------------------------
# Center plates verbatim from multirotor/frame_scaling.py: 24mm (x, width) x
# 90mm (z, length) x 2.5mm thick, two plates sandwiching 4 standoffs on a
# ~20x60mm footprint. Yes, 90mm is long for a 51mm-prop quad -- it models a
# fixed-size electronics stack; overhang past the props is intended.
PLATE_W = 24.0        # x width
PLATE_L = 90.0        # z length
PLATE_T = 2.5         # PLATE_THICKNESS_M
STANDOFF_D = 5.0
STANDOFF_H = 15.0
STANDOFF_X = 10.0     # standoffs at (+-10, *, +-30): ~20x60mm footprint
STANDOFF_Z = 30.0
ARM_W = 12.0
ARM_T = 2.5
# Squashed-X layout: motors at (+-42.5, +-28). The 24mm-wide plate only
# constrains x -- 42.5 puts each motor exactly arm_length_m() = prop_radius
# + PROP_TIP_CLEARANCE_M = 30.5mm out from the plate edge. In z the only
# limit is prop-prop clearance: +-28 gives 56mm front-back separation, i.e.
# 5mm between 51mm prop disks. Arm reach drops from 60.1mm (true-X at
# +-42.5/+-42.5) to ~50.9mm.
MOTOR_X = 42.5
MOTOR_Z = 28.0
PROP_DIAM = 51.0

# y levels: bottom plate centered at y=0 (spans -1.25..1.25); arms share the
# bottom-plate plane. Top plate spans 16.25..18.75.
ARM_Y = 0.0
MOTOR_BASE_Y = 1.25   # motor mounting flange sits on top of the arm
BELL_TOP_Y = MOTOR_BASE_Y + 1.5 + 3.0 + 6.5  # flange + stator + bell
PROP_Y = BELL_TOP_Y + 0.2  # target height of the prop's LOWEST vertex:
                           # sits almost flush on the bell, ~0.2mm shaft


def box(sx, sy, sz, center=(0, 0, 0)):
    m = trimesh.creation.box((sx, sy, sz))
    m.apply_translation(center)
    return m


def cyl(r, h, center=(0, 0, 0), sections=24):
    """Cylinder with its axis along +Y (trimesh builds along Z)."""
    m = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    m.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
    m.apply_translation(center)
    return m


def make_arm(motor_xy):
    """Flat arm along the direction from origin to motor_xy, in the XZ plane."""
    mx, mz = motor_xy
    dist = float(np.hypot(mx, mz))
    # run from slightly inside the center plate out past the motor center
    length = dist + 8.0
    start = -4.0
    arm = box(ARM_W, ARM_T, length, center=(0, ARM_Y, start + length / 2.0))
    angle = np.degrees(np.arctan2(mx, mz))  # rotate +Z toward motor direction
    arm.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
    )
    return arm


def make_motor():
    """1203 brushless per the mechanical drawing: 14.05mm total height,
    O1.5 shaft protruding 3mm below the base, a plus-shaped O9 mount
    carrying the 4xM2 holes on its arms with a small center boss, a
    barely-visible O12 stator, and an O15.5 bell (the widest part)
    that wraps over the stator and opens into a 5-spoke star around a
    O5 center hub.

    Stack (bottom to top, heights in mm): shaft 3.0 below base, cross
    1.5, boss ~0.9, stator sliver, bell skirt to +11.0 -> ~14.0 total
    ~ 14.05.
    """
    y = MOTOR_BASE_Y
    parts = [cyl(0.75, 3.0, (0, y - 1.5, 0), sections=12)]  # shaft below
    # Base mount is not a round flange: a plus-shaped cross spanning O9
    # with the 4xM2 holes out on the cross arms, a small center boss on
    # top, then a sliver of clearance before the stator.
    cross = trimesh.util.concatenate([
        box(9.6, 1.5, 3.6, (0, y + 0.75, 0)),
        box(3.6, 1.5, 9.6, (0, y + 0.75, 0))])
    cross.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0]))
    parts.append(cross)
    for sx, sz in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        parts.append(cyl(1.0, 0.4, (sx * 3.2, y + 1.5 + 0.2, sz * 3.2),
                         sections=10))
    parts.append(cyl(2.5, 0.9, (0, y + 1.5 + 0.45, 0)))      # center boss
    stator_y0 = y + 2.5                                       # ~0.1 clearance
    parts.append(cyl(6.0, 2.0, (0, stator_y0 + 1.0, 0)))     # stator
    # Bell (O15.5 annulus) wraps down over the stator so only a thin
    # sliver of stator shows beneath it, like the drawing.
    bell_y0 = stator_y0 + 1.2
    parts.append(trimesh.creation.annulus(
        r_min=6.9, r_max=7.75, height=11.0 - (bell_y0 - y)))
    parts[-1].apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
    parts[-1].apply_translation((0, (bell_y0 + y + 11.0) / 2.0, 0))
    spoke_y = y + 10.4
    for k in range(5):
        a = np.radians(72 * k)
        spoke = box(12.0, 1.2, 2.4, (0, spoke_y, 0))
        spoke.apply_transform(
            trimesh.transformations.rotation_matrix(a, [0, 1, 0]))
        parts.append(spoke)
    parts.append(cyl(2.5, 2.6, (0, spoke_y, 0)))               # O5 hub
    parts.append(cyl(0.75, 0.6, (0, y + 11.0 + 0.1, 0),        # shaft top:
                     sections=12))                             # ~0.2 visible
    return trimesh.util.concatenate(parts)


def _find_hub_center(v):
    """Fit the true rotation axis (XY) of the prop mesh.

    The AABB center of a 3-blade prop is NOT the hub axis, which is what
    caused the mirror seam gap. Iterate: start from the AABB center, take the
    innermost verts (r < 4mm -- hub material, which is axisymmetric; blade
    root fillets at larger radii are NOT symmetric and would bias the
    centroid), and recenter on their centroid.
    """
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    c = 0.5 * (lo + hi)[:2]
    for _ in range(4):
        r = np.hypot(v[:, 0] - c[0], v[:, 1] - c[1])
        hub = r < 0.004  # innermost verts only: blade roots bias the centroid
        c = v[hub, :2].mean(axis=0)
    return c


# Radius at which the extracted blade is sliced from the source hub. The
# slayer prop's hub is not axisymmetric (root fillets, spinner), so keeping it
# and mirroring 180 degrees leaves a visible seam gap. Instead we keep only
# the blade proper and add a clean procedural hub over the roots.
BLADE_ROOT_R = 0.010  # source units (meters)


def _slice_one_blade(mesh):
    """Keep one blade sector of the extracted 3-blade prop (axis = Z)."""
    v = mesh.vertices
    c = _find_hub_center(v)
    dx, dy = v[:, 0] - c[0], v[:, 1] - c[1]
    r = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx))
    keep = (r > BLADE_ROOT_R) & (th > -28.0) & (th < 38.0)
    face_keep = keep[mesh.faces].all(axis=1)
    m = mesh.submesh([np.nonzero(face_keep)[0]], append=True)
    # re-center hub on the origin
    m.apply_translation([-c[0], -c[1], 0.0])
    return m


def _prop_axis_to_y(mesh):
    """Rotate extracted prop so its axis (Z) becomes +Y."""
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
    )
    return mesh


def extract_slayer_prop():
    """Load vtx-slayer-3 Propeller_Baked, cut to 2 blades, scale to 51mm."""
    import UnityPy  # noqa: delayed optional dependency

    env = UnityPy.load(os.path.join(PR0P_DATA, "sharedassets0.assets"))
    src = None
    for obj in env.objects:
        if obj.type.name == "Mesh" and obj.read().m_Name == "Propeller_Baked":
            from UnityPy.export import MeshExporter
            data = MeshExporter.export_mesh(obj.read())
            cand = trimesh.load(
                trimesh.util.wrap_as_stream(
                    data.encode() if isinstance(data, str) else data),
                file_type="obj", force="mesh")
            # several meshes share the name (LODs / a degenerate 1mm-scale
            # copy); keep the largest
            if src is None or cand.extents.max() > src.extents.max():
                src = cand
    if src is None:
        raise RuntimeError("Propeller_Baked not found in sharedassets0")

    blade = _slice_one_blade(src)
    v = blade.vertices
    r = np.hypot(v[:, 0], v[:, 1])
    tip_r = r.max()
    scale = (PROP_DIAM / 2.0) / tip_r  # meters -> mm, tip at 25.5mm
    blade.apply_scale(scale)

    other = blade.copy()
    other.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))

    # Procedural hub: a cylinder covering the blade roots so the two halves
    # join cleanly. Radius a bit larger than the blade cut radius; height
    # spans the blade's z range near the root.
    hub_r = BLADE_ROOT_R * scale * 1.3
    root = blade.vertices[
        np.hypot(blade.vertices[:, 0], blade.vertices[:, 1])
        < BLADE_ROOT_R * scale * 1.8]
    z0, z1 = root[:, 2].min(), root[:, 2].max()
    hub = trimesh.creation.cylinder(
        radius=hub_r, height=(z1 - z0) + 1.0, sections=32)
    hub.apply_translation((0, 0, (z0 + z1) / 2.0))

    # Seam check: tip radii of the two halves must match (good axis fit) and
    # every blade vertex must sit within 0.5mm of the other half's surface
    # near the hub junction.
    r_b = np.hypot(blade.vertices[:, 0], blade.vertices[:, 1])
    r_o = np.hypot(other.vertices[:, 0], other.vertices[:, 1])
    print(f"[prop] tip radii: blade {r_b.max():.2f} mm, "
          f"mirror {r_o.max():.2f} mm")

    prop = trimesh.util.concatenate([blade, other, hub])
    _prop_axis_to_y(prop)
    print(f"[prop] extracted slayer prop, tip r={tip_r:.4f} m -> 51mm 2-blade")
    return prop


def procedural_prop():
    """Fallback: hub cylinder + two swept, slightly twisted blades."""
    hub = cyl(2.5, 2.5, (0, 0, 0))
    r_root, r_tip = 3.0, PROP_DIAM / 2.0
    stations = np.linspace(r_root, r_tip, 8)
    chord = np.linspace(8.0, 3.0, len(stations))
    pitch_deg = np.linspace(38.0, 16.0, len(stations))  # ~38mm pitch twist
    thick = 0.5
    verts, faces = [], []
    for s, c, p in zip(stations, chord, pitch_deg):
        pa = np.radians(p)
        for sign in (1, -1):  # blade cross-section: leading/trailing edge
            y = sign * c / 2 * np.sin(pa)
            z = sign * c / 2 * np.cos(pa)
            verts.append([s, y + sign * thick / 2, z])
            verts.append([s, y - sign * thick / 2, z])
    for i in range(len(stations) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(3):
            faces += [[a + k, b + k, a + k + 1], [a + k + 1, b + k, b + k + 1]]
    # caps
    faces += [[0, 2, 1], [1, 2, 3]]
    n = 4 * (len(stations) - 1)
    faces += [[n, n + 1, n + 2], [n + 1, n + 3, n + 2]]
    blade = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    other = blade.copy()
    other.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    print("[prop] procedural fallback prop")
    return trimesh.util.concatenate([hub, blade, other])


def make_prop():
    try:
        if os.path.isdir(PR0P_DATA):
            return extract_slayer_prop()
    except Exception as e:  # noqa: BLE001 - any failure -> procedural
        print(f"[prop] extraction failed ({e}); using procedural prop")
    return procedural_prop()


FPVCAM_STL = os.path.join(HERE, "fpvcam.stl")


def make_camera():
    """Micro cam centered vertically between the plates (y ~= 8.75mm) at the
    front (-Z) plate edge, tilted up ~27 degrees. Uses fpvcam.stl when
    present (recentred: the STL's origin sits at the top of the body);
    falls back to a box + lens stub."""
    cam_y = PLATE_T / 2 + STANDOFF_H / 2  # mid-height between plates
    cam_z = -(PLATE_L / 2 - 5.0)          # pushed to the front edge
    if os.path.isfile(FPVCAM_STL):
        cam = trimesh.load(FPVCAM_STL, force="mesh")
        cam.apply_translation((0, -0.5 * (cam.bounds[0][1]
                                          + cam.bounds[1][1]), 0))
        # STL is authored lens-up; pitch 90 + 27 so it faces -Z tilted up
        cam.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.radians(117), [1, 0, 0]))
        cam.apply_translation((0, cam_y, cam_z))
        return cam
    cam = box(14.0, 14.0, 14.0)
    cam.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(27), [1, 0, 0]))
    cam.apply_translation((0, cam_y, cam_z))
    lens = cyl(3.5, 2.0)
    lens.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(-63), [1, 0, 0]))
    lens.apply_translation((0, cam_y + 6.0 * np.sin(np.radians(27)),
                            cam_z - 7.0 * np.cos(np.radians(27))))
    return trimesh.util.concatenate([cam, lens])


def main():
    parts = {}

    parts["frame_bottom"] = box(PLATE_W, PLATE_T, PLATE_L, (0, 0, 0))
    top_y = PLATE_T / 2 + STANDOFF_H + PLATE_T / 2
    parts["frame_top"] = box(PLATE_W, PLATE_T, PLATE_L, (0, top_y, 0))

    for i, (sx, sz) in enumerate(
            [(1, 1), (1, -1), (-1, 1), (-1, -1)], start=1):
        parts[f"standoff{i}"] = cyl(
            STANDOFF_D / 2, STANDOFF_H,
            (sx * STANDOFF_X, PLATE_T / 2 + STANDOFF_H / 2, sz * STANDOFF_Z))

    motor_dirs = [(1, -1), (1, 1), (-1, -1), (-1, 1)]  # M1..M4 per config
    motor = make_motor()
    prop = make_prop()
    for i, (sx, sz) in enumerate(motor_dirs, start=1):
        parts[f"arm{i}"] = make_arm((sx * MOTOR_X, sz * MOTOR_Z))
        m = motor.copy()
        m.apply_translation((sx * MOTOR_X, 0, sz * MOTOR_Z))
        parts[f"motor{i}"] = m
        p = prop.copy()
        # prop's lowest point sits 0.2mm above the bell top (~flush)
        p.apply_translation(
            (sx * MOTOR_X, PROP_Y - p.bounds[0][1], sz * MOTOR_Z))
        parts[f"prop{i}"] = p

    parts["camera"] = make_camera()

    # 1S 680mAh pack on the top plate, long axis front-back
    batt_y = top_y + PLATE_T / 2 + 4.0
    parts["battery"] = box(17.0, 8.0, 58.0, (0, batt_y, 0))

    # whip antenna at the back, leaning back slightly
    ant = cyl(0.5, 40.0, sections=12)
    ant.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(15), [1, 0, 0]))
    ant.apply_translation((0, top_y + 18.0, PLATE_L / 2 - 2.0))
    parts["antenna"] = ant

    scene = trimesh.Scene()
    for name, mesh in parts.items():
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    scene.export(OUT_GLB)
    print(f"wrote {OUT_GLB} ({len(parts)} parts)")


if __name__ == "__main__":
    sys.exit(main())
