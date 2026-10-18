"""Procedural scenery from OpenStreetMap for the flying site: houses at the
real building footprints, the transmission lines on their mapped towers, and
light poles around the lit sports pitches.

Source: OpenStreetMap via the Overpass API, cached in
sim/assets/terrain/osm.json (ODbL: (c) OpenStreetMap contributors).  OSM has
no heights here, so heights are typical values (see the constants).

Output: sim/assets/terrain/scenery.bin, baked geometry in sim coordinates
(+x launch heading, +z pilot's right, +y up, metres above the pilot's
ground), draped on terrain/local.bin:
    int32 nTriVerts, int32 nLineVerts
    nTriVerts  x (float32 x, y, z; uint8 r, g, b, a)    -- triangles
    nLineVerts x (float32 x, y, z; uint8 r, g, b, a)    -- line segments
Colours are flat-shaded with the same sun as the terrain.

Run via `uv run python -m sim.terrain` (or `-m sim.scenery` to redo only this).
"""

import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np

from sim import terrain as T

OSM_CACHE = os.path.join(T.TERRAIN_DIR, "osm.json")
OUT = os.path.join(T.TERRAIN_DIR, "scenery.bin")
OSM_HALF = 2600.0                  # m, query box half-size (covers the mesh)

WALL = (214, 196, 160)             # sand
ROOF = (150, 64, 46)               # brick
FLAT_ROOF = (120, 112, 104)
HOUSE_EAVE = 5.5                   # m, typical 1-2 storey eave height
BIG_AREA, BIG_HEIGHT = 900.0, 8.0  # footprints above this get a flat roof
TOWER_STEEL = (150, 150, 150)
WIRE = (55, 55, 55)
POLE = (175, 175, 170)
LAMP = (240, 240, 225)
LIGHT_POLE_HEIGHT = 25.0           # m, typical ballfield light pole
LINE_HEIGHT = {230000: 40.0}       # m tower height by voltage; else DEFAULT
DEFAULT_LINE_HEIGHT = 28.0
SAG = 0.03                         # conductor sag / span


# --- data -------------------------------------------------------------------------


def fetch_osm():
    if os.path.exists(OSM_CACHE):
        with open(OSM_CACHE) as f:
            data = json.load(f)
        if data.get("_half") == OSM_HALF:
            return data
    dlat = OSM_HALF / 111320.0
    dlon = OSM_HALF / (111320.0 * math.cos(math.radians(T.PILOT_LAT)))
    bbox = (f"{T.PILOT_LAT - dlat},{T.PILOT_LON - dlon},"
            f"{T.PILOT_LAT + dlat},{T.PILOT_LON + dlon}")
    query = f"""[out:json][timeout:120];
(
  way["building"]({bbox});
  relation["building"]({bbox});
  way["power"~"^(line|minor_line)$"]({bbox});
  way["leisure"="pitch"]["lit"="yes"]({bbox});
);
out body geom;"""
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "flap_simulation"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    data["_half"] = OSM_HALF
    with open(OSM_CACHE, "w") as f:
        json.dump(data, f)
    return data


class Ground:
    """Bilinear heights from terrain/local.bin -- the grid the sim draws."""

    def __init__(self, path=T.LOCAL_OUT):
        with open(path, "rb") as f:
            nx, nz = np.fromfile(f, np.int32, 2)
            self.x0, self.z0, self.step = np.fromfile(f, np.float32, 3)
            self.h = np.fromfile(f, np.float32, nx * nz).reshape(nz, nx)

    def inside(self, x, z, margin=0.0):
        n = self.h.shape[0]
        lo = self.x0 + margin
        hi = self.x0 + (n - 1) * self.step - margin
        return lo <= x <= hi and lo <= z <= hi

    def at(self, x, z):
        fx = np.clip((np.asarray(x) - self.x0) / self.step, 0, self.h.shape[1] - 1.001)
        fz = np.clip((np.asarray(z) - self.z0) / self.step, 0, self.h.shape[0] - 1.001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        h = self.h
        return ((h[iz, ix] * (1 - tx) + h[iz, ix + 1] * tx) * (1 - tz)
                + (h[iz + 1, ix] * (1 - tx) + h[iz + 1, ix + 1] * tx) * tz)


def to_sim(lat, lon):
    east = (lon - T.PILOT_LON) * 111320.0 * math.cos(math.radians(T.PILOT_LAT))
    north = (lat - T.PILOT_LAT) * 111320.0
    return T.en_to_sim(east, north)


# --- geometry buffers ------------------------------------------------------------------


class Builder:
    def __init__(self):
        rel = math.radians(T.SUN_BEARING - T.LAUNCH_HEADING)
        el = math.radians(T.SUN_ELEV)
        self.sun = np.array([math.cos(el) * math.cos(rel), math.sin(el),
                             math.cos(el) * math.sin(rel)])
        self.tris, self.tri_col, self.lines, self.line_col = [], [], [], []

    def tri(self, a, b, c, rgb):
        a, b, c = (np.asarray(p, float) for p in (a, b, c))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            return
        n /= ln
        shade = 0.5 + 0.6 * abs(float(n @ self.sun))
        col = tuple(int(min(255, ch * shade)) for ch in rgb)
        self.tris += [a, b, c]
        self.tri_col += [col] * 3

    def quad(self, a, b, c, d, rgb):
        self.tri(a, b, c, rgb)
        self.tri(a, c, d, rgb)

    def line(self, a, b, rgb):
        self.lines += [np.asarray(a, float), np.asarray(b, float)]
        self.line_col += [rgb, rgb]

    def write(self, path):
        def pack(pts, cols):
            rec = np.zeros(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 4)])
            if pts:
                rec["p"] = np.array(pts)
                rec["c"][:, :3] = np.array(cols)
                rec["c"][:, 3] = 255
            return rec
        t, l = pack(self.tris, self.tri_col), pack(self.lines, self.line_col)
        with open(path, "wb") as f:
            np.array([len(t), len(l)], np.int32).tofile(f)
            t.tofile(f)
            l.tofile(f)


def ear_clip(poly):
    """Triangulate a simple polygon (list of (x, z)); returns index triples."""
    pts = [np.asarray(p, float) for p in poly]
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts)))
    idx = list(range(len(pts))) if area > 0 else list(range(len(pts)))[::-1]
    out, guard = [], 0
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(cross(a, b, pts[j]) >= 0 and cross(b, c, pts[j]) >= 0 and cross(c, a, pts[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            out.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break                       # degenerate: give up on the rest
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


# --- houses --------------------------------------------------------------------------------


def building_rings(data):
    for e in data["elements"]:
        if "building" not in e.get("tags", {}):
            continue
        if e["type"] == "way" and e.get("geometry"):
            yield e["geometry"]
        elif e["type"] == "relation":
            for m in e.get("members", []):
                if m.get("role") == "outer" and m.get("geometry"):
                    yield m["geometry"]


def obb(poly):
    """Minimum-area bounding rectangle: (centre, long axis unit, half-lengths)."""
    best = None
    for i in range(len(poly)):
        d = poly[(i + 1) % len(poly)] - poly[i]
        if np.linalg.norm(d) < 1e-6:
            continue
        u = d / np.linalg.norm(d)
        v = np.array([-u[1], u[0]])
        pu, pv = poly @ u, poly @ v
        area = (pu.max() - pu.min()) * (pv.max() - pv.min())
        if best is None or area < best[0]:
            c = u * (pu.max() + pu.min()) / 2 + v * (pv.max() + pv.min()) / 2
            hu, hv = (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2
            best = (area, c, u, v, hu, hv) if hu >= hv else (area, c, v, u, hv, hu)
    return best[1:]


def add_house(b, ground, ring):
    pts = np.array([to_sim(p["lat"], p["lon"]) for p in ring])
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3 or not ground.inside(*pts.mean(0), margin=5.0):
        return False
    area = 0.5 * abs(sum(pts[i, 0] * pts[(i + 1) % len(pts), 1]
                         - pts[(i + 1) % len(pts), 0] * pts[i, 1] for i in range(len(pts))))
    if area < 8.0:
        return False
    gh = ground.at(pts[:, 0], pts[:, 1])
    base = float(gh.min()) - 1.0          # sink 1 m so slopes don't show a gap
    big = area > BIG_AREA
    # eave above the pad (median ground), not the uphill corner: on hillside
    # lots the latter made downhill walls far taller than a stepped house
    eave = float(np.median(gh)) + (BIG_HEIGHT if big else HOUSE_EAVE)
    for i in range(len(pts)):
        a, c = pts[i], pts[(i + 1) % len(pts)]
        b.quad((a[0], base, a[1]), (c[0], base, c[1]), (c[0], eave, c[1]), (a[0], eave, a[1]), WALL)
    if big:
        for i0, i1, i2 in ear_clip(pts):
            b.tri(*((pts[i][0], eave, pts[i][1]) for i in (i0, i1, i2)), FLAT_ROOF)
        return True
    # hip roof: every eave edge slopes up to its projection on a ridge along
    # the long axis of the footprint's bounding rectangle
    c, u, v, hu, hv = obb(pts)
    rise = min(4.0, 0.36 * hv)
    half_ridge = max(0.0, hu - hv)
    top = eave + rise
    def ridge(p):
        t = float(np.clip((p - c) @ u, -half_ridge, half_ridge))
        r = c + u * t
        return (r[0], top, r[1])
    for i in range(len(pts)):
        a, d = pts[i], pts[(i + 1) % len(pts)]
        b.quad((a[0], eave, a[1]), (d[0], eave, d[1]), ridge(d), ridge(a), ROOF)
    return True


# --- transmission lines ----------------------------------------------------------------------


def tower_geometry(b, base, along, height, double_circuit):
    """Lattice tower drawn as lines; returns conductor attachment points."""
    across = np.array([-along[1], along[0]])
    bx, bz, by = base
    def P(r_al, r_ac, y):
        q = np.array([bx, bz]) + along * r_al + across * r_ac
        return (q[0], by + y, q[1])
    w0, w1 = 4.0, 1.0                  # half-width at the base / top
    levels = np.linspace(0, height, 9)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for k in range(len(levels) - 1):
        y0, y1 = levels[k], levels[k + 1]
        h0 = w0 + (w1 - w0) * y0 / height
        h1 = w0 + (w1 - w0) * y1 / height
        for j in range(4):
            sa, sc = corners[j]
            ta, tc = corners[(j + 1) % 4]
            b.line(P(sa * h0, sc * h0, y0), P(sa * h1, sc * h1, y1), TOWER_STEEL)   # leg
            b.line(P(sa * h1, sc * h1, y1), P(ta * h1, tc * h1, y1), TOWER_STEEL)   # ring
            b.line(P(sa * h0, sc * h0, y0), P(ta * h1, tc * h1, y1), TOWER_STEEL)   # brace
    arms = [0.62, 0.76, 0.90] if double_circuit else [0.80]
    reach = [6.5, 8.0, 6.5] if double_circuit else [6.0]
    attach = []
    for frac, r in zip(arms, reach):
        y = height * frac
        hw = w0 + (w1 - w0) * frac
        for side in ((-1, 1) if double_circuit else (-1, 0, 1)):
            if side == 0:
                attach.append(P(0, 0, height + 1.5))
                continue
            tip = P(0, side * (hw + r), y)
            b.line(P(0, side * hw, y), tip, TOWER_STEEL)
            b.line(P(0, side * hw, y + 2.0), tip, TOWER_STEEL)
            attach.append((tip[0], tip[1] - 2.0, tip[2]))      # insulator string
    if double_circuit:
        attach.append(P(0, 0, height + 3.0))                     # shield wire peak
        b.line(P(0, 0, height), attach[-1], TOWER_STEEL)
    return attach


def add_power_line(b, ground, way):
    tags = way.get("tags", {})
    try:
        volts = int(tags.get("voltage", "0").split(";")[0])
    except ValueError:
        volts = 0
    height = LINE_HEIGHT.get(volts, DEFAULT_LINE_HEIGHT)
    double = volts >= 230000
    pts = [np.array(to_sim(p["lat"], p["lon"])) for p in way["geometry"]]
    towers = []
    for i, p in enumerate(pts):
        if not ground.inside(*p, margin=10.0):
            towers.append(None)
            continue
        prev, nxt = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
        d = nxt - prev
        along = d / (np.linalg.norm(d) or 1.0)
        base = (p[0], p[1], float(ground.at(*p)))
        towers.append(tower_geometry(b, base, along, height, double))
    for a, c in zip(towers, towers[1:]):
        if a is None or c is None:
            continue
        for pa, pc in zip(a, c):
            pa, pc = np.array(pa), np.array(pc)
            span = np.linalg.norm((pc - pa)[[0, 2]])
            prev = pa
            for t in np.linspace(0, 1, 13)[1:]:
                q = pa + (pc - pa) * t
                q[1] -= 4 * SAG * span * t * (1 - t)               # parabolic sag
                b.line(prev, q, WIRE)
                prev = q
    return sum(t is not None for t in towers)


# --- stadium lights -----------------------------------------------------------------------------


def add_light_poles(b, ground, way):
    pts = np.array([to_sim(p["lat"], p["lon"]) for p in way["geometry"]])
    if len(pts) < 3 or not ground.inside(*pts.mean(0), margin=10.0):
        return 0
    c, u, v, hu, hv = obb(pts)
    spots = [c + u * su * hu + v * sv * (hv + 4.0) for su in (-1, 0, 1) for sv in (-1, 1)]
    for s in spots:
        gy = float(ground.at(*s))
        add_pole(b, (s[0], gy, s[1]), c)
    return len(spots)


def add_pole(b, base, aim):
    x, y, z = base
    r = 0.35
    ring = [(x + r * math.cos(a), z + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
    top = y + LIGHT_POLE_HEIGHT
    for i in range(6):
        a, d = ring[i], ring[(i + 1) % 6]
        b.quad((a[0], y, a[1]), (d[0], y, d[1]), (d[0], top, d[1]), (a[0], top, a[1]), POLE)
    # light bank: a 3 x 2 m panel facing the pitch
    to = np.array([aim[0] - x, aim[1] - z])
    to /= np.linalg.norm(to) or 1.0
    side = np.array([-to[1], to[0]]) * 1.5
    f = np.array([x, z]) + to * 0.5
    p = [(f[0] - side[0], top - 0.5, f[1] - side[1]), (f[0] + side[0], top - 0.5, f[1] + side[1]),
         (f[0] + side[0], top + 1.5, f[1] + side[1]), (f[0] - side[0], top + 1.5, f[1] - side[1])]
    b.quad(*p, LAMP)


def main():
    data = fetch_osm()
    ground = Ground()
    b = Builder()
    houses = sum(add_house(b, ground, ring) for ring in building_rings(data))
    towers = sum(add_power_line(b, ground, e) for e in data["elements"]
                 if e.get("tags", {}).get("power") in ("line", "minor_line") and e.get("geometry"))
    poles = sum(add_light_poles(b, ground, e) for e in data["elements"]
                if e.get("tags", {}).get("leisure") == "pitch" and e.get("tags", {}).get("lit") == "yes"
                and e.get("geometry"))
    b.write(OUT)
    print(f"wrote {OUT}: {houses} buildings, {towers} towers, {poles} light poles; "
          f"{len(b.tris) // 3} triangles, {len(b.lines) // 2} line segments")


if __name__ == "__main__":
    main()
