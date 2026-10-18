"""Terrain and placeholder scenery for the flying site, from open elevation data.

Elevation: AWS Open Data "Terrain Tiles" (Terrarium PNG encoding), which in
the US is USGS 3DEP.  Tiles are cached in sim/assets/terrain/tiles/; see
sim/assets/terrain/ATTRIBUTION.md.

Outputs
  sim/assets/terrain/local.bin
      Height grid around the pilot in sim coordinates, for the low-poly
      ground mesh and crash checks in the front end:
        int32 nx, int32 nz, float32 x0, z0, step, then nz*nx float32 heights
        (row z, column x), metres above the pilot's ground.
      Sim axes: +x = the launch heading, +z = the pilot's right, +y up.
  sim/assets/sky/placeholder_equirect.png
      8192x4096 equirectangular view from the pilot's eye: procedural sky
      over the real skyline, ray-cast from the wide terrain.  Same layout as
      a photo sphere: centre column = the launch heading, bearing increasing
      to the right; ticks and N/E/S/W letters are true bearings.

Run:  uv run python -m sim.terrain
"""

import json
import math
import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERRAIN_DIR = os.path.join(ROOT, "sim", "assets", "terrain")
TILE_DIR = os.path.join(TERRAIN_DIR, "tiles")
LOCAL_OUT = os.path.join(TERRAIN_DIR, "local.bin")
SKY_OUT = os.path.join(ROOT, "sim", "assets", "sky", "placeholder_equirect.png")
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Pilot spot: where the Black Mountain Ranch Park photo sphere was taken, and
# the direction it looks (the default launch heading).
PILOT_LAT, PILOT_LON = 32.9906425, -117.1282654
LAUNCH_HEADING = 308.79        # deg true
EYE_HEIGHT = 1.7               # m, matches the front end's LOS camera

# Runway: the dirt strip just NE of the photo-sphere spot, measured off a
# satellite view (scaled by the ballfield 338 m away at 218 deg): centre 29 m
# from the pilot at 40 deg, running along 40 deg (SW -> NE).
RUNWAY_CENTRE_DIST, RUNWAY_CENTRE_BEARING = 29.0, 40.0
RUNWAY_BEARING = 40.0          # deg true
RUNWAY_LENGTH, RUNWAY_WIDTH = 30.0, 6.0
FEATURES_OUT = os.path.join(TERRAIN_DIR, "features.json")

LOCAL_HALF = 2500.0            # m, half-size of the 3D ground mesh
LOCAL_STEP = 20.0              # m, low-poly cell size
SKY_W, SKY_H = 8192, 4096
SKY_RANGE = 30000.0            # m, how far the skyline ray-cast reaches
SUN_BEARING, SUN_ELEV = 225.0, 45.0   # deg; afternoon sun
EARTH_R = 6371000.0


# --- elevation tiles -----------------------------------------------------------


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y):
    os.makedirs(TILE_DIR, exist_ok=True)
    path = os.path.join(TILE_DIR, f"{z}_{x}_{y}.png")
    if not os.path.exists(path):
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y),
                                     headers={"User-Agent": "flap_simulation"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


class Dem:
    """A (2r+1)^2-tile Terrarium mosaic around the pilot, sampled in metres
    east/north of the pilot."""

    def __init__(self, z, r=1):
        self.z = z
        cx, cy = (int(v) for v in tile_xy(PILOT_LAT, PILOT_LON, z))
        span = range(-r, r + 1)
        rows = [np.hstack([fetch_tile(z, cx + dx, cy + dy) for dx in span])
                for dy in span]
        self.h = np.vstack(rows)
        self.px0, self.py0 = (cx - r) * 256, (cy - r) * 256

    def _pixels(self, east, north):
        lat = PILOT_LAT + north / 111320.0
        lon = PILOT_LON + east / (111320.0 * math.cos(math.radians(PILOT_LAT)))
        n = 256 * 2 ** self.z
        px = (lon + 180.0) / 360.0 * n - self.px0
        py = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n - self.py0
        return px - 0.5, py - 0.5

    def sample(self, east, north):
        """Bilinear elevation (m); clamps at the mosaic edge."""
        px, py = self._pixels(np.asarray(east, float), np.asarray(north, float))
        hgt, wid = self.h.shape
        px = np.clip(px, 0, wid - 1.001)
        py = np.clip(py, 0, hgt - 1.001)
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        tx, ty = px - x0, py - y0
        h = self.h
        return ((h[y0, x0] * (1 - tx) + h[y0, x0 + 1] * tx) * (1 - ty)
                + (h[y0 + 1, x0] * (1 - tx) + h[y0 + 1, x0 + 1] * tx) * ty)

    def half_extent(self):
        """Metres from the pilot to the nearest mosaic edge."""
        m_per_px = 156543.03 * math.cos(math.radians(PILOT_LAT)) / 2 ** self.z
        px, py = self._pixels(np.array(0.0), np.array(0.0))
        edge = min(px, py, self.h.shape[1] - px, self.h.shape[0] - py)
        return float(edge) * m_per_px


def en_to_sim(east, north):
    """East, north -> sim ground coords (+x launch heading, +z right)."""
    h = math.radians(LAUNCH_HEADING)
    return (east * math.sin(h) + north * math.cos(h),
            east * math.cos(h) - north * math.sin(h))


def find_runway():
    b = math.radians(RUNWAY_CENTRE_BEARING)
    ce, cn = RUNWAY_CENTRE_DIST * math.sin(b), RUNWAY_CENTRE_DIST * math.cos(b)
    x, z = en_to_sim(ce, cn)
    print(f"runway centre {ce:+.0f} E {cn:+.0f} N, bearing {RUNWAY_BEARING:.0f}")
    return {"x": x, "z": z, "bearingDeg": RUNWAY_BEARING,
            "headingInSimDeg": RUNWAY_BEARING - LAUNCH_HEADING,
            "length": RUNWAY_LENGTH, "width": RUNWAY_WIDTH}


def sim_to_en(x, z):
    """Sim ground coords (+x launch heading, +z right) -> east, north."""
    h = math.radians(LAUNCH_HEADING)
    east = x * math.sin(h) + z * math.cos(h)
    north = x * math.cos(h) - z * math.sin(h)
    return east, north


# --- colours ---------------------------------------------------------------------


def terrain_colour(slope_deg, rel_height, shade, dist):
    """San Diego scrub palette: flat ground = dry grass/turf, slopes = chaparral,
    high ground = browner; Lambert shade; haze with distance."""
    grass = np.array([150, 150, 95], float)
    turf = np.array([96, 128, 62], float)
    scrub = np.array([88, 96, 62], float)
    dirt = np.array([140, 118, 90], float)
    flat = np.clip(1 - slope_deg / 12.0, 0, 1)[..., None]
    near = np.clip(1 - dist / 400.0, 0, 1)[..., None]     # park turf near the pilot
    base = flat * (near * turf + (1 - near) * grass) + (1 - flat) * scrub
    high = np.clip((rel_height - 80) / 300, 0, 0.6)[..., None]
    base = base * (1 - high) + dirt * high
    col = base * (0.45 + 0.65 * shade[..., None])
    haze = np.array([190, 205, 220], float)
    a = (1 - np.exp(-dist / 18000.0))[..., None]
    return col * (1 - a) + haze * a


def sun_vector():
    b, e = math.radians(SUN_BEARING), math.radians(SUN_ELEV)
    return np.array([math.cos(e) * math.sin(b), math.cos(e) * math.cos(b), math.sin(e)])


def shade_and_slope(dem, east, north, d=10.0):
    """Lambert shade and slope (deg) from finite differences."""
    dhe = (dem.sample(east + d, north) - dem.sample(east - d, north)) / (2 * d)
    dhn = (dem.sample(east, north + d) - dem.sample(east, north - d)) / (2 * d)
    n = np.stack([-dhe, -dhn, np.ones_like(dhe)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    shade = np.clip(n @ sun_vector(), 0, 1)
    slope = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    return shade, slope


# --- local ground grid ------------------------------------------------------------


def write_local(dem, ground):
    xs = np.arange(-LOCAL_HALF, LOCAL_HALF + 1e-6, LOCAL_STEP)
    zz, xx = np.meshgrid(xs, xs, indexing="ij")
    east, north = sim_to_en(xx, zz)
    hts = (dem.sample(east, north) - ground).astype(np.float32)
    with open(LOCAL_OUT, "wb") as f:
        np.array([len(xs), len(xs)], np.int32).tofile(f)
        np.array([xs[0], xs[0], LOCAL_STEP], np.float32).tofile(f)
        hts.tofile(f)
    print(f"wrote {LOCAL_OUT}: {len(xs)}x{len(xs)} @ {LOCAL_STEP} m, "
          f"heights {hts.min():+.0f}..{hts.max():+.0f} m")


# --- sky ----------------------------------------------------------------------------


def value_noise(shape, cells, seed):
    rng = np.random.default_rng(seed)
    gy, gx = cells
    grid = rng.random((gy + 1, gx))
    grid = np.concatenate([grid, grid[:, :1]], axis=1)       # wraps in azimuth
    ys = np.linspace(0, gy, shape[0], dtype=np.float32)
    xs = np.linspace(0, gx, shape[1], dtype=np.float32)
    y0 = np.floor(ys).astype(int).clip(0, gy - 1)
    x0 = np.floor(xs).astype(int).clip(0, gx - 1)
    ty = (ys - y0)[:, None]
    tx = (xs - x0)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    top = grid[y0][:, x0] * (1 - tx) + grid[y0][:, x0 + 1] * tx
    bot = grid[y0 + 1][:, x0] * (1 - tx) + grid[y0 + 1][:, x0 + 1] * tx
    return (top * (1 - ty) + bot * ty).astype(np.float32)


def sky_rgb(bearing, elev):
    """Procedural sky for arrays of bearing/elevation (radians)."""
    t = np.clip(elev / (np.pi / 2), 0, 1) ** 0.6
    zenith = np.array([52, 102, 180], np.float32)
    haze = np.array([196, 214, 230], np.float32)
    sky = haze * (1 - t[..., None]) + zenith * t[..., None]
    sb, se = np.radians(SUN_BEARING), np.radians(SUN_ELEV)
    cosang = (np.sin(elev) * np.sin(se)
              + np.cos(elev) * np.cos(se) * np.cos(bearing - sb))
    glow = np.clip(cosang, 0, 1) ** 400 + np.clip(cosang, 0, 1) ** 12 * 0.25
    sky = sky + glow[..., None] * np.array([255, 245, 220], np.float32)
    shape = bearing.shape
    # square cells: the image is 2:1, so twice as many cells across as down
    clouds = (value_noise(shape, (12, 24), 1) * 0.6
              + value_noise(shape, (48, 96), 2) * 0.4)
    cover = np.clip((clouds - 0.55) * 4, 0, 1) * np.clip(1 - t * 1.2, 0, 1)
    return sky * (1 - cover[..., None] * 0.7) + 245 * cover[..., None] * 0.7


def render_sky(near, far, ground):
    """Ray-cast the terrain from the eye, column by column (voxel-space
    style): march outward, keep the highest elevation angle seen so far, and
    give each image row the colour of the first sample that rises above it."""
    W, H = SKY_W, SKY_H
    bearing = np.radians(LAUNCH_HEADING) + (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    elev = (0.5 - (np.arange(H) + 0.5) / H) * np.pi          # row centres
    r = np.geomspace(1.0, SKY_RANGE, 1400)
    eye = ground + EYE_HEIGHT
    near_r = near.half_extent() * 0.95

    sb, cb = np.sin(bearing), np.cos(bearing)
    east = r[:, None] * sb[None, :]
    north = r[:, None] * cb[None, :]
    hts = np.where(r[:, None] < near_r, near.sample(east, north), far.sample(east, north))
    drop = r[:, None] ** 2 / (2 * EARTH_R) * 0.87         # curvature less refraction
    ang = np.arctan2(hts - eye - drop, r[:, None])
    runmax = np.maximum.accumulate(ang, axis=0)
    shade, slope = shade_and_slope(near, east, north)
    shade_f, slope_f = shade_and_slope(far, east, north, d=60.0)
    use_near = r[:, None] < near_r
    shade = np.where(use_near, shade, shade_f)
    slope = np.where(use_near, slope, slope_f)
    colours = terrain_colour(slope, hts - ground, shade, np.broadcast_to(r[:, None], hts.shape))

    img = np.empty((H, W, 3), np.float32)
    bgrid, egrid = np.meshgrid(bearing, elev)
    img[:] = sky_rgb(bgrid.astype(np.float32), egrid.astype(np.float32))
    for c in range(W):
        top = runmax[-1, c]
        rows = np.nonzero(elev <= top)[0]
        if rows.size == 0:
            continue
        k = np.searchsorted(runmax[:, c], elev[rows], side="left")
        img[rows, c] = colours[np.minimum(k, len(r) - 1), c]
    return img


def annotate(img):
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=120)
    small = ImageFont.load_default(size=48)
    W, H = im.size
    horizon = H // 2
    col = lambda b: int(((b - LAUNCH_HEADING) / 360.0 + 0.5) % 1.0 * W)
    # markers sit ~12-18 deg up in the sky, clear of the skyline and the 3D
    # ground that draws over the lower part of this image
    mark = horizon - int(H * 15 / 180)
    for b in range(0, 360, 30):
        x = col(b)
        draw.line([(x, mark - 30), (x, mark + 30)], fill=(255, 255, 255), width=5)
        draw.text((x + 10, mark + 4), f"{b}", fill=(255, 255, 255), font=small,
                  stroke_width=2, stroke_fill=(0, 0, 0))
    for b, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        draw.text((col(b), mark - 110), name, fill=(255, 255, 255), font=font,
                  anchor="mm", stroke_width=5, stroke_fill=(0, 0, 0))
    draw.text((W // 2, mark - 230), "PLACEHOLDER - terrain: USGS 3DEP via AWS Terrain Tiles; "
              "map data (c) OpenStreetMap contributors",
              fill=(255, 255, 255), font=small, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))
    return im


def main():
    near = Dem(14, r=2)        # ~10 km square: covers the 5 km ground mesh
    far = Dem(11)
    ground = float(near.sample(0.0, 0.0))
    print(f"pilot ground {ground:.1f} m; near DEM reaches {near.half_extent():.0f} m, "
          f"far DEM {far.half_extent():.0f} m")
    write_local(near, ground)
    features = {"runway": find_runway()}
    with open(FEATURES_OUT, "w") as f:
        json.dump(features, f, indent=2)
    print(f"wrote {FEATURES_OUT}")
    from sim import scenery
    scenery.main()
    im = annotate(render_sky(near, far, ground))
    os.makedirs(os.path.dirname(SKY_OUT), exist_ok=True)
    im.save(SKY_OUT, optimize=True)
    print(f"wrote {SKY_OUT} ({im.size[0]}x{im.size[1]})")
    with open(os.path.join(TERRAIN_DIR, "site.json"), "w") as f:
        json.dump({"lat": PILOT_LAT, "lon": PILOT_LON, "groundElevation": ground,
                   "launchHeadingDeg": LAUNCH_HEADING, "eyeHeight": EYE_HEIGHT}, f,
                  indent=2)


if __name__ == "__main__":
    main()
