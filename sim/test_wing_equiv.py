"""C++ <-> Python equivalence test for the tailsitter aero model.

Builds a grid of aerospace-frame states (AoA +/-180 deg, sideslip, body
rates, elevon deflections, motor thrust), runs the C++ model via
`simitl-wing forces`, and compares against sim/wing_model.py.

Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest sim/test_wing_equiv.py -q
or directly:
    uv run python -m sim.test_wing_equiv
"""

import os
import subprocess
import sys

import numpy as np

from sim import wing_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "SimITL", "build", "linux", "tools",
                    "simitl-wing", "simitl-wing")
AS_BUILT = os.path.join(ROOT, "sim", "as_built.json")


def make_grid():
    """Rows: u,v,w,p,q,r,dl,dr,Tl,Tr  (aero frame, deg for dl/dr, N thrust)."""
    rows = []
    # AoA sweep -180..180 at 6 m/s
    for a_deg in range(-180, 181, 10):
        a = np.deg2rad(a_deg)
        v = 6.0
        rows.append([v * np.cos(a), 0.0, v * np.sin(a), 0, 0, 0,
                     0, 0, 0, 0])
    # sideslip sweep
    for b_deg in range(-30, 31, 15):
        b = np.deg2rad(b_deg)
        rows.append([6.0, 6.0 * np.tan(b), 0.5, 0, 0, 0, 0, 0, 0, 0])
    # hover + climb/descent columns
    for w in (-8.0, -4.0, -1.0, 1.0, 4.0, 8.0):
        rows.append([0.05, 0.0, w, 0, 0, 0, 0, 0, 0.3, 0.3])
    # body rates
    for rate in (-6.0, -2.0, 2.0, 6.0):
        rows.append([4.0, 0.0, 0.4, rate, 0, 0, 0, 0, 0, 0])   # roll
        rows.append([4.0, 0.0, 0.4, 0, rate, 0, 0, 0, 0, 0])   # pitch
        rows.append([4.0, 0.0, 0.4, 0, 0, rate, 0, 0, 0, 0])   # yaw
        rows.append([0.0, 0.0, -3.0, rate, 0, 0, 0, 0, 0.4, 0.4])
    # elevon sweeps at cruise and in the slipstream
    for d in (-18, -10, -5, 0, 5, 10, 18):
        rows.append([6.0, 0.0, 0.4, 0, 0, 0, d, d, 0, 0])
        rows.append([6.0, 0.0, 0.4, 0, 0, 0, d, -d, 0, 0])
        rows.append([0.5, 0.0, 2.0, 0, 0, 0, d, d, 0.4, 0.4])
    # asymmetric thrust in hover-ish conditions
    for tl, tr in ((0.0, 0.0), (0.3, 0.1), (0.6, 0.2), (0.0, 0.5)):
        rows.append([0.1, 0.0, 3.0, 0, 0, 0, 5, -5, tl, tr])
    # a few random states
    rng = np.random.default_rng(7)
    for _ in range(40):
        a = rng.uniform(-np.pi, np.pi)
        b = rng.uniform(-0.5, 0.5)
        v = rng.uniform(0.1, 12.0)
        u = v * np.cos(a) * np.cos(b)
        vv = v * np.sin(b)
        w = v * np.sin(a) * np.cos(b)
        p, q, r = rng.uniform(-8, 8, 3)
        dl, dr = rng.uniform(-18, 18, 2)
        tl, tr = rng.uniform(0, 0.7, 2)
        rows.append([u, vv, w, p, q, r, dl, dr, tl, tr])
    return np.array(rows)


def run_cpp(grid, out_csv):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        np.savetxt(f.name, grid, delimiter=",",
                   header="u,v,w,p,q,r,dl,dr,Tl,Tr", comments="")
        grid_path = f.name
    r = subprocess.run(
        [TOOL, "forces", "--wing", AS_BUILT, "--grid", grid_path,
         "--out", out_csv],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"simitl-wing failed: {r.stderr[-2000:]}")
    return np.loadtxt(out_csv, delimiter=",", skiprows=1)


def compare(grid):
    p, strips = wing_model.load_as_built(AS_BUILT)
    out_csv = os.path.join(ROOT, "sim", "forces_cpp.csv")
    data = run_cpp(grid, out_csv)

    errs_f, errs_m = [], []
    worst = (0.0, None)
    for row_in, row_out in zip(grid, data):
        u, v, w, pr, q, rr, dl, dr, tl, tr = row_in
        F, M = wing_model.forces_moments(
            p, strips, np.array([u, v, w]), np.array([pr, q, rr]),
            dl, dr, (tr, tl))          # python takes (right, left)
        Fc, Mc = row_out[10:13], row_out[13:16]
        ef = np.linalg.norm(F - Fc)
        em = np.linalg.norm(M - Mc)
        errs_f.append(ef)
        errs_m.append(em)
        scale = max(np.linalg.norm(F), 1e-3)
        if ef / scale > worst[0]:
            worst = (ef / scale, (row_in, F, Fc))
    return np.array(errs_f), np.array(errs_m), worst, data


def test_cpp_python_equivalence():
    grid = make_grid()
    errs_f, errs_m, worst, _ = compare(grid)

    f_tol = 1e-5   # N - implementations should be numerically identical
    m_tol = 1e-7   # N.m
    max_f = errs_f.max()
    max_m = errs_m.max()
    if max_f > f_tol or max_m > m_tol:
        row, F, Fc = worst[1]
        raise AssertionError(
            f"max force err {max_f:.3e} N, moment err {max_m:.3e} N.m\n"
            f"worst state {row}\n  py {F}\n  cpp {Fc}")


if __name__ == "__main__":
    grid = make_grid()
    errs_f, errs_m, worst, data = compare(grid)
    print(f"{len(grid)} states")
    print(f"max |F| err: {errs_f.max():.3e} N   mean {errs_f.mean():.3e}")
    print(f"max |M| err: {errs_m.max():.3e} N.m mean {errs_m.mean():.3e}")
    row, F, Fc = worst[1]
    print("worst state:", row)
    print("  py :", F)
    print("  cpp:", Fc)
