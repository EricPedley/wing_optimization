"""Refit prop_aero_model.py's CL_ALPHA and INDUCED_POWER_FACTOR against the
full real bench dataset gathered in multirotor/data/tmotor_*.csv -- ~15 real
props from 25mm to 101.6mm pitch, 31mm to 76.5mm diameter, replacing the
original 2-point calibration (45mm and 50.78mm props only).

Two separate fits, same split the original calibration used:
  - CL_ALPHA from thrust vs rpm (static thrust is a pure function of
    (rpm, diameter, pitch, blade_count) in this model -- no motor electricals
    needed at all).
  - INDUCED_POWER_FACTOR from torque vs rpm, backed out of bench current via
    each motor's own Kt = 8.3/kV (motor_model.KT_NUMERATOR) -- this DOES need
    a real per-row (kV, I0) to convert current to torque, so only rows whose
    motor has a resistance-free electrical spec (kV, I0) usable are included.
    R itself is not needed for this: motor_torque's I0 subtraction only needs
    kV and I0, not R.

Both fits drop low-throttle rows (<=30%) as ESC deadband, matching the
original calibration's approach ("both static throttle sweeps with the
low-throttle points dropped").

Run with: uv run python -m multirotor.calibrate_prop_aero
"""

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import multirotor.motor_model as mm
import multirotor.prop_aero_model as pa
from multirotor.data.prop_name_aliases import PROP_NAME_TO_CATALOGUE

DATA_DIR = Path(__file__).parent / "data"
MIN_THROTTLE_PCT = 30

# (csv filename, prop column name, throttle column name, rpm column name,
#  thrust column name, motor name/kv columns or None, current column name)
BENCH_FILES = [
    "tmotor_m1103_throttle_sweep.csv",
    "tmotor_m1104_throttle_sweep.csv",
    "tmotor_m1106_throttle_sweep.csv",
    "tmotor_m0803ii_throttle_sweep.csv",
    "tmotor_f1203_throttle_sweep.csv",
    "tmotor_f1204_throttle_sweep.csv",
    "tmotor_f1303_throttle_sweep.csv",
    "tmotor_f1404_throttle_sweep.csv",
]

# Motor idle/no-load current I0 (A) and kV, read off each datasheet's own
# spec table (see motor_datasheets.csv notes for each) -- needed to convert
# bench current into mechanical torque via motor_current_from_torque's
# inverse. Keyed by (motor, kv) as they appear in the throttle-sweep CSVs.
MOTOR_KV_I0 = {
    ("M1103", 8000): (8000, 0.48),
    ("M1103", 11000): (11000, 0.52),
    ("M1104", 7500): (7500, 0.44),
    ("M1106", 6000): (6000, 0.4),
    ("M0803II", 22000): (22000, 1.5),
    ("M0803II", 25000): (25000, 2.1),
    ("M0803II", 27000): (27000, 2.3),
    ("F1203", 7000): (7000, 0.64),
    ("F1204", 5000): (5000, 0.7),
    ("F1204", 6500): (6500, 0.9),
    ("F1303", 5000): (5000, 0.32),
    ("F1404", 2900): (2900, 0.4),
}


def _prop_geometry(prop_datasheets):
    by_name = {row["name"]: row for row in prop_datasheets}
    return by_name


# Bench series excluded outright: (throttle-sweep filename, prop label as it
# appears in that file's "prop" column). Found by cross-checking props of
# near-identical geometry across independent datasheets -- this one gives
# MORE thrust at a given rpm than a mechanically similar 3-blade prop
# (HQ3018, same 76.2mm diameter/1.8in pitch) from a different T-Motor
# datasheet, backwards from what blade count should do. Either a
# transcription error on this session's part or a real error on T-Motor's
# own page (both have happened before in this dataset -- see
# tmotor_m1103_throttle_sweep.csv's known-bad RPM row) -- not re-verified
# against the source page, so excluded rather than guessed at.
EXCLUDED_BENCH_SERIES = {
    ("tmotor_f1203_throttle_sweep.csv", "G3018-2"),
}


def load_bench_rows():
    with open(DATA_DIR / "prop_datasheets.csv", newline="") as f:
        prop_rows = list(csv.DictReader(f))
    props_by_name = _prop_geometry(prop_rows)

    rows = []
    for fname in BENCH_FILES:
        with open(DATA_DIR / fname, newline="") as f:
            for r in csv.DictReader(f):
                if int(float(r["throttle_pct"])) <= MIN_THROTTLE_PCT:
                    continue
                prop_label = r["prop"]
                if (fname, prop_label) in EXCLUDED_BENCH_SERIES:
                    continue
                catalogue_name = PROP_NAME_TO_CATALOGUE.get(prop_label)
                if catalogue_name is None or catalogue_name not in props_by_name:
                    continue
                prop = props_by_name[catalogue_name]
                if not prop["pitch_mm"]:
                    continue
                motor = r["motor"]
                kv = int(float(r["kv"]))
                key = (motor, kv)
                rows.append({
                    "file": fname,
                    "motor": motor,
                    "kv": kv,
                    "prop_label": prop_label,
                    "catalogue_name": catalogue_name,
                    "diameter_m": float(prop["diameter_mm"]) * 1e-3,
                    "pitch_m": float(prop["pitch_mm"]) * 1e-3,
                    "blade_count": float(prop["blade_count"]),
                    "rpm": float(r["rpm"]),
                    "thrust_g": float(r["thrust_g"]),
                    "current_a": float(r["current_a"]),
                    "motor_key": key,
                })
    return rows


def _grid_search_1d(vectorized_error_fn, lo, hi, n_candidates=201, n_refine=8):
    """Coarse-to-fine grid search over a scalar parameter, vectorized: each
    round evaluates all candidates in one vmapped call instead of looping in
    Python, so this is fast even over hundreds of bench rows."""
    for _ in range(n_refine):
        candidates = jnp.linspace(lo, hi, n_candidates)
        errors = vectorized_error_fn(candidates)
        best_i = int(jnp.argmin(errors))
        best = float(candidates[best_i])
        span = (hi - lo) / n_candidates * 2
        lo, hi = max(0.01, best - span), best + span
    return best


def fit_cl_alpha(rows, chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO):
    """Least-squares CL_ALPHA from bench thrust vs rpm across every row,
    holding CHORD_TO_DIAMETER_RATIO fixed (same degeneracy the original
    calibration notes -- only their product is identifiable from static
    thrust data alone).

    bemt_thrust_torque's static (vel=0) thrust is NOT linear in cl_alpha (u
    depends on cl_alpha through the quadratic solve), so this is a 1D
    nonlinear least-squares fit via a vmapped coarse-to-fine grid search
    rather than a closed form -- simple, robust, and this only needs to run
    once.
    """
    rpm = jnp.array([r["rpm"] for r in rows])
    diameter_m = jnp.array([r["diameter_m"] for r in rows])
    pitch_m = jnp.array([r["pitch_m"] for r in rows])
    blade_count = jnp.array([r["blade_count"] for r in rows])
    thrust_g_bench = jnp.array([r["thrust_g"] for r in rows])

    def sq_error_for_one(cl_alpha):
        thrust_n, _ = jax.vmap(
            lambda rp, d, p, b: pa.bemt_thrust_torque(
                rp, 0.0, d, p, b, chord_to_diameter_ratio, cl_alpha, pa.CD0,
                pa.INDUCED_POWER_FACTOR)
        )(rpm, diameter_m, pitch_m, blade_count)
        predicted_g = thrust_n * 1000.0 / 9.81
        return jnp.sum((predicted_g - thrust_g_bench) ** 2)

    vectorized_error_fn = jax.jit(jax.vmap(sq_error_for_one))
    return _grid_search_1d(vectorized_error_fn, 0.5, 12.0)


def fit_induced_power_factor(rows, cl_alpha, chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO):
    """Least-squares INDUCED_POWER_FACTOR from bench torque (backed out of
    current via each row's motor Kt/I0) vs rpm, holding the just-fit CL_ALPHA
    fixed. Only rows whose motor has a known (kV, I0) in MOTOR_KV_I0 are
    used.
    """
    usable = [r for r in rows if r["motor_key"] in MOTOR_KV_I0]
    torque_rows = []
    for r in usable:
        kv, i0 = MOTOR_KV_I0[r["motor_key"]]
        kt = mm.KT_NUMERATOR / kv
        # motor_current_from_torque(torque, kv) = torque * kv / KT_NUMERATOR, so
        # its inverse is torque = current * KT_NUMERATOR / kv = current * kt.
        # I0 subtracts off as friction current before this conversion (see
        # motor_model.motor_torque).
        current_after_friction = max(r["current_a"] - i0, 1e-6)
        torque_nm = current_after_friction * kt
        torque_rows.append({**r, "torque_nm": torque_nm})

    rpm = jnp.array([r["rpm"] for r in torque_rows])
    diameter_m = jnp.array([r["diameter_m"] for r in torque_rows])
    pitch_m = jnp.array([r["pitch_m"] for r in torque_rows])
    blade_count = jnp.array([r["blade_count"] for r in torque_rows])
    torque_nm_bench = jnp.array([r["torque_nm"] for r in torque_rows])

    def sq_error_for_one(induced_power_factor):
        _, torque_n = jax.vmap(
            lambda rp, d, p, b: pa.bemt_thrust_torque(
                rp, 0.0, d, p, b, chord_to_diameter_ratio, cl_alpha, pa.CD0,
                induced_power_factor)
        )(rpm, diameter_m, pitch_m, blade_count)
        return jnp.sum((torque_n - torque_nm_bench) ** 2)

    vectorized_error_fn = jax.jit(jax.vmap(sq_error_for_one))
    best = _grid_search_1d(vectorized_error_fn, 0.3, 5.0)
    return best, len(torque_rows)


def fit_cl_alpha_per_prop(rows):
    """CL_ALPHA fit independently for each distinct prop, to check whether a
    single-global-CL_ALPHA fit's large per-prop residuals are explained by a
    real per-prop CL_ALPHA that varies with geometry (e.g. pitch/diameter
    ratio) rather than just being noise a global fit averages over."""
    results = []
    for name in sorted(set(r["catalogue_name"] for r in rows)):
        prop_rows = [r for r in rows if r["catalogue_name"] == name]
        cl_alpha = fit_cl_alpha(prop_rows)
        pd_ratio = prop_rows[0]["pitch_m"] / prop_rows[0]["diameter_m"]
        results.append((name, cl_alpha, pd_ratio, prop_rows[0]["diameter_m"] * 1e3,
                         prop_rows[0]["blade_count"], len(prop_rows)))
    return results


def _report_residuals(subset, cl_alpha, induced_power_factor):
    print(f"\n{'prop':<30}{'n':>4}{'mean thrust err %':>19}{'worst thrust err %':>20}")
    for name in sorted(set(r["catalogue_name"] for r in subset)):
        prop_rows = [r for r in subset if r["catalogue_name"] == name]
        errs = []
        for r in prop_rows:
            thrust_n, _ = pa.bemt_thrust_torque(
                r["rpm"], 0.0, r["diameter_m"], r["pitch_m"], r["blade_count"],
                pa.CHORD_TO_DIAMETER_RATIO, cl_alpha, pa.CD0, induced_power_factor)
            predicted_g = float(thrust_n) * 1000.0 / 9.81
            errs.append(100.0 * (predicted_g - r["thrust_g"]) / r["thrust_g"])
        mean_err = sum(errs) / len(errs)
        worst_err = max(errs, key=abs)
        print(f"{name:<30}{len(prop_rows):>4}{mean_err:>18.1f}%{worst_err:>19.1f}%")


# Deliberately narrowed calibration subset -- see prop_aero_model.py's
# CL_ALPHA/INDUCED_POWER_FACTOR docstrings for the full story on why: no
# single CL_ALPHA reconciles the full ~300-row/18-prop dataset (per-prop
# fits range from ~1.4 to >50), so rather than force a global number that
# fits nothing well, this module fits only the props physically similar to
# what this design's optimizer actually searches (diameter capped at 76mm --
# see quad_model.MAX_PROP_DIAMETER_M -- and a moderate pitch/diameter ratio).
SUBSET_DIAMETER_MM_RANGE = (60.0, 77.0)
SUBSET_PITCH_TO_DIAMETER_RANGE = (0.55, 1.05)


def narrowed_subset(rows):
    lo_d, hi_d = SUBSET_DIAMETER_MM_RANGE
    lo_pd, hi_pd = SUBSET_PITCH_TO_DIAMETER_RANGE
    return [
        r for r in rows
        if lo_d <= r["diameter_m"] * 1e3 <= hi_d
        and lo_pd <= r["pitch_m"] / r["diameter_m"] <= hi_pd
    ]


def main():
    rows = load_bench_rows()
    print(f"Loaded {len(rows)} usable bench rows (throttle > {MIN_THROTTLE_PCT}%) "
          f"across {len(set(r['catalogue_name'] for r in rows))} distinct props "
          f"from {len(BENCH_FILES)} datasheets.\n")

    print("=== Diagnostic: per-prop CL_ALPHA (does one global value make sense?) ===")
    per_prop = fit_cl_alpha_per_prop(rows)
    print(f"{'prop':<30}{'CL_ALPHA':>10}{'P/D':>7}{'diam_mm':>9}{'blades':>7}{'n':>5}")
    for name, cl_alpha, pd, diam, blades, n in sorted(per_prop, key=lambda x: x[2]):
        flag = "  <- hit search ceiling" if cl_alpha > 11.9 else ""
        print(f"{name:<30}{cl_alpha:>10.3f}{pd:>7.2f}{diam:>9.1f}{blades:>7.0f}{n:>5}{flag}")
    print("\nRange spans ~1.4 to >50 (ceiling-limited) with no clean split by "
          "diameter or P/D alone -- no single global CL_ALPHA is defensible "
          "across this full dataset. See prop_aero_model.py's calibration "
          "comment for the full reasoning.\n")

    print(f"=== Actual calibration: narrowed subset "
          f"(diameter {SUBSET_DIAMETER_MM_RANGE}mm, P/D {SUBSET_PITCH_TO_DIAMETER_RANGE}) ===")
    subset = narrowed_subset(rows)
    print(f"{len(subset)} rows across {len(set(r['catalogue_name'] for r in subset))} props: "
          f"{sorted(set(r['catalogue_name'] for r in subset))}\n")

    cl_alpha = fit_cl_alpha(subset)
    induced_power_factor, n_torque_rows = fit_induced_power_factor(subset, cl_alpha)
    print(f"CL_ALPHA = {cl_alpha:.4f} (prop_aero_model.py currently: {pa.CL_ALPHA})")
    print(f"INDUCED_POWER_FACTOR = {induced_power_factor:.4f} "
          f"(prop_aero_model.py currently: {pa.INDUCED_POWER_FACTOR}), "
          f"from {n_torque_rows} rows with known motor I0/kV")
    _report_residuals(subset, cl_alpha, induced_power_factor)


if __name__ == "__main__":
    main()
