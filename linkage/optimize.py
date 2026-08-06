"""Geometry optimization for the flap-servo linkage.

Design variables are the five geometry sliders.  Each one carries a constraint
mode relative to its current slider value:

* ``fixed``   - pinned at the slider value, not a design variable
* ``le``      - free, but must stay at or below the slider value
* ``ge``      - free, but must stay at or above the slider value
* ``free``    - free anywhere inside the slider's own min/max range

Objectives maximize the area under the |torque/force| curve, its peak value, or
its minimum value (a maximin on worst-case advantage).  Two optional soft
constraints are folded in as penalties.
"""

import time

import numpy as np
from scipy.optimize import minimize

from linkage.core import auto_rod_length, simulate_flap, torque_force_ratio

DESIGN_VARS = ["servo_x", "servo_y", "servo_travel", "flap_x", "flap_y"]

CONSTRAINT_MODES = ["fixed", "le", "ge", "free"]

OBJECTIVES = ["none", "area", "peak", "min"]

# Objectives that map to a metric key and are maximized.
SCORED_OBJECTIVES = ("area", "peak", "min")

# Weight on each enabled soft constraint, and on losing part of the servo sweep.
PENALTY_WEIGHT = 1.0
INVALID_WEIGHT = 10.0
# The minimum-flap-angle requirement is a real constraint rather than a
# preference, so it outweighs any objective gain a shortfall could buy.
ANGLE_WEIGHT = 20.0


def var_bounds(mode, value, lo, hi):
    """Bounds for one design variable, or None if it is pinned."""
    if mode == "fixed":
        return None
    if mode == "le":
        return (lo, float(value))
    if mode == "ge":
        return (float(value), hi)
    return (lo, hi)


def evaluate(servo_x, servo_y, servo_travel, flap_x, flap_y,
             n_samples: int = 41, grid_points: int = 200):
    """Metrics for one candidate geometry, or None if it has no usable sweep."""
    rod_length = auto_rod_length(servo_x, servo_y, servo_travel, flap_x, flap_y,
                                 samples=30, grid_points=120)
    u = np.linspace(0.0, 1.0, n_samples)
    res = simulate_flap(u, servo_x, servo_y, servo_travel, flap_x, flap_y,
                        rod_length, grid_points=grid_points)

    ratio = torque_force_ratio(res, servo_travel)
    theta = res["flap_angle_rad"]
    ok = res["valid"] & np.isfinite(ratio)
    if ok.sum() < 3:
        return None

    u_ok = u[ok]
    mag = np.abs(ratio[ok])
    th = theta[ok]

    area = float(np.trapezoid(mag, u_ok))
    peak_i = int(np.argmax(mag))
    lo_i = int(np.argmin(th))
    hi_i = int(np.argmax(th))

    return {
        "rod_length": rod_length,
        "area": area,
        "peak": float(mag[peak_i]),
        # Worst-case advantage anywhere in the sweep; maximizing it is a maximin.
        "min": float(np.min(mag)),
        "theta_at_peak": float(th[peak_i]),
        "angle_span": float(th[hi_i] - th[lo_i]),
        # Largest deflection magnitude the linkage reaches, in degrees.  Taken
        # as |theta| so the measure does not depend on which way the flap swings.
        "max_angle_deg": float(np.degrees(np.max(np.abs(th)))),
        "mag_at_min_angle": float(mag[lo_i]),
        "mag_at_max_angle": float(mag[hi_i]),
        "valid_fraction": float(ok.sum()) / n_samples,
    }


def _penalties(m, soft):
    """Dimensionless penalty terms for the enabled soft constraints."""
    total = 0.0
    if "peak_at_zero" in soft:
        # Peak mechanical advantage should sit at flap angle 0.  Scale the
        # offset by half the swept angle range so the term is unitless.
        half_span = 0.5 * abs(m["angle_span"])
        total += (m["theta_at_peak"] / half_span) ** 2 if half_span > 1e-9 else 1.0
    if "symmetric_ends" in soft:
        lo, hi = m["mag_at_min_angle"], m["mag_at_max_angle"]
        mean = 0.5 * (lo + hi)
        total += ((lo - hi) / mean) ** 2 if mean > 1e-9 else 1.0
    return total


def _angle_penalty(m, min_max_angle):
    """One-sided shortfall against the required peak flap angle, in relative terms."""
    if not min_max_angle:
        return 0.0
    s = max(0.0, min_max_angle - m["max_angle_deg"]) / min_max_angle
    # Linear term as well as quadratic: a purely quadratic penalty flattens out
    # at the boundary, which leaves the optimizer parked just short of target.
    return s + s ** 2


def _cost(m, objective, soft, ref, min_max_angle=None):
    """Scalar cost to minimize.  ``ref`` normalizes the objective term."""
    if m is None:
        return 1e6
    raw = m[objective] if objective in SCORED_OBJECTIVES else 0.0
    score = raw / ref if ref > 1e-12 else raw
    return (
        -score
        + PENALTY_WEIGHT * _penalties(m, soft)
        + ANGLE_WEIGHT * _angle_penalty(m, min_max_angle)
        + INVALID_WEIGHT * (1.0 - m["valid_fraction"])
    )


def optimize(values, modes, ranges, objective, soft, min_max_angle=None,
             maxfev: int = 400, scan_points: int = 80):
    """Search for a better geometry.

    Parameters
    ----------
    values : dict
        Current slider value per name in ``DESIGN_VARS``.
    modes : dict
        Constraint mode per name (see ``CONSTRAINT_MODES``).
    ranges : dict
        ``(min, max)`` slider range per name.
    objective : str
        One of ``OBJECTIVES``.
    soft : sequence of str
        Enabled soft constraints: ``peak_at_zero``, ``symmetric_ends``.
    min_max_angle : float or None
        If set, the peak flap angle (degrees) the linkage must reach.

    Returns
    -------
    dict with ``values``, ``start_metrics``, ``best_metrics``, ``elapsed``,
    ``n_free``, ``message``.
    """
    start = time.perf_counter()
    base = {k: float(values[k]) for k in DESIGN_VARS}
    start_metrics = evaluate(**base)

    free = [k for k in DESIGN_VARS if modes.get(k, "fixed") != "fixed"]
    bounds = [var_bounds(modes[k], base[k], *ranges[k]) for k in free]

    if objective == "none":
        return {"values": base, "start_metrics": start_metrics,
                "best_metrics": start_metrics, "elapsed": 0.0,
                "n_free": len(free), "message": "Optimization off."}
    if not free:
        return {"values": base, "start_metrics": start_metrics,
                "best_metrics": start_metrics,
                "elapsed": time.perf_counter() - start, "n_free": 0,
                "message": "No free variables — every parameter is set to '='."}

    # Normalize against the starting geometry so the objective and the
    # penalties stay on comparable scales.
    ref = abs(start_metrics[objective]) if start_metrics else 1.0

    def wrapped(x):
        cand = dict(base)
        for k, xi, (lo, hi) in zip(free, x, bounds):
            cand[k] = float(np.clip(xi, lo, hi))
        return _cost(evaluate(**cand), objective, soft, ref, min_max_angle)

    # Clamp the start into its bounds; a 'le'/'ge' mode makes the slider value
    # itself an endpoint, which Powell tolerates fine.
    x0 = np.array([np.clip(base[k], lo, hi) for k, (lo, hi) in zip(free, bounds)])

    # Powell alone gets stuck on this surface once more than a couple of
    # variables are free, so scatter a seeded sample over the box first and
    # polish from whichever point looks best.
    rng = np.random.default_rng(0)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    scan = np.vstack([x0, lows + rng.random((scan_points, len(free))) * (highs - lows)])
    scan_costs = np.array([wrapped(x) for x in scan])
    x_seed = scan[int(np.argmin(scan_costs))]

    result = minimize(wrapped, x_seed, method="Powell", bounds=bounds,
                      options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4})

    best = dict(base)
    for k, xi, (lo, hi) in zip(free, np.atleast_1d(result.x), bounds):
        best[k] = float(np.clip(xi, lo, hi))
    best_metrics = evaluate(**best)

    # Powell can return a point worse than the start if the surface is rough.
    if best_metrics is None or _cost(
        best_metrics, objective, soft, ref, min_max_angle
    ) > _cost(start_metrics, objective, soft, ref, min_max_angle):
        best, best_metrics = base, start_metrics
        msg = f"No improvement found ({scan.shape[0] + result.nfev} evals)."
    else:
        msg = f"Converged in {scan.shape[0] + result.nfev} evals."

    return {"values": best, "start_metrics": start_metrics,
            "best_metrics": best_metrics, "elapsed": time.perf_counter() - start,
            "n_free": len(free), "message": msg}
