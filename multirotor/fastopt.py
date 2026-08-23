"""Multi-start bounded optimization of the quad design, jitted end to end,
with the objective, constraints, and physical assumptions all selectable at
call time rather than fixed at import time.

Same shape as linkage/fastopt.py: every design variable is always passed to
the solver (a "locked" one just gets a zero-width box), and the objective
and constraint choices are encoded as traced values rather than Python
branches, so one compiled function serves every combination the app's UI can
produce.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

import multirotor.prop_aero_model as pa
import multirotor.quad_model as qm

DESIGN_VARS = list(qm.DESIGN_VARS)
CONSTRAINT_MODES = ["fixed", "le", "ge", "free"]

# The four quantities a user can choose as the objective or leave as a
# constraint. Order matters -- it is the index objective_code and the
# constraint arrays are keyed by. "sense" is which direction is "better",
# which decides whether a non-objective row is a floor or a ceiling.
QUANTITIES = ["twr", "current_a", "spin_up_s", "tip_mach"]
QUANTITY_SENSE = {"twr": "max", "current_a": "min", "spin_up_s": "min",
                   "tip_mach": "min"}
QUANTITY_LABELS = {"twr": "Thrust-to-weight ratio", "current_a": "Current at full throttle (A)",
                    "spin_up_s": "Spin-up time, 10%→90% throttle (s)",
                    "tip_mach": "Tip Mach at full throttle"}
# Default constraint threshold for each quantity when a user turns its
# constraint on -- reasonable starting points, not claims about what is
# correct for every design.
QUANTITY_DEFAULT_THRESHOLD = {"twr": 4.0, "current_a": 12.0,
                               "spin_up_s": 0.050, "tip_mach": 0.5}
# Normalizes each quantity's constraint shortfall before squaring, so the
# four penalties (a TWR unit, an amp, a second, a Mach number) contribute
# comparably to the gradient. Same role as quad_model.py's *_SCALE constants.
QUANTITY_PENALTY_SCALE = {"twr": 1.0, "current_a": 1.0, "spin_up_s": 0.01,
                           "tip_mach": 0.05}

PENALTY_WEIGHT = 1e3

N_STARTS = 192
ADAM_STEPS = 350
ADAM_LR = 0.07
N_POLISH = 8
POLISH_ITERS = 25
LOCK_EPS = 1e-9

# Assumption defaults, in the order _solve/evaluate_configurable expects
# them: vbat, other_mass_kg, cl_alpha, induced_power_factor.
ASSUMPTION_DEFAULTS = {
    "vbat": qm.VBAT,
    "other_mass_kg": qm.OTHER_MASS_KG,
    "cl_alpha": pa.CL_ALPHA,
    "induced_power_factor": pa.INDUCED_POWER_FACTOR,
}


def var_bounds(mode, value, lo, hi):
    """Box for one design variable given its constraint mode. Same contract
    as linkage/fastopt.py's var_bounds."""
    value = float(value)
    if mode == "fixed":
        return (value - LOCK_EPS, value + LOCK_EPS)
    if mode == "le":
        return (lo, max(value, lo + LOCK_EPS))
    if mode == "ge":
        return (min(value, hi - LOCK_EPS), hi)
    return (lo, hi)


def evaluate_configurable(x, vbat, other_mass_kg, cl_alpha, induced_power_factor):
    """quad_model.evaluate with the tunable assumptions threaded through,
    chord_to_diameter_ratio and cd0 left at their calibrated defaults (not
    exposed in the app -- see prop_aero_model.py for why those two, unlike
    CL_ALPHA and INDUCED_POWER_FACTOR, are not independently identifiable
    from the bench data this session gathered)."""
    return qm.evaluate(x, vel=0.0, vbat=vbat, other_mass_kg=other_mass_kg,
                        cl_alpha=cl_alpha, induced_power_factor=induced_power_factor)


def cost_configurable(x, objective_code, c_enabled, c_threshold,
                       vbat, other_mass_kg, cl_alpha, induced_power_factor):
    """Scalar objective. objective_code indexes QUANTITIES (0=twr, 1=current_a,
    2=spin_up_s, 3=tip_mach); the objective quantity is maximized (twr) or
    minimized (the other three). c_enabled/c_threshold are length-4 arrays,
    one entry per QUANTITIES entry, applied as a floor (twr) or ceiling (the
    other three) penalty whenever c_enabled[i] > 0 -- including on the
    objective's own quantity, though the app's UI does not expose that
    combination since it is redundant with the objective itself.

    All selector values are traced, not Python bools/ints, so one compiled
    function serves every combination of objective and enabled constraints.
    """
    r = evaluate_configurable(x, vbat, other_mass_kg, cl_alpha, induced_power_factor)
    values = jnp.stack([r["twr"], r["current_a"], r["spin_up_s"], r["tip_mach"]])

    # "raw" is what we maximize: the quantity itself for twr, its negative
    # for the three lower-is-better quantities, selected by objective_code.
    raw = jnp.where(
        objective_code == 0, values[0],
        jnp.where(objective_code == 1, -values[1],
                  jnp.where(objective_code == 2, -values[2], -values[3])))

    thresholds = c_threshold
    scales = jnp.asarray([QUANTITY_PENALTY_SCALE[q] for q in QUANTITIES])
    # Floor for twr (violated when value < threshold), ceiling for the rest
    # (violated when value > threshold).
    shortfall = jnp.stack([
        thresholds[0] - values[0],
        values[1] - thresholds[1],
        values[2] - thresholds[2],
        values[3] - thresholds[3],
    ])
    penalty = jnp.sum(c_enabled * PENALTY_WEIGHT
                       * jnp.maximum(shortfall / scales, 0.0) ** 2)

    return -raw + penalty


@jax.jit
def _solve(x0s, lower, upper, objective_code, c_enabled, c_threshold,
           vbat, other_mass_kg, cl_alpha, induced_power_factor):
    def fun(x):
        return cost_configurable(x, objective_code, c_enabled, c_threshold,
                                  vbat, other_mass_kg, cl_alpha,
                                  induced_power_factor)

    grads = jax.vmap(jax.grad(fun))
    scale = jnp.maximum(upper - lower, 1e-12)

    def adam_step(state, k):
        x, m, v, t = state
        g = grads(x) / scale
        t = t + 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mhat = m / (1.0 - 0.9 ** t)
        vhat = v / (1.0 - 0.999 ** t)
        lr = ADAM_LR * 0.5 * (1.0 + jnp.cos(jnp.pi * k / ADAM_STEPS))
        x = jnp.clip(x - lr * scale * mhat / (jnp.sqrt(vhat) + 1e-8), lower, upper)
        return (x, m, v, t), None

    zeros = jnp.zeros_like(x0s)
    (xs, _, _, _), _ = jax.lax.scan(
        adam_step, (x0s, zeros, zeros, 0), jnp.arange(ADAM_STEPS)
    )

    costs = jnp.where(jnp.isfinite(jax.vmap(fun)(xs)), jax.vmap(fun)(xs), jnp.inf)
    _, top = jax.lax.top_k(-costs, N_POLISH)

    def polish(x0):
        res = LBFGSB(fun=fun, maxiter=POLISH_ITERS).run(x0, bounds=(lower, upper))
        x = jnp.clip(res.params, lower, upper)
        return x, fun(x)

    px, pcosts = jax.vmap(polish)(xs[top])
    pcosts = jnp.where(jnp.isfinite(pcosts), pcosts, jnp.inf)

    allx = jnp.concatenate([xs, px])
    allc = jnp.concatenate([costs, pcosts])
    best = jnp.argmin(allc)
    return allx[best], allc[best]


def _starts(x0, lower, upper):
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(lower), np.asarray(upper)
    scatter = rng.random((N_STARTS - 1, lo.size)) * (hi - lo) + lo
    return jnp.asarray(np.vstack([np.asarray(x0), scatter]))


def optimize(values, modes, ranges, objective, enabled, thresholds, assumptions):
    """values/modes/ranges: dicts keyed by DESIGN_VARS, same contract as
    linkage/fastopt.optimize. objective: one of QUANTITIES or "none".
    enabled/thresholds: dicts keyed by QUANTITIES. assumptions: dict with
    keys vbat, other_mass_kg, cl_alpha, induced_power_factor.
    """
    started = time.perf_counter()
    base = np.array([float(values[k]) for k in DESIGN_VARS], dtype=np.float64)

    a = {k: float(assumptions.get(k, ASSUMPTION_DEFAULTS[k])) for k in ASSUMPTION_DEFAULTS}
    assumption_args = (a["vbat"], a["other_mass_kg"], a["cl_alpha"],
                        a["induced_power_factor"])

    start_metrics = _to_floats(evaluate_configurable(jnp.asarray(base), *assumption_args))

    n_free = sum(1 for k in DESIGN_VARS if modes.get(k, "fixed") != "fixed")

    c_enabled = jnp.asarray([1.0 if enabled.get(q) else 0.0 for q in QUANTITIES],
                             dtype=jnp.float64)
    c_threshold = jnp.asarray([float(thresholds.get(q, QUANTITY_DEFAULT_THRESHOLD[q]))
                                for q in QUANTITIES], dtype=jnp.float64)

    if objective == "none":
        return {"values": dict(zip(DESIGN_VARS, base.tolist())),
                "start_metrics": start_metrics, "best_metrics": start_metrics,
                "elapsed": 0.0, "n_free": n_free, "message": "Optimization off."}
    if n_free == 0:
        return {"values": dict(zip(DESIGN_VARS, base.tolist())),
                "start_metrics": start_metrics, "best_metrics": start_metrics,
                "elapsed": time.perf_counter() - started, "n_free": 0,
                "message": "No free variables — every parameter is set to '='."}

    box = [var_bounds(modes[k], base[i], float(ranges[k][0]), float(ranges[k][1]))
           for i, k in enumerate(DESIGN_VARS)]
    lower = jnp.asarray([b[0] for b in box], dtype=jnp.float64)
    upper = jnp.asarray([b[1] for b in box], dtype=jnp.float64)

    objective_code = jnp.asarray(QUANTITIES.index(objective), dtype=jnp.int32)
    x0 = jnp.clip(jnp.asarray(base, dtype=jnp.float64), lower, upper)

    best_x, best_cost = _solve(_starts(x0, lower, upper), lower, upper,
                                objective_code, c_enabled, c_threshold,
                                *assumption_args)
    best_x.block_until_ready()

    start_cost = float(cost_configurable(x0, objective_code, c_enabled, c_threshold,
                                          *assumption_args))
    if not np.isfinite(float(best_cost)) or float(best_cost) > start_cost:
        best_x, message = x0, "No improvement found."
    else:
        message = f"Converged ({N_STARTS} starts)."

    best = np.asarray(best_x)
    return {"values": dict(zip(DESIGN_VARS, best.tolist())),
            "start_metrics": start_metrics,
            "best_metrics": _to_floats(evaluate_configurable(best_x, *assumption_args)),
            "elapsed": time.perf_counter() - started,
            "n_free": n_free, "message": message}


def _to_floats(metrics):
    return {k: (float(v) if not isinstance(v, dict) else _to_floats(v))
            for k, v in metrics.items()}


def warmup():
    """Trigger compilation once so the first real request is not the slow one."""
    values = dict(zip(DESIGN_VARS, [v for v in qm.BASELINE]))
    ranges = {"kv": (3000.0, 30000.0), "stator_volume_mm3": (150.0, 800.0),
              "prop_diameter_m": (0.03, 0.09), "blade_count": (2.0, 5.0),
              "pitch_m": (0.01, 0.08)}
    optimize(values, {k: "free" for k in DESIGN_VARS}, ranges, "twr",
             {"current_a": True, "spin_up_s": True, "tip_mach": True},
             QUANTITY_DEFAULT_THRESHOLD, ASSUMPTION_DEFAULTS)
