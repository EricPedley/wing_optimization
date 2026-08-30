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

# Blade count is physically discrete -- a real propeller has 2 or 3 blades,
# not 2.4 -- so it is not searched continuously like the other four design
# variables. Instead the app runs one full rollout of the continuous
# optimizer per candidate blade count (see optimize_over_blade_counts) and
# keeps whichever converges to the better cost. Kept to the two realistic
# small-prop options; every real propeller this session gathered bench data
# against was 2 or 3 blades.
BLADE_COUNT_CANDIDATES = [2.0, 3.0]

# The four quantities a user can choose as a *constraint* (and, other than
# current_at_throttle below, also as the objective). Order matters -- it is
# the index the constraint arrays (c_enabled/c_threshold/c_lambda) are keyed
# by. "sense" is which direction is "better", which decides whether a
# non-objective row is a floor or a ceiling.
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

# A fifth, objective-only choice: minimize current at a throttle *other*
# than full throttle (current_a above is always full-throttle, deliberately,
# since that's the worst case for the ESC). This does not appear in
# QUANTITIES -- it isn't something you'd constrain (there's no natural
# "keep current-at-50%-throttle under X" floor/ceiling the way TWR or
# full-throttle current are), only something you'd minimize, typically
# while holding TWR to a floor via QUANTITIES' own constraint row. It needs
# an extra parameter (which throttle) that the other four don't, which is
# why it is not just a fifth QUANTITIES entry.
OBJECTIVES = QUANTITIES + ["current_at_throttle"]
OBJECTIVE_LABELS = dict(QUANTITY_LABELS, current_at_throttle="Current at a chosen throttle (A)")
DEFAULT_OBJECTIVE_THROTTLE_FRAC = 0.5

# The augmented Lagrangian's fixed penalty weight -- see cost_configurable's
# docstring. Unlike a plain quadratic penalty's weight, this does NOT need to
# be pushed to infinity to get exact constraint satisfaction; the multiplier
# update in optimize() does that job instead. 1e3 just needs to be large
# enough that the inner LBFGSB/Adam solve feels the constraint at all.
AL_RHO = 1e3
# Outer augmented-Lagrangian iterations: the first uses c_lambda=0 (a plain
# quadratic penalty, i.e. exactly the old behaviour), and each subsequent one
# updates c_lambda by dual ascent and re-solves. Multiplier convergence for a
# handful of inequality constraints on a smooth low-dimensional problem is
# fast -- 5 has been enough in testing to drive a previously-persistent
# violation to numerical noise.
AL_OUTER_ITERS = 5

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


def constraint_violations(x, c_threshold, vbat, other_mass_kg, cl_alpha, induced_power_factor):
    """c_i(x), one per QUANTITIES entry: <= 0 means satisfied, > 0 means
    violated by that much (in units of QUANTITY_PENALTY_SCALE[q]). Shared by
    cost_configurable (which penalizes it) and the augmented-Lagrangian
    multiplier update in optimize() (which needs the raw violation, not the
    penalized cost)."""
    r = evaluate_configurable(x, vbat, other_mass_kg, cl_alpha, induced_power_factor)
    values = jnp.stack([r["twr"], r["current_a"], r["spin_up_s"], r["tip_mach"]])
    scales = jnp.asarray([QUANTITY_PENALTY_SCALE[q] for q in QUANTITIES])
    # Floor for twr (violated when value < threshold), ceiling for the rest
    # (violated when value > threshold).
    shortfall = jnp.stack([
        c_threshold[0] - values[0],
        values[1] - c_threshold[1],
        values[2] - c_threshold[2],
        values[3] - c_threshold[3],
    ])
    return shortfall / scales


def cost_configurable(x, objective_code, c_enabled, c_threshold, c_lambda,
                       objective_throttle_frac,
                       vbat, other_mass_kg, cl_alpha, induced_power_factor):
    """Scalar augmented-Lagrangian objective. objective_code indexes
    OBJECTIVES (0=twr, 1=current_a, 2=spin_up_s, 3=tip_mach,
    4=current_at_throttle); the objective quantity is maximized (twr) or
    minimized (everything else). c_enabled/c_threshold/c_lambda are length-4
    arrays, one entry per QUANTITIES entry -- current_at_throttle has no
    constraint row (see OBJECTIVES' docstring in this module), so it never
    appears in those three. objective_throttle_frac only matters when
    objective_code == 4; it is current_at_throttle_a's throttle fraction.

    This is not a plain quadratic exterior penalty: a fixed-weight quadratic
    penalty (cost = -raw + W*max(0, violation)^2) only enforces a binding
    constraint approximately, with residual violation at the optimum that
    shrinks as W grows but never actually reaches zero for finite W. Worse,
    that residual moves around when an *unrelated* constraint changes --
    tightening or loosening one constraint shifts where the whole penalized
    objective's unconstrained optimum sits, which can visibly worsen a
    different constraint's residual even though nothing about it changed.
    (This is exactly what motivated adding c_lambda: relaxing the spin-up
    threshold in the app made the current constraint start reading as
    slightly violated, with nothing about the current constraint itself
    having moved.)

    The fix is the standard one for this failure mode -- the Hestenes-Powell-
    Rockafellar augmented Lagrangian for inequality constraints:

        term_i = (1 / (2*RHO)) * (max(0, c_lambda_i + RHO*c_i(x))^2 - c_lambda_i^2)

    which reduces to the plain quadratic penalty when c_lambda_i = 0 (the
    first optimizer call in optimize()'s outer loop), but as c_lambda_i is
    updated by dual ascent (c_lambda_i <- max(0, c_lambda_i + RHO*c_i(x)) in
    optimize()) across a few outer iterations, the *linear* term it
    introduces lets the true constrained optimum be reached exactly, for a
    fixed, finite RHO -- no need to send the penalty weight to infinity.
    c_lambda_i converges to (an estimate of) the constraint's actual
    Lagrange multiplier: economically, "how much the objective would improve
    per unit this constraint were relaxed."

    All selector values are traced, not Python bools/ints, so one compiled
    function serves every combination of objective and enabled constraints.
    """
    r = evaluate_configurable(x, vbat, other_mass_kg, cl_alpha, induced_power_factor)
    values = jnp.stack([r["twr"], r["current_a"], r["spin_up_s"], r["tip_mach"]])

    current_at_throttle = qm.current_at_throttle_a(
        x, objective_throttle_frac, vel=0.0, vbat=vbat, other_mass_kg=other_mass_kg,
        cl_alpha=cl_alpha, induced_power_factor=induced_power_factor)

    # "raw" is what we maximize: the quantity itself for twr, its negative
    # for every lower-is-better quantity (including current_at_throttle),
    # selected by objective_code.
    raw = jnp.where(
        objective_code == 0, values[0],
        jnp.where(objective_code == 1, -values[1],
                  jnp.where(objective_code == 2, -values[2],
                            jnp.where(objective_code == 3, -values[3],
                                      -current_at_throttle))))

    c = constraint_violations(x, c_threshold, vbat, other_mass_kg, cl_alpha, induced_power_factor)
    al_term = (jnp.maximum(0.0, c_lambda + AL_RHO * c) ** 2 - c_lambda ** 2) / (2.0 * AL_RHO)
    penalty = jnp.sum(c_enabled * al_term)

    return -raw + penalty


@jax.jit
def _solve(x0s, lower, upper, objective_code, c_enabled, c_threshold, c_lambda,
           objective_throttle_frac, vbat, other_mass_kg, cl_alpha, induced_power_factor):
    def fun(x):
        return cost_configurable(x, objective_code, c_enabled, c_threshold, c_lambda,
                                  objective_throttle_frac, vbat, other_mass_kg, cl_alpha,
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


def _with_throttle_metric(metrics, x, throttle_frac, assumption_args):
    """best_metrics/start_metrics only come from evaluate_configurable, which
    knows nothing about current_at_throttle_a (it takes an extra argument
    evaluate() doesn't have) -- add it in so the app can display it
    regardless of which objective was actually run."""
    metrics = dict(metrics)
    metrics["current_at_throttle"] = float(
        qm.current_at_throttle_a(x, throttle_frac, vel=0.0, vbat=assumption_args[0],
                                  other_mass_kg=assumption_args[1], cl_alpha=assumption_args[2],
                                  induced_power_factor=assumption_args[3]))
    return metrics


def optimize(values, modes, ranges, objective, enabled, thresholds, assumptions,
              objective_throttle_frac=DEFAULT_OBJECTIVE_THROTTLE_FRAC):
    """values/modes/ranges: dicts keyed by DESIGN_VARS, same contract as
    linkage/fastopt.optimize. objective: one of OBJECTIVES or "none".
    enabled/thresholds: dicts keyed by QUANTITIES. assumptions: dict with
    keys vbat, other_mass_kg, cl_alpha, induced_power_factor.
    objective_throttle_frac: throttle fraction for the "current_at_throttle"
    objective; ignored for every other objective.
    """
    started = time.perf_counter()
    base = np.array([float(values[k]) for k in DESIGN_VARS], dtype=np.float64)

    a = {k: float(assumptions.get(k, ASSUMPTION_DEFAULTS[k])) for k in ASSUMPTION_DEFAULTS}
    assumption_args = (a["vbat"], a["other_mass_kg"], a["cl_alpha"],
                        a["induced_power_factor"])
    throttle_frac = jnp.asarray(float(objective_throttle_frac), dtype=jnp.float64)

    start_metrics = _with_throttle_metric(
        _to_floats(evaluate_configurable(jnp.asarray(base), *assumption_args)),
        jnp.asarray(base), throttle_frac, assumption_args)

    n_free = sum(1 for k in DESIGN_VARS if modes.get(k, "fixed") != "fixed")

    c_enabled = jnp.asarray([1.0 if enabled.get(q) else 0.0 for q in QUANTITIES],
                             dtype=jnp.float64)
    c_threshold = jnp.asarray([float(thresholds.get(q, QUANTITY_DEFAULT_THRESHOLD[q]))
                                for q in QUANTITIES], dtype=jnp.float64)

    zero_lambda = jnp.zeros(4, dtype=jnp.float64)
    objective_code_for_cost = jnp.asarray(
        OBJECTIVES.index(objective) if objective in OBJECTIVES else 0, dtype=jnp.int32)
    base_cost = float(cost_configurable(jnp.asarray(base), objective_code_for_cost,
                                         c_enabled, c_threshold, zero_lambda, throttle_frac,
                                         *assumption_args))

    if objective == "none":
        return {"values": dict(zip(DESIGN_VARS, base.tolist())),
                "start_metrics": start_metrics, "best_metrics": start_metrics,
                "elapsed": 0.0, "n_free": n_free, "message": "Optimization off.",
                "final_cost": base_cost}
    if n_free == 0:
        return {"values": dict(zip(DESIGN_VARS, base.tolist())),
                "start_metrics": start_metrics, "best_metrics": start_metrics,
                "elapsed": time.perf_counter() - started, "n_free": 0,
                "message": "No free variables — every parameter is set to '='.",
                "final_cost": base_cost}

    box = [var_bounds(modes[k], base[i], float(ranges[k][0]), float(ranges[k][1]))
           for i, k in enumerate(DESIGN_VARS)]
    lower = jnp.asarray([b[0] for b in box], dtype=jnp.float64)
    upper = jnp.asarray([b[1] for b in box], dtype=jnp.float64)

    objective_code = jnp.asarray(OBJECTIVES.index(objective), dtype=jnp.int32)
    x0 = jnp.clip(jnp.asarray(base, dtype=jnp.float64), lower, upper)

    # Augmented-Lagrangian outer loop -- see cost_configurable's docstring.
    # The first iteration (c_lambda all zero) is exactly the old plain
    # quadratic-penalty solve; each subsequent one re-solves with the
    # multiplier updated by dual ascent, converging constraint violation to
    # (numerical) zero rather than leaving a residual that depends on how
    # hard the objective happens to be pulling, or on unrelated constraints'
    # thresholds.
    c_lambda = jnp.zeros(4, dtype=jnp.float64)
    x_current = x0
    for _outer in range(AL_OUTER_ITERS):
        x_current, _ = _solve(_starts(x_current, lower, upper), lower, upper,
                               objective_code, c_enabled, c_threshold, c_lambda,
                               throttle_frac, *assumption_args)
        x_current.block_until_ready()
        c_val = constraint_violations(x_current, c_threshold, *assumption_args)
        c_lambda = jnp.maximum(0.0, c_lambda + AL_RHO * c_val) * c_enabled

    # Compare the AL-converged point against the untouched starting design
    # under the SAME final multiplier, so "did this help" is judged
    # consistently rather than across different penalty functions.
    final_cost = float(cost_configurable(x_current, objective_code, c_enabled,
                                          c_threshold, c_lambda, throttle_frac, *assumption_args))
    start_cost = float(cost_configurable(x0, objective_code, c_enabled,
                                          c_threshold, c_lambda, throttle_frac, *assumption_args))
    if not np.isfinite(final_cost) or final_cost > start_cost:
        best_x, message, final_cost = x0, "No improvement found.", start_cost
    else:
        best_x = x_current
        message = f"Converged ({N_STARTS} starts x {AL_OUTER_ITERS} AL iterations)."

    best = np.asarray(best_x)
    best_metrics = _with_throttle_metric(
        _to_floats(evaluate_configurable(best_x, *assumption_args)),
        best_x, throttle_frac, assumption_args)
    return {"values": dict(zip(DESIGN_VARS, best.tolist())),
            "start_metrics": start_metrics,
            "best_metrics": best_metrics,
            "elapsed": time.perf_counter() - started,
            "n_free": n_free, "message": message,
            "final_cost": final_cost}


def _to_floats(metrics):
    return {k: (float(v) if not isinstance(v, dict) else _to_floats(v))
            for k, v in metrics.items()}


def optimize_over_blade_counts(values, modes, ranges, objective, enabled, thresholds,
                                assumptions, blade_counts=BLADE_COUNT_CANDIDATES,
                                objective_throttle_frac=DEFAULT_OBJECTIVE_THROTTLE_FRAC):
    """Runs optimize() once per candidate blade count (see
    BLADE_COUNT_CANDIDATES), each with blade_count forced to "fixed" at that
    value regardless of what mode the caller passed for it, and returns the
    winner (lowest final_cost) plus every rollout for display.

    If the user's own blade_count constraint mode was already "fixed", every
    rollout still runs (comparing a fixed value against itself is cheap and
    keeps this function's contract simple), but they will all naturally
    converge to the same design except for the forced blade_count -- so the
    UI does not need a special case for "user already fixed it".
    """
    results = []
    for bc in blade_counts:
        rollout_values = dict(values, blade_count=bc)
        rollout_modes = dict(modes, blade_count="fixed")
        result = optimize(rollout_values, rollout_modes, ranges, objective,
                           enabled, thresholds, assumptions, objective_throttle_frac)
        result["blade_count"] = bc
        results.append(result)

    best = min(results, key=lambda r: r["final_cost"])
    return best, results


def warmup():
    """Trigger compilation once so the first real request is not the slow one."""
    values = dict(zip(DESIGN_VARS, [v for v in qm.BASELINE]))
    ranges = {"kv": (3000.0, 30000.0), "stator_volume_mm3": (150.0, 800.0),
              "prop_diameter_m": (0.02, qm.MAX_PROP_DIAMETER_M), "blade_count": (2.0, 3.0),
              "pitch_m": (0.01, 0.08)}
    optimize_over_blade_counts(
        values, {k: "free" for k in DESIGN_VARS}, ranges, "twr",
        {"current_a": True, "spin_up_s": True, "tip_mach": True},
        QUANTITY_DEFAULT_THRESHOLD, ASSUMPTION_DEFAULTS)
