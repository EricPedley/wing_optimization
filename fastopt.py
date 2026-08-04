"""Multi-start bounded optimization of the linkage geometry, jitted end to end.

Every design variable is always passed to the solver; a "locked" one simply gets
a zero-width box.  Keeping the array shapes and the solver configuration fixed
means one compiled function serves every combination of constraint modes,
objectives, and toggles, so only the first call pays for compilation.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

import fastmodel as fm
from fastmodel import N_SWEEP  # noqa: F401  (re-exported for callers)

DESIGN_VARS = ["servo_x", "servo_y", "servo_travel", "flap_x", "flap_y"]
CONSTRAINT_MODES = ["fixed", "le", "ge", "free"]
OBJECTIVES = ["none", "area", "peak", "min"]
_OBJECTIVE_CODE = {"none": 0, "area": 1, "peak": 2, "min": 3}

# Adam over a wide scatter of starts does the global search; a handful of the
# best points then get an LBFGSB polish.  Adam costs exactly one gradient per
# step, where LBFGSB's line search costs several, so doing the bulk of the work
# in Adam is ~10x cheaper for the same optimum.
N_STARTS = 128
ADAM_STEPS = 300
ADAM_LR = 0.08
N_POLISH = 8
# LBFGSB stops on its own convergence test, and under a vmap the loop runs until
# the *slowest* of the batch is done, so an uncapped polish costs ~3x for optima
# identical to Adam's.  A short cap keeps it purely as a safety net.
POLISH_ITERS = 12
# Zero-width boxes upset the projection, so a locked variable gets a hair of slack.
LOCK_EPS = 1e-9


def var_bounds(mode, value, lo, hi):
    """Box for one design variable given its constraint mode."""
    value = float(value)
    if mode == "fixed":
        return (value - LOCK_EPS, value + LOCK_EPS)
    if mode == "le":
        return (lo, max(value, lo + LOCK_EPS))
    if mode == "ge":
        return (min(value, hi - LOCK_EPS), hi)
    return (lo, hi)


@jax.jit
def _solve(x0s, lower, upper, objective, w_zero, w_sym, target, ref):
    def fun(x):
        return fm.cost(x, objective, w_zero, w_sym, target, ref)

    grads = jax.vmap(jax.grad(fun))
    scale = upper - lower

    def adam_step(state, k):
        x, m, v, t = state
        # Work in box-normalized units so variables with very different ranges
        # (servo travel vs flap offset) take comparably sized steps.
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
    # Keep the Adam point if the polish somehow made things worse.
    allx = jnp.concatenate([xs, px])
    allc = jnp.concatenate([costs, pcosts])
    best = jnp.argmin(allc)
    return allx[best], allc[best]


def _starts(x0, lower, upper):
    """The current geometry plus a deterministic scatter over the box."""
    rng = np.random.default_rng(0)
    scatter = rng.random((N_STARTS - 1, x0.size)) * (upper - lower) + lower
    return jnp.asarray(np.vstack([x0, scatter]))


def _to_floats(metrics):
    return {k: float(v) for k, v in metrics.items()}


def optimize(values, modes, ranges, objective, soft, min_max_angle=None):
    """Same contract as :func:`optimize.optimize`, but gradient-based and jitted."""
    import time

    started = time.perf_counter()
    base = np.array([float(values[k]) for k in DESIGN_VARS], dtype=np.float64)
    start_metrics = _to_floats(fm.metrics_jit(jnp.asarray(base)))

    n_free = sum(1 for k in DESIGN_VARS if modes.get(k, "fixed") != "fixed")
    if objective == "none":
        return {"values": dict(zip(DESIGN_VARS, base)), "start_metrics": start_metrics,
                "best_metrics": start_metrics, "elapsed": 0.0,
                "n_free": n_free, "message": "Optimization off."}
    if n_free == 0:
        return {"values": dict(zip(DESIGN_VARS, base)), "start_metrics": start_metrics,
                "best_metrics": start_metrics,
                "elapsed": time.perf_counter() - started, "n_free": 0,
                "message": "No free variables — every parameter is set to '='."}

    box = [var_bounds(modes[k], base[i], float(ranges[k][0]), float(ranges[k][1]))
           for i, k in enumerate(DESIGN_VARS)]
    # Everything below must be float64: an int-typed bound would be a different
    # jit signature and would silently trigger a full recompile.
    lower = jnp.asarray([b[0] for b in box], dtype=jnp.float64)
    upper = jnp.asarray([b[1] for b in box], dtype=jnp.float64)

    ref = abs(start_metrics[objective]) or 1.0
    args = (jnp.asarray(_OBJECTIVE_CODE[objective], dtype=jnp.int32),
            jnp.asarray(1.0 if "peak_at_zero" in soft else 0.0, dtype=jnp.float64),
            jnp.asarray(1.0 if "symmetric_ends" in soft else 0.0, dtype=jnp.float64),
            jnp.asarray(float(min_max_angle or 0.0), dtype=jnp.float64),
            jnp.asarray(float(ref), dtype=jnp.float64))

    x0 = jnp.clip(jnp.asarray(base, dtype=jnp.float64), lower, upper)
    best_x, best_cost = _solve(_starts(np.asarray(x0), np.asarray(lower),
                                       np.asarray(upper)), lower, upper, *args)
    best_x.block_until_ready()

    start_cost = float(fm.cost_jit(x0, *args))
    if not np.isfinite(float(best_cost)) or float(best_cost) > start_cost:
        best_x, message = x0, "No improvement found."
    else:
        message = f"Converged ({N_STARTS} starts)."

    best = np.asarray(best_x)
    return {"values": dict(zip(DESIGN_VARS, best.tolist())),
            "start_metrics": start_metrics,
            "best_metrics": _to_floats(fm.metrics_jit(best_x)),
            "elapsed": time.perf_counter() - started,
            "n_free": n_free, "message": message}


def warmup():
    """Trigger compilation once so the first real request is not the slow one."""
    values = dict(zip(DESIGN_VARS, [15.0, 6.0, 9.0, 0.0, 10.0]))
    ranges = {"servo_x": (10.0, 20.0), "servo_y": (0.0, 10.0),
              "servo_travel": (5.0, 15.0), "flap_x": (-5.0, 20.0),
              "flap_y": (5.0, 10.0)}
    optimize(values, {k: "free" for k in DESIGN_VARS}, ranges, "area",
             ["peak_at_zero", "symmetric_ends"], min_max_angle=20.0)
