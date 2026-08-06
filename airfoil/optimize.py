"""Multi-start bounded optimization of the tailsitter geometry.

Minimizes stall speed subject to floors on hover control authority, cruise yaw
authority, thrust-to-weight, tip Reynolds number, and packaging, all as squared
penalties so the whole problem stays differentiable.

Same shape as the linkage optimizer: Adam over a scatter of starts does the
global search, then a short LBFGSB polish on the best few.  Adam costs one
gradient per step where LBFGSB's line search costs several, so doing the bulk of
the work in Adam is much cheaper for the same optimum.

Run with ``uv run python -m airfoil.optimize``.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

import airfoil.airfoil_model as am

# Where the optimum is cached.  The solve takes several seconds, which is too
# long to repeat every time something wants to draw or report the result, and
# the answer only changes when the model or the bounds do.
RESULT_PATH = Path(__file__).parent / "optimum.json"

N_STARTS = 256
ADAM_STEPS = 400
ADAM_LR = 0.06
N_POLISH = 8
POLISH_ITERS = 30

# Box for each design variable, in DESIGN_VARS order.  These are what the
# optimizer is allowed to consider, so they encode real limits rather than
# taste: the root chord floor is the battery, the thickness floors are what the
# servo and battery need, and the hinge bounds keep the elevon between 10% and
# 40% of chord where thin-airfoil flap theory is still meaningful.
BOUNDS = {
    "root_chord": (0.070, am.MAX_CHORD),
    "tip_chord": (0.035, am.MAX_CHORD),
    "root_thickness": (0.010, 0.030),
    "tip_thickness": (0.003, 0.020),
    "x_hinge": (0.60, 0.90),
    "elevon_inboard_frac": (0.10, 0.70),
    "motor_frac": (0.20, 0.80),
    "servo_station": (0.20, 0.70),
    "servo_span_frac": (0.15, 0.85),
}


def _bounds_arrays():
    lower = jnp.asarray([BOUNDS[k][0] for k in am.DESIGN_VARS], dtype=jnp.float64)
    upper = jnp.asarray([BOUNDS[k][1] for k in am.DESIGN_VARS], dtype=jnp.float64)
    return lower, upper


@jax.jit
def _solve(x0s, lower, upper):
    def fun(x):
        return am.cost(x)

    grads = jax.vmap(jax.grad(fun))
    scale = upper - lower

    def adam_step(state, k):
        x, m, v, t = state
        # Work in box-normalized units so variables with very different ranges
        # (a chord in metres against a hinge fraction) take comparable steps.
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

    costs = jax.vmap(fun)(xs)
    costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
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


def _starts(lower, upper):
    """The baseline plus a deterministic scatter over the box."""
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(lower), np.asarray(upper)
    scatter = rng.random((N_STARTS - 1, lo.size)) * (hi - lo) + lo
    base = np.clip(np.asarray(am.BASELINE), lo, hi)
    return jnp.asarray(np.vstack([base, scatter]))


def optimize(save=True):
    """Best geometry found, with its constraint slacks."""
    lower, upper = _bounds_arrays()
    started = time.perf_counter()
    best_x, best_cost = _solve(_starts(lower, upper), lower, upper)
    best_x.block_until_ready()
    result = {
        "x": best_x,
        "cost": float(best_cost),
        "elapsed": time.perf_counter() - started,
    }
    if save:
        RESULT_PATH.write_text(json.dumps({
            "design_vars": list(am.DESIGN_VARS),
            "x": [float(v) for v in np.asarray(best_x)],
            "cost": result["cost"],
        }, indent=2) + "\n")
    return result


def load_optimum():
    """The cached optimum, or None if the optimizer has not been run.

    Returned as a plain array in DESIGN_VARS order.  The variable names are
    stored alongside it and checked, so a stale cache from before a design
    variable was added or reordered is rejected rather than silently
    misinterpreted as a valid geometry.
    """
    if not RESULT_PATH.exists():
        return None
    data = json.loads(RESULT_PATH.read_text())
    if data.get("design_vars") != list(am.DESIGN_VARS):
        return None
    return jnp.asarray(data["x"], dtype=jnp.float64)


def _print_comparison(x):
    """Baseline against optimum, on the quantities that drove the design."""
    rows = [
        ("stall speed, m/s", lambda r: float(r["v_stall"]), "{:.2f}"),
        ("area, cm^2", lambda r: float(r["area"]) * 1e4, "{:.1f}"),
        ("aspect ratio", lambda r: float(r["aspect_ratio"]), "{:.2f}"),
        ("mass, g", lambda r: float(r["mass"]) * 1e3, "{:.1f}"),
        ("TWR", lambda r: float(r["twr"]), "{:.2f}"),
        ("Re at tip", lambda r: float(r["re_tip"]), "{:,.0f}"),
        ("root t/c, %", lambda r: float(r["root_tc"]) * 100, "{:.1f}"),
        ("tip t/c, %", lambda r: float(r["tip_tc"]) * 100, "{:.1f}"),
        ("volume slack, mm", lambda r: float(r["volume_slack"]) * 1e3, "{:+.1f}"),
        ("hover pitch, rad/s^2",
         lambda r: abs(float(r["authority"]["hover"]["alpha_pitch"])), "{:.0f}"),
        ("hover roll, rad/s^2",
         lambda r: abs(float(r["authority"]["hover"]["alpha_roll"])), "{:.0f}"),
        ("cruise yaw, rad/s^2",
         lambda r: abs(float(r["authority"]["cruise"]["alpha_yaw"])), "{:.1f}"),
    ]
    base_r = am.evaluate(am.BASELINE)
    opt_r = am.evaluate(x)

    print(f"\n{'quantity':<24}{'baseline':>12}{'optimum':>12}")
    for label, fn, fmt in rows:
        print(f"  {label:<22}{fmt.format(fn(base_r)):>12}"
              f"{fmt.format(fn(opt_r)):>12}")

    print(f"\n{'design variable':<24}{'baseline':>12}{'optimum':>12}"
          f"{'at bound':>10}")
    for i, name in enumerate(am.DESIGN_VARS):
        b, o = float(am.BASELINE[i]), float(x[i])
        lo, hi = BOUNDS[name]
        at = "lower" if abs(o - lo) < 1e-6 else ("upper" if abs(o - hi) < 1e-6 else "")
        print(f"  {name:<22}{b:12.4f}{o:12.4f}{at:>10}")

    print(f"\n{'constraint':<24}{'shortfall':>12}")
    for name, value in am.constraints(x).items():
        v = float(value)
        mark = "" if v <= 1e-9 else "   <-- VIOLATED"
        print(f"  {name:<22}{v:12.4f}{mark}")


def main():
    print("Optimizing: minimize stall speed subject to authority, TWR,")
    print("Reynolds, and packaging floors.\n")
    print(f"  {N_STARTS} starts, {ADAM_STEPS} Adam steps, "
          f"{N_POLISH} polished for {POLISH_ITERS} iterations")

    result = optimize()
    print(f"  converged in {result['elapsed']:.1f} s"
          f"  (cost {result['cost']:.4f})")

    _print_comparison(result["x"])

    print("\nOptimum as a design vector:")
    print("BASELINE = jnp.array([")
    for name, value in zip(am.DESIGN_VARS, np.asarray(result["x"])):
        print(f"    {value:.4f},   # {name}")
    print("])")


if __name__ == "__main__":
    main()
