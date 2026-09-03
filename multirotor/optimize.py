"""Multi-start bounded optimization of the motor/propeller design point.

Maximizes thrust-to-weight ratio at full throttle subject to floors/ceilings
on current draw, spin-up responsiveness, and motor stator size, all as
squared penalties so the whole problem stays differentiable. Same shape as
airfoil/optimize.py and linkage/fastopt.py: Adam over a scatter of starts for
the global search, then a short LBFGSB polish on the best few.

Run with: uv run python -m multirotor.optimize
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

import multirotor.quad_model as qm

N_STARTS = 256
ADAM_STEPS = 400
ADAM_LR = 0.06
N_POLISH = 8
POLISH_ITERS = 30

# Box for each design variable, in DESIGN_VARS order. Wide enough to cover
# the range of realistic tiny-whoop/toothpick-class motors and props (see
# quad_model.py's docstring for why this is the target class); the lower
# bound on stator_volume_mm3 is set to the same 1002 floor the cost function
# already enforces as a penalty, so the box and the constraint agree.
BOUNDS = {
    "kv": (3000.0, 30000.0),
    "stator_volume_mm3": (qm.STATOR_VOLUME_FLOOR_MM3, 800.0),
    "prop_diameter_m": (0.04, 0.09),
    "blade_count": (2.0, 4.0),
    "pitch_m": (0.015, 0.08),
}


def _bounds_arrays():
    lower = jnp.asarray([BOUNDS[k][0] for k in qm.DESIGN_VARS], dtype=jnp.float64)
    upper = jnp.asarray([BOUNDS[k][1] for k in qm.DESIGN_VARS], dtype=jnp.float64)
    return lower, upper


@jax.jit
def _solve(x0s, lower, upper):
    def fun(x):
        return qm.cost(x)

    grads = jax.vmap(jax.grad(fun))
    scale = jnp.maximum(upper - lower, 1e-9)

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
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(lower), np.asarray(upper)
    scatter = rng.random((N_STARTS - 1, lo.size)) * (hi - lo) + lo
    base = np.clip(np.asarray(qm.BASELINE), lo, hi)
    return jnp.asarray(np.vstack([base, scatter]))


def optimize():
    lower, upper = _bounds_arrays()
    started = time.perf_counter()
    best_x, best_cost = _solve(_starts(lower, upper), lower, upper)
    best_x.block_until_ready()
    return {
        "x": best_x,
        "cost": float(best_cost),
        "elapsed": time.perf_counter() - started,
    }


def _print_report(x):
    r = qm.evaluate(x)
    c = qm.constraints(x)
    g = qm.unpack(x)

    print(f"\n{'design variable':<24}{'value':>14}{'at bound':>10}")
    for name, value in zip(qm.DESIGN_VARS, np.asarray(x)):
        lo, hi = BOUNDS[name]
        at = "lower" if abs(value - lo) < 1e-6 * max(abs(lo), 1.0) else (
            "upper" if abs(value - hi) < 1e-6 * max(abs(hi), 1.0) else "")
        print(f"  {name:<22}{value:14.5f}{at:>10}")

    print(f"\n{'quantity':<24}{'value':>14}")
    rows = [
        ("resistance, ohm", float(r["unit"]["resistance"]), "{:.4f}"),
        ("motor mass, g", float(r["unit"]["motor_mass"]) * 1e3, "{:.2f}"),
        ("prop mass, g", float(r["unit"]["prop_mass"]) * 1e3, "{:.2f}"),
        ("frame mass (est.), g", float(r["frame_mass"]) * 1e3, "{:.2f}"),
        ("total mass, g", float(r["total_mass"]) * 1e3, "{:.1f}"),
        ("rpm @ full throttle", float(r["rpm"]), "{:.0f}"),
        ("thrust/motor, N", float(r["thrust_n"]), "{:.3f}"),
        ("current/motor, A", float(r["current_a"]), "{:.2f}"),
        ("efficiency", float(r["efficiency"]), "{:.3f}"),
        ("temperature, C (unconstrained)", float(r["temperature_c"]), "{:.1f}"),
        ("TWR", float(r["twr"]), "{:.2f}"),
        ("spin-up 10-90%, ms", float(r["spin_up_s"]) * 1e3, "{:.1f}"),
        ("tip Mach @ full throttle", float(r["tip_mach"]), "{:.3f}"),
    ]
    for label, value, fmt in rows:
        print(f"  {label:<32}{fmt.format(value):>12}")

    print(f"\n{'constraint':<24}{'slack':>14}")
    for name, value in c.items():
        v = float(value)
        mark = "" if v >= -1e-9 else "   <-- VIOLATED"
        print(f"  {name:<22}{v:14.4f}{mark}")


def main():
    print("Optimizing: maximize TWR subject to current, spin-up, and")
    print("stator-size floors.\n")
    print(f"  {N_STARTS} starts, {ADAM_STEPS} Adam steps, "
          f"{N_POLISH} polished for {POLISH_ITERS} iterations")

    result = optimize()
    print(f"  converged in {result['elapsed']:.1f} s  (cost {result['cost']:.4f})")

    _print_report(result["x"])


if __name__ == "__main__":
    main()
