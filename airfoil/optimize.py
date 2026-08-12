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

import hashlib
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
    # Motor and servo spanwise positions are capped by how far the existing
    # wiring reaches.  Expressed as span fractions here because that is what the
    # design vector holds; the constraints check the same limits in metres, so a
    # violation is caught even if these bounds are widened.
    "motor_frac": (0.20, am.MAX_MOTOR_Y / (0.5 * am.SPAN)),
    "servo_chord_frac": (0.20, 0.70),
    "servo_span_frac": (0.15, am.MAX_SERVO_Y / (0.5 * am.SPAN)),
    # Leading-edge sweep.  Nothing in the objective rewards or penalizes it --
    # there is no stability model -- so the optimizer will leave it wherever it
    # starts.  The bounds are what keeps it in the range a tailless aircraft is
    # normally built in, and the reported quarter-chord sweep is the number to
    # watch: with taper it sits well forward of the leading-edge value.
    "le_sweep_deg": (0.0, 45.0),
    # Battery chordwise position, as the chord fraction of its forward face.
    # Bounded away from the leading edge because the nose has no depth there at
    # all, and away from the trailing edge because the battery still has to fit
    # ahead of it.
    "battery_station": (0.02, 0.40),
    # Camber, as Birnbaum-Glauert coefficients.  These are not directly
    # readable as a shape -- see the camber section of the model -- so the
    # bounds are set by what they produce rather than by what they look like:
    # the box spans roughly 0 to 4% camber, which covers everything from a
    # symmetric section to about as much camber as is useful at this Reynolds
    # number.
    #
    # A1 is floored at zero because negative camber is not a design this
    # aircraft wants.  A2 is allowed negative so the optimizer can reach
    # ordinary (nose-down) camber lines as well as reflexed ones: reflex is
    # A2 > A1, and pinning A2 positive would have quietly forbidden half the
    # space including the conventional cambered section.
    "camber_a1": (0.0, 0.15),
    "camber_a2": (-0.05, 0.15),
    # --- Linkage, all in millimetres.  These feed linkage_model directly, which
    # works in mm; airfoil.linkage_coupling is where the two unit systems meet.
    #
    # Servo body depth off the hinge axis.  Capped by how deep the section is
    # where the servo sits.  The servo_rail_depth constraint enforces that
    # properly against the real local thickness; this box is only a sane outer
    # limit so the optimizer does not waste starts on absurd geometry.
    "servo_height": (0.0, 6.0),
    # Where the control rod attaches on the servo end, as an offset from the
    # body depth above.  A short range because the pickup is a feature on the
    # servo arm, not a free-floating point: it is somewhere near the body, above
    # or below it, but not far from it.
    "servo_rod_dy": (-3.0, 3.0),
    # Fixed at the servo's actual stroke.  A degenerate box rather than a
    # special case, so DESIGN_VARS, BOUNDS, and unpack all stay uniform and
    # widening it later is a one-line edit.  _solve guards the zero width.
    "servo_travel_mm": (9.0, 9.0),
    # Horn offset along the flap.  The hinge-side attachment is nearly free --
    # it is a printed feature on a part that does not have to fit inside
    # anything -- so these bounds are deliberately loose and exist only to keep
    # the search in a region where a horn is still a horn.
    "flap_x_mm": (-15.0, 15.0),
    # Horn radius.  The design variable of this set: advantage scales with it
    # and throw inversely, their product pinned near servo_travel by virtual
    # work.  Floored at 5 mm because the linkage goes dead below that.  The
    # ceiling is structural, not aerodynamic -- a control horn is meant to stand
    # proud of the surface, so what limits it is how long a printed horn can be
    # before it flexes under the hinge load, not the section depth.
    "flap_y_mm": (5.0, 18.0),
}


def _fingerprint():
    """Hash of everything that decides what the optimum is.

    The cache is only valid for the model that produced it, and the thing that
    invalidates it in practice is not a new design variable -- that is rare and
    the name list already catches it -- but an edited constant.  Changing a
    servo force or a constraint floor leaves the design vector the same shape
    while moving the answer, so without this the stale optimum is reloaded and
    silently re-reported against limits it was never optimized for.

    Every upper-case module-level name in the model is included rather than a
    hand-picked list, because a hand-picked list is exactly the kind of thing
    that stops being complete the first time someone adds a constant and does
    not think of it.  The cost of over-including is a re-solve after an edit
    that did not matter; the cost of under-including is a wrong answer that
    looks right.  The bounds go in for the same reason.
    """
    parts = []
    # The linkage model is part of the answer now, so its constants belong in
    # the hash for the same reason the airfoil model's do.
    for module in (am, am.linkage_coupling, am.linkage_coupling.lm):
        for name in sorted(dir(module)):
            if not name.isupper() or name.startswith("_"):
                continue
            value = getattr(module, name)
            if hasattr(value, "tolist"):    # jnp arrays, e.g. BASELINE
                value = value.tolist()
            parts.append(f"{module.__name__}.{name}={value!r}")
    parts.append(f"BOUNDS={sorted(BOUNDS.items())!r}")
    # The search itself changes the answer too: fewer starts or steps can land
    # in a different basin, so a cache from a cheaper run is not interchangeable
    # with one from a thorough one.
    parts.append(f"SEARCH={(N_STARTS, ADAM_STEPS, ADAM_LR, N_POLISH, POLISH_ITERS)!r}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _bounds_arrays():
    lower = jnp.asarray([BOUNDS[k][0] for k in am.DESIGN_VARS], dtype=jnp.float64)
    upper = jnp.asarray([BOUNDS[k][1] for k in am.DESIGN_VARS], dtype=jnp.float64)
    return lower, upper


@jax.jit
def _solve(x0s, lower, upper):
    def fun(x):
        return am.cost(x)

    grads = jax.vmap(jax.grad(fun))
    # Guarded because a pinned variable (lower == upper, which is how a fixed
    # quantity like the servo stroke is expressed) would otherwise divide the
    # gradient by zero.  That does not just break the pinned variable: the inf
    # propagates through the shared Adam moment state and NaNs the whole vector.
    # The clip below pins the value exactly regardless of what scale says.
    scale = jnp.maximum(upper - lower, 1e-9)

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
            "fingerprint": _fingerprint(),
            "x": [float(v) for v in np.asarray(best_x)],
            "cost": result["cost"],
        }, indent=2) + "\n")
    return result


def load_optimum():
    """The cached optimum, or None if the optimizer has not been run.

    Returned as a plain array in DESIGN_VARS order.  The variable names are
    stored alongside it and checked, so a stale cache from before a design
    variable was added or reordered is rejected rather than silently
    misinterpreted as a valid geometry.  A fingerprint of the model constants
    and the bounds is checked for the same reason -- see _fingerprint -- so an
    optimum from before a constant was edited is rejected too.
    """
    if not RESULT_PATH.exists():
        return None
    data = json.loads(RESULT_PATH.read_text())
    if data.get("design_vars") != list(am.DESIGN_VARS):
        return None
    if data.get("fingerprint") != _fingerprint():
        # Say so rather than returning None quietly.  Every caller falls back to
        # the baseline, so a silent rejection looks identical to a plot of the
        # optimum -- which is the same failure this check exists to prevent,
        # just with the baseline in the stale optimum's place.
        print(f"{RESULT_PATH.name} is stale: the model constants or bounds have"
              " changed since it was written.  Falling back to the baseline;"
              " re-run 'python -m airfoil.optimize' to refresh it.")
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
