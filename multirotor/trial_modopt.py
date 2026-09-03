"""Trial: replace quad_model's squared-penalty constraints with modOpt's
JaxProblem + SLSQP, using real inequality constraints instead of penalty terms.

Does not modify quad_model.py or optimize.py -- reuses qm.evaluate() for the
objective (-TWR) and qm.constraints() (already named slacks, >=0 feasible)
mapped directly onto JaxProblem's cl/cu bounds. Single-start from the same
baseline optimize.py uses, just to see whether modOpt's SLSQP converges to a
comparable optimum with cleaner constraint handling (no penalty weights/scales
to tune).

Run with: uv run python -m multirotor.trial_modopt
"""

import jax
import jax.numpy as jnp
import numpy as np
import modopt as mo

import multirotor.quad_model as qm
from multirotor.optimize import BOUNDS

jax.config.update("jax_enable_x64", True)

CONSTRAINT_NAMES = ("current_slack_a", "spinup_slack_s", "volume_slack_mm3", "tip_mach_slack")


def jax_obj(x):
    return -qm.evaluate(x)["twr"]


def jax_con(x):
    c = qm.constraints(x)
    return jnp.array([c[name] for name in CONSTRAINT_NAMES])


def main():
    lower = np.array([BOUNDS[k][0] for k in qm.DESIGN_VARS])
    upper = np.array([BOUNDS[k][1] for k in qm.DESIGN_VARS])
    x0 = np.clip(np.asarray(qm.BASELINE, dtype=float), lower, upper)

    prob = mo.JaxProblem(
        x0=x0,
        nc=len(CONSTRAINT_NAMES),
        jax_obj=jax_obj,
        jax_con=jax_con,
        xl=lower,
        xu=upper,
        cl=np.zeros(len(CONSTRAINT_NAMES)),
        cu=np.full(len(CONSTRAINT_NAMES), np.inf),
        name="quad_design",
        order=1,
    )

    optimizer = mo.SLSQP(prob, solver_options={"maxiter": 200, "ftol": 1e-8})
    optimizer.solve()
    optimizer.print_results()

    x_star = np.asarray(optimizer.results["x"])
    r = qm.evaluate(jnp.asarray(x_star))
    c = qm.constraints(jnp.asarray(x_star))

    print("\ndesign point:")
    for name, value in zip(qm.DESIGN_VARS, x_star):
        print(f"  {name:<22}{value:14.5f}")

    print(f"\nTWR: {float(r['twr']):.4f}")
    print("\nconstraint slacks (>=0 feasible):")
    for name in CONSTRAINT_NAMES:
        v = float(c[name])
        mark = "" if v >= -1e-6 else "   <-- VIOLATED"
        print(f"  {name:<22}{v:14.4f}{mark}")


if __name__ == "__main__":
    main()
