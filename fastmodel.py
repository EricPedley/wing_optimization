"""JAX model of the flap linkage: closed-form solve, exact gradients, jitted.

The scipy model in :mod:`core` finds the flap angle by scanning a grid and
running ``brentq`` on every sign change, for every servo input, and then does it
again inside a bracketing search for the rod length.  None of that is necessary.

Writing the rod attachment point as a complex number collapses it::

    P(theta) = e^{-i theta} * (-flap_x + i flap_y)

so |P| = r = hypot(flap_x, flap_y) is constant and arg P = alpha - theta with
alpha = atan2(flap_y, -flap_x).  The rod-length constraint |S - P| = L expands to
|S|^2 - 2 S.P + r^2 = L^2, and S.P = r d cos(alpha - beta - theta) with
d = |S|, beta = atan2(S_y, S_x).  So the whole solve is one arccos::

    theta = alpha - beta -/+ arccos((d^2 + r^2 - L^2) / (2 r d))

The two signs are the two assembly branches; a linkage stays on one of them.
That form is vectorized, jit-compilable, and exactly differentiable, which is
what makes gradient-based optimization of the geometry practical.
"""

from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N_SWEEP = 41
U = jnp.linspace(0.0, 1.0, N_SWEEP)

# Mirrors optimize.py so results stay comparable.
PENALTY_WEIGHT = 1.0
INVALID_WEIGHT = 10.0
ANGLE_WEIGHT = 20.0
# Floor on |dtheta/du| so a dead-point geometry yields a large but finite
# advantage instead of an infinity that would poison the whole cost.
DTHETA_FLOOR = 1e-6
# arccos has an infinite derivative at +/-1, and clipping there would multiply
# that infinity by a zero clip-gradient, giving NaN.  Staying a hair inside keeps
# every derivative finite (bounded by ~1/sqrt(2*ARCCOS_EPS)).
ARCCOS_EPS = 1e-7
# The flap must keep turning the same way as the servo advances.  A sweep that
# stalls or reverses is a dead point: the advantage there is unbounded but the
# flap has stopped, so the design is useless.  Exact gradients find such points
# readily (the old finite differences hid them), so the signed turn rate must
# stay above this fraction of its uniform-sweep value.
#
# The same fraction caps the advantage *and* triggers the penalty, deliberately:
# dipping below it buys no extra advantage (the ratio saturates) yet still costs
# penalty, so "maximize peak advantage" has an interior optimum instead of
# running off to a dead point.  It allows the advantage to vary up to 1/0.25 = 4x
# its uniform-sweep value across the stroke, which is ample for real linkages.
DEAD_FRACTION = 0.25
DEAD_WEIGHT = 20.0


def _theta_and_c(u, p, rod_length, branch):
    """Flap angle at servo input ``u``, plus the cosine whose |.| must be <= 1."""
    servo_x, servo_y, servo_travel, flap_x, flap_y = p
    sx = servo_x + u * servo_travel
    d = jnp.hypot(sx, servo_y)
    beta = jnp.arctan2(servo_y, sx)
    r = jnp.maximum(jnp.hypot(flap_x, flap_y), 1e-9)
    alpha = jnp.arctan2(flap_y, -flap_x)
    c = (d * d + r * r - rod_length * rod_length) / (2.0 * r * d)
    safe = jnp.clip(c, -1.0 + ARCCOS_EPS, 1.0 - ARCCOS_EPS)
    theta = alpha - beta - branch * jnp.arccos(safe)
    return theta, c


def _theta(u, p, rod_length, branch):
    return _theta_and_c(u, p, rod_length, branch)[0]


def _bracket(p):
    """Rod lengths for which the linkage can close over the whole sweep."""
    servo_x, servo_y, servo_travel, flap_x, flap_y = p
    d = jnp.hypot(servo_x + U * servo_travel, servo_y)
    r = jnp.maximum(jnp.hypot(flap_x, flap_y), 1e-9)
    lo = jnp.max(jnp.abs(d - r))
    # A geometry that can never close has lo > hi.  Collapsing the bracket rather
    # than letting it invert keeps the bisection finite; the violation penalty is
    # what actually rejects such a candidate.
    return lo, jnp.maximum(jnp.min(d + r), lo)


def _endpoint_sum(rod_length, p, branch):
    """theta(0) + theta(1); zero when the flap range is symmetric about 0."""
    return (_theta(0.0, p, rod_length, branch)
            + _theta(1.0, p, rod_length, branch))


def _solve_rod_length(p, branch, iters: int = 50):
    """Rod length giving a symmetric flap range, by bisection.

    ``_endpoint_sum`` is monotonic in the rod length, so plain bisection is
    reliable.  Bisection alone has a piecewise-constant derivative, though, so a
    single Newton step is taken afterwards from a stop_gradient'd point: the
    value barely moves (the residual is already ~0) but the derivative becomes
    the implicit-function one, which is what the optimizer needs.
    """
    lo, hi = _bracket(p)
    pad = 1e-6 * (hi - lo) + 1e-9
    a0, b0 = lo + pad, hi - pad

    def body(_, state):
        a, b, fa = state
        m = 0.5 * (a + b)
        fm = _endpoint_sum(m, p, branch)
        same = (fm > 0) == (fa > 0)
        return (jnp.where(same, m, a),
                jnp.where(same, b, m),
                jnp.where(same, fm, fa))

    a, b, _ = jax.lax.fori_loop(
        0, iters, body, (a0, b0, _endpoint_sum(a0, p, branch))
    )
    rod_length = jax.lax.stop_gradient(0.5 * (a + b))

    f = _endpoint_sum(rod_length, p, branch)
    df = jax.grad(_endpoint_sum)(rod_length, p, branch)
    step = -f / jnp.where(jnp.abs(df) < 1e-12, 1e-12, df)
    # Cap the correction so geometries with no symmetric solution (where the
    # residual never reaches zero) cannot be flung out of the feasible bracket.
    step = jnp.clip(step, -0.1 * (hi - lo), 0.1 * (hi - lo))
    return jnp.clip(rod_length + step, a0, b0)


def solve(p):
    """Rod length and assembly branch for a geometry."""
    def score(branch):
        rod_length = _solve_rod_length(p, branch)
        residual = jnp.abs(_endpoint_sum(rod_length, p, branch))
        theta0 = jnp.abs(_theta(0.0, p, rod_length, branch))
        # Prefer a branch that actually achieves symmetry; among those, the one
        # sitting nearest zero degrees, which is how core.py picks its root.
        return rod_length, jnp.where(residual < 1e-8, theta0, 1e6 + residual)

    l_pos, s_pos = score(1.0)
    l_neg, s_neg = score(-1.0)
    take_pos = s_pos <= s_neg
    return jnp.where(take_pos, l_pos, l_neg), jnp.where(take_pos, 1.0, -1.0)


def sweep(p, u=None):
    """Flap angle, advantage, and constraint slack across the servo sweep.

    ``dtheta/du`` is an exact derivative rather than a finite difference, so the
    advantage curve is correct even where it is steep.
    """
    u = U if u is None else u
    rod_length, branch = solve(p)
    theta, c = jax.vmap(_theta_and_c, in_axes=(0, None, None, None))(
        u, p, rod_length, branch
    )
    dtheta = jax.vmap(jax.grad(_theta), in_axes=(0, None, None, None))(
        u, p, rod_length, branch
    )
    span = jnp.max(theta) - jnp.min(theta)
    floor = jnp.maximum(DEAD_FRACTION * jnp.abs(span), DTHETA_FLOOR)
    safe = jnp.sign(dtheta + 1e-30) * jnp.maximum(jnp.abs(dtheta), floor)
    ratio = p[2] / safe
    # How far outside the closable range each sample sits; 0 when reachable.
    violation = jnp.maximum(jnp.abs(c) - 1.0, 0.0)
    return rod_length, theta, ratio, violation, dtheta


def attach_point(theta, p):
    """Rod attachment point in global coordinates, from the same complex form."""
    _, _, _, flap_x, flap_y = p
    r = jnp.hypot(flap_x, flap_y)
    alpha = jnp.arctan2(flap_y, -flap_x)
    return r * jnp.cos(alpha - theta), r * jnp.sin(alpha - theta)


@partial(jax.jit, static_argnames=("n",))
def display_sweep(p, n: int = 201):
    """Everything the plots need for one geometry, in a single compiled call."""
    u = jnp.linspace(0.0, 1.0, n)
    rod_length, theta, ratio, violation, _ = sweep(p, u)
    ax, ay = attach_point(theta, p)
    return rod_length, u, theta, ratio, ax, ay, violation <= 0.0


def metrics(p):
    """Scores and diagnostics for one geometry, all differentiable."""
    rod_length, theta, ratio, violation, dtheta = sweep(p)
    mag = jnp.abs(ratio)
    peak_i = jnp.argmax(mag)
    lo_i = jnp.argmin(theta)
    hi_i = jnp.argmax(theta)

    # Uniform-rate reference: what dtheta/du would be if the flap swept its whole
    # range at a constant speed.  Measure the *signed* rate against the sweep's
    # own direction, so a flap that stalls and one that reverses are both caught.
    span = theta[hi_i] - theta[lo_i]
    direction = jnp.sign(jnp.mean(dtheta) + 1e-30)
    floor = DEAD_FRACTION * jnp.abs(span)
    shortfall = (jnp.maximum(floor - direction * dtheta, 0.0)
                 / jnp.maximum(floor, 1e-9))

    return {
        "rod_length": rod_length,
        "area": jnp.trapezoid(mag, U),
        "peak": mag[peak_i],
        "min": jnp.min(mag),
        "theta_at_peak": theta[peak_i],
        "angle_span": theta[hi_i] - theta[lo_i],
        "mag_at_min_angle": mag[lo_i],
        "mag_at_max_angle": mag[hi_i],
        "max_angle_deg": jnp.degrees(jnp.max(jnp.abs(theta))),
        "violation": jnp.mean(violation ** 2),
        # Reduce in float64 explicitly: jnp.mean over a bool array reduces in
        # float32, where an all-valid sweep comes back as 1 - 2^-24 rather than 1.
        "valid_fraction": jnp.mean((violation <= 0.0).astype(jnp.float64)),
        "deadness": jnp.mean(shortfall ** 2),
    }


def cost(p, objective: int, want_peak_at_zero, want_symmetric_ends,
         min_max_angle, ref):
    """Scalar objective.  ``objective``: 0 none, 1 area, 2 peak, 3 min.

    Mirrors optimize._cost so the fast path and the scipy path agree on what a
    good geometry is.  Flags are traced values, not Python bools, so a single
    compiled function serves every combination of settings.
    """
    m = metrics(p)
    raw = jnp.where(objective == 1, m["area"],
                    jnp.where(objective == 2, m["peak"],
                              jnp.where(objective == 3, m["min"], 0.0)))
    # log1p, not a plain ratio: it is monotonic in ``raw`` so it ranks geometries
    # identically, but it stops a near-zero starting metric (tiny ``ref``) from
    # inflating the objective to where the constraint penalties no longer matter.
    score = jnp.log1p(jnp.maximum(raw, 0.0) / jnp.where(ref > 1e-12, ref, 1.0))

    half_span = 0.5 * jnp.abs(m["angle_span"])
    p_zero = jnp.where(half_span > 1e-9,
                       (m["theta_at_peak"] / jnp.where(half_span > 1e-9, half_span, 1.0)) ** 2,
                       1.0)
    lo, hi = m["mag_at_min_angle"], m["mag_at_max_angle"]
    mean = 0.5 * (lo + hi)
    p_sym = jnp.where(mean > 1e-9,
                      ((lo - hi) / jnp.where(mean > 1e-9, mean, 1.0)) ** 2, 1.0)

    target = jnp.maximum(min_max_angle, 1e-9)
    s = jnp.maximum(0.0, target - m["max_angle_deg"]) / target
    p_angle = jnp.where(min_max_angle > 0.0, s + s * s, 0.0)

    return (-score
            + PENALTY_WEIGHT * (want_peak_at_zero * p_zero
                                + want_symmetric_ends * p_sym)
            + ANGLE_WEIGHT * p_angle
            + INVALID_WEIGHT * m["violation"]
            + DEAD_WEIGHT * m["deadness"])


# Eager JAX dispatches every primitive separately, which is far slower than the
# compiled versions; callers outside the solver should use these.
metrics_jit = jax.jit(metrics)
cost_jit = jax.jit(cost)
