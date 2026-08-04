"""Checks the JAX model against the original scipy model in :mod:`core`.

``core.py`` and ``optimize.py`` are no longer on the app's hot path, but they are
kept as the independent reference these tests measure against: one finds flap
angles with grid scans plus brentq, the other with a closed form, so agreeing to
1e-7 means the closed form is right.

Run with:  uv run python -m pytest test_fastmodel.py -q
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

import fastmodel as fm  # noqa: E402
import fastopt as fo  # noqa: E402
from core import _endpoint_angle_sum, simulate_flap, torque_force_ratio  # noqa: E402

RANGES = {"servo_x": (10.0, 20.0), "servo_y": (0.0, 10.0),
          "servo_travel": (5.0, 15.0), "flap_x": (-5.0, 20.0),
          "flap_y": (5.0, 10.0)}


def _random_geometries(n, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([RANGES[k][0] for k in fo.DESIGN_VARS])
    hi = np.array([RANGES[k][1] for k in fo.DESIGN_VARS])
    return rng.random((n, 5)) * (hi - lo) + lo


def test_closed_form_satisfies_rod_length_constraint():
    """theta from the arccos form must put the pin exactly one rod-length away."""
    worst = 0.0
    for p in _random_geometries(200, seed=1):
        rod_length, _, theta, _, ax, ay, ok = fm.display_sweep(jnp.asarray(p), 41)
        ok = np.asarray(ok)
        if not ok.any():
            continue
        sx = p[0] + np.linspace(0, 1, 41) * p[2]
        dist = np.hypot(sx - np.asarray(ax), p[1] - np.asarray(ay))
        worst = max(worst, np.max(np.abs(dist[ok] - float(rod_length))))
    # Not exact to machine precision by design: fm.ARCCOS_EPS holds the cosine a
    # hair inside +/-1 to keep arccos' derivative finite, which shifts near-tangent
    # poses by ~1e-7 on lengths of order 20 (a relative error of ~1e-8).
    assert worst < 1e-6, worst


def test_matches_core_where_core_converges():
    """Flap angle and attachment point must match the scipy solver."""
    d_theta = d_attach = 0.0
    compared = 0
    for p in _random_geometries(60, seed=2):
        rod_length, u, theta, _, ax, ay, ok = fm.display_sweep(jnp.asarray(p), 41)
        rod_length = float(rod_length)
        # Only meaningful where core's own symmetry search actually converged.
        if abs(_endpoint_angle_sum(rod_length, *p)) > 1e-6:
            continue
        ref = simulate_flap(np.asarray(u), *p, rod_length)
        mask = ref["valid"] & np.asarray(ok)
        if mask.sum() < 5:
            continue
        compared += 1
        d_theta = max(d_theta, np.max(np.abs(ref["flap_angle_rad"][mask] - np.asarray(theta)[mask])))
        d_attach = max(d_attach, np.max(np.abs(ref["attach_x"][mask] - np.asarray(ax)[mask])))
    assert compared >= 10, f"only {compared} comparable geometries"
    assert d_theta < 1e-6, d_theta
    assert d_attach < 1e-9, d_attach


def test_advantage_converges_to_core_under_refinement():
    """core's np.gradient advantage should approach the exact autodiff value."""
    p = np.array([15.0, 6.0, 9.0, 0.0, 10.0])
    rod_length, _, _, ratio, _, _, _ = fm.display_sweep(jnp.asarray(p), 41)
    exact = float(np.asarray(ratio)[20])

    errors = []
    for n in (201, 801, 3201):
        u = np.linspace(0, 1, n)
        ref = simulate_flap(u, *p, float(rod_length))
        errors.append(abs(torque_force_ratio(ref, p[2])[n // 2] - exact) / abs(exact))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] < 1e-6, errors


def test_no_nan_over_the_whole_design_box():
    """Cost and gradient must stay finite even for geometries that cannot close."""
    import jax

    args = (jnp.asarray(2, dtype=jnp.int32), jnp.asarray(1.0), jnp.asarray(1.0),
            jnp.asarray(30.0), jnp.asarray(10.0))
    xs = jnp.asarray(_random_geometries(2000, seed=3))
    costs = jax.jit(jax.vmap(lambda x: fm.cost(x, *args)))(xs)
    grads = jax.jit(jax.vmap(jax.grad(lambda x: fm.cost(x, *args))))(xs)
    assert np.all(np.isfinite(np.asarray(costs)))
    assert np.all(np.isfinite(np.asarray(grads)))


def test_valid_fraction_is_exactly_one_when_valid():
    """Guards the float32 reduction that made an all-valid sweep read as 0.99999994."""
    m = fm.metrics_jit(jnp.asarray([10.0, 0.0, 5.0, -5.0, 10.0]))
    assert float(m["valid_fraction"]) == 1.0


def test_optimizer_respects_bounds_and_constraints():
    values = dict(zip(fo.DESIGN_VARS, [15.0, 6.0, 9.0, 0.0, 10.0]))
    modes = {"servo_x": "free", "servo_y": "ge", "servo_travel": "fixed",
             "flap_x": "free", "flap_y": "le"}
    result = fo.optimize(values, modes, RANGES, "area",
                         ["peak_at_zero", "symmetric_ends"], min_max_angle=30.0)

    assert result["values"]["servo_travel"] == pytest.approx(9.0, abs=1e-6)
    assert result["values"]["servo_y"] >= 6.0 - 1e-6
    assert result["values"]["flap_y"] <= 10.0 + 1e-6
    for name, value in result["values"].items():
        lo, hi = RANGES[name]
        assert lo - 1e-6 <= value <= hi + 1e-6, (name, value)
    assert result["best_metrics"]["max_angle_deg"] >= 30.0 - 0.05


def test_min_angle_constraint_holds_across_targets():
    values = dict(zip(fo.DESIGN_VARS, [15.0, 6.0, 9.0, 0.0, 10.0]))
    modes = {k: "free" for k in fo.DESIGN_VARS}
    for objective in ("area", "peak", "min"):
        for target in (10.0, 25.0, 45.0):
            result = fo.optimize(values, modes, RANGES, objective,
                                 ["peak_at_zero", "symmetric_ends"], target)
            got = result["best_metrics"]["max_angle_deg"]
            assert got >= target - 0.05, (objective, target, got)


def test_peak_objective_stays_bounded():
    """Exact gradients otherwise drive dtheta/du to zero for unbounded advantage."""
    values = dict(zip(fo.DESIGN_VARS, [15.0, 6.0, 9.0, 0.0, 10.0]))
    modes = {k: "free" for k in fo.DESIGN_VARS}
    result = fo.optimize(values, modes, RANGES, "peak",
                         ["peak_at_zero", "symmetric_ends"], 30.0)
    best = result["best_metrics"]
    assert best["peak"] < 1e3, best["peak"]
    assert best["deadness"] < 1e-6, best["deadness"]
    assert best["max_angle_deg"] < 90.0, best["max_angle_deg"]
