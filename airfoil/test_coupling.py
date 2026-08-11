"""Checks on the assumptions the airfoil/linkage coupling is built from.

These are not tests of the aerodynamics, which is a model rather than a thing
with a right answer.  They pin the handful of structural properties the coupled
optimization *relies* on, each of which would otherwise fail silently: a wrong
number would still be a number, and the optimizer would still converge to it.

Run with:  uv run --with pytest python -m pytest airfoil/test_coupling.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import jax  # noqa: E402

import airfoil.airfoil_model as am  # noqa: E402
import airfoil.linkage_coupling as lc  # noqa: E402
import airfoil.optimize as ao  # noqa: E402


def test_flap_efficiency_is_flat_below_stall():
    """required_deflection_deg inverts a floor by dividing by the slope at one
    degree, which is only correct while the efficiency factor is exactly one.

    If flap_deflection_efficiency is ever changed to roll off gradually from
    zero deflection, that inversion silently under-predicts the deflection
    needed and every authority margin in the model is overstated.  This is the
    tripwire for that change.
    """
    for d in (0.0, 1.0, 7.5, am.FLAP_STALL_DEG - 1e-9):
        assert float(am.flap_deflection_efficiency(d)) == pytest.approx(1.0)
    # And it must actually decay above the knee, or the peak this model relies
    # on for delta_req's validity does not exist.
    assert float(am.flap_deflection_efficiency(am.FLAP_STALL_DEG * 2)) < 0.5


def test_authority_is_linear_below_stall():
    """The other half of the same assumption, end to end rather than on the
    efficiency term alone: angular acceleration must be proportional to
    deflection, so that inverting it is a division rather than a root find."""
    ratios = [
        float(am.evaluate(am.BASELINE, deflection_deg=d)["authority"]["hover"]
              ["alpha_pitch"]) / d
        for d in (1.0, 2.5, 10.0, 15.0)
    ]
    for r in ratios[1:]:
        assert r == pytest.approx(ratios[0], rel=1e-9)


def test_derived_deflection_meets_the_floors_it_was_solved_from():
    """delta_req is the deflection the authority floors demand, so evaluating
    at it must land on those floors -- not above, not below."""
    r = am.evaluate(am.BASELINE)
    hover = r["authority"]["hover"]
    assert abs(float(hover["alpha_pitch"])) >= am.MIN_ALPHA_PITCH_HOVER - 1e-6
    assert abs(float(hover["alpha_roll"])) >= am.MIN_ALPHA_ROLL_HOVER - 1e-6
    # Whichever axis is binding should sit *on* its floor rather than above it,
    # since delta_req is the max over axes.
    on_floor = (
        abs(float(hover["alpha_pitch"])) == pytest.approx(
            am.MIN_ALPHA_PITCH_HOVER, rel=1e-6)
        or abs(float(hover["alpha_roll"])) == pytest.approx(
            am.MIN_ALPHA_ROLL_HOVER, rel=1e-6))
    assert on_floor


def test_deflection_is_single_sided():
    """delta_max is half the linkage's peak-to-peak sweep.  Dropping the factor
    of two would double every throw the model reports, which reads as a design
    with twice the margin it has."""
    g = am.unpack(am.BASELINE)
    r = am.evaluate(am.BASELINE)
    lk = lc.linkage_metrics(
        float(r["pushrod_length"]), float(g["servo_rod_y"]),
        float(g["servo_travel_mm"]), float(g["flap_x_mm"]),
        float(g["flap_y_mm"]))
    import linkage.linkage_model as lm
    raw = lm.metrics(jnp.array([
        float(r["pushrod_length"]) * lc.MM_PER_M, float(g["servo_rod_y"]),
        float(g["servo_travel_mm"]), float(g["flap_x_mm"]),
        float(g["flap_y_mm"])]))
    assert float(lk["delta_max_deg"]) == pytest.approx(
        0.5 * float(jnp.degrees(jnp.abs(raw["angle_span"]))))
    assert float(lk["delta_max_deg"]) == pytest.approx(
        float(raw["max_angle_deg"]), rel=1e-6)


def test_linkage_is_in_the_gradient_graph():
    """The coupling is only real if the linkage variables move the linkage
    outputs.  A zero here means the two models share a design vector but not a
    problem, which looks identical to a working merge from the outside."""
    i = list(am.DESIGN_VARS).index("flap_y_mm")
    d_adv = jax.grad(lambda p: am.evaluate(p)["linkage_advantage"])(am.BASELINE)
    d_throw = jax.grad(lambda p: am.evaluate(p)["delta_geom"])(am.BASELINE)
    assert float(d_adv[i]) != 0.0
    assert float(d_throw[i]) != 0.0
    # And they must trade against each other: the servo's stroke is a fixed
    # budget, so buying advantage spends throw.
    assert float(d_adv[i]) * float(d_throw[i]) < 0.0


def test_advantage_times_throw_is_the_servo_stroke():
    """Virtual work, which is the whole reason the two models have to be solved
    together.  Holds to the extent the advantage is constant across the stroke,
    so it is checked loosely."""
    g = am.unpack(am.BASELINE)
    r = am.evaluate(am.BASELINE)
    product = (float(r["linkage_advantage"]) * lc.MM_PER_M
               * 2.0 * float(jnp.radians(r["delta_geom"])))
    assert product == pytest.approx(float(g["servo_travel_mm"]), rel=0.2)


def test_pinned_variable_does_not_poison_the_optimizer():
    """servo_travel_mm has a zero-width box.  Without the scale guard in
    _solve, dividing the gradient by that zero width sends an inf through the
    shared Adam state and NaNs every variable, not just the pinned one."""
    lower, upper = ao._bounds_arrays()
    i = list(am.DESIGN_VARS).index("servo_travel_mm")
    assert float(lower[i]) == float(upper[i]), "test assumes the box is pinned"
    x, cost = ao._solve(ao._starts(lower, upper), lower, upper)
    assert bool(jnp.all(jnp.isfinite(x)))
    assert float(cost) == float(cost)  # not NaN
    assert float(x[i]) == pytest.approx(float(lower[i]))


def test_every_design_variable_has_bounds():
    """_bounds_arrays and sensitivity both index BOUNDS by design variable name,
    so a variable added to one and not the other is a KeyError at run time."""
    assert set(ao.BOUNDS) == set(am.DESIGN_VARS)
    assert len(am.BASELINE) == len(am.DESIGN_VARS)


def test_unpack_is_positionally_consistent():
    """unpack indexes by name now, so a reordering of DESIGN_VARS should carry
    through rather than silently mis-assigning every variable after the change.
    """
    g = am.unpack(am.BASELINE)
    for i, name in enumerate(am.DESIGN_VARS):
        if name in g:
            assert float(g[name]) == pytest.approx(float(am.BASELINE[i]))
