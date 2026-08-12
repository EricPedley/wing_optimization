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


# --- Camber, balance, and trim ------------------------------------------------
#
# The trim sign was wrong once already: the Cl * static_margin term was written
# positive, which makes a stable aircraft appear to need down elevon and rewards
# exactly the wrong camber.  Everything still ran and the optimizer still
# converged, to a design bought with inverted physics.  These pin the signs.


def test_camber_line_closes_at_both_ends():
    """A camber line is measured from the chord, so it is zero at both ends by
    definition.  The raw Glauert integral is not -- the A2 term reaches -A2/3 at
    the trailing edge -- and camber_line subtracts that ramp.  If the correction
    is dropped, every camber and thickness figure is quietly measured from a
    line that is not the chord."""
    for a1, a2 in [(0.10, 0.05), (0.05, 0.12), (0.0, 0.08), (0.09, -0.03)]:
        assert float(am.camber_line(0.0, a1, a2)) == pytest.approx(0.0, abs=1e-9)
        assert float(am.camber_line(1.0, a1, a2)) == pytest.approx(0.0, abs=1e-9)


def test_reflex_is_exactly_a2_equals_a1():
    """The whole reason for parameterizing by Glauert coefficients rather than a
    geometric shape: zero pitching moment is an equality, not a root solve."""
    for a in (0.0, 0.05, 0.10, 0.15):
        assert float(am.cm_quarter_chord(a, a)) == pytest.approx(0.0, abs=1e-12)
    # And the sign either side of it, which is what "reflex" means.
    assert float(am.cm_quarter_chord(0.10, 0.04)) < 0.0   # plain camber, nose-down
    assert float(am.cm_quarter_chord(0.04, 0.10)) > 0.0   # reflexed, nose-up


def test_stable_wing_trims_with_up_elevon():
    """The sign that was wrong.  A stable aircraft carries its lift behind the
    centre of gravity, which is nose-down, so it needs *up* elevon -- negative
    deflection -- to hold level flight.  A positive answer here means the
    Cl * static_margin term has the wrong sign, which inverts the trim direction
    and makes the optimizer prefer camber it should reject."""
    x_hinge = 0.75
    # Symmetric section, stable: must need up elevon.
    assert float(am.trim_deflection_deg(0.0, 0.20, 0.10, x_hinge)) < 0.0
    # Plain camber makes it worse, reflex makes it better.
    plain = float(am.trim_deflection_deg(-0.04, 0.20, 0.10, x_hinge))
    reflexed = float(am.trim_deflection_deg(+0.04, 0.20, 0.10, x_hinge))
    assert plain < reflexed
    # Neutral margin with a symmetric section needs no trim at all.
    assert float(am.trim_deflection_deg(0.0, 0.20, 0.0, x_hinge)) == pytest.approx(0.0)


def test_static_margin_sign_and_sweep_direction():
    """Positive static margin means the CG is ahead of the aerodynamic centre,
    and sweeping the wing aft must increase it -- that is the only reason a
    tailless aircraft is swept.  A sign slip here would have the optimizer
    sweeping the wing forward to buy stability."""
    g = am.unpack(am.BASELINE)
    args = [float(g["root_chord"]), float(g["tip_chord"]),
            float(g["root_thickness"]), float(g["tip_thickness"])]
    _, mac = am.planform(args[0], args[1])
    tail = [float(g["battery_station"]), float(g["servo_chord_frac"]),
            float(g["servo_span_frac"]), float(mac)]
    margins = [float(am.static_margin(*args, s, *tail)) for s in (0.0, 20.0, 40.0)]
    assert margins[0] < margins[1] < margins[2]


def test_camber_raises_cl_max_and_lowers_stall_speed():
    """Camber's entire payoff.  If Cl_max stops depending on camber, the camber
    variables become invisible to the objective and the optimizer returns
    whatever the random start held -- the failure this model documents for motor
    position."""
    flat = float(am.max_lift_coefficient(0.0, 0.085, 40000.0))
    cambered = float(am.max_lift_coefficient(0.03, 0.085, 40000.0))
    assert cambered > flat

    i1 = am.DESIGN_VARS.index("camber_a1")
    d_stall = jax.grad(am.stall_only)(am.BASELINE)
    assert float(d_stall[i1]) < 0.0     # more camber, lower stall speed


def test_trim_consumes_control_throw():
    """Trim is paid for out of the elevon's travel.  If delta_max stops netting
    out the trim deflection, the model credits the design with authority it has
    already spent, which is the coupling this whole section exists to add."""
    r = am.evaluate(am.BASELINE)
    assert float(r["delta_geom_net"]) <= float(r["delta_geom"]) + 1e-12
    assert float(r["delta_max"]) <= float(r["delta_geom_net"]) + 1e-12


def test_cost_gradient_is_finite_across_the_box():
    """Camber and balance added arccos, log, and a division by a moment slope,
    each of which can produce a NaN that poisons the whole gradient vector
    rather than just its own entry."""
    lower, upper = ao._bounds_arrays()
    grad = jax.grad(am.cost)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = lower + frac * (upper - lower)
        assert bool(jnp.all(jnp.isfinite(grad(x)))), f"non-finite gradient at {frac}"


# --- Rigid boxes in a curved section ------------------------------------------


def test_box_depth_reduces_to_thickness_on_a_symmetric_section():
    """The straight-slot depth and the section thickness are the same thing only
    when there is no camber.  If they disagree at zero camber, box_depth has
    broken the symmetric case it replaced."""
    chord, thickness = 0.12, 0.015
    for x0, x1 in [(0.10, 0.60), (0.05, 0.20), (0.30, 0.35)]:
        slot = float(am.box_depth(chord, thickness, x0, x1, 0.0, 0.0))
        ends = min(float(am.thickness_at(x0, thickness)),
                   float(am.thickness_at(x1, thickness)))
        assert slot == pytest.approx(ends, abs=1e-9)


def test_camber_makes_the_usable_slot_shallower():
    """A battery is a rigid box and cannot follow the camber line, so the slot
    it can use is always shallower than the section is thick.  Checking
    thickness instead -- which is what the model did before camber existed --
    reports boxes fitting that do not."""
    chord, thickness = 0.126, 0.0152
    x0, x1 = 0.08, 0.08 + am.BATTERY_LENGTH / chord
    flat = float(am.box_depth(chord, thickness, x0, x1, 0.0, 0.0))
    curved = float(am.box_depth(chord, thickness, x0, x1, 0.083, 0.126))
    assert curved < flat


def test_packaging_accounts_for_camber():
    """End to end: the same geometry must report less room once the section is
    cambered.  This is the check that failed silently -- volume_slack read
    exactly zero on a design whose battery was 0.2 mm too deep to fit."""
    i1 = am.DESIGN_VARS.index("camber_a1")
    i2 = am.DESIGN_VARS.index("camber_a2")
    flat = am.BASELINE.at[i1].set(0.0).at[i2].set(0.0)
    curved = am.BASELINE.at[i1].set(0.12).at[i2].set(0.02)
    assert float(am.evaluate(curved)["volume_slack"]) < \
        float(am.evaluate(flat)["volume_slack"])


# --- Tip section --------------------------------------------------------------


def test_tip_camber_is_independent_of_the_root():
    """The tip carries its own shape, not a scaled copy of the root's.  If the
    increment stops reaching the tip section, washout silently becomes
    impossible and the tip-stall constraint can never be satisfied."""
    i = am.DESIGN_VARS.index("tip_camber_da1")
    twisted = am.BASELINE.at[i].set(-0.05)
    g = am.unpack(twisted)
    assert float(g["tip_camber_a1"]) == pytest.approx(
        float(g["camber_a1"]) - 0.05)
    r = am.evaluate(twisted)
    assert float(r["tip_camber"]) < float(r["camber"])
    assert float(r["washout"]) > 0.0


def test_washout_costs_lift():
    """Decambering the tip must reduce the wing's Cl_max, or washout is free and
    the optimizer takes it without trading anything for the stall protection."""
    i = am.DESIGN_VARS.index("tip_camber_da1")
    none = am.BASELINE.at[i].set(0.0)
    lots = am.BASELINE.at[i].set(-0.08)
    assert float(am.evaluate(lots)["cl_max"]) < float(am.evaluate(none)["cl_max"])


def test_wing_moment_lies_between_the_two_sections():
    """Cm is the wing's, not the root's.  With the tip free to differ, the
    area-weighted average has to sit between the two section values -- a result
    equal to either one means the loft is not being sampled."""
    i = am.DESIGN_VARS.index("tip_camber_da2")
    x = am.BASELINE.at[i].set(-0.06)
    r = am.evaluate(x)
    lo = min(float(r["cm_c4_root"]), float(r["cm_c4_tip"]))
    hi = max(float(r["cm_c4_root"]), float(r["cm_c4_tip"]))
    assert lo <= float(r["cm_c4"]) <= hi
    assert float(r["cm_c4_root"]) != pytest.approx(float(r["cm_c4_tip"]))


def test_drawn_boxes_sit_inside_the_section():
    """The drawing has to agree with the packaging model.

    Boxes were once drawn centred on the chord line, which is right only for a
    symmetric section.  On a cambered one the section sits above its chord, so a
    box centred on y=0 hung out through the lower surface and looked like a
    packaging failure the model was ignoring -- when the model was correct and
    the picture was not.  A drawing that disagrees with the constraint is worse
    than no drawing, because it gets believed.
    """
    plots = pytest.importorskip("airfoil.plots")
    np = pytest.importorskip("numpy")

    g = am.unpack(am.BASELINE)
    fig = plots.section_figure(am.BASELINE)
    traces = {t.name: t for t in fig.data if t.name}

    def surfaces(x_mm, chord, thickness, a1, a2):
        xs = np.asarray(x_mm) / 1e3 / chord
        half = 0.5 * np.asarray(am.thickness_at(xs, thickness))
        camber = np.asarray(am.camber_line(xs, a1, a2)) * chord
        return (camber - half) * 1e3, (camber + half) * 1e3

    box = traces["Battery"]
    lower, upper = surfaces(box.x[:2], float(g["root_chord"]),
                            float(g["root_thickness"]),
                            float(g["camber_a1"]), float(g["camber_a2"]))

    # The box rests on the inner skin, so its floor is exactly the highest point
    # of the lower surface over its own footprint.
    assert min(box.y) == pytest.approx(max(lower), abs=1e-6)

    # Its lid then agrees with the model's slack, sign and all.  Containment is
    # deliberately *not* asserted: the baseline battery does not fit, and a
    # drawing that cannot show that is a drawing that hides the constraint it
    # exists to illustrate.  What must hold is that the picture and the number
    # tell the same story.
    r = am.evaluate(am.BASELINE)
    drawn_gap = (min(upper) - max(box.y)) / 1e3
    assert drawn_gap == pytest.approx(
        float(r["battery_depth_available"]) - am.BATTERY_THICKNESS, abs=1e-9)
