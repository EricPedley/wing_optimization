"""Tests for quad_model.realized_design -- iterating between picking a real,
buyable motor and re-optimizing the propeller for it until the motor choice
stabilizes.

Run with: uv run --with pytest python -m pytest multirotor/test_realized_design.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.quad_model as qm  # noqa: E402


def test_best_uses_a_fully_specified_candidate():
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=3)
    assert result["best"] is not None
    assert result["best"]["motor"]["resistance_ohm"] is not None
    assert result["best"]["motor"]["mass_g"] is not None


def test_best_matches_a_manual_evaluation_of_the_chosen_motor():
    """The re-evaluated performance should be internally consistent: TWR
    computed from the returned thrust/mass should match the returned twr."""
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=3)
    best = result["best"]
    expected_twr = 4.0 * best["thrust_n"] / best["weight_n"]
    assert float(best["twr"]) == pytest.approx(float(expected_twr), rel=1e-6)


def test_iterations_reported_and_bounded():
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=4)
    assert 1 <= result["iterations"] <= 4
    assert isinstance(result["converged"], bool)


def test_a_second_round_starting_from_the_winner_does_not_flip_motors():
    """If realized_design has genuinely converged (motor stable), querying
    nearest_catalogue_motor directly at the winning motor's own (kV, volume)
    should return that same motor first -- i.e. the fixed point is real, not
    an artifact of stopping early."""
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=6)
    assert result["converged"]
    best = result["best"]
    top = qm.ms.nearest_catalogue_motor(
        best["motor"]["kv_rpm_per_v"], best["motor"]["volume_mm3"], n=1)[0]
    assert top["name"] == best["motor"]["name"]


def test_exact_catalogue_point_has_zero_distance_nearest_candidate():
    """Querying at a real motor's exact (kV, volume) should return that
    motor first with ~zero distance."""
    row = qm.ms._MOTOR_ROWS[0]
    kv = float(row["kv_rpm_per_v"])
    volume = qm.ms.stator_volume_mm3(
        float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
    candidates = qm.ms.nearest_catalogue_motor(kv, volume, n=1)
    assert candidates[0]["distance"] == pytest.approx(0.0, abs=1e-9)
    assert candidates[0]["name"] == row["name"]


def test_no_fully_specified_candidate_returns_none_best():
    """n_candidates=0 can never find a fully-specified candidate -- should
    not raise, just report no result found."""
    result = qm.realized_design(qm.BASELINE, n_candidates=0, max_iters=1)
    assert result["best"] is None
    assert not result["converged"]


def test_best_uses_a_real_catalogue_prop_by_default():
    """realized_design's default (use_catalogue_props=True) should return a
    real prop from prop_datasheets.csv, not an idealized (diameter, pitch,
    blade_count) point -- that is the whole point of the catalogue."""
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=3)
    best = result["best"]
    assert "prop" in best
    assert best["prop"]["name"] is not None
    # The evaluated diameter/pitch/blade_count should exactly match the
    # chosen catalogue prop's own specs, not some other value.
    assert float(best["prop_diameter_m"]) * 1e3 == pytest.approx(
        best["prop"]["diameter_mm"], rel=1e-9)
    assert float(best["pitch_m"]) * 1e3 == pytest.approx(
        best["prop"]["pitch_mm"], rel=1e-9)


def test_best_catalogue_prop_for_motor_picks_the_best_of_its_own_candidates():
    """_best_catalogue_prop_for_motor is a nearest-K heuristic, not an
    exhaustive search (same caveat as nearest_catalogue_motor/
    nearest_catalogue_prop) -- it is not guaranteed to find the GLOBAL best
    real prop for a motor, only the best among the n_prop_candidates nearest
    its query point. What it must do correctly is pick the best-scoring
    option among the exact candidate set it looked at -- checked directly
    here by calling nearest_catalogue_prop with the same query point and
    confirming none of THOSE candidates beats what was returned."""
    motor = qm.ms.nearest_catalogue_motor(qm.BASELINE[0], qm.BASELINE[1], n=1)[0]
    prop_x0 = qm.BASELINE[2:5]
    result, chosen_prop = qm._best_catalogue_prop_for_motor(
        motor, 0.0, qm.VBAT, qm.OTHER_MASS_KG, qm.pa.CHORD_TO_DIAMETER_RATIO,
        qm.pa.CL_ALPHA, qm.pa.CD0, qm.pa.INDUCED_POWER_FACTOR, prop_x0,
        n_prop_candidates=6)

    diameter_mm0 = float(prop_x0[0]) * 1e3
    pitch_mm0 = float(prop_x0[2]) * 1e3
    blade_count0 = float(prop_x0[1])
    candidates = qm.ps.nearest_catalogue_prop(
        diameter_mm0, pitch_mm0, blade_count0, n=6)
    best_cost = qm._default_prop_cost(result)
    for prop in candidates:
        r = qm._evaluate_motor_with_prop(
            motor, prop["diameter_mm"] * 1e-3, prop["blade_count"],
            prop["pitch_mm"] * 1e-3, 0.0, qm.VBAT, qm.OTHER_MASS_KG,
            qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA, qm.pa.CD0,
            qm.pa.INDUCED_POWER_FACTOR, prop_mass_kg_override=prop["mass_g"] * 1e-3)
        assert qm._default_prop_cost(r) >= best_cost - 1e-9, (
            f"{prop['name']} scores better than the chosen prop")


def test_idealized_prop_search_still_available_as_a_fallback():
    """use_catalogue_props=False should still run the old continuous LBFGSB
    search and return an idealized (non-catalogue) prop, for callers that
    specifically want that comparison."""
    result = qm.realized_design(qm.BASELINE, n_candidates=5, max_iters=3,
                                 use_catalogue_props=False)
    best = result["best"]
    assert "prop" not in best
