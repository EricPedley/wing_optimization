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


def test_prop_reoptimization_beats_the_unoptimized_prop():
    """The whole point of the iteration: re-optimizing the prop for the
    chosen real motor should do at least as well as evaluating that motor
    with the ORIGINAL continuous optimizer's prop unchanged."""
    x = qm.BASELINE
    g = qm.unpack(x)
    result = qm.realized_design(x, n_candidates=5, max_iters=3)
    best = result["best"]

    naive = qm._evaluate_motor_with_prop(
        best["motor"], g["prop_diameter_m"], g["blade_count"], g["pitch_m"],
        0.0, qm.VBAT, qm.OTHER_MASS_KG, qm.pa.CHORD_TO_DIAMETER_RATIO,
        qm.pa.CL_ALPHA, qm.pa.CD0, qm.pa.INDUCED_POWER_FACTOR)

    # Compare via cost (TWR minus constraint penalties), not raw TWR alone,
    # since a naive prop might violate current/spin-up/tip-Mach constraints
    # that the optimized prop respects.
    def cost_of(r):
        c = {
            "current_slack_a": qm.ESC_MAX_CURRENT_A - float(r["current_a"]),
            "spinup_slack_s": qm.SPIN_UP_BUDGET_S - float(r["spin_up_s"]),
            "tip_mach_slack": qm.MAX_TIP_MACH - float(r["tip_mach"]),
        }
        penalty = qm.PENALTY_WEIGHT * sum(
            max(-c[k] / scale, 0.0) ** 2 for k, scale in [
                ("current_slack_a", qm.CURRENT_SCALE_A),
                ("spinup_slack_s", qm.SPINUP_SCALE_S),
                ("tip_mach_slack", qm.TIP_MACH_SCALE),
            ])
        return -float(r["twr"]) + penalty

    assert cost_of(best) <= cost_of(naive) + 1e-6
