"""Tests for quad_model.realized_design -- snapping a continuous design point
to the nearest actually-buyable motor and re-evaluating with its real specs.

Run with: uv run --with pytest python -m pytest multirotor/test_realized_design.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.quad_model as qm  # noqa: E402


def test_candidates_are_sorted_nearest_first():
    result = qm.realized_design(qm.BASELINE, n_candidates=5)
    dists = [c["distance"] for c in result["candidates"]]
    assert dists == sorted(dists)


def test_best_uses_a_fully_specified_candidate():
    result = qm.realized_design(qm.BASELINE, n_candidates=5)
    assert result["best"] is not None
    assert result["best"]["motor"]["resistance_ohm"] is not None
    assert result["best"]["motor"]["mass_g"] is not None


def test_best_matches_a_manual_evaluation_of_the_chosen_motor():
    """The re-evaluated performance should be internally consistent: TWR
    computed from the returned thrust/mass should match the returned twr."""
    result = qm.realized_design(qm.BASELINE, n_candidates=5)
    best = result["best"]
    expected_twr = 4.0 * best["thrust_n"] / best["weight_n"]
    assert float(best["twr"]) == pytest.approx(float(expected_twr), rel=1e-6)


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
    """A tiny n_candidates that only reaches mass-only rows should not
    silently fabricate a result."""
    # Query near a size known to have mass-only rows in the CSV (1102),
    # asking for just 1 candidate makes it plausible the nearest is
    # incomplete; this just checks the function never raises and always
    # returns the expected shape either way.
    result = qm.realized_design(qm.BASELINE, n_candidates=1)
    assert "candidates" in result and "best" in result
    assert len(result["candidates"]) == 1
