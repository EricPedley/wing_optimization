"""Checks on quad_model.py's catalogue-lookup and hover-flight-time helpers.

Run with: uv run --with pytest python -m pytest multirotor/test_quad_model.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.quad_model as qm  # noqa: E402


def test_nearest_stator_sizes_exact_match_has_zero_delta():
    result = qm.nearest_stator_sizes(qm.ms.stator_volume_mm3(12.0, 2.5))
    assert result[0]["name"] == "1202.5"
    assert result[0]["delta_mm3"] == pytest.approx(0.0)


def test_nearest_stator_sizes_are_sorted_by_absolute_delta():
    result = qm.nearest_stator_sizes(300.0, n=len(qm.REALISTIC_STATOR_SIZES))
    deltas = [abs(r["delta_mm3"]) for r in result]
    assert deltas == sorted(deltas)


def test_nearest_stator_sizes_respects_n():
    assert len(qm.nearest_stator_sizes(300.0, n=2)) == 2
    assert len(qm.nearest_stator_sizes(300.0, n=3)) == 3


def test_hover_point_is_feasible_and_cheaper_than_full_throttle():
    """Hover needs less current than full throttle for any design that can
    actually hover -- if it needed more, something upstream would be wrong
    (full throttle is by definition the most thrust/current the design can
    produce)."""
    r = qm.hover_point(qm.BASELINE)
    full = qm.evaluate(qm.BASELINE)
    assert r["feasible"]
    assert 0.0 < r["hover_throttle_frac"] < 1.0
    assert r["hover_current_a_per_motor"] < float(full["current_a"])
    assert r["flight_time_min"] > 0.0


def test_hover_point_thrust_matches_weight():
    """The bisection's defining property: at the hover throttle it finds,
    four motors' thrust should equal the vehicle's weight."""
    g = qm.unpack(qm.BASELINE)
    unit = qm.motor_prop_unit(g["kv"], g["stator_volume_mm3"], g["prop_diameter_m"],
                               g["blade_count"], g["pitch_m"])
    total_mass = qm.OTHER_MASS_KG + 4.0 * (unit["motor_mass"] + unit["prop_mass"])
    weight_n = total_mass * qm.G

    r = qm.hover_point(qm.BASELINE)
    rpm = qm.mm.steady_state_rpm(
        r["hover_throttle_frac"] * qm.VBAT, 0.0, g["kv"], unit["resistance"], unit["i0"],
        unit["prop_a_factor"], unit["prop_torque_factor"], unit["prop_max_rpm"],
        unit["thrust_factor_x"], unit["thrust_factor_y"], unit["thrust_factor_z"])
    thrust, _ = qm.pa.bemt_thrust_torque(rpm, 0.0, g["prop_diameter_m"], g["pitch_m"],
                                          g["blade_count"])
    assert float(thrust) * 4.0 == pytest.approx(float(weight_n), rel=1e-3)


def test_hover_point_infeasible_when_too_heavy():
    """A design that cannot produce enough thrust even at full throttle
    should be flagged infeasible rather than silently reporting a fake
    flight time."""
    r = qm.hover_point(qm.BASELINE, other_mass_kg=100.0)  # absurdly heavy
    assert not r["feasible"]


def test_flight_time_scales_with_battery_capacity():
    small = qm.hover_point(qm.BASELINE, battery_mah=300.0)
    large = qm.hover_point(qm.BASELINE, battery_mah=900.0)
    assert large["flight_time_min"] == pytest.approx(
        small["flight_time_min"] * 3.0, rel=1e-6)
