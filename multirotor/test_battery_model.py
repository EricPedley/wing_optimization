"""Tests for battery_model.py (OCV curve + constant-internal-resistance
terminal voltage) and quad_model.hover_point's use of it for flight-time
estimation.

Run with: uv run --with pytest python -m pytest multirotor/test_battery_model.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.battery_model as bm  # noqa: E402
import multirotor.quad_model as qm  # noqa: E402


def test_open_circuit_voltage_decreases_with_soc():
    v_full = float(bm.open_circuit_voltage(0.0))
    v_half = float(bm.open_circuit_voltage(0.5))
    v_empty = float(bm.open_circuit_voltage(1.0))
    assert v_full > v_half > v_empty
    assert v_full == pytest.approx(4.2, abs=0.01)


def test_terminal_voltage_sags_with_current():
    v_no_load = bm.terminal_voltage(0.3, 0.0, bm.r_int_ohm("680mAh"))
    v_loaded = bm.terminal_voltage(0.3, 20.0, bm.r_int_ohm("680mAh"))
    assert float(v_loaded) < float(v_no_load)


def test_all_three_batteries_present_with_expected_masses():
    assert set(bm.BATTERIES) == {"480mAh", "580mAh", "680mAh"}
    assert bm.mass_kg("480mAh") == pytest.approx(12.6e-3)
    assert bm.mass_kg("580mAh") == pytest.approx(14.1e-3)
    assert bm.mass_kg("680mAh") == pytest.approx(16.2e-3)


def test_larger_pack_gives_longer_hover_flight_time():
    r_small = qm.hover_point(qm.BASELINE, battery_name="480mAh")
    r_large = qm.hover_point(qm.BASELINE, battery_name="680mAh")
    assert r_small["feasible"] and r_large["feasible"]
    assert r_large["flight_time_min"] > r_small["flight_time_min"]


def test_hover_point_reports_feasible_and_positive_flight_time():
    r = qm.hover_point(qm.BASELINE, battery_name="680mAh")
    assert r["feasible"]
    assert 0.0 < r["hover_throttle_frac"] <= 1.0
    assert r["hover_current_a_total"] > 0.0
    assert r["flight_time_min"] > 0.0


def test_infeasible_design_reports_zero_flight_time():
    # A design with far too little thrust: huge other_mass_kg forces
    # infeasibility regardless of motor/prop.
    r = qm.hover_point(qm.BASELINE, battery_name="680mAh", other_mass_kg=100.0)
    assert not r["feasible"]
    assert r["flight_time_min"] == 0.0
