"""Checks prop_aero_model.py against real bench-test data from its
calibration set, and pins the model's basic structural properties.

Run with: uv run --with pytest python -m pytest multirotor/test_prop_aero_model.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.prop_aero_model as pa  # noqa: E402

# Two of the four real static thrust-stand bench props CL_ALPHA/
# INDUCED_POWER_FACTOR are calibrated against (see
# multirotor/calibrate_prop_aero.py and prop_aero_model.py's calibration
# comment for the full 4-prop/89-row fit and why the calibration was
# deliberately narrowed to a 60-77mm-diameter, moderate-pitch/diameter-ratio
# subset rather than the full ~300-row/18-prop dataset, which has no honest
# single-CL_ALPHA fit at all). Throttle 55-100% (below that is ESC deadband,
# same convention the fit itself used at >30%; a slightly higher floor here
# just keeps this fixture short). rpm and measured thrust (converted grams ->
# N), from data/tmotor_f1203_throttle_sweep.csv (Gemfan Hurricane 3018-2,
# tested on a T-Motor F1203) and data/tmotor_f1204_throttle_sweep.csv
# (HQProp T3x2x3, tested on a T-Motor F1204).
_BENCH_PROPS = {
    "Gemfan Hurricane 3018-2: 76.5mm/2bl/45.7mm pitch": dict(
        diameter_m=0.0765, pitch_m=0.0457, blades=2.0,
        rpm=jnp.array([20378.0, 21574.0, 22835.0, 23931.0, 24914.0,
                        25789.0, 26814.0, 27776.0, 28657.0, 29447.0]),
        thrust_n=jnp.array([75.25, 86.78, 97.67, 106.02, 115.10, 126.63,
                             135.60, 145.85, 154.59, 165.51]) / 1000.0 * 9.81,
    ),
    "HQProp T3x2x3: 76.2mm/3bl/50.8mm pitch": dict(
        diameter_m=0.0762, pitch_m=0.0508, blades=3.0,
        rpm=jnp.array([20441.0, 21925.0, 23023.0, 24046.0, 25282.0,
                        26404.0, 27447.0, 28505.0, 29282.0, 29834.0]),
        thrust_n=jnp.array([77.63, 87.87, 96.45, 108.87, 120.29, 132.24,
                             143.50, 155.09, 164.96, 171.75]) / 1000.0 * 9.81,
    ),
}


@pytest.mark.parametrize("name", list(_BENCH_PROPS))
def test_static_thrust_matches_the_bench_data_within_forty_percent(name):
    """CL_ALPHA/INDUCED_POWER_FACTOR are a joint fit across 4 real props (see
    prop_aero_model.py's calibration comment), so no single one matches as
    tightly as a per-propeller fit would -- the wide margin here reflects
    that real, documented per-prop spread (mean error 0-20%, worst single
    row up to ~38% for these two props specifically -- see
    calibrate_prop_aero.py's residual report), not slack test-writing."""
    prop = _BENCH_PROPS[name]
    for rpm, measured in zip(prop["rpm"], prop["thrust_n"]):
        predicted, _ = pa.bemt_thrust_torque(
            rpm, 0.0, prop["diameter_m"], prop["pitch_m"], prop["blades"])
        assert float(predicted) == pytest.approx(float(measured), rel=0.40)


def test_static_thrust_is_an_exact_rpm_squared_law():
    """The core prediction the calibration itself checked before fitting
    CL_ALPHA at all: at vel=0 this model has no linear-in-rpm term."""
    r1, r2 = 15000.0, 30000.0
    t1, _ = pa.bemt_thrust_torque(r1, 0.0, 0.05, 0.03, 3.0)
    t2, _ = pa.bemt_thrust_torque(r2, 0.0, 0.05, 0.03, 3.0)
    assert float(t2 / t1) == pytest.approx((r2 / r1) ** 2, rel=1e-6)


def test_thrust_decreases_with_forward_speed_at_fixed_rpm():
    rpm, D, P, B = 25000.0, 0.05, 0.03, 3.0
    t0, _ = pa.bemt_thrust_torque(rpm, 0.0, D, P, B)
    t1, _ = pa.bemt_thrust_torque(rpm, 5.0, D, P, B)
    t2, _ = pa.bemt_thrust_torque(rpm, 10.0, D, P, B)
    assert float(t2) < float(t1) < float(t0)


def test_to_simitl_params_static_b_coefficient_is_zero():
    """prop_a_factor alone must reproduce the static thrust exactly, i.e.
    SimITL's b(vel=0) coefficient is exactly zero -- the whole reason
    to_simitl_params doesn't need to fit propAFactor, only read it off."""
    D, P, B = 0.05, 0.03, 3.0
    params = pa.to_simitl_params(D, P, B)
    t0, _ = pa.bemt_thrust_torque(pa.REFERENCE_RPM, 0.0, D, P, B)
    assert float(params["prop_a_factor"] * pa.REFERENCE_RPM ** 2) == \
        pytest.approx(float(t0), rel=1e-6)
    assert float(params["thrust_factor_z"]) == pytest.approx(float(t0), rel=1e-6)


def test_gradients_are_finite():
    import jax

    def f(params):
        d, p, b = params
        t, q = pa.bemt_thrust_torque(25000.0, 3.0, d, p, b)
        return t + q

    grad = jax.grad(f)(jnp.array([0.05, 0.03, 3.0]))
    assert bool(jnp.all(jnp.isfinite(grad)))
