"""Checks prop_aero_model.py against the one real bench-test data point it
was calibrated against, and pins the model's basic structural properties.

Run with: uv run --with pytest python -m pytest multirotor/test_prop_aero_model.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.prop_aero_model as pa  # noqa: E402

# Two real static thrust-stand bench tests, throttle 30-100% (below that is
# ESC deadband -- see the session these were calibrated in). rpm and
# measured thrust (converted grams -> N). CL_ALPHA is a joint fit across
# both; see prop_aero_model.py's calibration comment for why a single global
# CL_ALPHA still misses each individually by up to ~10%, in opposite
# directions -- that is why the tolerance below is wide.
_BENCH_PROPS = {
    "A: 45mm/3bl/1.5in": dict(
        diameter_m=0.045, pitch_m=1.5 * 25.4e-3, blades=3.0,
        rpm=jnp.array([16647.0, 22146.0, 27210.0, 31087.0, 35172.0,
                        38324.0, 41283.0, 43745.0]),
        thrust_n=jnp.array([9.5, 17.4, 27.4, 36.3, 44.5, 53.9, 62.5, 71.7])
        / 1000.0 * 9.81,
    ),
    "B: 50.78mm/3bl/1.9in": dict(
        diameter_m=0.05078, pitch_m=1.9 * 25.4e-3, blades=3.0,
        rpm=jnp.array([21577.0, 27203.0, 31920.0, 35992.0, 38788.0,
                        42056.0, 44996.0, 47758.0]),
        thrust_n=jnp.array([27.0, 44.0, 61.0, 77.0, 92.0, 108.0, 124.0, 139.0])
        / 1000.0 * 9.81,
    ),
}


@pytest.mark.parametrize("name", list(_BENCH_PROPS))
def test_static_thrust_matches_the_bench_data_within_fifteen_percent(name):
    """CL_ALPHA is a joint fit across both propellers (see
    prop_aero_model.py's calibration comment), so no single one matches as
    tightly as a per-propeller fit would -- the wide margin here reflects
    that real, documented spread, not slack test-writing."""
    prop = _BENCH_PROPS[name]
    for rpm, measured in zip(prop["rpm"], prop["thrust_n"]):
        predicted, _ = pa.bemt_thrust_torque(
            rpm, 0.0, prop["diameter_m"], prop["pitch_m"], prop["blades"])
        assert float(predicted) == pytest.approx(float(measured), rel=0.15)


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
