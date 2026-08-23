"""Pins the scaling-law fits in motor_scaling.py to their calibration data.

These are not tests of accuracy -- the fits are openly rough, two-to-six-point
placeholders (see motor_scaling.py's docstrings) -- they just catch a fit
silently breaking (e.g. NaN from a bad log, or a sign flip in the exponent)
and pin the exact numbers so a future recalibration is a visible diff rather
than a silent drift.

Run with: uv run --with pytest python -m pytest multirotor/test_motor_scaling.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.motor_scaling as ms  # noqa: E402


def test_stator_volume_matches_the_size_code():
    assert ms.stator_volume_mm3(10.0, 2.0) == pytest.approx(200.0)
    assert ms.stator_volume_mm3(12.0, 3.0) == pytest.approx(432.0)


def test_mass_fit_reproduces_its_two_calibration_points():
    assert float(ms.motor_mass_kg(ms.stator_volume_mm3(10.0, 2.0))) == \
        pytest.approx(2.5e-3, rel=1e-3)
    assert float(ms.motor_mass_kg(ms.stator_volume_mm3(12.0, 2.5))) == \
        pytest.approx(4.5e-3, rel=1e-3)


def test_mass_increases_with_volume():
    """The one property any future recalibration must keep: a bigger stator
    should never come out lighter."""
    small = float(ms.motor_mass_kg(150.0))
    large = float(ms.motor_mass_kg(500.0))
    assert large > small


def test_km_is_roughly_constant_within_a_stator_size():
    """The physical premise the whole fit rests on: Km should not depend
    strongly on kV at a fixed stator size, since it is meant to be a property
    of the winding-independent geometry. If a future recalibration violates
    this by a wide margin, fitting Km(volume) at all stops being justified."""
    km_1002 = ms._CAL_KM[:3]
    km_1203 = ms._CAL_KM[3:]
    assert float(jnp.std(km_1002) / jnp.mean(km_1002)) < 0.10
    assert float(jnp.std(km_1203) / jnp.mean(km_1203)) < 0.10


def test_km_fit_reproduces_group_means():
    mean_1002 = float(jnp.mean(ms._CAL_KM[:3]))
    mean_1203 = float(jnp.mean(ms._CAL_KM[3:]))
    assert float(ms.motor_constant_km(200.0)) == pytest.approx(mean_1002, rel=0.02)
    assert float(ms.motor_constant_km(432.0)) == pytest.approx(mean_1203, rel=0.02)


def test_resistance_roundtrips_the_calibration_data():
    """motor_resistance_ohm inverts Km = Kt/sqrt(R); feeding it a calibration
    kV and volume should land close to the datasheet R it was fit from --
    not exactly, since Km is only the group mean, but within the same
    ballpark the raw per-motor Km values spread over."""
    for kv, r, diameter, height in [
        (14000.0, 0.175, 10.0, 2.0),
        (19000.0, 0.089, 10.0, 2.0),
        (22000.0, 0.075, 10.0, 2.0),
        (6000.0, 0.320, 12.0, 3.0),
        (8000.0, 0.167, 12.0, 3.0),
        (11500.0, 0.100, 12.0, 3.0),
    ]:
        volume = ms.stator_volume_mm3(diameter, height)
        predicted = float(ms.motor_resistance_ohm(kv, volume))
        assert predicted == pytest.approx(r, rel=0.20)


def test_resistance_decreases_with_kv():
    """Higher kV means fewer turns, hence lower resistance, for a fixed
    stator -- the direction a sign error in the Kt/Km inversion would flip."""
    lo = float(ms.motor_resistance_ohm(6000.0, 432.0))
    hi = float(ms.motor_resistance_ohm(15000.0, 432.0))
    assert hi < lo


def test_km_gradients_are_finite():
    import jax

    grad = jax.grad(lambda v: ms.motor_constant_km(v))
    assert bool(jnp.isfinite(grad(300.0)))

    grad_r = jax.grad(lambda kv: ms.motor_resistance_ohm(kv, 300.0))
    assert bool(jnp.isfinite(grad_r(10000.0)))
