"""Pins the scaling-law fits in motor_scaling.py to their calibration data.

These are not tests of accuracy -- the fits are openly rough, datasheet-point
placeholders (see motor_scaling.py's docstrings and data/motor_datasheets.csv)
-- they just catch a fit silently breaking (e.g. NaN from a bad log, or a
sign flip in the exponent) and check every calibration row round-trips
through its own fit to within a stated tolerance, so a future recalibration
(editing the CSV) is a visible test failure rather than a silent drift.

Run with: uv run --with pytest python -m pytest multirotor/test_motor_scaling.py -q
"""

from collections import defaultdict

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.motor_scaling as ms  # noqa: E402


def test_stator_volume_matches_the_size_code():
    assert ms.stator_volume_mm3(10.0, 2.0) == pytest.approx(200.0)
    assert ms.stator_volume_mm3(12.0, 3.0) == pytest.approx(432.0)


def test_mass_fit_reproduces_every_calibration_row_within_25_percent():
    """Looser than the original two-point line (which passed through its own
    points exactly) since this is now a least-squares fit over many rows --
    but every row should still round-trip reasonably closely, or the fit is
    badly mis-specified."""
    for row in ms._MOTOR_ROWS:
        if not row["mass_g"]:
            continue
        volume = ms.stator_volume_mm3(
            float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
        predicted_g = float(ms.motor_mass_kg(volume)) * 1e3
        actual_g = float(row["mass_g"])
        ratio = predicted_g / actual_g
        assert 1.0 / 1.25 < ratio < 1.25, (
            f"{row['name']}: predicted {predicted_g:.2f}g vs actual {actual_g}g")


def test_mass_increases_with_volume():
    """The one property any future recalibration must keep: a bigger stator
    should never come out lighter."""
    small = float(ms.motor_mass_kg(150.0))
    large = float(ms.motor_mass_kg(500.0))
    assert large > small


def _km_groups_by_volume():
    groups = defaultdict(list)
    for volume, km in zip(ms._CAL_VOLUME_MM3_KM, ms._CAL_KM):
        groups[float(volume)].append(float(km))
    return groups


def test_km_is_roughly_constant_within_a_stator_size():
    """The physical premise the whole fit rests on: Km should not depend
    strongly on kV at a fixed stator size, since it is meant to be a property
    of the winding-independent geometry. Only checked for sizes with more
    than one winding on record -- a single-winding size has nothing to be
    consistent with yet."""
    for volume, kms in _km_groups_by_volume().items():
        if len(kms) < 2:
            continue
        kms = jnp.array(kms)
        cv = float(jnp.std(kms) / jnp.mean(kms))
        assert cv < 0.30, f"volume={volume}mm^3: Km coefficient of variation {cv:.2f}"


def test_km_fit_reproduces_group_means():
    for volume, kms in _km_groups_by_volume().items():
        mean_km = sum(kms) / len(kms)
        assert float(ms.motor_constant_km(volume)) == pytest.approx(mean_km, rel=0.35)


def test_resistance_roundtrips_the_calibration_data():
    """motor_resistance_ohm inverts Km = Kt/sqrt(R); feeding it a calibration
    kV and volume should land close to the datasheet R it was fit from --
    not exactly, since Km is only the group mean, but within the same
    ballpark the raw per-motor Km values spread over."""
    for row in ms._MOTOR_ROWS:
        if not row["resistance_ohm"]:
            continue
        volume = ms.stator_volume_mm3(
            float(row["stator_diameter_mm"]), float(row["stator_height_mm"]))
        predicted = float(ms.motor_resistance_ohm(float(row["kv_rpm_per_v"]), volume))
        assert predicted == pytest.approx(float(row["resistance_ohm"]), rel=0.70), row["name"]


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
