"""Sanity-checks for frame_scaling.py's geometric estimate.

Not accuracy tests against real hardware -- there is no calibration data,
this is a first-principles guess (see frame_scaling.py's module docstring)
-- these just check the geometry behaves the way a physical frame should
(bigger prop never means less material, gradients are finite for the
optimizer, mass is in a plausible ballpark) and pin the fixed-geometry floor
so a future edit to the constants shows up as a diff.

Run with: uv run --with pytest python -m pytest multirotor/test_frame_scaling.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.frame_scaling as fs  # noqa: E402


def test_standoff_volume_matches_cylinder_formula():
    radius_m = 0.5 * fs.STANDOFF_DIAMETER_M
    expected = jnp.pi * radius_m ** 2 * fs.STANDOFF_HEIGHT_M
    assert float(fs.standoff_volume_m3()) == pytest.approx(float(expected))

def test_frame_mass_is_never_less_than_the_fixed_geometry_floor():
    """Center plates + standoffs don't depend on prop diameter, so mass can
    never drop below their combined mass regardless of how small the prop
    is."""
    floor_kg = (
        fs.N_CENTER_PLATES * fs.CENTER_PLATE_LENGTH_M * fs.CENTER_PLATE_WIDTH_M
        * fs.PLATE_THICKNESS_M * fs.CARBON_FIBER_DENSITY_KG_M3
        + fs.N_STANDOFFS * float(fs.standoff_volume_m3()) * fs.ALUMINUM_DENSITY_KG_M3
    )
    for diameter_m in (0.02, 0.05, 0.09, 0.15):
        assert float(fs.frame_mass_kg(diameter_m)) >= floor_kg - 1e-12


def test_frame_mass_increases_with_prop_diameter():
    small = float(fs.frame_mass_kg(0.05))
    large = float(fs.frame_mass_kg(0.15))
    assert large > small


def test_frame_mass_is_a_plausible_ballpark_for_a_toothpick_quad():
    """Loose sanity bound, not a precise target: a tiny-whoop/toothpick frame
    with 15mm aluminum standoffs and 2mm carbon plate should land somewhere
    in the tens of grams, not fractions of a gram or hundreds of grams."""
    mass_g = float(fs.frame_mass_kg(0.065)) * 1e3
    assert 5.0 < mass_g < 100.0


def test_gradient_is_finite():
    import jax

    grad = jax.grad(fs.frame_mass_kg)
    assert bool(jnp.isfinite(grad(0.07)))
