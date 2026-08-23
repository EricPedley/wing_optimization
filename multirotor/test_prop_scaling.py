"""Pins prop_scaling.py's fits to their calibration data.

Not accuracy tests -- the mass fit is openly noisy (R^2 ~ 0.77 in log-log
space, worst single point off by ~40%, see prop_scaling.py) -- these catch a
fit breaking silently (NaN, sign flip) and pin the derived constants so a
recalibration shows up as a diff.

Run with: uv run --with pytest python -m pytest multirotor/test_prop_scaling.py -q
"""

import pytest

jnp = pytest.importorskip("jax.numpy")

import multirotor.prop_scaling as ps  # noqa: E402


def test_diameter_mm_converts_inches():
    assert float(ps.diameter_mm(3.0)) == pytest.approx(76.2)


def test_mass_fit_is_within_its_own_stated_error_bound():
    """Every calibration point should round-trip through the fit to within
    the ~1.4x multiplicative error the module's docstring claims -- if a
    future edit makes that bound tighter or looser, this is where it shows."""
    for diameter, blades, mass_g in zip(
            ps._CAL_DIAMETER_MM, ps._CAL_BLADE_COUNT, ps._CAL_MASS_G):
        predicted_g = float(ps.prop_mass_kg(diameter, blades)) * 1e3
        ratio = predicted_g / float(mass_g)
        assert 1.0 / 1.5 < ratio < 1.5, (
            f"diameter={float(diameter)}mm blades={float(blades)}: "
            f"predicted {predicted_g:.3f}g vs actual {float(mass_g):.3f}g")


def test_mass_increases_with_diameter_at_fixed_blade_count():
    small = float(ps.prop_mass_kg(50.0, 3))
    large = float(ps.prop_mass_kg(150.0, 3))
    assert large > small


def test_mass_increases_with_blade_count_at_fixed_diameter():
    two = float(ps.prop_mass_kg(76.2, 2))
    three = float(ps.prop_mass_kg(76.2, 3))
    assert three > two


def test_inertia_shape_factor_is_physically_bounded():
    """k = I/(m r^2) must sit strictly between 0 (all mass at the hub) and 1
    (all mass at the tip) to be a physically possible mass distribution."""
    assert 0.0 < ps._INERTIA_SHAPE_FACTOR < 1.0


def test_inertia_scales_as_mass_times_radius_squared():
    base = float(ps.prop_inertia_kg_m2(0.002, 76.2))
    double_mass = float(ps.prop_inertia_kg_m2(0.004, 76.2))
    double_diameter = float(ps.prop_inertia_kg_m2(0.002, 152.4))
    assert double_mass == pytest.approx(2.0 * base, rel=1e-9)
    assert double_diameter == pytest.approx(4.0 * base, rel=1e-9)


def test_gradients_are_finite():
    import jax

    grad_mass = jax.grad(lambda d: ps.prop_mass_kg(d, 3.0))
    assert bool(jnp.isfinite(grad_mass(76.2)))

    grad_inertia = jax.grad(lambda d: ps.prop_inertia_kg_m2(0.002, d))
    assert bool(jnp.isfinite(grad_inertia(76.2)))
