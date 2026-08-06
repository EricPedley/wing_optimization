"""Sweep design variables and constants to see what each one actually buys.

The model is cheap enough that a sweep is more informative than a single
optimum: it shows which variables the design is sensitive to and which are
flat, which is what tells you where to spend effort.

Run with ``uv run python -m airfoil.sweep``.
"""

import jax.numpy as jnp

import airfoil.airfoil_model as am


def sweep(name, values, fn, fmt="{:.3f}"):
    """Print one metric against one swept quantity."""
    print(f"\n=== {name} ===")
    for v in values:
        row = fn(v)
        formatted = "  ".join(
            f"{k}={fmt.format(float(x)) if not isinstance(x, str) else x}"
            for k, x in row.items()
        )
        print(f"  {v:>8}  {formatted}")


def with_var(index, value, base=None):
    """Baseline design vector with one entry replaced."""
    base = am.BASELINE if base is None else base
    return base.at[index].set(value)


def main():
    idx = {n: i for i, n in enumerate(am.DESIGN_VARS)}

    # Hinge position is the elevon sizing decision.  dCm/deta peaks near 0.75
    # because a larger flap moves its load centre toward the quarter-chord
    # reference point, shortening the moment arm faster than the extra area
    # helps.  So this sweep has an interior optimum rather than a monotone
    # trend, which is the whole reason to look at it.
    sweep(
        "Hinge position (elevon chord fraction)",
        [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        lambda x: {
            "elevon%c": (1 - x) * 100,
            "dCl/deta": am.flap_lift_slope(x),
            "dCm/deta": am.flap_moment_slope(x),
            "eff": am.flap_effectiveness_ratio(x),
            "hinge_mNm": am.evaluate(with_var(idx["x_hinge"], x))
            ["authority"]["cruise"]["hinge"] * 1e3,
        },
    )

    # Motor position trades yaw authority against slipstream coverage.  Wash
    # fraction saturates once the slipstream is fully inside the elevon span,
    # but the yaw moment arm keeps growing, so the two do not peak together.
    sweep(
        "Motor spanwise position (fraction of semi-span)",
        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
        lambda x: {
            "y_mm": x * 0.5 * am.SPAN * 1e3,
            "wash": am.evaluate(with_var(idx["motor_frac"], x))["wash_fraction"],
            "yaw_mNm": am.evaluate(with_var(idx["motor_frac"], x))
            ["yaw_moment"] * 1e3,
            "hover_pitch": am.evaluate(with_var(idx["motor_frac"], x))
            ["authority"]["hover"]["pitch"] * 1e3,
        },
    )

    # Root chord is the lever on thickness ratio.  The servo fixes root
    # thickness in absolute terms, so a longer chord is the only way to get the
    # thickness ratio down to something that works at this Reynolds number.
    sweep(
        "Root chord (tip follows at fixed taper 0.70)",
        [0.075, 0.085, 0.095, 0.105, 0.115, 0.125],
        lambda x: {
            "t/c%": am.evaluate(
                with_var(idx["tip_chord"], 0.70 * x,
                         with_var(idx["root_chord"], x)))["root_tc"] * 100,
            "area_cm2": am.evaluate(
                with_var(idx["tip_chord"], 0.70 * x,
                         with_var(idx["root_chord"], x)))["area"] * 1e4,
            "mass_g": am.evaluate(
                with_var(idx["tip_chord"], 0.70 * x,
                         with_var(idx["root_chord"], x)))["mass"] * 1e3,
            "TWR": am.evaluate(
                with_var(idx["tip_chord"], 0.70 * x,
                         with_var(idx["root_chord"], x)))["twr"],
            "Re_tip": am.evaluate(
                with_var(idx["tip_chord"], 0.70 * x,
                         with_var(idx["root_chord"], x)))["re_tip"],
        },
        fmt="{:.1f}",
    )

    # Taper drives tip Reynolds number and tip stall.  Aggressive taper looks
    # efficient on paper but starves the tip of both chord and Re, which is
    # exactly where a swept flying wing is already most likely to stall first.
    sweep(
        "Taper ratio",
        [0.45, 0.55, 0.65, 0.75, 0.85, 1.00],
        lambda x: {
            "tip_mm": 0.105 * x * 1e3,
            "area_cm2": am.evaluate(
                with_var(idx["tip_chord"], 0.105 * x))["area"] * 1e4,
            "AR": am.evaluate(
                with_var(idx["tip_chord"], 0.105 * x))["aspect_ratio"],
            "Re_tip": am.evaluate(
                with_var(idx["tip_chord"], 0.105 * x))["re_tip"],
        },
        fmt="{:.1f}",
    )

    # Thrust is the least certain input and it gates hovering entirely.  Worth
    # knowing how much margin the current guess has before it stops closing.
    print("\n=== Thrust sensitivity (UNMEASURED input) ===")
    saved = am.THRUST_PER_MOTOR
    for t in [0.020, 0.025, 0.030, 0.035, 0.040, 0.045]:
        am.THRUST_PER_MOTOR = t
        r = am.evaluate(am.BASELINE)
        verdict = "hovers" if float(r["twr"]) > 1.0 else "CANNOT HOVER"
        print(f"  {t * 1e3:5.0f} g/motor   TWR={float(r['twr']):.2f}"
              f"   hover_q={float(r['authority']['hover']['q']):5.1f} Pa"
              f"   {verdict}")
    am.THRUST_PER_MOTOR = saved

    # Skin density is the other soft number: whether the LW-PLA actually foams
    # changes wing mass by a factor of two or more, and it lands directly on TWR.
    print("\n=== Printed density sensitivity ===")
    saved_rho = am.SKIN_AREAL_DENSITY
    for rho in [400.0, 600.0, 900.0, 1240.0]:
        am.SKIN_AREAL_DENSITY = am.WALL_THICKNESS * rho
        r = am.evaluate(am.BASELINE)
        label = "foamed" if rho < 700 else ("partly" if rho < 1000 else "solid PLA")
        print(f"  {rho:6.0f} kg/m^3  ({label:9s})"
              f"  wing={float(am.wing_mass(0.105, 0.074, 0.016, 0.006)) * 1e3:5.1f} g"
              f"  all-up={float(r['mass']) * 1e3:5.1f} g"
              f"  TWR={float(r['twr']):.2f}")
    am.SKIN_AREAL_DENSITY = saved_rho

    # The servo sets root thickness through the hinge-thickness fraction, and
    # that fraction is an assumption rather than a measurement.  It is worth
    # seeing how much of the centre-section thickness rides on it.
    print("\n=== Servo depth and hinge thickness fraction ===")
    print(f"  {'servo':>6}  " + "  ".join(f"f={f:.2f}" for f in (0.4, 0.5, 0.6, 0.7)))
    for servo in [0.005, 0.006, 0.008, 0.010]:
        cells = []
        for f in (0.4, 0.5, 0.6, 0.7):
            cells.append(f"{servo / f * 1e3:6.1f}")
        print(f"  {servo * 1e3:4.0f}mm  " + "  ".join(cells)
              + "   <- required root thickness, mm")


if __name__ == "__main__":
    main()
