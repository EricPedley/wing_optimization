"""Whole-quad forward model: four identical motor+propeller units.

Combines motor_model.py (SimITL-mirrored electrical/thermal/spin-up
physics), motor_scaling.py and prop_scaling.py (catalogue-free mass/
resistance sizing), and prop_aero_model.py (parametric propeller
aerodynamics) into one evaluate() a gradient-based optimizer can drive.

Free variables: motor kV, motor stator volume, propeller diameter, blade
count, and pitch -- the five named at the start of this design (kV and
stator volume set the motor's electrical properties via motor_scaling.py;
diameter/blades/pitch set the propeller's aerodynamics via
prop_aero_model.py and its mass/inertia via prop_scaling.py).

Assumes all four motor/propeller units are identical and the vehicle hovers
level, so per-motor thrust and current are the same on all four arms -- this
model has no notion of yaw/roll/pitch authority, only the propulsion
system's aggregate thrust, current, and spin-up time.
"""

import jax
import jax.numpy as jnp

import multirotor.motor_model as mm
import multirotor.motor_scaling as ms
import multirotor.prop_aero_model as pa
import multirotor.prop_scaling as ps

G = 9.81  # m/s^2

# 1S nominal, matching the 3.7V/12A ESC this design targets (see
# conversation). Nominal storage voltage, not full-charge (~4.2V) -- full
# charge is the true worst case for current draw and should be checked
# before trusting the current constraint close to its limit.
VBAT = 3.7
ESC_MAX_CURRENT_A = 12.0

# Smallest stator size with a datasheet showing it can take the ESC's ~50W
# burst (see conversation) -- used as a hard floor instead of a thermal
# model, so the optimizer cannot pick a motor smaller than something known
# to survive this ESC's current.
STATOR_VOLUME_FLOOR_MM3 = ms.stator_volume_mm3(10.0, 2.0)  # 1002

# "10% throttle to 90% throttle within this long" -- see motor_model.
# spin_up_time_s for the closed-form derivation.
SPIN_UP_START_FRAC = 0.10
SPIN_UP_END_FRAC = 0.90
SPIN_UP_BUDGET_S = 0.050

# prop_aero_model.py is incompressible BEMT with no tip-loss, stall, or
# compressibility-drag model, and its static thrust curve is an unbounded
# rpm^2 law -- so nothing stops the optimizer from extrapolating rpm well
# past anything the model has been checked against, where it just keeps
# handing back quadratically more "free" thrust. That is exactly what
# happened at MAX_TIP_MACH=0.5 (the original guess here): the optimizer
# landed at 72,800 rpm on a 45mm prop, predicting 337g of thrust from a
# model whose only two real calibration points topped out at 139g/47,758rpm
# and 71.7g/43,745rpm -- more than double anything measured, entirely from
# extrapolation with no physical mechanism (tip loss, blade stall,
# compressibility drag rise) to push back.
#
# History of this constant, because it has been wrong twice in ways worth
# not repeating: first set to 0.5 as a guess, which the optimizer's own
# extrapolation abused (72,800 rpm, 2x any measured thrust); tightened to
# 0.35 from the two small-prop bench tests, which then turned out to
# actively exclude a real, commercially-sold 65mm-prop combo (a
# 1202.5/15000kV motor) operating successfully around Mach 0.40 on 1S --
# i.e. the cap was excluding working hardware, not protecting against
# nonsense.
#
# Small props in particular routinely run tip speeds well above what a
# full-size prop or rotor would (they are not constrained by the same
# noise/efficiency/structural tradeoffs that keep full-size propellers well
# below Mach 1), so pinning this near a small-prop-specific "typical"
# operating point was the wrong kind of bound to begin with. What this
# constraint should actually do is stop the model from being trusted into
# genuinely unmodeled territory -- transonic flow over the blade -- not
# second-guess whether a given rpm is a "reasonable" design choice; that
# judgment belongs to the current, spin-up, and size constraints instead.
# 0.9 leaves a margin below Mach 1 for that purpose alone.
SPEED_OF_SOUND_M_S = 343.0
MAX_TIP_MACH = 0.9

# Placeholder for everything not modeled elsewhere: frame, FC/ESC stack,
# battery, VTX, wiring. Not fit from anything -- a rough guess sized for a
# tiny 1S whoop/toothpick build (consistent with the 3.7V/12A ESC) to get
# the optimizer running end to end. Replace once an airframe is chosen; TWR
# is directly sensitive to this number.
OTHER_MASS_KG = 0.040

# Motor thermal resistance is not modeled (see conversation: the stator
# floor above stands in for a thermal constraint), but motor_model.equilibrium
# still computes a temperature for visibility. This value is an unfit
# placeholder purely so that number is not nonsense; it does not feed any
# constraint or the objective.
PLACEHOLDER_MOTOR_RTH = 12.0

# Hard packaging limit for this build: the frame this is being sized for
# cannot fit a bigger prop. Not derived from anything in the physics model --
# it is an airframe constraint, so it belongs here as a bound the app applies
# to the design-variable box, not as a penalty term competing with the
# physical constraints.
MAX_PROP_DIAMETER_M = 3.0 * 25.4e-3  # 3 inches

DESIGN_VARS = ("kv", "stator_volume_mm3", "prop_diameter_m", "blade_count",
               "pitch_m")

BASELINE = jnp.array([20000.0, 250.0, 0.05, 3.0, 0.03])


def no_load_current_a(kv):
    """Placeholder I0(kV), UNFIT.

    motor_scaling.py deliberately left I0 out of its calibration -- noisy,
    and a smaller effect on the physics here than Km and mass are. This is a
    rough placeholder in the ballpark of the SimITL catalogue's I0 values
    (0.6-1.2A), increasing weakly with kV on the physical grounds that
    higher electrical frequency raises iron losses, but it is NOT fit to any
    data. Revisit if a result turns out to be sensitive to it.
    """
    return 0.5 + 0.3 * (kv / 10000.0)


def unpack(x):
    return dict(zip(DESIGN_VARS, x))


def motor_prop_unit(kv, stator_volume_mm3, prop_diameter_m, blade_count,
                     pitch_m, chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                     cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                     induced_power_factor=pa.INDUCED_POWER_FACTOR):
    """Every derived property of one motor+propeller pair.

    The four aero constants are accepted as overrides (defaults are the
    calibrated module values) for the same reason bemt_thrust_torque takes
    them -- see prop_aero_model.py -- so a caller can explore sensitivity to
    their real, documented uncertainty without mutating module state.
    """
    resistance = ms.motor_resistance_ohm(kv, stator_volume_mm3)
    i0 = no_load_current_a(kv)
    motor_mass = ms.motor_mass_kg(stator_volume_mm3)

    diameter_mm = prop_diameter_m * 1e3
    prop_mass = ps.prop_mass_kg(diameter_mm, blade_count)
    prop_inertia = ps.prop_inertia_kg_m2(prop_mass, diameter_mm)

    aero = pa.to_simitl_params(prop_diameter_m, pitch_m, blade_count,
                                chord_to_diameter_ratio, cl_alpha, cd0,
                                induced_power_factor)

    return dict(resistance=resistance, i0=i0, motor_mass=motor_mass,
                prop_mass=prop_mass, prop_inertia=prop_inertia, **aero)


def evaluate(x, vel=0.0, vbat=VBAT, other_mass_kg=OTHER_MASS_KG,
             chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
             cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
             induced_power_factor=pa.INDUCED_POWER_FACTOR):
    """Full operating point of the quad at full throttle, plus spin-up time.

    Full throttle (volts=vbat) is what TWR and the current constraint are
    evaluated at -- TWR is meant as "thrust available", and current draw is
    worst-case at max commanded throttle. vel=0 (hover) throughout: this
    model does not represent forward flight.

    vbat, other_mass_kg, and the four aero constants all default to the
    module-level assumptions but are accepted as overrides -- see
    motor_prop_unit -- so app.py can vary them per-request without shared
    mutable state.
    """
    g = unpack(x)
    unit = motor_prop_unit(g["kv"], g["stator_volume_mm3"],
                            g["prop_diameter_m"], g["blade_count"],
                            g["pitch_m"], chord_to_diameter_ratio, cl_alpha,
                            cd0, induced_power_factor)

    total_mass = other_mass_kg + 4.0 * (unit["motor_mass"] + unit["prop_mass"])
    weight_n = total_mass * G

    full_throttle = mm.equilibrium(
        vbat, vel, g["kv"], unit["resistance"], unit["i0"],
        PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"], unit["prop_torque_factor"],
        unit["prop_max_rpm"], unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"])

    twr = 4.0 * full_throttle["thrust_n"] / jnp.maximum(weight_n, 1e-9)

    spin_up_s = mm.spin_up_time_s(
        SPIN_UP_START_FRAC * vbat, SPIN_UP_END_FRAC * vbat, vel,
        g["kv"], unit["resistance"], unit["i0"], unit["prop_a_factor"],
        unit["prop_torque_factor"], unit["prop_max_rpm"],
        unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"], unit["prop_inertia"])

    tip_speed_m_s = jnp.pi * g["prop_diameter_m"] * full_throttle["rpm"] / 60.0
    tip_mach = tip_speed_m_s / SPEED_OF_SOUND_M_S

    return dict(unit=unit, total_mass=total_mass, weight_n=weight_n,
                twr=twr, spin_up_s=spin_up_s, tip_mach=tip_mach,
                **full_throttle)


def current_at_throttle_a(x, throttle_frac, vel=0.0, vbat=VBAT, other_mass_kg=OTHER_MASS_KG,
                           chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                           cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                           induced_power_factor=pa.INDUCED_POWER_FACTOR):
    """Per-motor current at an arbitrary throttle fraction, not just full
    throttle.

    Separate from evaluate() because TWR and the current constraint there are
    deliberately worst-case (full throttle); this is for the opposite
    question -- "how much current does this design draw at the throttle it
    will actually spend most of a flight at" -- e.g. minimizing current at a
    cruise/loiter throttle instead of minimizing it where it is already
    capped by the ESC. other_mass_kg is accepted only so this function's
    signature matches evaluate()'s and can be driven by the same "assumptions"
    bundle app.py passes around; mass does not otherwise enter this
    calculation, since current at a given throttle does not depend on it.
    """
    g = unpack(x)
    unit = motor_prop_unit(g["kv"], g["stator_volume_mm3"], g["prop_diameter_m"],
                            g["blade_count"], g["pitch_m"], chord_to_diameter_ratio,
                            cl_alpha, cd0, induced_power_factor)
    r = mm.equilibrium(throttle_frac * vbat, vel, g["kv"], unit["resistance"], unit["i0"],
                        PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"], unit["prop_torque_factor"],
                        unit["prop_max_rpm"], unit["thrust_factor_x"], unit["thrust_factor_y"],
                        unit["thrust_factor_z"])
    return r["current_a"]


def throttle_sweep(x, throttle_fracs, vel=0.0, vbat=VBAT, other_mass_kg=OTHER_MASS_KG,
                    chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                    cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                    induced_power_factor=pa.INDUCED_POWER_FACTOR):
    """Per-motor rpm/thrust/current and tip Mach across a throttle sweep, for
    plotting -- the equivalent of the bench-test tables this session's
    calibration was built from, but for a candidate design rather than a
    real part. Motor/prop sizing is fixed across the sweep (it depends only
    on the design variables, not throttle); only the electrical operating
    point moves.
    """
    g = unpack(x)
    unit = motor_prop_unit(g["kv"], g["stator_volume_mm3"], g["prop_diameter_m"],
                            g["blade_count"], g["pitch_m"], chord_to_diameter_ratio,
                            cl_alpha, cd0, induced_power_factor)

    def at_throttle(frac):
        volts = frac * vbat
        r = mm.equilibrium(volts, vel, g["kv"], unit["resistance"], unit["i0"],
                            PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"],
                            unit["prop_torque_factor"], unit["prop_max_rpm"],
                            unit["thrust_factor_x"], unit["thrust_factor_y"],
                            unit["thrust_factor_z"])
        tip_mach = (jnp.pi * g["prop_diameter_m"] * r["rpm"] / 60.0) / SPEED_OF_SOUND_M_S
        return r["rpm"], r["thrust_n"], r["current_a"], tip_mach

    rpm, thrust_n, current_a, tip_mach = jax.vmap(at_throttle)(throttle_fracs)
    return {"throttle_frac": throttle_fracs, "rpm": rpm, "thrust_n": thrust_n,
            "current_a": current_a, "tip_mach": tip_mach}


# --- Realistic stator sizes ---------------------------------------------------
#
# stator_volume_mm3 is a continuous design variable so the optimizer can
# search it freely, but a motor only actually ships in a handful of sizes.
# This is the catalogue of sizes worth building against -- the ones a small
# 1S whoop/toothpick build (the class this design targets) is actually sold
# in -- so a continuous answer can be read off against real parts rather
# than left as an abstract mm^3 number.
REALISTIC_STATOR_SIZES = [
    ("1002", 10.0, 2.0), ("1102", 11.0, 2.0), ("1103", 11.0, 3.0),
    ("1104", 11.0, 4.0), ("1202.5", 12.0, 2.5), ("1203", 12.0, 3.0),
    ("1204", 12.0, 4.0),
]


def nearest_stator_sizes(volume_mm3, n=2):
    """The n catalogue sizes (see REALISTIC_STATOR_SIZES) closest in volume
    to a continuous design's stator_volume_mm3, nearest first.

    Ranked by absolute volume difference, not kV/R fit -- two sizes can have
    the same volume from a very different diameter/height split (e.g. 1102
    and 1104 differ a lot in shape despite both being "close" in volume to
    something between them), which this ranking does not distinguish. Good
    enough for "which off-the-shelf part is closest", not a substitute for
    checking the actual diameter/height split fits the build.
    """
    volume_mm3 = float(volume_mm3)
    scored = []
    for name, diameter_mm, height_mm in REALISTIC_STATOR_SIZES:
        v = ms.stator_volume_mm3(diameter_mm, height_mm)
        delta = v - volume_mm3
        scored.append({
            "name": name, "volume_mm3": v, "delta_mm3": delta,
            "delta_pct": 100.0 * delta / volume_mm3 if volume_mm3 > 0 else 0.0,
        })
    scored.sort(key=lambda s: abs(s["delta_mm3"]))
    return scored[:n]


# --- Hover flight time ---------------------------------------------------------
#
# TWR and the current constraint are evaluated at full throttle -- the right
# regime for "how much thrust is available" and "will the ESC survive max
# commanded current" -- but neither says how long the battery lasts, because
# a hovering quad does not fly at full throttle: it flies at whatever
# throttle makes thrust equal weight. That operating point has no closed
# form here (thrust is a closed-form function of rpm, and rpm of volts, but
# composing them and inverting for the volts that hits a target thrust is
# not itself closed-form), so it is found by bisection on throttle fraction
# instead -- thrust is monotonic in throttle, so bisection converges
# reliably without needing a derivative.


def hover_point(x, battery_mah=680.0, vel=0.0, vbat=VBAT, other_mass_kg=OTHER_MASS_KG,
                 chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                 cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                 induced_power_factor=pa.INDUCED_POWER_FACTOR, iters=40):
    """Hover throttle, current draw, and estimated flight time for a design.

    Not part of the optimizer's cost function -- this runs a bisection
    search, so it is a plain (non-jitted, non-differentiable-through) Python
    function meant for display, called once per UI update rather than once
    per optimizer gradient step.

    flight_time_min is battery_mah / (4 * hover current) with no reserve
    margin -- i.e. "time to fully discharge at a constant hover load", not a
    safe usable flight time. A real flight plan should keep a reserve (e.g.
    stop at 80% discharge); this deliberately reports the unpadded number so
    that choice stays visible rather than being silently baked in.
    """
    g = unpack(x)
    unit = motor_prop_unit(g["kv"], g["stator_volume_mm3"], g["prop_diameter_m"],
                            g["blade_count"], g["pitch_m"], chord_to_diameter_ratio,
                            cl_alpha, cd0, induced_power_factor)
    total_mass = other_mass_kg + 4.0 * (unit["motor_mass"] + unit["prop_mass"])
    weight_n = total_mass * G
    thrust_needed_per_motor = weight_n / 4.0

    def thrust_at(frac):
        volts = frac * vbat
        rpm = mm.steady_state_rpm(
            volts, vel, g["kv"], unit["resistance"], unit["i0"],
            unit["prop_a_factor"], unit["prop_torque_factor"], unit["prop_max_rpm"],
            unit["thrust_factor_x"], unit["thrust_factor_y"], unit["thrust_factor_z"])
        return pa.bemt_thrust_torque(rpm, vel, g["prop_diameter_m"], g["pitch_m"],
                                      g["blade_count"], chord_to_diameter_ratio,
                                      cl_alpha, cd0, induced_power_factor)[0]

    max_thrust = float(thrust_at(1.0))
    feasible = max_thrust >= float(thrust_needed_per_motor)

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(thrust_at(mid)) < float(thrust_needed_per_motor):
            lo = mid
        else:
            hi = mid
    hover_frac = hi if feasible else 1.0

    hover = mm.equilibrium(
        hover_frac * vbat, vel, g["kv"], unit["resistance"], unit["i0"],
        PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"], unit["prop_torque_factor"],
        unit["prop_max_rpm"], unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"])
    total_current_a = 4.0 * float(hover["current_a"])
    flight_time_min = ((battery_mah / 1000.0) / total_current_a * 60.0
                        if total_current_a > 1e-9 else float("inf"))

    return {
        "feasible": feasible,
        "hover_throttle_frac": float(hover_frac),
        "hover_current_a_per_motor": float(hover["current_a"]),
        "hover_current_a_total": total_current_a,
        "battery_mah": float(battery_mah),
        "flight_time_min": flight_time_min,
    }


# --- Cost: maximize TWR subject to current, spin-up, and size floors --------
#
# Exterior penalty method, same shape as airfoil/optimize.py's cost: minimize
# -TWR plus a squared penalty for every violated constraint, each shortfall
# normalized by a characteristic scale before squaring so the three
# constraints (amps, seconds, mm^3) contribute comparably to the gradient
# rather than whichever has the largest raw units dominating.

PENALTY_WEIGHT = 1e3
CURRENT_SCALE_A = 1.0
SPINUP_SCALE_S = 0.01
VOLUME_SCALE_MM3 = 20.0
TIP_MACH_SCALE = 0.05


def constraints(x, vel=0.0):
    """Constraint slacks; positive means satisfied, negative means violated."""
    r = evaluate(x, vel)
    g = unpack(x)
    return {
        "current_slack_a": ESC_MAX_CURRENT_A - r["current_a"],
        "spinup_slack_s": SPIN_UP_BUDGET_S - r["spin_up_s"],
        "volume_slack_mm3": g["stator_volume_mm3"] - STATOR_VOLUME_FLOOR_MM3,
        "tip_mach_slack": MAX_TIP_MACH - r["tip_mach"],
    }


def cost(x, vel=0.0):
    r = evaluate(x, vel)
    c = constraints(x, vel)

    def penalty(shortfall, scale):
        return jnp.maximum(-shortfall / scale, 0.0) ** 2

    total_penalty = PENALTY_WEIGHT * (
        penalty(c["current_slack_a"], CURRENT_SCALE_A)
        + penalty(c["spinup_slack_s"], SPINUP_SCALE_S)
        + penalty(c["volume_slack_mm3"], VOLUME_SCALE_MM3)
        + penalty(c["tip_mach_slack"], TIP_MACH_SCALE)
    )
    return -r["twr"] + total_penalty
