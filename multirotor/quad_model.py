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
import numpy as np
from jaxopt import LBFGSB

import multirotor.battery_model as bm
import multirotor.frame_scaling as fs
import multirotor.motor_model as mm
import multirotor.motor_scaling as ms
import multirotor.prop_factor_graph_model as pa
import multirotor.prop_scaling as ps

G = 9.81  # m/s^2

# 1S nominal, matching the 3.7V/12A ESC this design targets (see
# conversation). Nominal storage voltage, not full-charge (~4.2V) -- full
# charge is the true worst case for current draw and should be checked
# before trusting the current constraint close to its limit.
VBAT = 3.7
ESC_MAX_CURRENT_A = 12.0

# ESC throttle exponent from the prop_factor_graph.py bench calibration.
# Effective voltage applied to the motor is VBAT * throttle_frac^DUTY_GAMMA
# rather than the naive throttle_frac * VBAT.  Fitted value ~0.79.
DUTY_GAMMA = 0.789


def effective_voltage(vbat, throttle_frac):
    """Effective DC link voltage after the ESC's throttle mapping."""
    return vbat * (throttle_frac ** DUTY_GAMMA)

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

# Placeholder for everything not modeled elsewhere: FC/ESC stack, VTX,
# antenna, camera, wiring. NOT the battery -- battery mass is added
# separately and explicitly wherever a battery is chosen (see
# battery_model.py's battery_mass_kg, threaded through hover_point/
# _hover_point_jit and optimize_efficiency.py's realize step). Frame mass is
# also separate (frame_scaling.py, added into total_mass in
# evaluate()/hover_point() below).
#
# 0.016kg (16g), from a user estimate of a real FC+VTX+antenna+camera stack
# for this build. Previously 0.023kg -- a rougher guess that turned out too
# high once checked against a real reference build; see git history for that
# prior value's own history (it in turn replaced an even older 0.040kg that
# used to also cover frame mass before frame_scaling.py existed).
OTHER_MASS_KG = 0.016

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

    frame_mass = fs.frame_mass_kg(g["prop_diameter_m"])
    total_mass = (other_mass_kg + frame_mass
                  + 4.0 * (unit["motor_mass"] + unit["prop_mass"]))
    weight_n = total_mass * G

    full_throttle = mm.equilibrium(
        vbat, vel, g["kv"], unit["resistance"], unit["i0"],
        PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"], unit["prop_torque_factor"],
        unit["prop_max_rpm"], unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"])

    twr = 4.0 * full_throttle["thrust_n"] / jnp.maximum(weight_n, 1e-9)

    spin_up_s = mm.spin_up_time_s(
        effective_voltage(vbat, SPIN_UP_START_FRAC),
        effective_voltage(vbat, SPIN_UP_END_FRAC), vel,
        g["kv"], unit["resistance"], unit["i0"], unit["prop_a_factor"],
        unit["prop_torque_factor"], unit["prop_max_rpm"],
        unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"], unit["prop_inertia"])

    tip_speed_m_s = jnp.pi * g["prop_diameter_m"] * full_throttle["rpm"] / 60.0
    tip_mach = tip_speed_m_s / SPEED_OF_SOUND_M_S

    return dict(unit=unit, frame_mass=frame_mass, total_mass=total_mass,
                weight_n=weight_n, twr=twr, spin_up_s=spin_up_s,
                tip_mach=tip_mach, **full_throttle)


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
    r = mm.equilibrium(effective_voltage(vbat, throttle_frac), vel, g["kv"],
                        unit["resistance"], unit["i0"], PLACEHOLDER_MOTOR_RTH,
                        unit["prop_a_factor"], unit["prop_torque_factor"],
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
        volts = effective_voltage(vbat, frac)
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
#
# Pack voltage is not constant across a flight (see battery_model.py): it
# sags with both load and state of charge, and vbat itself is one of the
# equilibrium's inputs, so the hover throttle/current found above is only
# the *initial* operating point. As the pack discharges, its terminal
# voltage under the same hover load keeps falling, which means the throttle
# fraction needed to still produce hover thrust keeps rising (and current
# with it) -- a hovering quad draws more current near the end of a battery
# than at the start, not the same current the whole way, so
# battery_mah / (4 * initial current) overestimates flight time. This is
# handled by simulating the discharge in fixed time steps: at each step,
# re-run the same bisection at the pack's *current* terminal voltage (a
# function of mAh already drawn and the present current, via
# battery_model.terminal_voltage) to find the new hover throttle/current,
# then advance mAh drawn by current * dt. Stops at whichever of "rated
# capacity fully drawn" or "pack voltage sagged to a 3.0V floor" comes
# first -- LiPo terminal voltage falls off a cliff past that point (see the
# OCV curve's own tail), and a real ESC/flight controller would call this
# "empty" well before the model's voltage term goes non-physical.

DISCHARGE_VOLTAGE_FLOOR_V = 3.0
DISCHARGE_DT_S = 2.0
DISCHARGE_N_STEPS = 1800  # DISCHARGE_DT_S * DISCHARGE_N_STEPS = 3600s cap
_HOVER_NEWTON_ITERS = 8


def _hover_throttle_frac(vbat_now, thrust_needed_per_motor, vel, kv, resistance, i0,
                          prop_a_factor, prop_torque_factor, prop_max_rpm,
                          thrust_factor_x, thrust_factor_y, thrust_factor_z,
                          diameter_m, pitch_m, blade_count, chord_to_diameter_ratio,
                          cl_alpha, cd0, induced_power_factor):
    """Throttle fraction (of vbat_now) at which one motor/prop's thrust
    equals thrust_needed_per_motor, by Newton's method on
    f(frac) = thrust(effective_voltage(vbat_now, frac)) - thrust_needed_per_motor.

    thrust(volts) is a closed-form composition of two quadratic solves
    (steady_state_rpm, then bemt_thrust_torque) -- smooth and, on the
    feasible branch, monotonically increasing in volts -- so jax.grad gives
    an exact derivative and a handful of Newton steps converges far faster
    than bisection while staying inside one jit/scan trace instead of
    escaping to Python floats every iteration.

    Clamped to [0, 1] every step: thrust is only monotonic increasing near
    its equilibrium branch, and volts=0 or the propeller's max-rpm regime can
    otherwise send Newton's step outside the physically meaningful throttle
    range.
    """
    def thrust_at(volts):
        rpm = mm.steady_state_rpm(
            volts, vel, kv, resistance, i0, prop_a_factor, prop_torque_factor,
            prop_max_rpm, thrust_factor_x, thrust_factor_y, thrust_factor_z)
        return pa.bemt_thrust_torque(rpm, vel, diameter_m, pitch_m, blade_count,
                                      chord_to_diameter_ratio, cl_alpha, cd0,
                                      induced_power_factor)[0]

    def f(frac):
        return thrust_at(effective_voltage(vbat_now, frac)) - thrust_needed_per_motor

    df = jax.grad(f)

    def newton_step(frac, _):
        slope = jnp.maximum(df(frac), 1e-9)
        frac = jnp.clip(frac - f(frac) / slope, 0.0, 1.0)
        return frac, None

    frac0 = jnp.array(0.5)
    frac, _ = jax.lax.scan(newton_step, frac0, None, length=_HOVER_NEWTON_ITERS)
    return frac


def _hover_point_jit(vel, other_mass_kg, chord_to_diameter_ratio, cl_alpha, cd0,
                      induced_power_factor, kv, stator_volume_mm3, prop_diameter_m,
                      blade_count, pitch_m, battery_mass_kg, capacity_mah, r_int_ohm):
    unit = motor_prop_unit(kv, stator_volume_mm3, prop_diameter_m, blade_count, pitch_m,
                            chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)
    frame_mass = fs.frame_mass_kg(prop_diameter_m)
    total_mass = (other_mass_kg + frame_mass + battery_mass_kg
                  + 4.0 * (unit["motor_mass"] + unit["prop_mass"]))
    weight_n = total_mass * G
    thrust_needed_per_motor = weight_n / 4.0

    def solve_frac(vbat_now):
        return _hover_throttle_frac(
            vbat_now, thrust_needed_per_motor, vel, kv, unit["resistance"], unit["i0"],
            unit["prop_a_factor"], unit["prop_torque_factor"], unit["prop_max_rpm"],
            unit["thrust_factor_x"], unit["thrust_factor_y"], unit["thrust_factor_z"],
            prop_diameter_m, pitch_m, blade_count, chord_to_diameter_ratio, cl_alpha,
            cd0, induced_power_factor)

    def current_at(vbat_now, frac):
        hover = mm.equilibrium(
            effective_voltage(vbat_now, frac), vel, kv, unit["resistance"], unit["i0"],
            PLACEHOLDER_MOTOR_RTH, unit["prop_a_factor"], unit["prop_torque_factor"],
            unit["prop_max_rpm"], unit["thrust_factor_x"], unit["thrust_factor_y"],
            unit["thrust_factor_z"])
        return 4.0 * hover["current_a"]

    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int_ohm)  # OCV at full charge
    max_thrust0 = pa.bemt_thrust_torque(
        mm.steady_state_rpm(vbat0, vel, kv, unit["resistance"], unit["i0"],
                             unit["prop_a_factor"], unit["prop_torque_factor"],
                             unit["prop_max_rpm"], unit["thrust_factor_x"],
                             unit["thrust_factor_y"], unit["thrust_factor_z"]),
        vel, prop_diameter_m, pitch_m, blade_count, chord_to_diameter_ratio,
        cl_alpha, cd0, induced_power_factor)[0]
    feasible0 = max_thrust0 >= thrust_needed_per_motor

    hover_frac0 = solve_frac(vbat0)
    current0 = current_at(vbat0, hover_frac0)

    def step(carry, _):
        mah_drawn, t_s, current, still_flying = carry
        soc_frac = mah_drawn / capacity_mah
        vbat_now = bm.terminal_voltage(soc_frac, current, r_int_ohm)

        rpm_max = mm.steady_state_rpm(vbat_now, vel, kv, unit["resistance"], unit["i0"],
                                       unit["prop_a_factor"], unit["prop_torque_factor"],
                                       unit["prop_max_rpm"], unit["thrust_factor_x"],
                                       unit["thrust_factor_y"], unit["thrust_factor_z"])
        max_thrust_now = pa.bemt_thrust_torque(
            rpm_max, vel, prop_diameter_m, pitch_m, blade_count,
            chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)[0]

        can_continue = (still_flying
                         & (vbat_now > DISCHARGE_VOLTAGE_FLOOR_V)
                         & (mah_drawn < capacity_mah)
                         & (max_thrust_now >= thrust_needed_per_motor))

        frac_now = solve_frac(vbat_now)
        current_now = current_at(vbat_now, frac_now)

        new_current = jnp.where(can_continue, current_now, current)
        new_mah = mah_drawn + jnp.where(can_continue,
                                         new_current * (DISCHARGE_DT_S / 3600.0) * 1000.0,
                                         0.0)
        new_t_s = t_s + jnp.where(can_continue, DISCHARGE_DT_S, 0.0)
        return (new_mah, new_t_s, new_current, can_continue), None

    init = (jnp.array(0.0), jnp.array(0.0), current0, feasible0)
    (_, t_s_final, _, _), _ = jax.lax.scan(step, init, None, length=DISCHARGE_N_STEPS)

    return dict(feasible=feasible0, hover_frac0=hover_frac0, current0=current0,
                flight_time_s=t_s_final)


_hover_point_jitted = jax.jit(_hover_point_jit, static_argnums=())


def hover_point(x, battery_name="680mAh", vel=0.0, other_mass_kg=OTHER_MASS_KG,
                 chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                 cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                 induced_power_factor=pa.INDUCED_POWER_FACTOR):
    """Hover throttle/current at the start of a flight, plus estimated flight
    time from simulating the pack's discharge under constant hover thrust.

    Everything -- the per-timestep hover-throttle solve (Newton's method
    inside _hover_throttle_frac, exact-gradient rather than bisection since
    thrust(volts) is smooth and closed-form) and the discharge time-stepping
    itself -- runs inside one jax.jit'd lax.scan (_hover_point_jit), the same
    style as _solve_prop_for_motor's vmapped LBFGSB above: no Python-level
    loop dispatches an untraced JAX op per iteration, so a several-hundred-
    step discharge simulation stays fast enough for a UI callback.

    battery_name selects one of battery_model.BATTERIES (480mAh/580mAh/
    680mAh BetaFPV LAVA II 1S HV-LiPo) for capacity, mass, and internal
    resistance; vbat is no longer a free parameter here since the whole
    point of this function is to model how it sags, but the pack's own mass
    is folded into total_mass (a heavier or lighter battery changes hover
    thrust needed per motor, which is exactly the effect this function
    should reflect).

    The discharge loop is a fixed-length scan (DISCHARGE_N_STEPS steps of
    DISCHARGE_DT_S each) with a "still_flying" mask that freezes state once
    the pack empties, sags past DISCHARGE_VOLTAGE_FLOOR_V, or can no longer
    produce hover thrust even at full throttle -- so flight_time_s is exact
    to within one timestep regardless of when within the fixed length that
    happens, at the cost of always running the full step count. Raise
    DISCHARGE_N_STEPS if a design's flight time can exceed
    DISCHARGE_DT_S * DISCHARGE_N_STEPS (3600s / 60min by default).

    flight_time_min has no reserve margin -- i.e. "time to fully discharge
    (or hit the voltage floor) at a constant hover load", not a safe usable
    flight time. A real flight plan should keep a reserve (e.g. stop at 80%
    discharge); this deliberately reports the unpadded number so that choice
    stays visible rather than being silently baked in.
    """
    g = unpack(x)
    capacity_mah = bm.capacity_mah(battery_name)
    battery_mass_kg = bm.mass_kg(battery_name)
    r_int = bm.r_int_ohm(battery_name)

    r = _hover_point_jitted(
        vel, other_mass_kg, chord_to_diameter_ratio, cl_alpha, cd0,
        induced_power_factor, g["kv"], g["stator_volume_mm3"], g["prop_diameter_m"],
        g["blade_count"], g["pitch_m"], battery_mass_kg, capacity_mah, r_int)

    feasible = bool(r["feasible"])
    current0 = float(r["current0"])
    if not feasible:
        return {
            "feasible": False,
            "hover_throttle_frac": float(r["hover_frac0"]),
            "hover_current_a_per_motor": current0 / 4.0,
            "hover_current_a_total": current0,
            "battery_name": battery_name,
            "battery_mah": capacity_mah,
            "flight_time_min": 0.0,
        }

    return {
        "feasible": True,
        "hover_throttle_frac": float(r["hover_frac0"]),
        "hover_current_a_per_motor": current0 / 4.0,
        "hover_current_a_total": current0,
        "battery_name": battery_name,
        "battery_mah": capacity_mah,
        "flight_time_min": float(r["flight_time_s"]) / 60.0,
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


# --- Realized design: snap to an actually-buyable motor, then re-fit the prop
#
# The optimizer searches kV and stator_volume_mm3 as continuous variables and
# motor_scaling.py's fits stand in for "what would a motor here be like" --
# useful for search, but nothing says a motor at the optimizer's exact
# (kV, volume) is sold. This answers the question a build actually needs:
# given the optimizer's design point, which real cataloged motor (see
# data/motor_datasheets.csv) is closest, and what does the whole quad's
# performance look like using THAT motor's real datasheet mass/R/I0 instead
# of the fitted estimates -- i.e. the number you'd actually get, not the
# number the continuous search believes.
#
# Picking a motor purely by (kV, volume) distance to the *original*
# continuous optimum is not quite right, though: that optimum was found
# jointly with a particular prop, and a real motor's kV rarely matches the
# continuous kV exactly, which shifts what prop is actually best for it (a
# lower-kV motor wants a different pitch/diameter tradeoff to reach the same
# rpm-limited operating point, changing its torque load, which is what a
# motor is actually matched to -- not the raw kV/volume numbers in isolation).
# So this is genuinely a coupled fixed-point problem, not a one-shot lookup:
# fix a motor, find its best prop; that motor+prop's performance is what
# should decide which candidate motor is "best," not proximity alone; and in
# principle a different prop implies a different ideal motor, which could
# flip which real motor is closest. realized_design iterates: propose the
# n_candidates nearest motors by (kV, volume) distance to the current design
# point, optimize the prop for each, keep whichever motor+prop combination
# has the best TWR as the new design point, and repeat until the chosen
# motor stops changing (each round's winning motor+its optimized prop become
# next round's query point for a fresh nearest-candidates lookup -- since the
# winner's own real kV/volume are now exactly on the grid, that lookup
# reliably includes it again as a candidate, so the loop can only change
# course if a *different* candidate, evaluated with ITS optimal prop, now
# wins).


def _prop_only_cost(prop_x, kv, resistance, i0, other_mass_kg, motor_mass, vel, vbat,
                     chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor):
    """cost(), specialized to a fixed motor (kV/R/I0/mass) with only the three
    propeller variables (diameter, blade_count, pitch) free. Same penalty
    shape as cost() -- see its docstring -- just without the stator-volume
    floor penalty, since volume is not a free variable once a real motor is
    chosen."""
    diameter_m, blade_count, pitch_m = prop_x
    diameter_mm_ = diameter_m * 1e3
    prop_mass = ps.prop_mass_kg(diameter_mm_, blade_count)
    prop_inertia = ps.prop_inertia_kg_m2(prop_mass, diameter_mm_)
    aero = pa.to_simitl_params(diameter_m, pitch_m, blade_count,
                                chord_to_diameter_ratio, cl_alpha, cd0,
                                induced_power_factor)

    total_mass = other_mass_kg + fs.frame_mass_kg(diameter_m) + 4.0 * (motor_mass + prop_mass)
    weight_n = total_mass * G

    full_throttle = mm.equilibrium(
        vbat, vel, kv, resistance, i0, PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    twr = 4.0 * full_throttle["thrust_n"] / jnp.maximum(weight_n, 1e-9)

    spin_up_s = mm.spin_up_time_s(
        effective_voltage(vbat, SPIN_UP_START_FRAC),
        effective_voltage(vbat, SPIN_UP_END_FRAC), vel, kv, resistance, i0,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
        prop_inertia)

    tip_speed_m_s = jnp.pi * diameter_m * full_throttle["rpm"] / 60.0
    tip_mach = tip_speed_m_s / SPEED_OF_SOUND_M_S

    def penalty(shortfall, scale):
        return jnp.maximum(-shortfall / scale, 0.0) ** 2

    total_penalty = PENALTY_WEIGHT * (
        penalty(ESC_MAX_CURRENT_A - full_throttle["current_a"], CURRENT_SCALE_A)
        + penalty(SPIN_UP_BUDGET_S - spin_up_s, SPINUP_SCALE_S)
        + penalty(MAX_TIP_MACH - tip_mach, TIP_MACH_SCALE)
    )
    return -twr + total_penalty


_PROP_BOUNDS_LOWER = jnp.array([0.04, 2.0, 0.015])
_PROP_BOUNDS_UPPER = jnp.array([0.09, 4.0, 0.08])
_PROP_N_STARTS = 24
_PROP_POLISH_ITERS = 60


def _make_solve_prop_for_motor(prop_cost_fn):
    """Builds a jitted, vmapped-multi-start LBFGSB solver for a given
    per-prop cost function, so realized_design's search machinery is reusable
    for objectives other than "maximize TWR" -- e.g.
    optimize_efficiency.py's "minimize hover current subject to a TWR floor"
    needs the exact same nearest-motor/re-fit-the-prop loop, just scored
    differently. prop_cost_fn must have the same signature as
    _prop_only_cost (prop_x, kv, resistance, i0, other_mass_kg, motor_mass,
    vel, vbat, chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)
    -> scalar cost to minimize."""
    @jax.jit
    def _solve(starts, kv, resistance, i0, motor_mass, vel, vbat, other_mass_kg,
               chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor):
        def fun(prop_x):
            return prop_cost_fn(prop_x, kv, resistance, i0, other_mass_kg, motor_mass,
                                 vel, vbat, chord_to_diameter_ratio, cl_alpha, cd0,
                                 induced_power_factor)

        def polish(x0):
            res = LBFGSB(fun=fun, maxiter=_PROP_POLISH_ITERS).run(
                x0, bounds=(_PROP_BOUNDS_LOWER, _PROP_BOUNDS_UPPER))
            x = jnp.clip(res.params, _PROP_BOUNDS_LOWER, _PROP_BOUNDS_UPPER)
            return x, fun(x)

        xs, costs = jax.vmap(polish)(starts)
        costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
        best = jnp.argmin(costs)
        return xs[best], costs[best]

    return _solve


# vmapped LBFGSB over a scatter of starts, jitted as one unit -- same shape as
# optimize.py's _solve, so the per-candidate/per-round multi-start search
# this is called from (potentially dozens of times across realized_design's
# iterations x n_candidates) stays fast: everything inside is traced once
# and reused, instead of re-tracing a fresh LBFGSB call from Python for every
# start. Built once at import time for the default (TWR-maximizing) cost;
# other cost functions build their own via _make_solve_prop_for_motor.
_solve_prop_for_motor = _make_solve_prop_for_motor(_prop_only_cost)


def _optimize_prop_for_motor(kv, resistance, i0, motor_mass, vel, vbat, other_mass_kg,
                              chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor,
                              prop_x0, solve_fn=_solve_prop_for_motor):
    """Best (diameter, blade_count, pitch) for a fixed, fully-specified real
    motor. A handful of LBFGSB starts scattered plus the caller's previous
    prop as a seed -- this is a 3-variable, well-behaved sub-problem (the
    hard multi-modal search is what optimize.py's Adam/multi-start already
    solved to find a motor neighborhood; this only needs to locally refine
    the prop for one fixed motor), so it does not need Adam's global search.

    solve_fn defaults to the TWR-maximizing solver but accepts one built by
    _make_solve_prop_for_motor for a different objective (see
    optimize_efficiency.py).
    """
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(_PROP_BOUNDS_LOWER), np.asarray(_PROP_BOUNDS_UPPER)
    scatter = rng.random((_PROP_N_STARTS - 1, 3)) * (hi - lo) + lo
    starts = jnp.asarray(np.vstack([np.clip(np.asarray(prop_x0), lo, hi), scatter]))

    best_x, best_cost = solve_fn(
        starts, kv, resistance, i0, motor_mass, vel, vbat, other_mass_kg,
        chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)
    return best_x, best_cost


def _throttle_limited_operating_point(kv, resistance, i0, aero, prop_inertia,
                                       vbat, vel, esc_max=ESC_MAX_CURRENT_A,
                                       n_bisect=30):
    """Return the steady-state operating point with a software throttle cap.

    If full-throttle current is below esc_max, returns the true full-throttle
    point and t_lim=1.0.  Otherwise bisects the throttle fraction until the
    per-motor current is exactly esc_max, and returns the point at that
    voltage.  Spin-up time is recomputed as the 10% -> 90% of the *capped*
    max throttle (so a bigger prop that needs limiting is penalized less on
    spin-up, since it only has to reach the capped rpm).
    """
    full = mm.equilibrium(
        vbat, vel, kv, resistance, i0, PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    full_current = float(full["current_a"])

    if full_current <= esc_max + 1e-9:
        t_lim = 1.0
        limited = full
    else:
        lo, hi = 0.0, 1.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            mid_throttle = effective_voltage(vbat, mid)
            r = mm.equilibrium(
                mid_throttle, vel, kv, resistance, i0, PLACEHOLDER_MOTOR_RTH,
                aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
                aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
            if float(r["current_a"]) > esc_max:
                hi = mid
            else:
                lo = mid
        t_lim = 0.5 * (lo + hi)
        limited = mm.equilibrium(
            effective_voltage(vbat, t_lim), vel, kv, resistance, i0, PLACEHOLDER_MOTOR_RTH,
            aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
            aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])

    # Spin-up is evaluated from 10% to 90% of the capped max throttle.
    spin_end_frac = SPIN_UP_END_FRAC * t_lim
    spin_end_frac = max(spin_end_frac, SPIN_UP_START_FRAC + 1e-3)
    spin_up_s = mm.spin_up_time_s(
        effective_voltage(vbat, SPIN_UP_START_FRAC),
        effective_voltage(vbat, spin_end_frac), vel,
        kv, resistance, i0, aero["prop_a_factor"], aero["prop_torque_factor"],
        aero["prop_max_rpm"], aero["thrust_factor_x"], aero["thrust_factor_y"],
        aero["thrust_factor_z"], prop_inertia)

    return limited, spin_up_s, t_lim


def _evaluate_motor_with_prop(c, prop_diameter_m, blade_count, pitch_m, vel, vbat,
                               other_mass_kg, chord_to_diameter_ratio, cl_alpha, cd0,
                               induced_power_factor, prop_mass_kg_override=None):
    """Full evaluate()-shaped result for one catalogue motor candidate `c`
    (see motor_scaling.nearest_catalogue_motor) paired with a specific prop.

    prop_mass_kg_override, when given, replaces prop_scaling's fitted mass
    estimate with a real part's own datasheet mass (see
    prop_scaling.nearest_catalogue_prop) -- used when the prop itself is also
    a real catalogue part, not just an idealized (diameter, pitch,
    blade_count) point.

    Real motor/prop pairs are allowed to hit the ESC current cap; if they do,
    this routine applies a software throttle limit so the reported full-throttle
    thrust, current, spin-up and tip Mach are all at the capped operating point,
    while hover (and therefore hover efficiency) is unchanged.
    """
    motor_mass = c["mass_g"] * 1e-3
    resistance = c["resistance_ohm"]
    i0 = c["i0_a"] if c["i0_a"] is not None else no_load_current_a(c["kv_rpm_per_v"])

    diameter_mm_ = prop_diameter_m * 1e3
    prop_mass = (prop_mass_kg_override if prop_mass_kg_override is not None
                 else ps.prop_mass_kg(diameter_mm_, blade_count))
    prop_inertia = ps.prop_inertia_kg_m2(prop_mass, diameter_mm_)
    aero = pa.to_simitl_params(prop_diameter_m, pitch_m, blade_count,
                                chord_to_diameter_ratio, cl_alpha, cd0,
                                induced_power_factor)

    frame_mass = fs.frame_mass_kg(prop_diameter_m)
    total_mass = other_mass_kg + frame_mass + 4.0 * (motor_mass + prop_mass)
    weight_n = total_mass * G

    limited, spin_up_s, t_lim = _throttle_limited_operating_point(
        c["kv_rpm_per_v"], resistance, i0, aero, prop_inertia, vbat, vel)

    twr = 4.0 * limited["thrust_n"] / jnp.maximum(weight_n, 1e-9)
    tip_speed_m_s = jnp.pi * prop_diameter_m * limited["rpm"] / 60.0
    tip_mach = tip_speed_m_s / SPEED_OF_SOUND_M_S

    return dict(
        motor=c, motor_mass=motor_mass, prop_mass=prop_mass, frame_mass=frame_mass,
        prop_diameter_m=prop_diameter_m, blade_count=blade_count, pitch_m=pitch_m,
        total_mass=total_mass, weight_n=weight_n, twr=twr,
        spin_up_s=spin_up_s, tip_mach=tip_mach, throttle_limit=t_lim,
        **limited)


def _default_prop_cost(result, min_twr=None):
    """Default scoring for _best_catalogue_prop_for_motor: maximize TWR (same
    objective the continuous _prop_only_cost defaults to), expressed as a
    cost to MINIMIZE so lower is better, for consistency with every other
    cost function in this module. min_twr is accepted for interface
    symmetry with optimize_efficiency's scorer but unused here."""
    return -float(result["twr"])


def _best_catalogue_prop_for_motor(c, vel, vbat, other_mass_kg, chord_to_diameter_ratio,
                                    cl_alpha, cd0, induced_power_factor, prop_x0,
                                    n_prop_candidates=30, prop_cost_fn=_default_prop_cost):
    """Best REAL, buyable propeller (see prop_scaling.nearest_catalogue_prop)
    for a fixed, fully-specified real motor `c` -- the discrete counterpart
    to _optimize_prop_for_motor's continuous LBFGSB search.

    Both motor and prop are now small (~20 and ~27 entry) real catalogues, so
    this is a brute-force evaluate-and-rank rather than a gradient search:
    look up the n_prop_candidates real props nearest prop_x0 (the previous
    round's diameter/pitch/blade_count, continuous or real), evaluate the
    whole quad with each paired against motor `c` using ITS OWN datasheet
    mass (not the fitted estimate), and keep whichever prop_cost_fn scores
    best. prop_cost_fn takes an _evaluate_motor_with_prop-shaped result dict
    and returns a scalar cost to minimize (default: -TWR); pass a different
    one for a different objective, e.g. optimize_efficiency.py's hover
    current subject to a TWR floor.

    Returns (result, prop_row) -- the winning evaluate()-shaped dict and the
    nearest_catalogue_prop row it came from -- or (None, None) if no real
    prop is on record (should not happen with a non-empty catalogue, but
    kept explicit rather than assumed).
    """
    diameter_mm0 = float(prop_x0[0]) * 1e3
    pitch_mm0 = float(prop_x0[2]) * 1e3
    blade_count0 = float(prop_x0[1])
    prop_candidates = ps.nearest_catalogue_prop(
        diameter_mm0, pitch_mm0, blade_count0, n=n_prop_candidates)

    best_result, best_prop, best_cost = None, None, float("inf")
    for prop in prop_candidates:
        result = _evaluate_motor_with_prop(
            c, prop["diameter_mm"] * 1e-3, prop["blade_count"], prop["pitch_mm"] * 1e-3,
            vel, vbat, other_mass_kg, chord_to_diameter_ratio, cl_alpha, cd0,
            induced_power_factor, prop_mass_kg_override=prop["mass_g"] * 1e-3)
        result["prop"] = prop
        cost = prop_cost_fn(result)
        if cost < best_cost:
            best_result, best_prop, best_cost = result, prop, cost

    return best_result, best_prop


def realized_design(x, vel=0.0, vbat=VBAT, other_mass_kg=OTHER_MASS_KG,
                     n_candidates=21, n_prop_candidates=30, max_iters=6,
                     chord_to_diameter_ratio=pa.CHORD_TO_DIAMETER_RATIO,
                     cl_alpha=pa.CL_ALPHA, cd0=pa.CD0,
                     induced_power_factor=pa.INDUCED_POWER_FACTOR,
                     prop_cost_fn=_default_prop_cost,
                     use_catalogue_props=True, solve_fn=_solve_prop_for_motor):
    """Re-evaluate a design point's whole-quad performance using a real,
    buyable motor AND a real, buyable propeller (see
    prop_scaling.nearest_catalogue_prop) -- iterated to a fixed point, since
    picking a motor and picking a prop are coupled (see the module comment
    above).

    Each round: propose the n_candidates nearest catalogue motors to the
    current design point (motor_scaling.nearest_catalogue_motor), skip any
    missing mass/resistance (can't be simulated), find the best REAL prop for
    each fully-specified candidate motor (via _best_catalogue_prop_for_motor:
    brute-force evaluate the n_prop_candidates nearest catalogue props,
    scored by prop_cost_fn -- default TWR-maximizing; pass a different one
    for a different objective, e.g. optimize_efficiency.py's "minimize hover
    current subject to a TWR floor"), and keep the motor+prop combination
    prop_cost_fn scores best. That winner becomes next round's design point.
    Stops when the chosen motor is the same real part two rounds in a row, or
    after max_iters rounds if it keeps oscillating between two similarly-good
    motors (rare, but with only ~20 catalogue motors and no strict
    monotonicity proof across the swap, worth capping rather than assuming).

    use_catalogue_props=False falls back to the OLD behavior: a continuous
    LBFGSB search over idealized (diameter, pitch, blade_count) via solve_fn,
    with no real prop backing it. Kept for callers that specifically want the
    idealized-prop answer (e.g. to compare against the catalogue-snapped
    one) -- but the idealized search has no penalty for wandering into a
    pitch/diameter ratio no real prop in this size class has ever been sold
    at (real open props top out around P/D~1.1, ducted cinewhoop props
    ~1.2 -- see prop_scaling.nearest_catalogue_prop's docstring), which is
    exactly the failure mode the catalogue-backed default exists to avoid.

    Returns a dict with "best" (the winning round's evaluate()-shaped result,
    plus "motor", "prop_diameter_m", "blade_count", "pitch_m", and -- when
    use_catalogue_props -- "prop", the winning prop_scaling.nearest_catalogue_prop
    row; None if no round ever found a fully-specified candidate),
    "converged" (True if the motor choice stabilized before max_iters), and
    "iterations" (rounds run).
    """
    g = unpack(x)
    kv, volume = g["kv"], g["stator_volume_mm3"]
    prop_x0 = jnp.array([g["prop_diameter_m"], g["blade_count"], g["pitch_m"]])

    best = None
    prev_motor_name = None
    converged = False
    iterations = 0

    for iterations in range(1, max_iters + 1):
        candidates = ms.nearest_catalogue_motor(kv, volume, n=n_candidates)
        round_best = None
        round_best_cost = float("inf")

        for c in candidates:
            if c["mass_g"] is None or c["resistance_ohm"] is None:
                continue

            if use_catalogue_props:
                result, prop = _best_catalogue_prop_for_motor(
                    c, vel, vbat, other_mass_kg, chord_to_diameter_ratio, cl_alpha,
                    cd0, induced_power_factor, prop_x0,
                    n_prop_candidates=n_prop_candidates, prop_cost_fn=prop_cost_fn)
                if result is None:
                    continue
                cost = prop_cost_fn(result)
            else:
                i0 = c["i0_a"] if c["i0_a"] is not None else no_load_current_a(c["kv_rpm_per_v"])
                prop_vec, prop_cost = _optimize_prop_for_motor(
                    c["kv_rpm_per_v"], c["resistance_ohm"], i0, c["mass_g"] * 1e-3,
                    vel, vbat, other_mass_kg, chord_to_diameter_ratio, cl_alpha, cd0,
                    induced_power_factor, prop_x0, solve_fn=solve_fn)
                diameter_m, blade_count, pitch_m = (float(v) for v in prop_vec)
                result = _evaluate_motor_with_prop(
                    c, diameter_m, blade_count, pitch_m, vel, vbat, other_mass_kg,
                    chord_to_diameter_ratio, cl_alpha, cd0, induced_power_factor)
                cost = float(prop_cost)

            if cost < round_best_cost:
                round_best_cost = cost
                round_best = result

        if round_best is None:
            break

        best = round_best
        if best["motor"]["name"] == prev_motor_name:
            converged = True
            break
        prev_motor_name = best["motor"]["name"]

        kv = best["motor"]["kv_rpm_per_v"]
        volume = best["motor"]["volume_mm3"]
        prop_x0 = jnp.array([best["prop_diameter_m"], best["blade_count"], best["pitch_m"]])

    return {"best": best, "converged": converged, "iterations": iterations}
