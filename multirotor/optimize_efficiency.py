"""Multi-start bounded optimization for the most efficient design that still
hits a minimum thrust-to-weight ratio, rather than optimize.py's "maximize
TWR" objective.

"Most efficient" is taken to mean minimum hover ELECTRICAL POWER (current x
hover voltage) on a fixed battery -- not minimum hover current alone. Current
by itself is not a fair comparison across designs: different props reach
their hover thrust at different throttle fractions of the pack's fixed OCV,
so two designs can draw the same current at very different voltages (and
hence very different power). Minimizing current in isolation was found to
reward small, high-kV motor/prop combos that hover at low throttle/voltage
even though they draw more actual power than a bigger, lower-kV combo
hovering at higher throttle -- exactly backwards from real multirotor
efficiency intuition (bigger/slower props are normally more efficient per
unit thrust), and confirmed by checking: a real Gemfan SPEEDX2 1202.5
bench-test data point (data/speedx2_1202_5_throttle_sweep.csv) shows a
larger prop achieving MORE grams per watt than a smaller one at comparable
thrust, the opposite of what a current-only objective would predict. Power
is the correct apples-to-apples quantity.

No thrust normalization (g/W) is used here: the objective is "minimize the
watts this specific vehicle burns to hover itself," not "compare efficiency
across vehicles of different mass" -- a smaller/lighter design that draws
less power to hover ITSELF is doing better on the metric that actually
determines this vehicle's flight time, without needing a separate
thrust-per-watt figure of merit.

Flight time itself would be the more literal target, but the discharge
simulation inside hover_point is a fixed-length lax.scan with a boolean
still-flying mask -- flight_time_s is piecewise-constant almost everywhere
in design-variable space (it only changes when a perturbation flips which
timestep the mask trips), so its gradient is zero almost everywhere and
useless to a gradient-based optimizer (checked directly: jax.grad of
flight_time_s on quad_model.BASELINE returns all zeros). Hover power from
the same function's Newton solve (_hover_throttle_frac) is smooth and has a
real, nonzero gradient, and minimizing it is a good proxy for maximizing
flight time for a fixed battery capacity (exactly equivalent if the pack's
terminal voltage did not sag with state of charge; close in practice since
this is evaluated at full charge), so that is the actual objective here.

Same shape as optimize.py otherwise: Adam over a scatter of starts for the
global search, then a short LBFGSB polish on the best few. The full-throttle
TWR/current/spin-up/tip-Mach/stator-volume constraints from optimize.py's
cost are kept (a design has to still be buildable, not just efficient at
hover), plus a new TWR-floor constraint.

Run with: uv run python -m multirotor.optimize_efficiency
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

import multirotor.battery_model as bm
import multirotor.optimize as opt
import multirotor.quad_model as qm

N_STARTS = 256
ADAM_STEPS = 400
ADAM_LR = 0.06
N_POLISH = 8
POLISH_ITERS = 30

MIN_TWR = 4.0
BATTERY_NAME = "680mAh"

PENALTY_WEIGHT = 1e3
TWR_SCALE = 0.5
CURRENT_SCALE_A = 1.0
SPINUP_SCALE_S = 0.01
VOLUME_SCALE_MM3 = 20.0
TIP_MACH_SCALE = 0.05

# Reuse optimize.py's box -- same realistic tiny-whoop/toothpick range, no
# reason for the efficiency search to cover different hardware.
BOUNDS = opt.BOUNDS


def _bounds_arrays():
    return opt._bounds_arrays()


def hover_power_w(x, vel=0.0, battery_name=BATTERY_NAME,
                   other_mass_kg=qm.OTHER_MASS_KG):
    """Total hover electrical power (all 4 motors, at full charge) -- the
    smooth objective this module actually optimizes. See module docstring
    for why power (current x voltage) rather than current alone: hover
    voltage (throttle fraction x pack OCV) differs between designs, so
    current by itself is not comparable across them."""
    g = qm.unpack(x)
    capacity_mah = bm.capacity_mah(battery_name)
    battery_mass_kg = bm.mass_kg(battery_name)
    r_int = bm.r_int_ohm(battery_name)
    r = qm._hover_point_jitted(
        vel, other_mass_kg, qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA,
        qm.pa.CD0, qm.pa.INDUCED_POWER_FACTOR, g["kv"], g["stator_volume_mm3"],
        g["prop_diameter_m"], g["blade_count"], g["pitch_m"], battery_mass_kg,
        capacity_mah, r_int)
    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)  # OCV at full charge, matches _hover_point_jit
    hover_voltage = r["hover_frac0"] * vbat0
    return r["current0"] * hover_voltage


def cost(x, vel=0.0, min_twr=MIN_TWR, battery_name=BATTERY_NAME,
         other_mass_kg=qm.OTHER_MASS_KG):
    """Minimize hover power subject to a TWR floor plus the same
    buildability constraints optimize.py's cost enforces (current cap,
    spin-up budget, stator-volume floor, tip-Mach ceiling) -- a design that
    is efficient at hover but can't reach min_twr, draws over the ESC's
    limit at full throttle, or is otherwise unbuildable is not useful."""
    r = qm.evaluate(x, vel, other_mass_kg=other_mass_kg)
    c = qm.constraints(x, vel)
    g = qm.unpack(x)

    def penalty(shortfall, scale):
        return jnp.maximum(-shortfall / scale, 0.0) ** 2

    twr_slack = r["twr"] - min_twr
    total_penalty = PENALTY_WEIGHT * (
        penalty(twr_slack, TWR_SCALE)
        + penalty(c["current_slack_a"], CURRENT_SCALE_A)
        + penalty(c["spinup_slack_s"], SPINUP_SCALE_S)
        + penalty(g["stator_volume_mm3"] - qm.STATOR_VOLUME_FLOOR_MM3, VOLUME_SCALE_MM3)
        + penalty(c["tip_mach_slack"], TIP_MACH_SCALE)
    )
    return hover_power_w(x, vel, battery_name, other_mass_kg) + total_penalty


@jax.jit
def _solve(x0s, lower, upper):
    def fun(x):
        return cost(x)

    grads = jax.vmap(jax.grad(fun))
    scale = jnp.maximum(upper - lower, 1e-9)

    def adam_step(state, k):
        x, m, v, t = state
        g = grads(x) / scale
        t = t + 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mhat = m / (1.0 - 0.9 ** t)
        vhat = v / (1.0 - 0.999 ** t)
        lr = ADAM_LR * 0.5 * (1.0 + jnp.cos(jnp.pi * k / ADAM_STEPS))
        x = jnp.clip(x - lr * scale * mhat / (jnp.sqrt(vhat) + 1e-8), lower, upper)
        return (x, m, v, t), None

    zeros = jnp.zeros_like(x0s)
    (xs, _, _, _), _ = jax.lax.scan(
        adam_step, (x0s, zeros, zeros, 0), jnp.arange(ADAM_STEPS)
    )

    costs = jax.vmap(fun)(xs)
    costs = jnp.where(jnp.isfinite(costs), costs, jnp.inf)
    _, top = jax.lax.top_k(-costs, N_POLISH)

    def polish(x0):
        res = LBFGSB(fun=fun, maxiter=POLISH_ITERS).run(x0, bounds=(lower, upper))
        x = jnp.clip(res.params, lower, upper)
        return x, fun(x)

    px, pcosts = jax.vmap(polish)(xs[top])
    pcosts = jnp.where(jnp.isfinite(pcosts), pcosts, jnp.inf)

    allx = jnp.concatenate([xs, px])
    allc = jnp.concatenate([costs, pcosts])
    best = jnp.argmin(allc)
    return allx[best], allc[best]


def _starts(lower, upper):
    rng = np.random.default_rng(0)
    lo, hi = np.asarray(lower), np.asarray(upper)
    scatter = rng.random((N_STARTS - 1, lo.size)) * (hi - lo) + lo
    base = np.clip(np.asarray(qm.BASELINE), lo, hi)
    return jnp.asarray(np.vstack([base, scatter]))


def optimize():
    lower, upper = _bounds_arrays()
    started = time.perf_counter()
    best_x, best_cost = _solve(_starts(lower, upper), lower, upper)
    best_x.block_until_ready()
    return {
        "x": best_x,
        "cost": float(best_cost),
        "elapsed": time.perf_counter() - started,
    }


def _print_report(x):
    r = qm.evaluate(x)
    c = qm.constraints(x)
    g = qm.unpack(x)
    hover = qm.hover_point(x, battery_name=BATTERY_NAME)

    print(f"\n{'design variable':<24}{'value':>14}{'at bound':>10}")
    for name, value in zip(qm.DESIGN_VARS, np.asarray(x)):
        lo, hi = BOUNDS[name]
        at = "lower" if abs(value - lo) < 1e-6 * max(abs(lo), 1.0) else (
            "upper" if abs(value - hi) < 1e-6 * max(abs(hi), 1.0) else "")
        print(f"  {name:<22}{value:14.5f}{at:>10}")

    print(f"\n{'quantity':<24}{'value':>14}")
    rows = [
        ("resistance, ohm", float(r["unit"]["resistance"]), "{:.4f}"),
        ("motor mass, g", float(r["unit"]["motor_mass"]) * 1e3, "{:.2f}"),
        ("prop mass, g", float(r["unit"]["prop_mass"]) * 1e3, "{:.2f}"),
        ("frame mass (est.), g", float(r["frame_mass"]) * 1e3, "{:.2f}"),
        ("total mass, g", float(r["total_mass"]) * 1e3, "{:.1f}"),
        ("rpm @ full throttle", float(r["rpm"]), "{:.0f}"),
        ("thrust/motor, N", float(r["thrust_n"]), "{:.3f}"),
        ("current/motor @ full throttle, A", float(r["current_a"]), "{:.2f}"),
        ("TWR @ full throttle", float(r["twr"]), "{:.2f}"),
        ("spin-up 10-90%, ms", float(r["spin_up_s"]) * 1e3, "{:.1f}"),
        ("tip Mach @ full throttle", float(r["tip_mach"]), "{:.3f}"),
        ("hover throttle frac", hover["hover_throttle_frac"], "{:.3f}"),
        ("hover current, A (total)", hover["hover_current_a_total"], "{:.3f}"),
        ("hover power, W (total)", float(hover_power_w(x, battery_name=BATTERY_NAME)), "{:.3f}"),
        (f"flight time, min ({BATTERY_NAME})", hover["flight_time_min"], "{:.1f}"),
    ]
    for label, value, fmt in rows:
        print(f"  {label:<32}{fmt.format(value):>12}")

    print(f"\n{'constraint':<24}{'slack':>14}")
    all_constraints = dict(c)
    all_constraints["twr_slack (>= %.1f)" % MIN_TWR] = float(r["twr"]) - MIN_TWR
    for name, value in all_constraints.items():
        v = float(value)
        mark = "" if v >= -1e-9 else "   <-- VIOLATED"
        print(f"  {name:<22}{v:14.4f}{mark}")


# --- Realize against a real motor, using the SAME efficiency objective -----
#
# quad_model.realized_design's default prop sub-optimizer maximizes TWR, so
# calling it as-is on an efficiency-optimized continuous design point would
# re-fit the prop for max thrust again once a real motor is chosen --
# defeating the point (confirmed: doing exactly that overshoots to TWR ~9.3
# and burns more hover power than needed for a TWR-4 design). Instead this
# builds a prop sub-solver scored by the SAME "minimize hover power subject
# to a TWR floor" cost used above (see module docstring for why power, not
# current alone), via quad_model._make_solve_prop_for_motor, and passes it
# into realized_design so the real-motor search stays consistent with what
# was actually optimized.


def _prop_only_efficiency_cost(prop_x, kv, resistance, i0, other_mass_kg, motor_mass,
                                vel, vbat, chord_to_diameter_ratio, cl_alpha, cd0,
                                induced_power_factor, min_twr=MIN_TWR,
                                battery_name=BATTERY_NAME):
    """Same signature as quad_model._prop_only_cost (so it plugs into
    _make_solve_prop_for_motor/realized_design), scored by hover POWER
    instead of -TWR -- see module docstring for why power, not current
    alone. vbat is unused for the objective itself (hover power is computed
    from the battery's own OCV, not the fixed-throttle vbat quad_model.cost
    uses) but kept in the signature for interface compatibility with the
    TWR-maximizing cost."""
    diameter_m, blade_count, pitch_m = prop_x
    diameter_mm_ = diameter_m * 1e3
    prop_mass = qm.ps.prop_mass_kg(diameter_mm_, blade_count)
    prop_inertia = qm.ps.prop_inertia_kg_m2(prop_mass, diameter_mm_)
    aero = qm.pa.to_simitl_params(diameter_m, pitch_m, blade_count,
                                   chord_to_diameter_ratio, cl_alpha, cd0,
                                   induced_power_factor)

    capacity_mah = bm.capacity_mah(battery_name)
    battery_mass_kg = bm.mass_kg(battery_name)
    r_int = bm.r_int_ohm(battery_name)

    total_mass = (other_mass_kg + qm.fs.frame_mass_kg(diameter_m) + battery_mass_kg
                  + 4.0 * (motor_mass + prop_mass))
    weight_n = total_mass * qm.G
    thrust_needed_per_motor = weight_n / 4.0

    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)
    hover_frac = qm._hover_throttle_frac(
        vbat0, thrust_needed_per_motor, vel, kv, resistance, i0,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
        diameter_m, pitch_m, blade_count, chord_to_diameter_ratio, cl_alpha, cd0,
        induced_power_factor)
    hover = qm.mm.equilibrium(
        hover_frac * vbat0, vel, kv, resistance, i0, qm.PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    hover_voltage = hover_frac * vbat0
    hover_power_total = 4.0 * hover["current_a"] * hover_voltage

    full_throttle = qm.mm.equilibrium(
        vbat, vel, kv, resistance, i0, qm.PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    twr = 4.0 * full_throttle["thrust_n"] / jnp.maximum(weight_n, 1e-9)
    spin_up_s = qm.mm.spin_up_time_s(
        qm.SPIN_UP_START_FRAC * vbat, qm.SPIN_UP_END_FRAC * vbat, vel, kv, resistance, i0,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
        prop_inertia)
    tip_speed_m_s = jnp.pi * diameter_m * full_throttle["rpm"] / 60.0
    tip_mach = tip_speed_m_s / qm.SPEED_OF_SOUND_M_S

    def penalty(shortfall, scale):
        return jnp.maximum(-shortfall / scale, 0.0) ** 2

    total_penalty = PENALTY_WEIGHT * (
        penalty(twr - min_twr, TWR_SCALE)
        + penalty(qm.ESC_MAX_CURRENT_A - full_throttle["current_a"], CURRENT_SCALE_A)
        + penalty(qm.SPIN_UP_BUDGET_S - spin_up_s, SPINUP_SCALE_S)
        + penalty(qm.MAX_TIP_MACH - tip_mach, TIP_MACH_SCALE)
    )
    return hover_power_total + total_penalty


_solve_prop_for_efficiency = qm._make_solve_prop_for_motor(_prop_only_efficiency_cost)


def _result_efficiency_cost(result, min_twr=MIN_TWR, battery_name=BATTERY_NAME,
                             other_mass_kg=qm.OTHER_MASS_KG):
    """prop_cost_fn for quad_model.realized_design's catalogue-prop search:
    same "minimize hover power subject to a TWR floor" objective as
    _prop_only_efficiency_cost (see module docstring for why power, not
    current alone), but scored from an already-evaluated
    _evaluate_motor_with_prop result (a real motor+real prop pair) instead of
    a raw (diameter, blade_count, pitch) vector -- this is what lets
    realize_efficient_design rank real catalogue props by the same objective
    the continuous search used, rather than falling back to TWR."""
    m = result["motor"]
    resistance = m["resistance_ohm"]
    i0 = m["i0_a"] if m["i0_a"] is not None else qm.no_load_current_a(m["kv_rpm_per_v"])
    capacity_mah = bm.capacity_mah(battery_name)
    battery_mass_kg = bm.mass_kg(battery_name)
    r_int = bm.r_int_ohm(battery_name)

    # total_mass in `result` already includes the motor+prop but not the
    # battery pack itself (evaluate()/_evaluate_motor_with_prop don't know
    # about a battery) -- add it here, same as hover_point does.
    total_mass = float(result["total_mass"]) + battery_mass_kg
    weight_n = total_mass * qm.G
    thrust_needed_per_motor = weight_n / 4.0

    aero = qm.pa.to_simitl_params(
        result["prop_diameter_m"], result["pitch_m"], result["blade_count"])

    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)
    hover_frac = qm._hover_throttle_frac(
        vbat0, thrust_needed_per_motor, 0.0, m["kv_rpm_per_v"], resistance, i0,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
        result["prop_diameter_m"], result["pitch_m"], result["blade_count"],
        qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA, qm.pa.CD0,
        qm.pa.INDUCED_POWER_FACTOR)
    hover = qm.mm.equilibrium(
        hover_frac * vbat0, 0.0, m["kv_rpm_per_v"], resistance, i0, qm.PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    hover_voltage = float(hover_frac) * vbat0
    hover_power_total = float(4.0 * hover["current_a"]) * hover_voltage

    def penalty(shortfall, scale):
        return max(-shortfall / scale, 0.0) ** 2

    total_penalty = PENALTY_WEIGHT * (
        penalty(float(result["twr"]) - min_twr, TWR_SCALE)
        + penalty(qm.ESC_MAX_CURRENT_A - float(result["current_a"]), CURRENT_SCALE_A)
        + penalty(qm.SPIN_UP_BUDGET_S - float(result["spin_up_s"]), SPINUP_SCALE_S)
        + penalty(qm.MAX_TIP_MACH - float(result["tip_mach"]), TIP_MACH_SCALE)
    )
    return hover_power_total + total_penalty


def realize_efficient_design(x, n_candidates=21, n_prop_candidates=30, max_iters=6,
                              min_twr=MIN_TWR, battery_name=BATTERY_NAME,
                              other_mass_kg=qm.OTHER_MASS_KG, use_catalogue_props=True):
    """realized_design, scored by this module's efficiency objective instead
    of TWR -- see module comment above for why a plain realized_design(x)
    call is the wrong tool once the continuous design point was itself
    optimized for efficiency rather than max thrust.

    use_catalogue_props=True (the default, matching quad_model.realized_design)
    picks from real, buyable props (data/prop_datasheets.csv), scored by
    _result_efficiency_cost. Pass False to fall back to the old idealized
    continuous prop search via _solve_prop_for_efficiency."""
    def prop_cost_fn(result):
        return _result_efficiency_cost(result, min_twr, battery_name, other_mass_kg)

    return qm.realized_design(
        x, other_mass_kg=other_mass_kg, n_candidates=n_candidates,
        n_prop_candidates=n_prop_candidates, max_iters=max_iters,
        prop_cost_fn=prop_cost_fn, use_catalogue_props=use_catalogue_props,
        solve_fn=_solve_prop_for_efficiency)


def _result_hover_point(result, battery_name=BATTERY_NAME, other_mass_kg=qm.OTHER_MASS_KG):
    """hover_point-equivalent computed from an already-evaluated
    _evaluate_motor_with_prop result (real motor, and possibly real prop --
    if result carries "prop", its real datasheet mass is already reflected
    in result["total_mass"], unlike qm.hover_point which always uses the
    fitted prop_scaling estimate)."""
    m = result["motor"]
    resistance = m["resistance_ohm"]
    i0 = m["i0_a"] if m["i0_a"] is not None else qm.no_load_current_a(m["kv_rpm_per_v"])
    capacity_mah = bm.capacity_mah(battery_name)
    battery_mass_kg = bm.mass_kg(battery_name)
    r_int = bm.r_int_ohm(battery_name)

    total_mass = float(result["total_mass"]) + battery_mass_kg
    weight_n = total_mass * qm.G
    thrust_needed_per_motor = weight_n / 4.0

    aero = qm.pa.to_simitl_params(
        result["prop_diameter_m"], result["pitch_m"], result["blade_count"])

    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)
    max_thrust0 = qm.pa.bemt_thrust_torque(
        qm.mm.steady_state_rpm(vbat0, 0.0, m["kv_rpm_per_v"], resistance, i0,
                                aero["prop_a_factor"], aero["prop_torque_factor"],
                                aero["prop_max_rpm"], aero["thrust_factor_x"],
                                aero["thrust_factor_y"], aero["thrust_factor_z"]),
        0.0, result["prop_diameter_m"], result["pitch_m"], result["blade_count"],
        qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA, qm.pa.CD0,
        qm.pa.INDUCED_POWER_FACTOR)[0]
    feasible = bool(max_thrust0 >= thrust_needed_per_motor)

    hover_frac = qm._hover_throttle_frac(
        vbat0, thrust_needed_per_motor, 0.0, m["kv_rpm_per_v"], resistance, i0,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
        result["prop_diameter_m"], result["pitch_m"], result["blade_count"],
        qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA, qm.pa.CD0,
        qm.pa.INDUCED_POWER_FACTOR)
    hover = qm.mm.equilibrium(
        hover_frac * vbat0, 0.0, m["kv_rpm_per_v"], resistance, i0, qm.PLACEHOLDER_MOTOR_RTH,
        aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
        aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
    current0 = float(4.0 * hover["current_a"])

    if not feasible:
        return {
            "feasible": False,
            "hover_throttle_frac": float(hover_frac),
            "hover_current_a_total": current0,
            "flight_time_min": 0.0,
        }

    # quad_model._hover_point_jit derives motor R/I0/mass from the FITTED
    # scaling laws via motor_prop_unit, which would silently discard this
    # real motor's actual datasheet values -- so the discharge simulation is
    # re-run here directly against the real (kV, R, I0), matching
    # _hover_point_jit's own step() logic exactly except for that swap.
    return _discharge_with_real_motor(
        m, result["prop_diameter_m"], result["pitch_m"], result["blade_count"],
        thrust_needed_per_motor, vbat0, capacity_mah, r_int, hover_frac, current0)


def _discharge_with_real_motor(m, prop_diameter_m, pitch_m, blade_count,
                                thrust_needed_per_motor, vbat0, capacity_mah, r_int,
                                hover_frac0, current0):
    """The same fixed-length discharge simulation as quad_model._hover_point_jit's
    step(), but parameterized by a real motor's own (kV, R, I0) instead of
    deriving them from motor_prop_unit's fitted scaling laws -- needed
    because this module's realized designs use a real catalogue motor whose
    R/I0 usually differ from what the fit would predict at its nominal
    stator volume."""
    resistance = m["resistance_ohm"]
    i0 = m["i0_a"] if m["i0_a"] is not None else qm.no_load_current_a(m["kv_rpm_per_v"])
    kv = m["kv_rpm_per_v"]
    aero = qm.pa.to_simitl_params(prop_diameter_m, pitch_m, blade_count)

    def solve_frac(vbat_now):
        return qm._hover_throttle_frac(
            vbat_now, thrust_needed_per_motor, 0.0, kv, resistance, i0,
            aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
            aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"],
            prop_diameter_m, pitch_m, blade_count, qm.pa.CHORD_TO_DIAMETER_RATIO,
            qm.pa.CL_ALPHA, qm.pa.CD0, qm.pa.INDUCED_POWER_FACTOR)

    def current_at(vbat_now, frac):
        hover = qm.mm.equilibrium(
            frac * vbat_now, 0.0, kv, resistance, i0, qm.PLACEHOLDER_MOTOR_RTH,
            aero["prop_a_factor"], aero["prop_torque_factor"], aero["prop_max_rpm"],
            aero["thrust_factor_x"], aero["thrust_factor_y"], aero["thrust_factor_z"])
        return 4.0 * hover["current_a"]

    def step(carry, _):
        mah_drawn, t_s, current, still_flying = carry
        soc_frac = mah_drawn / capacity_mah
        vbat_now = bm.terminal_voltage(soc_frac, current, r_int)

        rpm_max = qm.mm.steady_state_rpm(
            vbat_now, 0.0, kv, resistance, i0, aero["prop_a_factor"],
            aero["prop_torque_factor"], aero["prop_max_rpm"], aero["thrust_factor_x"],
            aero["thrust_factor_y"], aero["thrust_factor_z"])
        max_thrust_now = qm.pa.bemt_thrust_torque(
            rpm_max, 0.0, prop_diameter_m, pitch_m, blade_count,
            qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA, qm.pa.CD0,
            qm.pa.INDUCED_POWER_FACTOR)[0]

        can_continue = (still_flying
                         & (vbat_now > qm.DISCHARGE_VOLTAGE_FLOOR_V)
                         & (mah_drawn < capacity_mah)
                         & (max_thrust_now >= thrust_needed_per_motor))

        frac_now = solve_frac(vbat_now)
        current_now = current_at(vbat_now, frac_now)

        new_current = jnp.where(can_continue, current_now, current)
        new_mah = mah_drawn + jnp.where(
            can_continue, new_current * (qm.DISCHARGE_DT_S / 3600.0) * 1000.0, 0.0)
        new_t_s = t_s + jnp.where(can_continue, qm.DISCHARGE_DT_S, 0.0)
        return (new_mah, new_t_s, new_current, can_continue), None

    init = (jnp.array(0.0), jnp.array(0.0), jnp.array(current0), jnp.array(True))
    (_, t_s_final, _, _), _ = jax.lax.scan(step, init, None, length=qm.DISCHARGE_N_STEPS)

    return {
        "feasible": True,
        "hover_throttle_frac": float(hover_frac0),
        "hover_current_a_total": current0,
        "flight_time_min": float(t_s_final) / 60.0,
    }


def _print_realized_report(realized):
    print(f"\nRealized against a real catalogue motor AND propeller "
          f"({realized['iterations']} iteration(s), "
          f"{'converged' if realized['converged'] else 'DID NOT CONVERGE'}):")
    best = realized["best"]
    if best is None:
        print("  No fully-specified candidate found -- widen n_candidates/n_prop_candidates.")
        return

    m = best["motor"]
    print(f"  Motor: {m['name']} ({m['vendor']}), kV={m['kv_rpm_per_v']:.0f}, "
          f"R={m['resistance_ohm']:.4f} ohm, mass={m['mass_g']:.2f} g")
    if "prop" in best:
        p = best["prop"]
        pd = p["pitch_to_diameter"]
        print(f"  Prop: {p['name']} ({p['vendor']}), diameter={p['diameter_mm']:.1f} mm, "
              f"pitch={p['pitch_mm']:.1f} mm, blades={p['blade_count']:.0f}, "
              f"mass={p['mass_g']:.2f} g, P/D={pd:.2f}")
    else:
        print(f"  Prop (IDEALIZED, not a real part): diameter={best['prop_diameter_m'] * 1e3:.1f} mm, "
              f"blades={best['blade_count']:.2f}, pitch={best['pitch_m'] * 1e3:.1f} mm, "
              f"P/D={best['pitch_m'] / best['prop_diameter_m']:.2f}")

    hover = _result_hover_point(best, battery_name=BATTERY_NAME, other_mass_kg=qm.OTHER_MASS_KG)
    r_int = bm.r_int_ohm(BATTERY_NAME)
    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)
    hover_power_total = hover["hover_current_a_total"] * hover["hover_throttle_frac"] * vbat0

    print(f"\n  {'quantity':<32}{'value':>12}")
    rows = [
        ("total mass, g", float(best["total_mass"]) * 1e3, "{:.1f}"),
        ("TWR @ full throttle", float(best["twr"]), "{:.2f}"),
        ("current/motor @ full throttle, A", float(best["current_a"]), "{:.2f}"),
        ("spin-up 10-90%, ms", float(best["spin_up_s"]) * 1e3, "{:.1f}"),
        ("tip Mach @ full throttle", float(best["tip_mach"]), "{:.3f}"),
        ("hover throttle frac", hover["hover_throttle_frac"], "{:.3f}"),
        ("hover current, A (total)", hover["hover_current_a_total"], "{:.3f}"),
        ("hover power, W (total)", hover_power_total, "{:.3f}"),
        (f"flight time, min ({BATTERY_NAME})", hover["flight_time_min"], "{:.1f}"),
    ]
    for label, value, fmt in rows:
        print(f"  {label:<32}{fmt.format(value):>12}")


# --- Pareto frontier: spin-up time vs. hover efficiency ----------------------
#
# SPIN_UP_BUDGET_S (quad_model.py, 50ms) was tuned for whoop-class racing
# responsiveness; once frame mass is modeled realistically for a more robust
# build, no real motor/prop combo in the catalogue clears TWR>=4 within that
# budget at all -- the smallest whoop-class motors that CAN spin up that fast
# don't have the torque for a big enough prop, and the bigger/more-torquey
# motors that can drive a big prop take 100-200ms+ to spin up. Rather than
# pick one arbitrary relaxed threshold, this sweeps every real motor x prop
# combo that clears TWR>=4 and the ESC's current cap (both non-negotiable
# regardless of spin-up budget), and reports the Pareto frontier: the subset
# where no other combo is both faster to spin up AND lower hover power. Every
# frontier point is a genuine trade -- faster response costs efficiency, and
# vice versa -- so the choice of where to sit on it is a build decision, not
# something the optimizer should decide via one hardcoded budget.


def _catalogue_combos(query_diameter_mm=50.0, query_pitch_mm=40.0, query_blade_count=2.0,
                       min_twr=MIN_TWR, esc_max_current_a=None, other_mass_kg=qm.OTHER_MASS_KG,
                       battery_name=BATTERY_NAME):
    """Every real motor x real prop combo (full catalogues, ~21 x ~27) that
    clears the TWR floor and ESC current cap, with hover power/spin-up/flight
    time attached. query_* only seed prop_scaling.nearest_catalogue_prop's
    ranking (n=30 covers every prop regardless), so they do not bias which
    combos are found -- only their reported "distance", which is unused here.
    """
    esc_max_current_a = esc_max_current_a if esc_max_current_a is not None else qm.ESC_MAX_CURRENT_A
    motors = qm.ms._MOTOR_ROWS
    props = qm.ps.nearest_catalogue_prop(
        query_diameter_mm, query_pitch_mm, query_blade_count, n=30)
    r_int = bm.r_int_ohm(battery_name)
    vbat0 = bm.terminal_voltage(0.0, 0.0, r_int)

    combos = []
    for row in motors:
        if not row["resistance_ohm"] or not row["mass_g"]:
            continue
        c = {
            "name": row["name"], "vendor": row["vendor"],
            "kv_rpm_per_v": float(row["kv_rpm_per_v"]),
            "resistance_ohm": float(row["resistance_ohm"]),
            "mass_g": float(row["mass_g"]),
            "i0_a": float(row["i0_a"]) if row["i0_a"] else None,
        }
        for p in props:
            res = qm._evaluate_motor_with_prop(
                c, p["diameter_mm"] * 1e-3, p["blade_count"], p["pitch_mm"] * 1e-3,
                0.0, qm.VBAT, other_mass_kg, qm.pa.CHORD_TO_DIAMETER_RATIO, qm.pa.CL_ALPHA,
                qm.pa.CD0, qm.pa.INDUCED_POWER_FACTOR, prop_mass_kg_override=p["mass_g"] * 1e-3)
            twr = float(res["twr"])
            current = float(res["current_a"])
            if twr < min_twr or current > esc_max_current_a:
                continue

            hover = _result_hover_point(res, battery_name=battery_name, other_mass_kg=other_mass_kg)
            hover_power_w = hover["hover_current_a_total"] * hover["hover_throttle_frac"] * vbat0

            combos.append({
                "motor": c["name"], "prop": p["name"], "diameter_mm": p["diameter_mm"],
                "twr": twr, "current_a": current,
                "spinup_ms": float(res["spin_up_s"]) * 1e3,
                "hover_power_w": hover_power_w,
                "flight_time_min": hover["flight_time_min"],
            })
    return combos


def pareto_frontier(min_twr=MIN_TWR, esc_max_current_a=None, other_mass_kg=qm.OTHER_MASS_KG,
                     battery_name=BATTERY_NAME):
    """The Pareto-optimal (spinup_ms, hover_power_w) combos among every real
    motor x real prop pairing that clears the TWR floor and current cap (see
    module comment above). Sorted by spin-up time ascending; each entry
    strictly improves on hover power over every faster entry before it, so
    the list is exactly the trade-off curve -- no entry is beaten on both
    axes by another.
    """
    combos = _catalogue_combos(min_twr=min_twr, esc_max_current_a=esc_max_current_a,
                                other_mass_kg=other_mass_kg, battery_name=battery_name)
    combos.sort(key=lambda c: c["spinup_ms"])
    frontier = []
    best_power_so_far = float("inf")
    for c in combos:
        if c["hover_power_w"] < best_power_so_far:
            frontier.append(c)
            best_power_so_far = c["hover_power_w"]
    return frontier


def _print_pareto_frontier(frontier):
    print(f"\nPareto frontier: spin-up time vs. hover power "
          f"(TWR >= {MIN_TWR}, current <= {qm.ESC_MAX_CURRENT_A}A, real parts only)")
    print(f"  {len(frontier)} points -- each beats every faster point on hover power\n")
    print(f"  {'diam mm':>8}  {'prop':<30}{'motor':<26}{'TWR':>6}{'A':>7}"
          f"{'spinup ms':>11}{'power W':>10}{'flight min':>12}")
    for c in frontier:
        print(f"  {c['diameter_mm']:>8.1f}  {c['prop']:<30}{c['motor']:<26}"
              f"{c['twr']:>6.2f}{c['current_a']:>7.2f}{c['spinup_ms']:>11.1f}"
              f"{c['hover_power_w']:>10.2f}{c['flight_time_min']:>12.1f}")


def main():
    print(f"Optimizing: minimize hover power subject to TWR >= {MIN_TWR}, plus")
    print("current, spin-up, stator-size, and tip-Mach floors/ceilings.\n")
    print(f"  {N_STARTS} starts, {ADAM_STEPS} Adam steps, "
          f"{N_POLISH} polished for {POLISH_ITERS} iterations")

    result = optimize()
    print(f"  converged in {result['elapsed']:.1f} s  (cost {result['cost']:.4f})")

    _print_report(result["x"])

    realized = realize_efficient_design(result["x"])
    _print_realized_report(realized)

    _print_pareto_frontier(pareto_frontier())


if __name__ == "__main__":
    main()
