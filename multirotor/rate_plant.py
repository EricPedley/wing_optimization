"""Linearized per-axis rate plant for PID analysis/derivation.

Plant: motor-command delta u (throttle fraction) ->
  effective voltage  v = max(1 V, sagged battery voltage) * filtered PWM
  motor torque       T_m(v, w)   (SimITL's Kt = 8.3/kV, not the optimizer's)
  prop drag torque   T_p(w)      (quadratic in rpm)
  prop rpm dynamics  J_prop * dw/dt = T_m - T_p   -> first-order lag
  thrust             T(w) = b*rpm + a*rpm^2       (SimITL prop model)
  axis torque        lever * dT   (roll/pitch) or propTorque (yaw)
  body rate          I * d(omega)/dt = torque     -> integrator

Linearizing at hover gives the textbook quad rate plant

    P(s) = K / (s * (tau*s + 1)) * e^(-Ls)

per axis, where tau is the motor/prop spin-up time constant, K folds in
lever arm, dT/drpm, the voltage map slope, and 1/I. Filters and delay are
still reduced separately by the PID analysis below, not simulated here.
The axis input u is the difference between opposing pairs' commands:
individual motors change by +/-u/2. Balanced differential commands do not
change mean PWM, hence do not change battery sag to first order.

The realized parts, mass and body inertia come from the pinned SimITL
quad/component JSONs. The JAX voltage, battery and motor functions mirror
SimITL/src/sim/physics.cpp without changing the design optimizer. Battery
noise samples are explicit randf() values in [-1, 1], defaulting to zero
for deterministic analysis. battery_step returns voltage evaluated BEFORE
its capacity update, matching SimITL's updateBat ordering.

This remains a nominal symmetric, zero-inflow rate reduction: individual
motorVariance, noise, damage and thermal dynamics are excluded. Yaw uses
steady prop drag reaction, not SimITL's transient motor-reaction torque.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import multirotor.motor_model as mm

RPM2RADS = 2.0 * jnp.pi / 60.0
KT_NUMERATOR = 8.3

# The realized build (matches config/quad/hq-51mm-micro.json)
KV = 11500.0
STATOR_VOLUME_MM3 = 339.3          # 1203
PROP_DIAMETER_M = 0.051
BLADE_COUNT = 2.0
PITCH_M = 0.0381
DEFAULT_QUAD_CONFIG = (Path(__file__).resolve().parents[1] / "SimITL" / "tools"
                       / "simitl-playback" / "config" / "quad" / "hq-51mm-micro.json")
_OCV_DRAWN = jnp.array([-.06, 0., .01, .04, .50, .60, .85, 1., 1.01, 1.03, 1.06, 1.08])
_OCV_VOLTS = jnp.array([4.4, 4.2, 4.05, 3.97, 3.85, 3.7, 3.63, 3.49, 3.4, 3.3, 3., 0.])


def _unit(config_path=DEFAULT_QUAD_CONFIG):
    config_path = Path(config_path)
    quad = json.loads(config_path.read_text())
    components = {}
    for kind in ("motor", "propeller", "battery"):
        for path in sorted((config_path.parent.parent / kind).glob("*.json")):
            component = json.loads(path.read_text())
            if component["name"] == quad[kind]:
                components[kind] = component
                break
        else:
            raise ValueError(f"Missing {kind} config for {quad[kind]!r}")
    motor, prop, battery = (components[k] for k in ("motor", "propeller", "battery"))
    return dict(
        kv=motor["motorKV"], resistance=motor["motorR"], i0=motor["motorI0"],
        motor_mass=motor["mass"], prop_mass=prop["mass"], prop_inertia=prop["propInertia"],
        prop_a_factor=prop["propAFactor"], prop_torque_factor=prop["propTorqueFactor"],
        prop_max_rpm=prop["propMaxRpm"],
        **{f"thrust_factor_{a}": prop["propThrustFactor"][a] for a in "xyz"},
        frame_electronics_mass=quad["mass"], battery_mass=battery["mass"],
        battery_capacity_mah=battery["batCapacity"],
        battery_capacity_charged_mah=battery["batCapacityCharged"],
        battery_cell_count=battery["batCellCount"], max_voltage_sag=battery["maxVoltageSag"],
        inertia=np.array([1.0 / quad["invInertia"][a] for a in "xyz"]),
        motor_positions=np.array([[quad[f"motor{i}Pos"][a] for a in "xyz"]
                                  for i in range(1, 5)]))


def battery_voltage(remaining_mah, motor_pwm, unit, noise_sample=0.0):
    drawn = 1.0 - remaining_mah / jnp.maximum(unit["battery_capacity_mah"], 1.0)
    ocv = jnp.maximum(jnp.interp(drawn, _OCV_DRAWN, _OCV_VOLTS)
                      * unit["battery_cell_count"], 0.1)
    charge_inv = 1.0 - remaining_mah / jnp.maximum(unit["battery_capacity_charged_mah"], 1.0)
    mean_pwm = jnp.maximum(jnp.mean(jnp.broadcast_to(motor_pwm, (4,))), 0.0)
    sag = unit["max_voltage_sag"] * mean_pwm**2 * (1.0 + charge_inv**2)
    return jnp.clip(ocv - sag - jnp.abs(noise_sample * 0.01), 0.0, 100.0)


def effective_voltage(vbat, throttle_frac):
    return jnp.maximum(vbat, 1.0) * throttle_frac


def battery_step(remaining_mah, motor_pwm, motor_current_a, dt, unit,
                 voltage_noise_sample=0.0, consumption_noise_sample=0.0):
    vbat = battery_voltage(remaining_mah, motor_pwm, unit, voltage_noise_sample)
    pwm = jnp.broadcast_to(motor_pwm, (4,))
    currents = jnp.broadcast_to(motor_current_a, (4,))
    current = jnp.sum(jnp.abs(currents) * jnp.clip(pwm, 0.0, 1.0))
    minimum = jnp.minimum(0.2, (0.5 + consumption_noise_sample * 0.25)
                          / jnp.maximum(vbat, 0.01))
    drain = jnp.maximum(current / 3.6, minimum)
    remaining = remaining_mah - drain * dt
    return dict(vbat=vbat, current_a=drain * 3.6, remaining_mah=remaining,
                mah_drawn=unit["battery_capacity_charged_mah"] - remaining)


def motor_torque(volts, rpm, unit):
    kv = jnp.maximum(unit["kv"], 0.0001)
    kt = KT_NUMERATOR / kv
    raw = (volts - rpm / kv) / jnp.maximum(unit["resistance"], 0.0001) * kt
    friction = unit["i0"] * kt
    return jnp.where(rpm > 1.0, raw - friction,
                     jnp.where(rpm < -1.0, raw + friction,
                               raw - jnp.clip(raw, -friction, friction)))


def motor_current(torque, unit):
    return torque * unit["kv"] / KT_NUMERATOR


def prop_thrust(rpm, unit, vel=0.0):
    return mm.prop_thrust(rpm, vel, unit["prop_a_factor"], unit["prop_max_rpm"],
                          unit["thrust_factor_x"], unit["thrust_factor_y"],
                          unit["thrust_factor_z"])


def steady_state_rpm(volts, unit, vel=0.0):
    kv = jnp.maximum(unit["kv"], 0.0001)
    resistance = jnp.maximum(unit["resistance"], 0.0001)
    kt = KT_NUMERATOR / kv
    maxrpm = jnp.maximum(unit["prop_max_rpm"], 0.01)
    thrust_factor = mm.prop_thrust_factor(vel, unit["thrust_factor_x"],
                                         unit["thrust_factor_y"], unit["thrust_factor_z"])
    prop_b = (thrust_factor - unit["prop_a_factor"] * maxrpm**2) / maxrpm
    a = unit["prop_torque_factor"] * unit["prop_a_factor"]
    b = unit["prop_torque_factor"] * prop_b + kt / (kv * resistance)
    c = kt * (unit["i0"] - volts / resistance)
    rpm = -2.0 * c / (b + jnp.sqrt(jnp.maximum(b*b - 4*a*c, 0.0)))
    return jnp.maximum(rpm, 0.0)


def hover_point(unit=None, remaining_mah=None):
    """Nominal symmetric hover at the given remaining capacity (mAh).

    Defaults to the configured initial charge. Solves throttle and sag
    together, with zero battery noise and settled actuator filters.
    """
    unit = unit or _unit()
    mass = unit["frame_electronics_mass"] + unit["battery_mass"]  # frame+elec+battery from config
    # use the realized config's mass breakdown directly
    mass += 4 * (unit["motor_mass"] + unit["prop_mass"])
    need = mass * 9.81 / 4.0
    if remaining_mah is None:
        remaining_mah = unit["battery_capacity_charged_mah"]

    def operating_point(frac):
        vbat = battery_voltage(remaining_mah, jnp.full(4, frac), unit)
        volts = effective_voltage(vbat, frac)
        rpm = steady_state_rpm(volts, unit)
        return vbat, volts, rpm, prop_thrust(rpm, unit)

    if float(operating_point(1.0)[3]) < need:
        raise ValueError("Insufficient thrust to hover at this battery charge")

    def bisect(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        below = operating_point(mid)[3] < need
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 60, bisect, (0.0, 1.0))
    frac = float(0.5 * (lo + hi))
    vbat, volts, rpm, thrust = map(float, operating_point(frac))
    return dict(frac=frac, volts=volts, rpm=rpm, thrust_n=thrust, mass=mass,
                vbat=vbat, remaining_mah=remaining_mah)


def plant(axis="roll", unit=None, hover=None):
    """Return (K, tau) for P(s) = K / (s*(tau*s+1)), plus intermediate gains.

    axis: 'roll' (torque about z, lever = |x| of motor pos),
          'pitch' (about x, lever = |z|), or
          'yaw' (about y, steady reaction torque from prop drag).
    u is the command difference between opposing pairs (+/-u/2 per motor).
    Pass hover_point(unit, remaining_mah=...) to select battery charge.
    """
    unit = unit or _unit()
    hover = hover or hover_point(unit)
    rpm = hover["rpm"]
    volts = hover["volts"]
    frac = hover["frac"]

    tqf = float(unit["prop_torque_factor"])  # vel=0 -> thrust factor = z only

    # --- prop-side slopes at hover rpm (SI, rad/s) -------------------------
    # SimITL thrust model: T(rpm) = b*rpm + a*rpm^2
    dT_drpm = jax.grad(lambda r: prop_thrust(r, unit))(rpm)  # N per rpm
    dT_dw = dT_drpm / RPM2RADS               # N per (rad/s)
    dQ_dw = tqf * dT_dw                       # prop drag torque slope, N.m.s

    # --- motor-side slopes -------------------------------------------------
    # T_m = (v - w_rpm/kv)/R * kt - friction; dTm/dw_rpm = -kt/(kv*R)
    dTm_dw = jax.grad(lambda r: motor_torque(volts, r, unit))(rpm) / RPM2RADS  # N.m per (rad/s)
    dTm_dv = jax.grad(lambda v: motor_torque(v, rpm, unit))(volts)  # N.m per volt

    # rpm lag: J dw/dt = (dTm_dw - dQ_dw) w + ... ; tau = J/(dQ_dw - dTm_dw)
    J = max(float(unit["prop_inertia"]), 1e-8)
    tau = J / (dQ_dw - dTm_dw)

    # input gain: volts per unit throttle at hover frac, balanced differential PWM
    direction = jnp.array([1.0, 1.0, -1.0, -1.0])
    dv_du = jax.grad(lambda delta: effective_voltage(
        battery_voltage(hover["remaining_mah"], frac + delta * direction, unit),
        frac + delta))(0.0)

    # rpm response per throttle unit: dTm_dv * dv_du / (dQ_dw - dTm_dw)
    dw_du = dTm_dv * dv_du / (dQ_dw - dTm_dw)   # rad/s per unit throttle

    # thrust per throttle unit
    dT_du = dT_dw * dw_du                       # N per unit throttle

    I = unit["inertia"]
    pos = unit["motor_positions"]
    if axis == "roll":      # rotation about z; left pair vs right pair
        lever = abs(pos[0][0])
        # 2 motors each side: dTorque = 2*lever*dT
        K = 2.0 * lever * dT_du / I[2]
    elif axis == "pitch":   # about x; front pair vs rear pair
        lever = abs(pos[0][2])
        K = 2.0 * lever * dT_du / I[0]
    elif axis == "yaw":     # about y; reaction torque, 2 CW vs 2 CCW
        dQ_du = tqf * dT_du
        K = 2.0 * dQ_du / I[1]
    else:
        raise ValueError(axis)

    return dict(K=float(K), tau=float(tau), dv_du=float(dv_du), dw_du=float(dw_du),
                dT_du=float(dT_du), J=J, hover=hover)


def closed_loop_pid(K, tau, kp, ki, kd, delay=0.0, T=0.4, dt=1e-4):
    """Simulate the linearized closed loop (PID on rate error) in numpy.

    Plant state: [omega (body rate), w_prop (prop shaft rad/s lag state)].
    Controller: u = kp*e + ki*int(e) + kd*de/dt, e = setpoint - omega.
    delay is modeled as a simple first-order Pade-ish lag on the plant input
    (one extra pole at ~1/delay) -- adequate for margin discussion.
    Returns (t, omega, u).
    """
    steps = int(T / dt)
    omega = 0.0
    w_prop = 0.0
    integ = 0.0
    prev_omega = 0.0
    u_del = 0.0  # delayed/relaxed command state
    u_max = 0.6  # differential throttle headroom available around hover
    out = np.empty((steps, 3))
    for i in range(steps):
        t = i * dt
        sp = 670.0 if t >= 0.05 else 0.0  # deg/s step, matching the sim run
        e = sp * np.pi / 180.0 - omega    # rad/s
        integ += e * dt
        # Betaflight takes D on the measurement (gyro), not the error --
        # no derivative kick on a setpoint step
        de = -(omega - prev_omega) / dt
        prev_omega = omega
        u = kp * e + ki * integ + kd * de
        u = np.clip(u, -u_max, u_max)
        # approximate loop delay as a first-order lag on u
        alpha = dt / max(delay, 1e-9) if delay > 0 else 1.0
        u_del += alpha * (u - u_del)
        # plant: prop lag then integrator
        w_prop += (K * u_del - w_prop) / tau * dt
        omega += w_prop * dt   # K defined as rate-input gain: dw_du already
        # NOTE: w_prop here is actually "rate input"; see plant() docstring.
        out[i] = (t, omega * 180.0 / np.pi, u)
    return out


def place_poles(axis, wn_rad_s, zeta=0.7, delay_s=0.003):
    """Pole-place a rate-loop PID on P(s) = K/(s(tau s+1)) e^{-Ls}.

    With u = kp*e + kd*d(-omega)/dt (D on measurement) and the lag+integrator
    plant, the closed-loop characteristic polynomial is

        tau s^2 + (1 + K kd) s + K kp = tau (s^2 + 2 zeta wn s + wn^2)

    giving  kp = tau wn^2 / K,  kd = (2 zeta wn tau - 1) / K.

    ki is chosen separately for disturbance rejection: the I term must be
    fast enough to cancel steady torque asymmetry but its integrator pole
    adds phase lag at crossover; ki = kp * wn / 5 is a safe default.

    Also reports loop phase margin: plant phase at gain crossover of the
    *open* loop L(s) = (kp + kd s) K e^{-Ls} / (s(tau s+1)) is
    -90 - atan(wc tau) - wc*L + atan(kd wc / kp); PM = 180 + that.
    """
    p = plant(axis)
    K, tau = p["K"], p["tau"]
    kp = tau * wn_rad_s**2 / K
    kd = (2.0 * zeta * wn_rad_s * tau - 1.0) / K
    ki = kp * wn_rad_s / 5.0
    if kd < 0:
        kd = 0.0

    # gain crossover: |L(jw)|=1. Solve numerically on a grid.
    w = np.logspace(0, 4, 20001)
    L = np.abs((kp + 1j * kd * w) * K / (1j * w * (1j * w * tau + 1.0)))
    wc = float(w[np.argmin(np.abs(L - 1.0))])
    phase = (-np.pi / 2 - np.arctan(wc * tau) - wc * delay_s
             + np.arctan2(kd * wc, kp))
    pm = np.degrees(np.pi + phase)
    return dict(kp=kp, ki=ki, kd=kd, wc=wc, phase_margin_deg=pm)


def loop_with_filters(axis, kp, ki, kd, delay_s=0.0,
                      f_gyro_hz=250.0, f_dterm_hz=75.0, f_dterm2_hz=150.0):
    """Open loop L(jw) with the real measurement-path filters.

    e = sp - G_g(s)*y          (P/I see gyro-filtered measurement)
    D path: kd*s*G_g*G_d(s)*y  (D sees gyro+D lowpassed measurement)

    L(s) = P(s)*e^(-Ls) * [ (kp + ki/s)*G_g + kd*s*G_g*G_d ]
    """
    p = plant(axis)
    K, tau = p["K"], p["tau"]
    w = np.logspace(0, 4, 20001)
    s = 1j * w
    G_g = 1.0 / (1.0 + s / (2 * np.pi * f_gyro_hz))
    G_d = G_g / (1.0 + s / (2 * np.pi * f_dterm_hz))
    if f_dterm2_hz:
        G_d = G_d / (1.0 + s / (2 * np.pi * f_dterm2_hz))
    P = K / (s * (tau * s + 1.0))
    C = (kp + ki / s) * G_g + kd * s * G_d
    L = P * C * np.exp(-s * delay_s)
    return w, L


def design_margins(axis, wc, pm_deg, ki=0.0, delay_s=0.0,
                   f_gyro_hz=250.0, f_dterm_hz=75.0, f_dterm2_hz=150.0):
    """Solve kp,kd so L(jw) has gain=1 at w=wc with phase margin pm_deg,
    using the full open loop (filters + delay). ki is held fixed.
    Two equations: |L(jwc)|=1 and angle(L(jwc))=-180+pm_deg."""
    p = plant(axis)
    K, tau = p["K"], p["tau"]
    jw = 1j * wc
    G_g = 1.0 / (1.0 + jw / (2 * np.pi * f_gyro_hz))
    G_d = G_g / (1.0 + jw / (2 * np.pi * f_dterm_hz))
    if f_dterm2_hz:
        G_d = G_d / (1.0 + jw / (2 * np.pi * f_dterm2_hz))
    P = K / (jw * (tau * jw + 1.0)) * np.exp(-jw * delay_s)
    # C(jw) = (kp + ki/s)*G_g + kd*s*G_d  =>  C = kp*A + kd*B + ki*A/s
    A = G_g
    B = jw * G_d
    Cki = ki * A / jw
    # need P*(kp*A + kd*B + Cki) = 1 at phase -180+pm
    target = np.exp(1j * (-np.pi + np.radians(pm_deg)))  # L(jwc) value
    needed = target / P                                  # required C(jwc)
    M = np.array([[A.real, B.real], [A.imag, B.imag]])
    rhs = np.array([(needed - Cki).real, (needed - Cki).imag])
    kp, kd = np.linalg.solve(M, rhs)
    return dict(kp=kp, kd=kd, ki=ki, wc=wc, pm_deg=pm_deg)


def stability_check(axis, kp, ki, kd, **kw):
    """Phase margin + predicted stability for given gains."""
    w, L = loop_with_filters(axis, kp, ki, kd, **kw)
    mag = np.abs(L)
    i = np.argmin(np.abs(mag - 1.0))
    wc = w[i]
    pm = np.degrees(np.pi + np.angle(L[i]))
    # instability signature: does |L| exceed 1 while phase < -180?
    bad = np.any((np.abs(L) > 1.0) & (np.unwrap(np.angle(L)) < -np.pi))
    return dict(wc=wc, phase_margin_deg=pm, unstable=bool(bad))


def main():
    unit = _unit()
    hover = hover_point(unit)
    print(f"hover: battery {hover['vbat']:.3f} V at {hover['remaining_mah']:.0f} mAh, "
          f"throttle {hover['frac']:.3f}, motor {hover['volts']:.2f} V, "
          f"{hover['rpm']:.0f} rpm, {hover['thrust_n']/9.81*1000:.1f} g-f/motor, "
          f"mass {hover['mass']*1e3:.1f} g")
    for axis in ("roll", "pitch", "yaw"):
        p = plant(axis, unit, hover)
        print(f"{axis:5s}: K={p['K']:.3f} (rad/s^2 per pair-command difference), "
              f"tau={p['tau']*1e3:.1f} ms, dT/du={p['dT_du']:.3f} N/throttle")

    print()
    print("pole-placed roll PID vs target bandwidth "
          "(zeta=0.7, delay=3ms):")
    for f_hz in (5, 10, 15, 20, 25, 30, 40):
        r = place_poles("roll", 2 * np.pi * f_hz, delay_s=3e-3)
        print(f"  {f_hz:3d} Hz: kp={r['kp']:.4f} ki={r['ki']:.3f} "
              f"kd={r['kd']:.5f} wc={r['wc']:.0f} rad/s "
              f"PM={r['phase_margin_deg']:.0f} deg")


if __name__ == "__main__":
    main()
