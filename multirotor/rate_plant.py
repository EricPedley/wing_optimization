"""Linearized per-axis rate plant for PID analysis/derivation.

Plant: motor-command delta u (throttle fraction) ->
  effective voltage  v = VBAT * frac^DUTY_GAMMA
  motor torque       T_m(v, w)   (affine in rpm, drops with back-EMF)
  prop drag torque   T_p(w)      (quadratic in rpm)
  prop rpm dynamics  J_prop * dw/dt = T_m - T_p   -> first-order lag
  thrust             T(w) = b*rpm + a*rpm^2       (SimITL prop model)
  axis torque        lever * dT   (roll/pitch) or propTorque (yaw)
  body rate          I * d(omega)/dt = torque     -> integrator

Linearizing at hover gives the textbook quad rate plant

    P(s) = K / (s * (tau*s + 1)) * e^(-Ls)

per axis, where tau is the motor/prop spin-up time constant, K folds in
lever arm, dT/drpm, the voltage map slope, and 1/I, and L is a lumped delay
for gyro filtering + PID-loop + ESC update (estimated ~2-4 ms, the dominant
phase-margin killer and the main thing this module does NOT get from
physics -- it's a measurement-side quantity).

All jacobians come from jax.grad on motor_model/prop functions, so the
plant inherits the same calibrated constants the design optimizer used.
"""

import jax
import jax.numpy as jnp
import numpy as np

import multirotor.motor_model as mm
import multirotor.quad_inertia as qi
import multirotor.quad_model as qm

RPM2RADS = 2.0 * jnp.pi / 60.0

# The realized build (matches config/quad/hq-51mm-whoop.json)
KV = 11500.0
STATOR_VOLUME_MM3 = 339.3          # 1203
PROP_DIAMETER_M = 0.051
BLADE_COUNT = 2.0
PITCH_M = 0.0381
VBAT = qm.VBAT


def _unit():
    return qm.motor_prop_unit(KV, STATOR_VOLUME_MM3, PROP_DIAMETER_M,
                              BLADE_COUNT, PITCH_M)


def hover_point(unit=None):
    """Hover throttle fraction / volts / rpm / thrust for the real build."""
    unit = unit or _unit()
    mass = (qm.OTHER_MASS_KG + 0.0 + 4 * (unit["motor_mass"] + unit["prop_mass"])
            + 0.0424 * 0)  # see note: frame+elec+battery baked in quad_inertia
    # use the inertia module's mass breakdown directly
    mass = (qi.FRAME_ELECTRONICS_MASS_KG + qi.BATTERY_MASS_KG
            + 4 * (unit["motor_mass"] + unit["prop_mass"]))
    need = mass * qm.G / 4.0

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        v = qm.effective_voltage(VBAT, mid)
        rpm = float(mm.steady_state_rpm(
            v, 0.0, KV, unit["resistance"], unit["i0"], unit["prop_a_factor"],
            unit["prop_torque_factor"], unit["prop_max_rpm"],
            unit["thrust_factor_x"], unit["thrust_factor_y"],
            unit["thrust_factor_z"]))
        t = float(mm.prop_thrust(
            rpm, 0.0, unit["prop_a_factor"], unit["prop_max_rpm"],
            unit["thrust_factor_x"], unit["thrust_factor_y"],
            unit["thrust_factor_z"]))
        if t < need:
            lo = mid
        else:
            hi = mid
    frac = 0.5 * (lo + hi)
    volts = float(qm.effective_voltage(VBAT, frac))
    rpm = float(mm.steady_state_rpm(
        volts, 0.0, KV, unit["resistance"], unit["i0"], unit["prop_a_factor"],
        unit["prop_torque_factor"], unit["prop_max_rpm"],
        unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"]))
    thrust = float(mm.prop_thrust(
        rpm, 0.0, unit["prop_a_factor"], unit["prop_max_rpm"],
        unit["thrust_factor_x"], unit["thrust_factor_y"],
        unit["thrust_factor_z"]))
    return dict(frac=frac, volts=volts, rpm=rpm, thrust_n=thrust, mass=mass)


def plant(axis="roll", unit=None, hover=None):
    """Return (K, tau) for P(s) = K / (s*(tau*s+1)), plus intermediate gains.

    axis: 'roll' (torque about z, lever = |x| of motor pos),
          'pitch' (about x, lever = |z|), or
          'yaw' (about y, reaction torque from prop drag).
    """
    unit = unit or _unit()
    hover = hover or hover_point(unit)
    rpm = hover["rpm"]
    volts = hover["volts"]
    frac = hover["frac"]

    kt = float(mm.motor_constant(KV))
    R = float(unit["resistance"])
    a_f = float(unit["prop_a_factor"])
    tqf = float(unit["prop_torque_factor"])
    maxrpm = float(unit["prop_max_rpm"])
    tfz = float(unit["thrust_factor_z"])  # vel=0 -> thrust factor = z only

    # --- prop-side slopes at hover rpm (SI, rad/s) -------------------------
    # SimITL thrust model: T(rpm) = b*rpm + a*rpm^2
    b = (tfz - a_f * maxrpm**2) / maxrpm
    dT_drpm = b + 2.0 * a_f * rpm            # N per rpm
    dT_dw = dT_drpm / RPM2RADS               # N per (rad/s)
    dQ_dw = tqf * dT_dw                       # prop drag torque slope, N.m.s

    # --- motor-side slopes -------------------------------------------------
    # T_m = (v - w_rpm/kv)/R * kt - friction; dTm/dw_rpm = -kt/(kv*R)
    dTm_dw = -kt / (KV * R) / RPM2RADS        # N.m per (rad/s)
    dTm_dv = kt / R                           # N.m per volt

    # rpm lag: J dw/dt = (dTm_dw - dQ_dw) w + ... ; tau = J/(dQ_dw - dTm_dw)
    J = float(unit["prop_inertia"])
    tau = J / (dQ_dw - dTm_dw)

    # input gain: volts per unit throttle at hover frac
    dv_du = VBAT * qm.DUTY_GAMMA * frac ** (qm.DUTY_GAMMA - 1.0)

    # rpm response per throttle unit: dTm_dv * dv_du / (dQ_dw - dTm_dw)
    dw_du = dTm_dv * dv_du / (dQ_dw - dTm_dw)   # rad/s per unit throttle

    # thrust per throttle unit
    dT_du = dT_dw * dw_du                       # N per unit throttle

    I = qi.quad_inertia()["I"]
    pos = qi.MOTOR_POSITIONS_M
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

    return dict(K=K, tau=tau, dv_du=dv_du, dw_du=dw_du, dT_du=dT_du,
                J=J, hover=hover)


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


def main():
    unit = _unit()
    hover = hover_point(unit)
    print(f"hover: throttle {hover['frac']:.3f}, {hover['volts']:.2f} V, "
          f"{hover['rpm']:.0f} rpm, {hover['thrust_n']*1000:.0f} g-f/motor, "
          f"mass {hover['mass']*1e3:.1f} g")
    for axis in ("roll", "pitch", "yaw"):
        p = plant(axis, unit, hover)
        print(f"{axis:5s}: K={p['K']:.3f} (rad/s^2 per throttle unit), "
              f"tau={p['tau']*1e3:.1f} ms, dT/du={p['dT_du']*1000:.0f} gf")

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
