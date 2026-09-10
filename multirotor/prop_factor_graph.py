"""Joint inference of propeller and motor constants as a GTSAM factor graph.

Why a factor graph instead of the per-series fits in calibrate_prop_aero.py:
a bench row measures a PROPELLER through a MOTOR, and the repo has no
independent measurement of either. Backing mechanical power out of bench
current with a fixed datasheet (kV, I0) -- what calibrate_prop_aero.py's
fit_induced_power_factor does -- treats the motor spec as exact and dumps all
of its error into the prop. That is measurably wrong: the same prop measured
on two different motors then disagrees with itself by a median 15.6% in kT
(up to 78.6%), and the implied figure of merit exceeds 1.0 (energetically
impossible) on many series, worst 4.38, even though the raw ELECTRICAL data
respects the momentum-theory bound everywhere (FM 0.37-0.58).

So the motor spec is not ground truth, it is just another noisy measurement.
This module says so explicitly: motor constants are latent variables with
priors whose strength reflects how much the datasheet is actually trusted,
and every prop/motor pair that shares a bench series constrains both ends
jointly. A prop measured on several motors and a motor measured with several
props triangulate each other.

State (all in log space, so every variable is positive by construction and
priors are multiplicative/relative):
  prop nominal: [log kT_nom, log kP_nom, log p]  with
                T = kT*rpm^p,  P_mech = kP*rpm^3.
                Thrust-only props have [log kT_nom, log p].
  prop instance: [log kT_inst, log kP_inst] per (propeller, motor) pair.
                 Each is the coefficient set that actually produced the bench row.
                 (1D for thrust-only instances, 2D for powered instances.)
  motor node: [log kt_scale, log i0_scale, log r_scale] multiplicative
              corrections on Kt = Ke = 60/(2*pi*kV), datasheet I0 and R.
              The Kt correction also changes back EMF, preserving equivalent-DC
              energy consistency. Resistance stays positive by construction.

Factors:
  thrust (instance + nominal) : T_pred = kT_inst*measured_rpm^p, for every row.
  motor (instance + motor)    : fixed battery voltage and duty = throttle_pct/100
                                -> closed-form equilibrium RPM and battery current
                                -> residual on measured RPM and current A only.
  prop instance prior (between) : instance[:2] - nominal[:2] ~ N(0, prop_prior_sigma).
  prop p prior (unary)          : log p ~ N(log mu_p, sigma_log_p) from diameter,
                                  pitch and blade_count heuristics.
  motor prior (unary)           : uniform tight corrections centred on 1.0,
                                  configurable (default Kt 5%, I0 10%, R 20%).

No fitted duty or ESC losses. Datasheet resistance conventions, linear throttle
mapping and constant I0 remain model assumptions. Measurements have independent
fixed noise scales (5% plus absolute floors), with a row-level Huber kernel.
These weights include model discrepancy and are not sensor precisions.

Robust (Huber) noise models on the data factors so the known-corrupt
tmotor_m1103 rows are downweighted by the estimator instead of being
hand-excluded.

Run with: uv run python -m multirotor.prop_factor_graph
"""

import numpy as np
import gtsam
from gtsam import symbol
from collections import defaultdict

import csv
from multirotor.calibrate_prop_aero import load_bench_rows, MOTOR_KV_I0, DATA_DIR

RHO = 1.225
KT_NUMERATOR = 60.0 / (2.0 * np.pi)

# Geometry-informed prior on the thrust power-law exponent p.
# p = 2.0 is the ideal actuator-disk static-thrust result. Deviations in either
# direction are possible for small-diameter, high-pitch/diameter, and multi-blade
# props, so the prior is centered at 2.0 with a geometry-dependent width.
# These constants are engineering heuristics, not fitted from bench data.
P_PRIOR_D_REF_MM = 75.0          # reference 3-inch prop
P_PRIOR_PD0 = 0.5                # reference pitch/diameter
P_PRIOR_BASE_SIGMA = 0.04
P_PRIOR_RE_SIGMA = 0.10          # small-diameter props have less certain p
P_PRIOR_PD_SIGMA = 0.08          # aggressive pitch makes p less certain
P_PRIOR_BLADE_SIGMA = 0.03       # extra blades make p less certain
P_PRIOR_MISSING_PITCH_SIGMA = 0.08
P_PRIOR_MAX_SIGMA = 0.10         # keep the anchor on p = 2 meaningful


def _load_motor_resistance():
    with open(DATA_DIR / "motor_datasheets.csv", newline="") as f:
        by_name = {r["name"]: r for r in csv.DictReader(f)}
    return {m: float(by_name[f"{m[0]}-{m[1]}"]["resistance_ohm"])
            for m in MOTOR_KV_I0 if by_name[f"{m[0]}-{m[1]}"]["resistance_ohm"]}


MOTOR_RESISTANCE = _load_motor_resistance()


def prop_p_prior(diameter_mm, pitch_mm, blade_count):
    """Geometry-informed Gaussian prior for the thrust exponent p.

    Returns (mu_p, sigma_p) in linear p-space. The factor graph then places
    a prior on log p ~ N(log(mu_p), sigma_p / mu_p) for small-to-moderate
    sigma_p (delta-method approximation).
    """
    pitch_missing = pitch_mm is None or not np.isfinite(pitch_mm) or pitch_mm <= 0
    if pitch_missing:
        PD = P_PRIOR_PD0
    else:
        PD = pitch_mm / diameter_mm

    re_term = max(0.0, 1.0 - diameter_mm / P_PRIOR_D_REF_MM)
    pd_term = max(0.0, PD - P_PRIOR_PD0)
    blade_term = max(0.0, blade_count - 2)

    # Keep the prior centered on the ideal p = 2; use geometry only to widen it.
    mu_p = 2.0
    sigma_p = (P_PRIOR_BASE_SIGMA
               + P_PRIOR_RE_SIGMA * re_term
               + P_PRIOR_PD_SIGMA * pd_term
               + P_PRIOR_BLADE_SIGMA * blade_term
               + (P_PRIOR_MISSING_PITCH_SIGMA if pitch_missing else 0.0))
    sigma_p = min(sigma_p, P_PRIOR_MAX_SIGMA)
    return mu_p, sigma_p


def equilibrium_predictions(voltage_v, duty, kT, kP, kt, i0, resistance,
                            jacobian=False):
    """Predict [RPM, battery current A, thrust N] from fixed voltage and duty.

    With Q = kP*rpm^2*30/pi, voltage and torque balance reduce to
    (R*kP*30/pi/Kt)*rpm^2 + (Kt*pi/30)*rpm + R*I0 - duty*V = 0.
    The positive root is evaluated without cancellation, including R = 0.
    Below the constant-friction starting threshold, the rotor is stalled.
    Optional derivatives are with respect to log(kT, kP, Kt, I0, R).
    """
    applied = duty * voltage_v
    drive = applied - resistance * i0
    if drive <= 0:
        current = duty * applied / resistance if resistance > 0 else 0.0
        prediction = np.array([0.0, current, 0.0])
        if not jacobian:
            return prediction
        derivatives = np.zeros((3, 5))
        derivatives[1, 4] = -current
        return prediction, derivatives
    q_per_rpm2 = kP * 30.0 / np.pi
    a = resistance * q_per_rpm2 / kt
    b = kt * np.pi / 30.0
    root = np.sqrt(b ** 2 + 4.0 * a * drive)
    rpm = 2.0 * drive / (b + root)
    torque = q_per_rpm2 * rpm ** 2        # = kP*rpm^3/omega
    motor_current = torque / kt + i0
    prediction = np.array([rpm, duty * motor_current, kT * rpm ** 2])
    if not jacobian:
        return prediction
    drpm = np.array([0.0, -a * rpm ** 2, a * rpm ** 2 - b * rpm,
                     -resistance * i0, -a * rpm ** 2 - resistance * i0]) / (2.0 * a * rpm + b)
    dcurrent = duty * (2.0 * q_per_rpm2 * rpm / kt * drpm
                      + np.array([0.0, torque / kt, -torque / kt, i0, 0.0]))
    dthrust = 2.0 * kT * rpm * drpm + np.array([kT * rpm ** 2, 0.0, 0.0, 0.0, 0.0])
    return prediction, np.vstack([drpm, dcurrent, dthrust])

# Prior sigma (in log space, ~fractional) on each motor's Kt, I0 and R correction.
# Uniform tight priors anchor motors to datasheets, including noisy motors.
# These strengths are modeling assumptions, not published spec uncertainties.
# Measurement weights also include model discrepancy; they are not sensor specs.
# Resistance is anchored to its datasheet; its convention remains an assumption.
DEFAULT_KT_PRIOR_SIGMA = 0.05   # 5% on the torque/back-EMF constant
DEFAULT_I0_PRIOR_SIGMA = 0.10   # 10% on the no-load current
DEFAULT_RESISTANCE_PRIOR_SIGMA = 0.20
DEFAULT_PROP_PRIOR_SIGMA = 0.10  # 10% log-sigma on per-instance kT/kP spread
RPM_RELATIVE_SIGMA = 0.05      # baseline RPM measurement/model-discrepancy weight
CURRENT_RELATIVE_SIGMA = 0.05
THRUST_RELATIVE_SIGMA = 0.05


def _keys(rows):
    props = sorted({r["catalogue_name"] for r in rows})
    motors = sorted({r["motor_key"] for r in rows if r["motor_key"] in MOTOR_RESISTANCE})
    pk = {n: symbol("p", i) for i, n in enumerate(props)}
    mk = {m: symbol("m", i) for i, m in enumerate(motors)}
    instances = sorted({(r["catalogue_name"], r["motor_key"]) for r in rows})
    ik = {(n, m): symbol("i", i) for i, (n, m) in enumerate(instances)}

    # Props that have at least one row with a known-resistance motor need kP.
    powered = {r["catalogue_name"] for r in rows if r["motor_key"] in MOTOR_RESISTANCE}
    # pk: [log kT, log p] for thrust-only, [log kT, log kP, log p] for powered.
    # ik: instance coefficients, dimension = pk_dim - 1 (p is a family-level parameter).
    pk_dim = {n: (3 if n in powered else 2) for n in props}
    ik_dim = {n: (2 if n in powered else 1) for n in props}

    prop_specs = {}
    for r in rows:
        n = r["catalogue_name"]
        if n not in prop_specs:
            d_mm = r["diameter_m"] * 1000.0
            P_mm = r["pitch_m"] * 1000.0 if r["pitch_m"] is not None else None
            prop_specs[n] = (d_mm, P_mm, r["blade_count"])

    return props, motors, pk, ik, mk, pk_dim, ik_dim, prop_specs


def _thrust_factor(ikey, pkey, rpm, thrust_n, sigma_n):
    """Binary factor: T = kT_inst * rpm^p, with kT from ikey and p from pkey."""
    def err(this, values, H=None):
        ik = values.atVector(ikey)
        pk = values.atVector(pkey)
        kT = np.exp(ik[0])
        p = np.exp(pk[-1])
        pred = kT * rpm ** p
        if H is not None:
            H0 = np.zeros((1, len(ik)))
            H0[0, 0] = pred                      # d/d log kT
            H1 = np.zeros((1, len(pk)))
            # d/d log p = p * pred * log(rpm)
            H1[0, -1] = p * pred * np.log(rpm)
            H[0] = H0
            H[1] = H1
        return np.array([pred - thrust_n])
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma_n))
    return gtsam.CustomFactor(noise, [ikey, pkey], err)


def _motor_factor(ikey, mkey, row, kt_nom, i0_nom, resistance):
    """Residuals on measured RPM and battery current from the forward model."""
    voltage = row["voltage_v"]
    duty = row["throttle_pct"] / 100.0
    measured = np.array([row["rpm"], row["current_a"]])
    sigmas = measured * np.array([RPM_RELATIVE_SIGMA, CURRENT_RELATIVE_SIGMA]) + np.array([100.0, 0.05])

    def err(this, values, H=None):
        _, kP = np.exp(values.atVector(ikey))
        kt_scale, i0_scale, r_scale = np.exp(values.atVector(mkey))
        prediction, derivatives = equilibrium_predictions(
            voltage, duty, 1.0, kP, kt_nom * kt_scale, i0_nom * i0_scale,
            resistance * r_scale, jacobian=True)
        residual = prediction[:2] - measured
        if H is not None:
            # instance prop: kT has no effect on RPM/current; only kP matters
            H[0] = np.column_stack((np.zeros(2), derivatives[:2, 1].copy()))
            H[1] = derivatives[:2, 1:].copy()  # d/d log kt_scale, log i0_scale, log r_scale
        return residual
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Diagonal.Sigmas(sigmas))
    return gtsam.CustomFactor(noise, [ikey, mkey], err)


def _prop_prior_factor(ikey, pkey, sigmas):
    """Between factor: instance coefficients stay near the nominal (first ik_dim components)."""
    def err(this, values, H=None):
        x = values.atVector(ikey)
        y = values.atVector(pkey)
        r = x - y[:len(x)]
        if H is not None:
            H[0] = np.eye(len(x))
            H[1] = np.hstack((-np.eye(len(x)),
                              np.zeros((len(x), len(y) - len(x)))))
        return r
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    return gtsam.CustomFactor(noise, [ikey, pkey], err)


def _p_prior_factor(pkey, p_index, mu_log_p, sigma_log_p):
    """Unary factor on the p component of the prop nominal vector."""
    def err(this, values, H=None):
        x = values.atVector(pkey)
        r = np.array([x[p_index] - mu_log_p])
        if H is not None:
            H0 = np.zeros((1, len(x)))
            H0[0, p_index] = 1.0
            H[0] = H0
        return r
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma_log_p)
    return gtsam.CustomFactor(noise, [pkey], err)


def build_graph(rows, kt_prior_sigma=DEFAULT_KT_PRIOR_SIGMA,
                i0_prior_sigma=DEFAULT_I0_PRIOR_SIGMA,
                resistance_prior_sigma=DEFAULT_RESISTANCE_PRIOR_SIGMA,
                prop_prior_sigma=DEFAULT_PROP_PRIOR_SIGMA):
    props, motors, pk, ik, mk, pk_dim, ik_dim, prop_specs = _keys(rows)
    graph = gtsam.NonlinearFactorGraph()
    sig = np.array([kt_prior_sigma, i0_prior_sigma, resistance_prior_sigma])
    if not np.all(np.isfinite(sig) & (sig > 0)):
        raise ValueError("Motor prior sigmas must be positive and finite")
    if not (np.isfinite(prop_prior_sigma) and prop_prior_sigma > 0):
        raise ValueError("prop_prior_sigma must be positive and finite")

    for m in motors:
        graph.add(gtsam.PriorFactorVector(
            mk[m], np.zeros(3), gtsam.noiseModel.Diagonal.Sigmas(sig)))

    for (n, m), ikey in ik.items():
        graph.add(_prop_prior_factor(
            ikey, pk[n], np.full(ik_dim[n], prop_prior_sigma)))

    for n in props:
        d_mm, P_mm, blade_count = prop_specs[n]
        mu_p, sigma_p = prop_p_prior(d_mm, P_mm, blade_count)
        p_index = pk_dim[n] - 1
        # prior on log p: mean = log(mu_p), sigma approximated as sigma_p / mu_p
        graph.add(_p_prior_factor(
            pk[n], p_index, np.log(mu_p), sigma_p / mu_p))

    for r in rows:
        for field in ("rpm", "current_a", "thrust_g"):
            if not np.isfinite(r[field]) or r[field] <= 0:
                raise ValueError(f"Bench factors require positive finite {field}")
        n = r["catalogue_name"]
        ikey = ik[(n, r["motor_key"])]
        pkey = pk[n]
        thrust_n = r["thrust_g"] * 9.81e-3
        graph.add(_thrust_factor(ikey, pkey, r["rpm"], thrust_n,
                                 THRUST_RELATIVE_SIGMA * thrust_n + 0.005))
        if r["motor_key"] not in mk:
            continue
        if not np.isfinite(r["voltage_v"]) or r["voltage_v"] <= 0:
            raise ValueError("Bench factors require positive finite voltage_v")
        duty = r["throttle_pct"] / 100.0
        if not np.isfinite(duty) or not 0 < duty <= 1:
            raise ValueError("Bench factors require 0 < throttle_pct <= 100")
        kv, i0 = MOTOR_KV_I0[r["motor_key"]]
        graph.add(_motor_factor(ikey, mk[r["motor_key"]], r,
                                KT_NUMERATOR / kv, i0, MOTOR_RESISTANCE[r["motor_key"]]))
    return graph, props, motors, pk, ik, mk


def initial_values(rows, props, motors, pk, ik, mk, pk_dim, ik_dim, prop_specs):
    v = gtsam.Values()
    by_prop = defaultdict(list)
    by_inst = defaultdict(list)
    for r in rows:
        by_prop[r["catalogue_name"]].append(r)
        by_inst[(r["catalogue_name"], r["motor_key"])].append(r)

    for n in props:
        d_mm, P_mm, blade_count = prop_specs[n]
        mu_p, _ = prop_p_prior(d_mm, P_mm, blade_count)
        log_p = np.log(mu_p)

        rs = by_prop[n]
        rpm = np.array([x["rpm"] for x in rs])
        tn = np.array([x["thrust_g"] for x in rs]) * 9.81e-3
        # Least-squares kT for the design given the geometry-informed p.
        rpm_p = rpm ** mu_p
        kT = float(np.sum(tn * rpm_p) / np.sum(rpm_p ** 2))

        if pk_dim[n] == 2:
            v.insert(pk[n], np.array([np.log(max(kT, 1e-14)), log_p]))
            continue

        powered = [x for x in rs if x["motor_key"] in mk]
        rpm = np.array([x["rpm"] for x in powered])
        cur = np.array([x["current_a"] / (x["throttle_pct"] / 100.0) for x in powered])
        kt = np.array([KT_NUMERATOR / MOTOR_KV_I0[x["motor_key"]][0] for x in powered])
        i0 = np.array([MOTOR_KV_I0[x["motor_key"]][1] for x in powered])
        q = np.maximum(cur - i0, 1e-3) * kt
        kP = float(np.median(q / (rpm ** 2 * 30.0 / np.pi)))
        v.insert(pk[n], np.array([np.log(max(kT, 1e-14)),
                                   np.log(max(kP, 1e-18)),
                                   log_p]))

    for (n, m), ikey in ik.items():
        d_mm, P_mm, blade_count = prop_specs[n]
        mu_p, _ = prop_p_prior(d_mm, P_mm, blade_count)
        inst_rows = by_inst[(n, m)]
        rpm = np.array([x["rpm"] for x in inst_rows])
        tn = np.array([x["thrust_g"] for x in inst_rows]) * 9.81e-3
        rpm_p = rpm ** mu_p
        kT_inst = float(np.sum(tn * rpm_p) / np.sum(rpm_p ** 2))

        if ik_dim[n] == 1:
            v.insert(ikey, np.array([np.log(max(kT_inst, 1e-14))]))
            continue

        if m in mk:
            kv, i0 = MOTOR_KV_I0[m]
            cur = np.array([r["current_a"] / (r["throttle_pct"] / 100.0) for r in inst_rows])
            q = np.maximum(cur - i0, 1e-3) * (KT_NUMERATOR / kv)
            kP_inst = float(np.median(q / (rpm ** 2 * 30.0 / np.pi)))
        else:
            # No current data for this motor; start at the nominal kP.
            kP_inst = np.exp(v.atVector(pk[n])[1])
        v.insert(ikey, np.array([np.log(max(kT_inst, 1e-14)),
                                 np.log(max(kP_inst, 1e-18))]))

    for m in motors:
        v.insert(mk[m], np.zeros(3))
    return v


def solve(rows, kt_prior_sigma=DEFAULT_KT_PRIOR_SIGMA,
          i0_prior_sigma=DEFAULT_I0_PRIOR_SIGMA,
          resistance_prior_sigma=DEFAULT_RESISTANCE_PRIOR_SIGMA,
          prop_prior_sigma=DEFAULT_PROP_PRIOR_SIGMA):
    graph, props, motors, pk, ik, mk = build_graph(
        rows, kt_prior_sigma, i0_prior_sigma, resistance_prior_sigma,
        prop_prior_sigma)
    _, _, _, _, _, pk_dim, ik_dim, prop_specs = _keys(rows)
    init = initial_values(rows, props, motors, pk, ik, mk, pk_dim, ik_dim, prop_specs)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(300)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    result = opt.optimize()
    return graph, result, props, motors, pk, ik, mk, init


def main():
    rows = [r for r in load_bench_rows() if r["motor_key"] in MOTOR_KV_I0]
    graph, res, props, motors, pk, ik, mk, init = solve(rows)
    dimension = sum(len(res.atVector(k)) for k in res.keys())
    print(f"{len(rows)} bench rows | {len(props)} props | {len(motors)} motors | "
          f"{graph.size()} factors | {dimension} variables")
    print(f"error: {graph.error(init):.1f} -> {graph.error(res):.1f}\n")
    motor_rows = [r for r in rows if r["motor_key"] in mk]
    print(f"Thrust factor on all {len(rows)} rows; motor factor on {len(motor_rows)} "
          f"rows with known resistance")
    print(f"Motor prior log-sigmas: Kt={DEFAULT_KT_PRIOR_SIGMA}, "
          f"I0={DEFAULT_I0_PRIOR_SIGMA}, R={DEFAULT_RESISTANCE_PRIOR_SIGMA}")
    print(f"Prop instance prior log-sigma: {DEFAULT_PROP_PRIOR_SIGMA}")
    print("Geometry-informed p prior on each prop family from diameter, pitch, blade count")
    print("Measurement sigmas: RPM 5% + 100 rpm, current 5% + 0.05 A, "
          "thrust 5% + 0.005 N (working model-discrepancy allowances)")

    thrust_errors = defaultdict(lambda: defaultdict(list))
    motor_errors = defaultdict(list)
    for r in rows:
        n = r["catalogue_name"]
        ikey = ik[(n, r["motor_key"])]
        pkey = pk[n]
        kT_inst = np.exp(res.atVector(ikey)[0])
        p = np.exp(res.atVector(pkey)[-1])
        t_err = (kT_inst * r["rpm"] ** p) / (r["thrust_g"] * 9.81e-3) - 1.0
        thrust_errors[n][r["motor_key"]].append(t_err)
        thrust_errors[n]["ALL"].append(t_err)
        thrust_errors["ALL"]["ALL"].append(t_err)
        if r["motor_key"] not in mk:
            continue
        kt_scale, i0_scale, r_scale = np.exp(res.atVector(mk[r["motor_key"]]))
        kv, i0 = MOTOR_KV_I0[r["motor_key"]]
        kP_inst = np.exp(res.atVector(ikey)[1])
        pred = equilibrium_predictions(
            r["voltage_v"], r["throttle_pct"] / 100.0, 1.0, kP_inst,
            KT_NUMERATOR / kv * kt_scale, i0 * i0_scale,
            MOTOR_RESISTANCE[r["motor_key"]] * r_scale)
        m_err = pred[:2] / np.array([r["rpm"], r["current_a"]]) - 1.0
        for label in [r["motor"], "ALL"]:
            motor_errors[label].append(m_err)
    print("\n=== Thrust residual: kT_inst*rpm^p / measured_thrust - 1 ===")
    print(f"{'propeller':<32}{'n':>4}{'RMS%':>8}{'max%':>8}")
    for label in sorted(thrust_errors):
        a = np.array(thrust_errors[label]["ALL"])
        print(f"{label:<32}{len(a):>4}{100*np.sqrt(np.mean(a*a)):>8.1f}"
              f"{100*max(abs(a.min()), a.max()):>8.1f}")
    print("\n=== Motor residual by motor: (predicted - measured) / measured ===")
    print(f"{'motor':<29}{'n':>4}{'RPM%':>8}{'current%':>11}")
    for label in sorted(motor_errors):
        a = np.array(motor_errors[label])
        rms = 100.0 * np.sqrt(np.mean(np.square(a), axis=0))
        print(f"{label:<29}{len(a):>4}{rms[0]:>8.1f}{rms[1]:>11.1f}")

    marg = gtsam.Marginals(graph, res)

    print("=== motors: multiplicative corrections the data wants on the datasheet ===")
    print(f"{'motor':<18}{'Kt x':>9}{'+/-':>7}{'I0 x':>9}{'+/-':>7}"
          f"{'R x':>9}{'+/-':>7}{'R ohm':>9}{'corr Kt,R':>11}")
    for m in motors:
        x = res.atVector(mk[m]); c = marg.marginalCovariance(mk[m])
        s = np.sqrt(np.diag(c))
        correlation = c[0, 2] / (s[0] * s[2])
        print(f"{m[0]+'-'+str(m[1]):<18}{np.exp(x[0]):>9.3f}{s[0]:>7.3f}"
              f"{np.exp(x[1]):>9.3f}{s[1]:>7.3f}{np.exp(x[2]):>9.3f}{s[2]:>7.3f}"
              f"{MOTOR_RESISTANCE[m] * np.exp(x[2]):>9.4f}{correlation:>11.3f}")

    print("\n=== prop nominal coefficients (log-sigma = fractional uncertainty) ===")
    print(f"{'prop':<28}{'kTnom*1e9':>10}{'sig':>6}{'kPnom*1e13':>12}{'sig':>6}"
          f"{'p':>6}{'sig':>6}{'FM':>7}")
    out = {}
    for n in props:
        x = res.atVector(pk[n]); s = np.sqrt(np.diag(marg.marginalCovariance(pk[n])))
        kT_nom = np.exp(x[0])
        p = np.exp(x[-1])
        if len(x) == 2:
            print(f"{n:<28}{kT_nom*1e9:>10.3f}{s[0]:>6.3f}"
                  f"{'--':>12}{'--':>6}{p:>6.3f}{s[1]:>6.3f}{'--':>7}")
            continue
        kP_nom = np.exp(x[1])
        d = [r for r in rows if r["catalogue_name"] == n][0]["diameter_m"]
        A = np.pi * (d / 2) ** 2
        rpm = np.median([r["rpm"] for r in rows if r["catalogue_name"] == n])
        T = kT_nom * rpm ** p; P = kP_nom * rpm ** 3
        fm = (T ** 1.5 / np.sqrt(2 * RHO * A)) / P
        out[n] = (kT_nom, kP_nom, p, fm)
        print(f"{n:<28}{kT_nom*1e9:>10.3f}{s[0]:>6.3f}"
              f"{kP_nom*1e13:>12.3f}{s[1]:>6.3f}{p:>6.3f}{s[2]:>6.3f}{fm:>7.2f}")

    if out:
        fms = np.array([v[3] for v in out.values()])
        print(f"\nfigure of merit: min={fms.min():.2f} median={np.median(fms):.2f} "
              f"max={fms.max():.2f}   (>1 is physically impossible)")
        print(f"  FM>1 props: {int((fms>1).sum())}/{len(fms)}")

    print("\n=== prop per-motor instance spread (multiplier vs nominal) ===")
    print(f"{'prop':<28}{'kT x min':>10}{'kT x max':>10}"
          f"{'kP x min':>10}{'kP x max':>10}")
    for n in props:
        x_nom = res.atVector(pk[n])
        kT_nom = np.exp(x_nom[0])
        kP_nom = np.exp(x_nom[1]) if len(x_nom) > 2 else None
        kT_mults = []
        kP_mults = []
        for (pn, m), ikey in ik.items():
            if pn != n:
                continue
            xi = res.atVector(ikey)
            kT_mults.append(np.exp(xi[0]) / kT_nom)
            if len(xi) > 1 and kP_nom is not None:
                kP_mults.append(np.exp(xi[1]) / kP_nom)
        kt_min = min(kT_mults) if kT_mults else 1.0
        kt_max = max(kT_mults) if kT_mults else 1.0
        if kP_nom is not None and kP_mults:
            kp_min = min(kP_mults); kp_max = max(kP_mults)
            print(f"{n:<28}{kt_min:>10.3f}{kt_max:>10.3f}"
                  f"{kp_min:>10.3f}{kp_max:>10.3f}")
        else:
            print(f"{n:<28}{kt_min:>10.3f}{kt_max:>10.3f}"
                  f"{'--':>10}{'--':>10}")


if __name__ == "__main__":
    main()
