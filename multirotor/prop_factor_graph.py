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
  prop  node: [log kT, log kP]  with  T = kT*rpm^2,  P_mech = kP*rpm^3
  motor node: [log kt_scale, log i0_scale]  multiplicative corrections on the
              datasheet's own Kt = KT_NUMERATOR/kV and I0.

Factors:
  thrust  (unary on prop) : T_pred  = kT*rpm^2                    vs bench thrust
  current (prop + motor)  : I_pred  = Q/(Kt*kt_scale) + I0*i0_scale
                            Q       = kP*rpm^2*30/pi
                          : vs bench current -- this is the factor that ties
                            the two node types together.
  priors  (unary)         : motor corrections centred on 1.0; strength set
                            per motor by how the datasheet was sourced.

Robust (Huber) noise models on the data factors so the known-corrupt
tmotor_m1103 rows are downweighted by the estimator instead of being
hand-excluded.

Run with: uv run python -m multirotor.prop_factor_graph
"""

import numpy as np
import gtsam
from gtsam import symbol
from collections import defaultdict

import multirotor.motor_model as mm
from multirotor.calibrate_prop_aero import load_bench_rows, MOTOR_KV_I0

RHO = 1.225

# Prior sigma (in log space, ~fractional) on each motor's Kt and I0 correction.
# This is the "unary factors with differing strengths" idea: a motor whose
# spec came off the manufacturer's own table is trusted harder than one whose
# resistance was scraped from a reseller listing (see motor_datasheets.csv --
# M1103's R is from a third-party listing and its sweep has known-bad rows).
DEFAULT_KT_PRIOR_SIGMA = 0.10   # 10% -- kV is usually well specified
DEFAULT_I0_PRIOR_SIGMA = 0.35   # I0 varies strongly with rpm/voltage; loose
LOOSE_MOTORS = {"M1103"}        # noisy datasheet -> let the data move it more
LOOSE_KT_PRIOR_SIGMA = 0.25
LOOSE_I0_PRIOR_SIGMA = 0.60


def _keys(rows):
    props = sorted({r["catalogue_name"] for r in rows})
    motors = sorted({r["motor_key"] for r in rows})
    pk = {n: symbol("p", i) for i, n in enumerate(props)}
    mk = {m: symbol("m", i) for i, m in enumerate(motors)}
    return props, motors, pk, mk


def _thrust_factor(key, rpm, thrust_n, sigma_n):
    """Unary on the prop node: T = kT*rpm^2 (kT held as log kT)."""
    def err(this, values, H=None):
        lkT, _ = values.atVector(key)
        pred = np.exp(lkT) * rpm ** 2
        if H is not None:
            H[0] = np.array([[pred, 0.0]])
        return np.array([pred - thrust_n])
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma_n))
    return gtsam.CustomFactor(noise, [key], err)


def _current_factor(pkey, mkey, rpm, current_a, kt_nom, i0_nom, sigma_a):
    """Binary: bench current sees the prop's shaft torque THROUGH the motor."""
    def err(this, values, H=None):
        _, lkP = values.atVector(pkey)
        lkts, li0s = values.atVector(mkey)
        kP = np.exp(lkP)
        torque = kP * rpm ** 2 * 30.0 / np.pi        # = kP*rpm^3/omega
        kt = kt_nom * np.exp(lkts)
        i_torque = torque / kt
        i_idle = i0_nom * np.exp(li0s)
        pred = i_torque + i_idle
        if H is not None:
            H[0] = np.array([[0.0, i_torque]])        # d/d log kP
            H[1] = np.array([[-i_torque, i_idle]])    # d/d log kt_scale, log i0_scale
        return np.array([pred - current_a])
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, sigma_a))
    return gtsam.CustomFactor(noise, [pkey, mkey], err)


def build_graph(rows):
    props, motors, pk, mk = _keys(rows)
    graph = gtsam.NonlinearFactorGraph()

    for m in motors:
        loose = m[0] in LOOSE_MOTORS
        sig = np.array([LOOSE_KT_PRIOR_SIGMA if loose else DEFAULT_KT_PRIOR_SIGMA,
                        LOOSE_I0_PRIOR_SIGMA if loose else DEFAULT_I0_PRIOR_SIGMA])
        graph.add(gtsam.PriorFactorVector(
            mk[m], np.zeros(2), gtsam.noiseModel.Diagonal.Sigmas(sig)))

    for r in rows:
        thrust_n = r["thrust_g"] * 9.81e-3
        # measurement sigmas: relative floor + absolute floor
        graph.add(_thrust_factor(pk[r["catalogue_name"]], r["rpm"], thrust_n,
                                 0.05 * thrust_n + 0.005))
        kv, i0 = MOTOR_KV_I0[r["motor_key"]]
        kt_nom = mm.KT_NUMERATOR / kv
        graph.add(_current_factor(pk[r["catalogue_name"]], mk[r["motor_key"]],
                                  r["rpm"], r["current_a"], kt_nom, i0,
                                  0.05 * r["current_a"] + 0.05))
    return graph, props, motors, pk, mk


def initial_values(rows, props, motors, pk, mk):
    v = gtsam.Values()
    by_prop = defaultdict(list)
    for r in rows:
        by_prop[r["catalogue_name"]].append(r)
    for n in props:
        rs = by_prop[n]
        rpm = np.array([x["rpm"] for x in rs])
        tn = np.array([x["thrust_g"] for x in rs]) * 9.81e-3
        kT = float(np.sum(tn * rpm ** 2) / np.sum(rpm ** 4))
        # crude kP seed from datasheet Kt, refined by the optimizer
        cur = np.array([x["current_a"] for x in rs])
        kt = np.array([mm.KT_NUMERATOR / MOTOR_KV_I0[x["motor_key"]][0] for x in rs])
        i0 = np.array([MOTOR_KV_I0[x["motor_key"]][1] for x in rs])
        q = np.maximum(cur - i0, 1e-3) * kt
        kP = float(np.median(q / (rpm ** 2 * 30.0 / np.pi)))
        v.insert(pk[n], np.array([np.log(max(kT, 1e-14)), np.log(max(kP, 1e-18))]))
    for m in motors:
        v.insert(mk[m], np.zeros(2))
    return v


def solve(rows):
    graph, props, motors, pk, mk = build_graph(rows)
    init = initial_values(rows, props, motors, pk, mk)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(300)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    result = opt.optimize()
    return graph, result, props, motors, pk, mk, init


def main():
    rows = [r for r in load_bench_rows() if r["motor_key"] in MOTOR_KV_I0]
    graph, res, props, motors, pk, mk, init = solve(rows)
    print(f"{len(rows)} bench rows | {len(props)} props | {len(motors)} motors | "
          f"{graph.size()} factors | {2*(len(props)+len(motors))} variables")
    print(f"error: {graph.error(init):.1f} -> {graph.error(res):.1f}\n")

    marg = gtsam.Marginals(graph, res)

    print("=== motors: multiplicative corrections the data wants on the datasheet ===")
    print(f"{'motor':<18}{'Kt x':>9}{'+/-':>7}{'I0 x':>9}{'+/-':>7}")
    for m in motors:
        x = res.atVector(mk[m]); c = marg.marginalCovariance(mk[m])
        s = np.sqrt(np.diag(c))
        print(f"{m[0]+'-'+str(m[1]):<18}{np.exp(x[0]):>9.3f}{s[0]:>7.3f}"
              f"{np.exp(x[1]):>9.3f}{s[1]:>7.3f}")

    print("\n=== props: jointly-estimated coefficients (log-sigma = frac. uncertainty) ===")
    print(f"{'prop':<28}{'kT*1e9':>10}{'sig':>7}{'kP*1e13':>10}{'sig':>7}{'FM':>7}")
    out = {}
    for n in props:
        x = res.atVector(pk[n]); s = np.sqrt(np.diag(marg.marginalCovariance(pk[n])))
        kT, kP = np.exp(x[0]), np.exp(x[1])
        d = [r for r in rows if r["catalogue_name"] == n][0]["diameter_m"]
        A = np.pi * (d / 2) ** 2
        # FM at a representative rpm: ideal induced power / actual shaft power
        rpm = np.median([r["rpm"] for r in rows if r["catalogue_name"] == n])
        T = kT * rpm ** 2; P = kP * rpm ** 3
        fm = (T ** 1.5 / np.sqrt(2 * RHO * A)) / P
        out[n] = (kT, kP, fm)
        print(f"{n:<28}{kT*1e9:>10.3f}{s[0]:>7.3f}{kP*1e13:>10.3f}{s[1]:>7.3f}{fm:>7.2f}")

    fms = np.array([v[2] for v in out.values()])
    print(f"\nfigure of merit: min={fms.min():.2f} median={np.median(fms):.2f} "
          f"max={fms.max():.2f}   (>1 is physically impossible)")
    print(f"  FM>1 props: {int((fms>1).sum())}/{len(fms)}")


if __name__ == "__main__":
    main()
