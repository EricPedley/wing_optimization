// Fixed-size MPC driver around the symforce-generated kernels in generated/.
//
// Mirrors multirotor/optimal_control_symforce.py::RiccatiMPC running the
// four_motor_control_experiment.py controller in deviation coordinates:
//   z = [x_dev_scaled; u_prev_dev],  cost = sum z'Qz z + 2 z'Nz u + u'Rz u
//                                        + 2 z'c_z + 2 u'c_u,  terminal Pf.
// init() linearizes the discrete plant at hover trim by composing the
// generated per-substep jacobians (chain rule over kSubsteps); solve() runs
// one backward Riccati pass (riccati_step x kHorizon) and a forward rollout
// with per-stage command clipping (forward_step x kHorizon).  All state is
// stack/static arrays - no heap, no dynamic allocation, no Eigen dependency
// (the generated headers see the shim in eigen_shim/).
//
// Feed update() with the measured plant state each control tick: gyro rates
// (rad/s, BF frame order roll/pitch/yaw), rotor speed (rad/s), filtered PWM,
// and the rate reference (rad/s).  It returns absolute motor PWM commands.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cmath>
#include <cstring>

#include "Eigen/Core"  // eigen_shim/ satisfies this; real Eigen works too
#include "generated/substep_jac.h"
#include "generated/riccati_step.h"
#include "generated/forward_step.h"
#include "generated/collective_thrust_jac.h"
#include "generated/collective.h"
#include "generated/mpc_constants.h"

namespace quad_mpc {

// Column-major helpers over raw doubles (the Eigen shim provides storage
// and indexing only, so the driver does its own loops).
inline void set_zero(double* m, int n) { std::memset(m, 0, n * sizeof(double)); }

// out(i,j) += a(i,:) . b(:,j); column-major, r x k by k x c.
inline void matmul_acc(int r, int k, int c, const double* a, const double* b,
                       double* out) {
  for (int j = 0; j < c; ++j)
    for (int i = 0; i < r; ++i) {
      double s = 0.;
      for (int l = 0; l < k; ++l) s += a[i + l * r] * b[l + j * k];
      out[i + j * r] += s;
    }
}

inline void matvec(int r, int c, const double* a, const double* x, double* y) {
  for (int i = 0; i < r; ++i) {
    double s = 0.;
    for (int j = 0; j < c; ++j) s += a[i + j * r] * x[j];
    y[i] = s;
  }
}

class Mpc {
 public:
  using Vec4 = Eigen::Matrix<double, kInputs, 1>;
  using Vec3 = Eigen::Matrix<double, kVIn, 1>;
  using Vec11 = Eigen::Matrix<double, kPhysicalStates, 1>;
  using VecN = Eigen::Matrix<double, kStates, 1>;
  using VecZ = Eigen::Matrix<double, kAugZ, 1>;
  using MatN = Eigen::Matrix<double, kStates, kStates>;
  using MatB = Eigen::Matrix<double, kStates, kInputs>;
  using MatV = Eigen::Matrix<double, kStates, kVIn>;
  using MatZ = Eigen::Matrix<double, kAugZ, kAugZ>;
  using MatK = Eigen::Matrix<double, kVIn, kAugZ>;

  // Linearize the discrete map at one operating point (physical 11-state +
  // applied input) and scale into trim-anchored deviation coordinates.
  static void linearize(const Vec11& x_op, const Vec4& u_op, MatN* As,
                        MatB* Bs) {
    double acc[11 * 19], acc2[11 * 19];
    set_zero(acc, 11 * 19);
    for (int i = 0; i < 11; ++i) acc[i * 12] = 1.;  // acc(i,i)=1, col-major
    Vec11 phys = x_op;
    Vec4 delayed = u_op, sag = u_op;
    Eigen::Matrix<double, 11, 11> jx;
    Eigen::Matrix<double, 11, 4> jd, js;
    for (int k = 0; k < kSubsteps; ++k) {
      Vec11 next = sym::SubstepJac<double>(phys, delayed, sag, kRemainingMah,
                                         &jx, &jd, &js);
      for (int j = 0; j < 19; ++j)
        for (int i = 0; i < 11; ++i) {
          double s = 0.;
          for (int l = 0; l < 11; ++l) s += jx(i, l) * acc[l + j * 11];
          if (j >= 11 && j < 15) s += jd(i, j - 11);
          if (j >= 15) s += js(i, j - 15);
          acc2[i + j * 11] = s;
        }
      std::memcpy(acc, acc2, sizeof(acc));
      phys = next;
    }
    MatN A;
    MatB B;
    A.setZero();
    B.setZero();
    for (int i = 0; i < 11; ++i)
      for (int j = 0; j < 11; ++j) A(i, j) = acc[i + j * 11];
    for (int i = 0; i < 11; ++i)
      for (int j = 0; j < 4; ++j) {
        A(i, 11 + j) += acc[i + (11 + j) * 11];            // delayed = head slot
        A(i, kStates - 4 + j) += acc[i + (15 + j) * 11];   // sag = tail slot
      }
    for (int i = 0; i + 4 < 4 * kDelaySteps; ++i)
      A(11 + i, 15 + i) = 1.;  // queue shifts toward the head (oldest) slot
    for (int i = 0; i < 4; ++i) B(kStates - 4 + i, i) = 1.;
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kStates; ++j)
        (*As)(i, j) = A(i, j) * kScale[j] / kScale[i];
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kInputs; ++j)
        (*Bs)(i, j) = B(i, j) / kScale[i];
  }

  // Gain-scheduled MPC: the rate loop loses margin as rotor speed rises, so
  // the plant is relinearized at kNumSched operating points (kSchedX/U) and a
  // full gain table is built per point.  Per tick, tables are interpolated by
  // the mean measured rotor speed.
  void init() {
    // v-space weights: applied command u = m*1 + M*v with the collective
    // channel m pinned to the target mean, so the QP only optimizes the
    // differential v.  M'M = 4I -> weights scale by 4 (kRv/kSv constants).
    for (int i = 0; i < kVIn; ++i)
      for (int j = 0; j < kVIn; ++j) {
        Rv_(i, j) = i == j ? kRv : 0.;
        Sv_(i, j) = i == j ? kSv : 0.;
      }
    // Pz = diag(kTerminalCost, 0) on the augmented state (shared across the
    // schedule: weights are design constants, the trim DARE is the terminal
    // approximation).
    Pz_.setZero();
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kStates; ++j)
        Pz_(i, j) = kTerminalCost[i + j * kStates];
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kStates; ++j) Q_(i, j) = kQ[i + j * kStates];
    // Startup history: motors were at 0 while held/arming -> queue of zeros,
    // previous applied command all-equal -> zero differential.
    for (int i = 0; i < kVIn; ++i) v_prev_[i] = 0.;
    for (int i = 0; i < 4 * kDelaySteps; ++i) queue_[i] = 0.;

    // The affine feedforward f_k is linear in the target rhs (bias, ref,
    // collective request).  Probe the seven basis columns of kTargetInv per
    // operating point: G[s][k][i][j] = f_k for unit rhs_j.  Per tick the
    // backward pass collapses to f_k = G_k * rhs - a 3x7 matvec instead of 20
    // Riccati steps.  The probe passes fold in BOTH constant affine terms:
    // the disturbance increment dz = kDt*bias (rate rows) and the pinned
    // collective feedthrough m*1 into the queue tail.
    for (int s = 0; s < kNumSched; ++s) {
      Vec11 xop;
      Vec4 uop;
      for (int i = 0; i < 11; ++i) xop(i, 0) = kSchedX[s][i];
      for (int i = 0; i < 4; ++i) uop(i, 0) = kSchedU[s][i];
      linearize(xop, uop, &AsS_[s], &BsS_[s]);
      // v-space input matrix: Bv = B * M (the plant sees m + Mv).
      for (int i = 0; i < kStates; ++i)
        for (int j = 0; j < kVIn; ++j) {
          double acc = 0.;
          for (int l = 0; l < 4; ++l)
            acc += BsS_[s](i, l) * kMixV[l * kVIn + j];
          BvS_[s](i, j) = acc;
        }
      MatZ P, P2;
      VecZ g, g2;
      Vec3 f;
      for (int j = 0; j < 7; ++j) {
        VecN xr;
        Vec4 ur;
        for (int i = 0; i < kStates; ++i)
          xr(i, 0) = kTargetInv[i + j * kAugmented];
        for (int i = 0; i < 4; ++i)
          ur(i, 0) = kTargetInv[kStates + i + j * kAugmented];
        // Decompose the motor-space reference into (collective, v).
        const double mj = 0.25 * (ur(0, 0) + ur(1, 0) + ur(2, 0) + ur(3, 0));
        Vec3 vr;
        for (int a = 0; a < kVIn; ++a) {
          double acc = 0.;
          for (int i = 0; i < 4; ++i)
            acc += kMixV[i * kVIn + a] * (ur(i, 0) - mj);
          vr(a, 0) = 0.25 * acc;
        }
        // Constant affine terms for this probe: disturbance increment on the
        // rate rows, pinned collective m into the queue tail slots.
        VecZ a;
        a.setZero();
        if (j < 3) a(j, 0) = kDt;
        for (int i = 0; i < 4; ++i) a(kStates - 4 + i, 0) = mj;
        P = Pz_;
        for (int i = 0; i < kStates; ++i) {
          double acc = 0.;
          for (int l = 0; l < kStates; ++l)
            acc -= kTerminalCost[i + l * kStates] * xr(l, 0);
          g(i, 0) = acc;
        }
        for (int i = kStates; i < kAugZ; ++i) g(i, 0) = 0.;
        for (int k = kHorizon - 1; k >= 0; --k) {
          VecZ geff;
          for (int i = 0; i < kAugZ; ++i) {
            double acc = g(i, 0);
            for (int l = 0; l < kAugZ; ++l) acc += P(i, l) * a(l, 0);
            geff(i, 0) = acc;
          }
          sym::RiccatiStep<double>(AsS_[s], BvS_[s], Q_, Rv_, Sv_, xr, vr, P,
                                   geff, &K_[s][k], &f, &P2, &g2);
          for (int i = 0; i < kVIn; ++i) G_[s][k][i][j] = f(i, 0);
          P = P2;
          g = g2;
        }
      }
    }
  }

  // One control tick: measured state + rate reference -> absolute PWM.
  // gyro/body rates in rad/s (BF order), omega rotor rad/s, filtered 0..1,
  // ref rad/s, collective_target as a fraction of hover thrust.
  std::array<double, 4> update(const double gyro[3], const double omega[4],
                               const double filtered[4], const double ref[3],
                               double collective_target = 1.) {
    VecN x_meas;
    for (int i = 0; i < 3; ++i) x_meas(i, 0) = gyro[i];
    for (int i = 0; i < 4; ++i) x_meas(3 + i, 0) = omega[i];
    for (int i = 0; i < 4; ++i) x_meas(7 + i, 0) = filtered[i];
    for (int i = 0; i < 4 * kDelaySteps; ++i) x_meas(11 + i, 0) = queue_[i];
    VecN x_dev;
    for (int i = 0; i < kStates; ++i)
      x_dev(i, 0) = (x_meas(i, 0) - kTrimState[i]) / kScale[i];

    // Schedule index: mean measured rotor speed selects the linearization
    // point (and interpolates between neighbors).  Low-passed: the raw
    // telemetry jitter would otherwise wander the gain tables tick to tick.
    const double w_meas =
        0.25 * (omega[0] + omega[1] + omega[2] + omega[3]);
    wbar_ += 0.05 * (w_meas - wbar_);
    const double wbar = wbar_;
    int seg = 0;
    double alpha = 0.;
    if (wbar >= kSchedW[kNumSched - 1]) {
      seg = kNumSched - 2;
      alpha = 1.;
    } else {
      while (seg < kNumSched - 2 && wbar > kSchedW[seg + 1]) ++seg;
      const double span = kSchedW[seg + 1] - kSchedW[seg];
      alpha = span > 1e-9 ? (wbar - kSchedW[seg]) / span : 0.;
      alpha = std::max(0., std::min(1., alpha));
    }
    MatN As;
    MatB Bs;
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kStates; ++j)
        As(i, j) = AsS_[seg](i, j) +
                   alpha * (AsS_[seg + 1](i, j) - AsS_[seg](i, j));
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kInputs; ++j)
        Bs(i, j) = BsS_[seg](i, j) +
                   alpha * (BsS_[seg + 1](i, j) - BsS_[seg](i, j));
    MatV Bv;
    for (int i = 0; i < kStates; ++i)
      for (int j = 0; j < kVIn; ++j)
        Bv(i, j) = BvS_[seg](i, j) +
                   alpha * (BvS_[seg + 1](i, j) - BvS_[seg](i, j));

    // Rate-disturbance observer: the gap between the measured rates and the
    // model's one-step prediction integrates into bias_ (Kalman gain baked as
    // kBiasGain).  bias_ shifts the equilibrium target so a persistent
    // disturbance torque produces the right steady-state command.
    if (have_pred_) {
      double e[3];
      for (int i = 0; i < 3; ++i) e[i] = x_dev(i, 0) - pred_rate_[i];
      for (int i = 0; i < 3; ++i) {
        double b = bias_[i];
        for (int j = 0; j < 3; ++j) b += kBiasGain[i + j * 3] * e[j];
        bias_[i] = std::max(-kBiasMax, std::min(kBiasMax, b));
      }
    }

    // Equilibrium target for this rate reference.  collective_target is
    // clamped to the model's feasible maximum so the clipped rollout gets a
    // reachable equilibrium instead of thrashing at the input bound.
    const double coll = std::max(0., std::min(kCollMax, collective_target));
    double rhs7[7] = {bias_[0], bias_[1], bias_[2], ref[0] / kRateScale,
                      ref[1] / kRateScale, ref[2] / kRateScale, 0.};
    double target[kAugmented];
    // The collective row is linearized at hover and thrust ~ w^2 is convex, so
    // the linear solve's target lands off the requested thrust at large
    // collective.  The target is affine in the collective request, so bisect
    // the request value until the model's true nonlinear thrust lands on
    // coll: target(lambda) = base + lambda * kTargetInv[:,6].
    {
      double base[kAugmented];
      double rhs0[7] = {bias_[0], bias_[1], bias_[2], ref[0] / kRateScale,
                        ref[1] / kRateScale, ref[2] / kRateScale, 0.};
      set_zero(base, kAugmented);
      matmul_acc(kAugmented, 7, 1, kTargetInv, rhs0, base);
      const int c6 = 6 * kAugmented;
      auto thrustAt = [&](double lam) {
        Eigen::Matrix<double, 11, 1> xabs;
        for (int i = 0; i < 11; ++i)
          xabs(i, 0) =
              kTrimState[i] + (base[i] + lam * kTargetInv[c6 + i]) * kScale[i];
        return sym::Collective<double>(xabs, 0.) / kHoverThrust;
      };
      // lam < 0 is legitimate (below-hover collective), so bracket downward
      // too.
      double lo = coll - 3., hi = coll;
      while (thrustAt(hi) < coll && hi < 32.) hi *= 2.;
      while (thrustAt(lo) > coll && lo > -4.) lo -= 0.5;
      for (int it = 0; it < 18; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (thrustAt(mid) < coll) lo = mid; else hi = mid;
      }
      // Feasibility cap: the equilibrium command u_ref = base_u + lam*col6_u
      // is affine in lam, so cap lam where any motor's target hits an input
      // bound (upper: ~0.1 headroom below saturation; lower: the idle floor).
      // Under damage the bias term lifts base_u, so this automatically backs
      // the collective off to what the quad can still produce.
      double lam = 0.5 * (lo + hi);
      constexpr double kUrefCap = 0.5;                    // applied ~ trim+.5
      constexpr double kUrefFloor = 0.05 - 0.404;         // applied ~ 0.05
      double lam_lo = -1e9, lam_hi = 1e9;
      for (int i = 0; i < 4; ++i) {
        const double du = kTargetInv[c6 + kStates + i];
        if (du > 1e-9) {
          lam_hi = std::min(lam_hi, (kUrefCap - base[kStates + i]) / du);
          lam_lo = std::max(lam_lo, (kUrefFloor - base[kStates + i]) / du);
        } else if (du < -1e-9) {
          lam_hi = std::min(lam_hi, (kUrefFloor - base[kStates + i]) / du);
          lam_lo = std::max(lam_lo, (kUrefCap - base[kStates + i]) / du);
        }
      }
      lam = std::max(lam_lo, std::min(lam_hi, lam));
      rhs7[6] = lam;
      for (int i = 0; i < kAugmented; ++i)
        target[i] = base[i] + lam * kTargetInv[c6 + i];
    }
    VecN& x_ref = x_ref_;
    Vec4& u_ref = u_ref_;
    for (int i = 0; i < kStates; ++i) x_ref(i, 0) = target[i];
    for (int i = 0; i < 4; ++i) u_ref(i, 0) = target[kStates + i];
    // Mixer decomposition of the equilibrium command: the collective channel
    // m is pinned to the target mean (exactly like BF's mixer pinning the
    // mean command to throttle); the QP optimizes the differential v.
    const double m = 0.25 * (u_ref(0, 0) + u_ref(1, 0) + u_ref(2, 0) +
                             u_ref(3, 0));

    // Per-stage affine feedforward: f_k = G_k * rhs.  G was probed with both
    // constant affine terms folded in (disturbance dz on the rate rows and
    // the pinned collective m into the queue tail), so this matvec
    // reproduces the full Riccati+disturbance+collective map exactly.
    const double dz0 = kDt * bias_[0], dz1 = kDt * bias_[1],
                dz2 = kDt * bias_[2];
    for (int k = 0; k < kHorizon; ++k) {
      Vec3& f = f_[k];
      for (int i = 0; i < kVIn; ++i) {
        double s = 0.;
        for (int j = 0; j < 7; ++j)
          s += (G_[seg][k][i][j] +
                alpha * (G_[seg + 1][k][i][j] - G_[seg][k][i][j])) * rhs7[j];
        f(i, 0) = s;
      }
    }

    // Forward rollout with per-stage clipping of the APPLIED command
    // u = m + M v - mixer-style: the collective channel is pinned, per-motor
    // bounds bite on the differential (rate authority shrinks as m nears the
    // bounds, which is the physics, not a projection hack).
    VecZ z;
    for (int i = 0; i < kStates; ++i) z(i, 0) = x_dev(i, 0);
    for (int i = 0; i < kVIn; ++i) z(kStates + i, 0) = v_prev_[i];
    Vec4 u_min, u_max;
    for (int i = 0; i < 4; ++i) {
      u_min(i, 0) = kUDevMin[i];
      u_max(i, 0) = kUDevMax[i];
    }
    Vec4 u_app, u0;
    Vec3 v;
    VecZ z2;
    MatK Kk;
    for (int k = 0; k < kHorizon; ++k) {
      for (int i = 0; i < kVIn; ++i)
        for (int j = 0; j < kAugZ; ++j)
          Kk(i, j) = K_[seg][k](i, j) +
                     alpha * (K_[seg + 1][k](i, j) - K_[seg][k](i, j));
      // v = -(K z + f)
      for (int i = 0; i < kVIn; ++i) {
        double s = f_[k](i, 0);
        for (int j = 0; j < kAugZ; ++j) s += Kk(i, j) * z(j, 0);
        v(i, 0) = -s;
      }
      // Applied motor command: u = clip(m + M v) per motor.
      for (int i = 0; i < 4; ++i) {
        double s = m;
        for (int j = 0; j < kVIn; ++j) s += kMixV[i * kVIn + j] * v(j, 0);
        u_app(i, 0) = std::max(u_min(i, 0), std::min(u_max(i, 0), s));
      }
      // z' = Az z + Bu u_applied (x part); the slew tail records the
      // EFFECTIVE applied differential (post-clip), not the commanded v -
      // the slew penalty must anchor to what the actuators really did, or a
      // saturated plan leaves a phantom slew reference that feeds back.
      for (int i = 0; i < kStates; ++i) {
        double s = 0.;
        for (int j = 0; j < kStates; ++j) s += As(i, j) * z(j, 0);
        for (int j = 0; j < 4; ++j) s += Bs(i, j) * u_app(j, 0);
        z2(i, 0) = s;
      }
      for (int a = 0; a < kVIn; ++a) {
        double s = 0.;
        for (int i = 0; i < 4; ++i)
          s += kMixV[i * kVIn + a] * (u_app(i, 0) - m);
        z2(kStates + a, 0) = 0.25 * s;  // M+ = M'/4 (columns orthonormal)
      }
      // The disturbance persists across the horizon: z' += dz (rate rows).
      z2(0, 0) += dz0; z2(1, 0) += dz1; z2(2, 0) += dz2;
      z = z2;
      if (k == 0) {
        u0 = u_app;
        for (int i = 0; i < kVIn; ++i) v0_(i, 0) = z2(kStates + i, 0);
        // One-step rate prediction for next tick's observer innovation.
        for (int i = 0; i < 3; ++i) pred_rate_[i] = z2(i, 0);
        have_pred_ = true;
      }
    }
    for (int i = 0; i < kVIn; ++i) v_prev_[i] = v0_(i, 0);

    std::array<double, 4> command;
    for (int i = 0; i < 4; ++i)
      command[i] = kTrimCommand[i] + u0(i, 0);
    if (kDelaySteps) {
      for (int i = 0; i + 4 < 4 * kDelaySteps; ++i) queue_[i] = queue_[i + 4];
      for (int i = 0; i < 4; ++i) queue_[4 * (kDelaySteps - 1) + i] = command[i];
    }
    return command;
  }

  // Command the ESC delay line is currently acting on (queue head).
  double delayedCommand(int i) const { return queue_[i]; }

  // Estimated rate disturbance (scaled-rate-per-step units).
  double bias(int i) const { return bias_[i]; }

  // Resync the command queue / slew state to what is actually being applied
  // (e.g. while another controller drives the motors) so switching over is
  // bumpless.
  void trackCommands(const double applied[4]) {
    have_pred_ = false;  // prediction predates the tracked commands
    double mu = 0.;
    for (int i = 0; i < 4; ++i) mu += applied[i] - kTrimCommand[i];
    mu *= 0.25;
    for (int i = 0; i < 4; ++i)
      for (int d = 0; d < kDelaySteps; ++d) queue_[d * 4 + i] = applied[i];
    for (int a = 0; a < kVIn; ++a) {
      double s = 0.;
      for (int i = 0; i < 4; ++i)
        s += kMixV[i * kVIn + a] * (applied[i] - kTrimCommand[i] - mu);
      v_prev_[a] = 0.25 * s;
    }
  }

  // Diagnostics expose the hover (schedule point 0) linearization.
  const MatN& A() const { return AsS_[0]; }
  const MatB& B() const { return BsS_[0]; }
  const MatK& gain(int k) const { return K_[0][k]; }
  const Vec3& ff(int k) const { return f_[k]; }
  const VecN& last_x_ref() const { return x_ref_; }
  const Vec4& last_u_ref() const { return u_ref_; }

 private:
  MatN Q_;
  MatN AsS_[kNumSched];
  MatB BsS_[kNumSched];
  MatV BvS_[kNumSched];
  MatZ Pz_;
  VecN x_ref_;
  Vec4 u_ref_;
  Vec3 v0_;
  Eigen::Matrix<double, kVIn, kVIn> Rv_, Sv_;
  MatK K_[kNumSched][kHorizon];
  Vec3 f_[kHorizon];
  // rhs (bias, ref, coll) -> affine feedforward f, per schedule point
  double G_[kNumSched][kHorizon][kVIn][7];
  double queue_[4 * kDelaySteps > 0 ? 4 * kDelaySteps : 4];
  double v_prev_[kVIn];
  double bias_[3] = {0., 0., 0.};
  double wbar_ = 0.;  // filtered schedule variable (mean rotor speed)
  double pred_rate_[3] = {0., 0., 0.};
  bool have_pred_ = false;
};

}  // namespace quad_mpc
