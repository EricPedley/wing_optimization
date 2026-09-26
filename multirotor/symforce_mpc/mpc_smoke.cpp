// Host smoke test for the generated quad MPC: linearize at trim, run a few
// control ticks against a roll-rate step, print the commands.  Compile:
//   g++ -O2 -std=c++17 -I. -Ieigen_shim mpc_smoke.cpp -o /tmp/mpc_smoke
#include <array>
#include <chrono>
#include <cstdio>

#include "quad_mpc.h"

int main() {
  quad_mpc::Mpc mpc;
  const auto t0 = std::chrono::steady_clock::now();
  mpc.init();
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("init_us %.1f\n",
              std::chrono::duration<double, std::micro>(t1 - t0).count());

#ifdef DUMP_MATS
  for (int i = 0; i < quad_mpc::kStates; ++i) {
    for (int j = 0; j < quad_mpc::kStates; ++j)
      std::printf("A %d %d %.17g\n", i, j, mpc.A()(i, j));
    for (int j = 0; j < quad_mpc::kInputs; ++j)
      std::printf("B %d %d %.17g\n", i, j, mpc.B()(i, j));
  }
#endif

  // plant state at hover trim: zero rates, trim rotor speed, trim pwm
  double gyro[3] = {0., 0., 0.};
  double omega[4], filtered[4];
  for (int i = 0; i < 4; ++i) {
    omega[i] = quad_mpc::kTrimState[3 + i];
    filtered[i] = quad_mpc::kTrimState[7 + i];
  }
  const double ref[3] = {2.0, 0., 0.};  // ~115 deg/s roll step
  for (int k = 0; k < 5; ++k) {
    const auto s = std::chrono::steady_clock::now();
    const std::array<double, 4> cmd = mpc.update(gyro, omega, filtered, ref);
    const auto e = std::chrono::steady_clock::now();
    std::printf("tick %d cmd %.6f %.6f %.6f %.6f solve_us %.1f\n", k,
                cmd[0], cmd[1], cmd[2], cmd[3],
                std::chrono::duration<double, std::micro>(e - s).count());
    if (k == 0) {
      for (int i = 0; i < 15; ++i)
        std::printf("x_ref %d %.17g\n", i, mpc.last_x_ref()(i, 0));
      for (int i = 0; i < 4; ++i)
        std::printf("u_ref %d %.17g\n", i, mpc.last_u_ref()(i, 0));
      for (int j = 0; j < quad_mpc::kAugZ; ++j)
        std::printf("K0 %d %.17g\n", j, mpc.gain(0)(0, j));
      for (int i = 0; i < quad_mpc::kVIn; ++i)
        std::printf("f0 %d %.17g\n", i, mpc.ff(0)(i, 0));
    }
  }
  return 0;
}
