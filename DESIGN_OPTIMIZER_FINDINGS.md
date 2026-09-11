# Multirotor design optimizer — calibrated parameters and software throttle cap

This document summarises what happened when the motor/prop parameters calibrated by `multirotor/prop_factor_graph.py` were plugged back into the continuous multirotor design optimizer, and what happened when a software throttle limit was added for real motor/prop combos that would otherwise exceed the ESC current cap.

## 1. Calibrated parameters that were plugged in

Three changes bridge the factor-graph calibration to the design optimizer:

1. **ESC throttle exponent (`duty_gamma`) in `multirotor/quad_model.py`**
   - `DUTY_GAMMA = 0.789`, fitted by the factor graph.
   - Added `effective_voltage(vbat, throttle_frac) = vbat * throttle_frac ** DUTY_GAMMA`.
   - Replaced all partial-throttle voltage mappings in hover, throttle sweep, spin-up, and `optimize_efficiency` discharge paths.

2. **Motor torque constant (`KT_NUMERATOR`) in `multirotor/motor_model.py`**
   - Changed from SimITL's `8.3` to the SI value `60 / (2π) ≈ 9.5493`.
   - `multirotor/motor_scaling.py` recalibrates its `K_m` fit automatically from this.

3. **Reduced-order propeller model (`multirotor/prop_factor_graph_model.py`)**
   - Replaces the unconstrained BEMT (`multirotor/prop_aero_model.py`) in the optimizer path.
   - Predicts static thrust and torque at a reference RPM from log-linear regressions on diameter, pitch, and blade count, using the factor-graph per-prop `kT`/`kP` fits.
   - `quad_model.py` now imports it as `pa`.

## 2. Baseline vs. calibrated `optimize.py` (max TWR)

| quantity | baseline (uncalibrated) | after plugging in params |
|---|---|---|
| kV | 17 343 | 12 293 |
| stator volume, mm³ | 528 | 562 |
| prop diameter, mm | 49.8 | 57.6 |
| blade count | 2.09 | 2.0 |
| pitch, mm | 57.9 | 53.5 |
| total mass, g | 66.5 | 69.2 |
| full-throttle RPM | 57 977 | 37 745 |
| thrust per motor, N | 1.77 | 1.23 |
| current per motor, A | 12.00 (at limit) | 11.88 |
| efficiency | 0.785 | 0.830 |
| TWR | 10.88 | 7.25 |
| spin-up 10–90 %, ms | 31.9 | 37.3 |
| tip Mach | 0.441 | 0.332 |

The optimizer now picks a larger, lower-kV motor/prop combination and a much less extreme TWR. The current constraint is no longer the only active binding constraint; the design sits inside the ESC cap.

## 3. `optimize_efficiency.py` (min hover power)

Continuous optimum:

| design variable | value |
|---|---|
| kV | 11 367 |
| stator volume, mm³ | 429 |
| prop diameter, mm | 57.0 |
| blade count | 2 |
| pitch, mm | 63.0 |

Realized against real catalogue parts:

| quantity | value |
|---|---|
| motor | GTS-V3 1203-11500 (11 500 kV, 0.100 Ω, 4.50 g) |
| prop | HQProp 51 mmx2 (51.0 mm, 38.1 mm pitch, 2 blades) |
| total mass | 61.8 g |
| TWR | 4.39 |
| current per motor @ full throttle | 5.27 A |
| spin-up 10–90 % | 39.3 ms |
| tip Mach | 0.276 |
| hover throttle fraction | 0.275 |
| hover current, total | 7.30 A |
| flight time, 680 mAh | 5.6 min |

This is a realistic whoop-class result: well under the 12 A ESC limit, good TWR, and ~5–6 min flight time.

## 4. Software throttle limit

Real motor/prop pairs that exceed the ESC current at full throttle are now allowed, but their full-throttle operating point is recomputed at a throttle fraction `t_lim` that caps per-motor current at exactly `12 A`. Thrust, TWR, spin-up time, and tip Mach are all reported at that capped point. Hover is computed independently, so hover efficiency is preserved.

### 4.1 Example of a current-violating combo

The max-TWR continuous design realised against a real 1303-11500 motor + HQProp T3x2x3 76.2 mm prop:

| | uncapped | with throttle cap |
|---|---|---|
| throttle limit | 1.000 | 0.958 |
| current per motor, A | 12.64 | 12.00 |
| thrust per motor, N | 1.216 | 1.155 |
| TWR | 6.76 | 6.42 |
| spin-up 10–90 %, ms | 199.7 | 202.2 |

As expected, the current is brought inside the cap and thrust/TWR drop, but this particular combo still violates the 50 ms spin-up budget — the motor simply does not have the torque/inertia to spin the big prop fast enough.

### 4.2 Pareto frontier (spin-up time vs. hover power)

Running the full real motor × prop Pareto sweep with the throttle cap produces a richer trade-off curve. The lowest-power combos all violate the 50 ms spin-up budget; the first point that still clears both current and spin-up is the same 1203-11500 + 51 mmx2 combo found above.

| motor | prop | TWR | current/A | spin-up ms | hover power W | flight min |
|---|---|---:|---:|---:|---:|---:|
| M1103-8000 | Gemfan Hurricane 3018-2 (76.5 mm, 2-blade) | 4.60 | 5.18 | 136 | **9.72** | **7.3** |
| GTS-V3 1203-11500 | Gemfan 75 mm 2-blade Toothpick | 6.28 | 8.22 | 98 | 9.72 | 6.3 |
| GTS-V3 1203-11500 | HQProp Ultralight 65 mm 2-blade | 5.43 | 6.70 | 64 | 10.23 | 6.6 |
| GTS-V3 1002-19000 | HQProp Ultralight 2×0.9 51 mm | 7.28 | 11.02 | 57 | 10.99 | 5.2 |
| GTS-V3 1203-11500 | HQProp 51 mmx2 | 4.39 | 5.27 | **39** | 11.48 | 6.7 |

Key observation: the throttle cap **does** expose substantially more efficient real designs (down to ~9.7 W hover power and ~7.3 min flight), but they sit on the wrong side of the 50 ms spin-up wall. The cap helps current but does not fix the inertia/motor-torque problem that limits spin-up.

## 5. Caveats and limitations

- The reduced-order prop model is a 3-parameter log-linear fit over 18 catalogue props. Residuals are ~40 % for thrust and ~50 % for torque, so it is a broad-brush model rather than a precise per-prop BEMT. It is, however, far more physically defensible than the unconstrained BEMT outside the 60–77 mm range.
- It uses `p = 2` (quadratic thrust/torque scaling), whereas the factor graph found per-prop `p` values between ~1.57 and ~2.10. High-RPM extrapolation for props with strongly non-quadratic thrust curves is therefore approximate.
- The throttle cap is currently applied only in the realized/Pareto path (`_evaluate_motor_with_prop`). The continuous `optimize.py` / `optimize_efficiency` cost paths still penalise uncapped current. Adding the cap there too is straightforward and would let the gradient search itself propose current-limited designs.
- `realized_design` and `realize_efficient_design` use the same catalogue constraints; the 50 ms spin-up budget is the binding reason the lowest-power Pareto points are not buildable.

## 6. Files changed

- `multirotor/quad_model.py`: `DUTY_GAMMA`, `effective_voltage`, throttle-limit helper `_throttle_limited_operating_point`, and `_evaluate_motor_with_prop`.
- `multirotor/motor_model.py`: `KT_NUMERATOR = 60 / (2π)`.
- `multirotor/prop_factor_graph_model.py`: new reduced-order prop model.
- `multirotor/optimize_efficiency.py`: `effective_voltage` in hover/discharge calculations.
- `multirotor/test_quad_model.py`: updated hover-throttle test and `battery_mah` test.

## 7. Suggested next steps

1. **Propagate the throttle cap into the continuous cost functions** (`quad_model.evaluate`, `_prop_only_cost`, `optimize_efficiency.cost`, `_prop_only_efficiency_cost`) so the gradient search can discover capped designs directly.
2. **Relax or re-derive the spin-up budget** if the priority is flight time; the Pareto frontier shows 6.6–7.3 min is available at the cost of 64–136 ms spin-up.
3. **Re-fit the prop model** with interaction or higher-order terms if more catalogue props become available, reducing the ~40 % residual.
4. **Investigate 2S / higher-current ESC assumptions**, which would move both the voltage and the current cap and likely change the optimum.
