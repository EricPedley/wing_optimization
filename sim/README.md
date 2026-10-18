# Tailsitter LOS simulator

In-process Betaflight SITL (via the SimITL fork) + tailsitter 6-DOF physics +
raylib/SDL2 line-of-sight front end. Implements `FLIGHT_SIM_PLAN.md`.

## Layout

- `as_built.py` — generates `as_built.json`: planform/geometry/linkage params
  for the built aircraft (mass 80 g measured, not CAD). `thrustScale` is the
  unmeasured static-thrust correction — tune it to match the real hover
  throttle.
- `load_cli.py` — pushes the real Betaflight CLI dump into the SITL eeprom.
- `wing_model.py` + `test_wing_equiv.py` — Python reference model; the C++
  `TailsitterPhysics` must match it (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  uv run python -m pytest sim/test_wing_equiv.py`).
- `frontend/` — `wingsim`, the playable LOS simulator.
- `SimITL/tools/simitl-wing` — headless runner used for diagnostics
  (`idle`, `forces`, `--probe`, `--frozen --inj`, `--hold`, `--nopulses`).

## Build

```sh
cd SimITL && ./build.sh                       # SimITL + tools (once)
cd sim/frontend && cmake -B build && cmake --build build -j
```

raylib comes from the prebuilt bundle at `~/code/PufferLib/raylib-5.5_linux_amd64`
(override with `-DRAYLIB_ROOT=...`). SDL2 comes from the system.

## Run

```sh
cd sim/frontend
./build/wingsim --wing ../as_built.json \
  --eeprom ../../SimITL/build/linux/tools/simitl-wing/wing.bin --preset 1
```

- Presets: `1` hand launch, `2` hover start, `3` cruise; `R` resets the current
  preset; `1/2/3` relaunch after a crash; `+/-` or wheel zoom; `H` help.
- Joystick: SDL2 raw axes; the RadioMaster Pocket reads as 8 axes
  (`--listaxes` prints live values). Channel mapping is in `calib.json`
  (created on first run); order is Ail/Ele/Thr/Rud = axes 0-3, arm switch on
  axis 4.
- Keyboard fallback: arrows = pitch/roll, W/S = throttle, A/D = yaw,
  space = arm.
- `--thr <v>` / `--autoarm` drive throttle/arm without the radio (testing).

The wing model is procedural: planform extruded from `as_built.json`
(chord/taper/sweep/thickness/camber), elevons pivoted about the measured
hinge axes, prop discs at the motor stations.

## Validation against flight log (M5)

`validate_flight.py` decodes a Blackbox CSV, converts the logged
`rcCommand[0..3]` (µs, semantic order ROLL/PITCH/YAW/THROTTLE) back to
normalized sim channels, replays them through `simitl-wing replay`, and
compares gyro rates and control power per axis.

```sh
blackbox_decode --output-dir /tmp/bbl_wing ~/Downloads/wing_gliderport_flights.bbl
uv run python sim/validate_flight.py \
    --csv /tmp/bbl_wing/wing_gliderport_flights.02.csv --t0 70 --t1 90
```

Findings from `wing_gliderport_flights.02` (the 92 s cruise flight,
window 70-90 s):

- The replay runs end-to-end and stays armed (79 % of the window), but the
  aircraft diverges into a ~600 deg/s yaw tumble with elevons saturated at
  the +18 deg limit — the untuned plant does not yet track the closed loop.
- The real wing carries a persistent cruise trim (~-5.8 deg elevon
  equivalent, pidSum_pitch ~= -105 us) with a small continuous pitch drift
  (-5.7 deg/s) that the pilot catches by hand; expected with `i_pitch = 0`.
- Tuning targets, in rough priority: `thrustScale` (unmeasured static
  thrust), elevon neutral/trim offset (`elevonNeutralDeg`-style bias),
  then CG and aero coefficients.

## Known gaps / validation status

- **Hover realism is unvalidated** — no hover-attempt Blackbox log exists.
- Static thrust per motor is assumed, not measured; `thrustScale` in
  `as_built.json` corrects it.
- The FC runs the aircraft's real acro config (`i_roll = i_pitch = 0`, pure
  PD): holding a hover requires continuous stick correction, same as the
  real wing — a no-I acro loop cannot hold attitude by itself.
- Betaflight will not arm with throttle above idle and runs a gyro
  calibration during boot — the sim respects both, so hold the throttle low
  and still until armed (the tools do this automatically).
