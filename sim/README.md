# Tailsitter LOS simulator

In-process Betaflight SITL (via the SimITL fork) + tailsitter 6-DOF physics +
raylib/SDL2 line-of-sight front end. Implements `FLIGHT_SIM_PLAN.md`.

## Layout

- `as_built.py` — generates `as_built.json`. Airfoil body from the optimizer;
  elevon hinge line/span and prop position from the Onshape `Final Assembly`
  revolute mates (cached with the render meshes in `assets/`; refresh with
  `uv run python -m sim.as_built --fetch`). Mass 80 g measured, not CAD.
  `thrustScale` is the unmeasured static-thrust correction.
- `load_cli.py` + `wing.bin` — builds the SITL eeprom from the aircraft's
  Betaflight CLI dump and verifies it by booting a fresh SITL on the saved
  file and reading back every setting, mixer rule and aux range. Rebuild after
  changing the real config:
  `uv run --with websocket-client python -m sim.load_cli ~/Documents/BTFL_cli_backup_....txt`.
  It reports what it skips (board identity, `resource`, `serial`), what the
  SITL firmware lacks (dynamic notch, bidirectional DShot, battery profiles),
  and one sim override (`motor_pwm_protocol = PWM`: no DShot in the SITL, and
  idle is the same fraction of throttle either way).
- `wing_model.py` + `test_wing_equiv.py` — Python reference model; the C++
  `TailsitterPhysics` must match it.
- `test_control_signs.py` — with Betaflight in the loop: every axis is
  negative feedback, and sticks command the right direction. This is what
  pins the gyro frame and the servo/motor output mapping.
- Run both: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest sim/`;
  render transforms: `sim/frontend/build/test_wingmesh sim/as_built.json`.
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
./build/wingsim --preset 2      # defaults: --wing ../as_built.json --eeprom ../wing.bin
```

- Presets (`--preset` and keys): `1` hand launch, `2` hover start, `3` cruise;
  `R` resets the current preset; `+/-` or wheel zoom; `H` help.
- Radio: treated as the receiver. Joystick axis *i* is RC channel *i+1*,
  passed to Betaflight untouched, so the aircraft's own `map`/`aux` config
  decides what each channel does (arming is AUX1 = ch5, 1700-2100). Use the
  radio model you fly the wing with. `calib.json` (optional, `"ch1"`..`"ch8"`:
  axis/inv/lo/hi/center) only exists to patch a radio model that differs.
  `--listaxes` prints live axis values.
- Keyboard (only when no joystick): arrows = roll/pitch (up = stick forward),
  W/S = throttle, A/D = yaw, space = ch5 switch.
- Arming behaves like the real FC: arm switch off and throttle low at boot,
  then flip it. The HUD shows ch1-8 and Betaflight's arming-disable flags.
- Launch hold: after a launch/reset the aircraft is held at the preset pose
  until Betaflight is armed and driving the motors above idle, then let go
  with the preset's launch velocity. The camera re-aims on every launch.
- The sim flies on a copy (`frontend/wing_runtime.bin`), so `wing.bin` only
  changes when `load_cli.py` rebuilds it.

- Site: Black Mountain Ranch Park, pilot at the photo-sphere spot
  (32.99064, -117.12827), launching toward 308.8 deg true. `terrain.py`
  fetches USGS 3DEP elevation (AWS Terrain Tiles, cached in
  `assets/terrain/tiles/`, see `ATTRIBUTION.md`) and writes
  `assets/terrain/local.bin`, a 2.4 km low-poly ground mesh drawn in 3D
  (shadow and crash checks follow it; `--no-terrain` for flat ground), and
  the 8192x4096 placeholder sky: procedural sky over the real skyline,
  ray-cast from the pilot's eye out to 30 km. Change the site/heading at the
  top of `terrain.py` and rerun it.
- On the ground mesh (5 km square, 20 m cells): a 30 m x 6 m dirt runway
  running SW-NE through the centre of the flat area the pilot stands in
  (`features.json`), and OpenStreetMap scenery from `scenery.py`
  (`scenery.bin`): houses extruded from the real building footprints (sand
  walls, brick hip roofs; flat roofs over 900 m^2), the two transmission
  lines on their mapped towers (lattice towers with sagging conductors; 40 m
  for the 230 kV line, 28 m otherwise), and 25 m light poles around the lit
  pitches at 4S Ranch Sports Park. OSM has no heights here, so all heights
  are typical values set at the top of `scenery.py`. `scenery.bin` is not
  committed (10 MB, derived): build it once with
  `uv run python -m sim.scenery` (offline, from the cached `osm.json`).
- Scenery: an equirectangular photo sphere (`--sky <file>`, default the
  placeholder; `--no-sky` for none). The eye never leaves the pilot spot, so
  a sphere shot from where you stand to fly (at eye height) is exactly right.
  `--sky-heading <deg>` = the heading in the image, measured from its centre
  column and increasing clockwise, that you face at launch (sim +x); the
  placeholder's centre column is the launch heading and its ticks and N/E/S/W
  letters are true bearings, to compare a real photo against. Use the highest resolution you can: the 35 deg LOS view shows
  ~55 px/deg on a 1080p screen versus ~23 px/deg in an 8192-wide sphere.

The aircraft is the Onshape CAD (`assets/*.stl`, right half mirrored), with
each elevon rotated about its mated hinge axis.

## Strip aerodynamics: sweep, sideslip, reversed flow

Each spanwise strip uses simple sweep theory: the section sees the flow
perpendicular to its swept quarter-chord line (36 deg, mirrored left/right),
so sideslip loads the two halves differently -- roll away from the sideslip
and weak weathervaning (`test_aero_signs.py`). Flow along the quarter-chord
line adds skin friction only. The elevon effect is rescaled to its own hinge
sweep (21 deg, from the Onshape mate). With flow from the trailing edge the
section is mirrored: force at 3/4 chord, no camber moment, and the elevon
acts as a weak, reversed leading-edge flap -- so backward flight is
pitch-unstable, as on the real wing. Unvalidated without flight data; the
strips also have no finite-span lift-slope correction (aspect ratio ~2.8), so
lift slope is still on the high side.

## Hover ANGLE mode (patched Betaflight)

`SimITL/patches/betaflight/0001-tailsitter-hover-angle-mode.patch` (applied
by CMake at configure time) adds `angle_tailsitter` to the PID profile. When
it is ON and ANGLE mode is active, self-levelling runs in a hover frame (the
body pitched 90 deg nose-up), and its rate setpoints are mapped back onto the
body axes. The rate PIDs, mixer and servo mixing are unchanged:

Sticks keep their 3D-plane (acro hover) meaning:

| stick | hover motion | body axis / actuator | levelled |
|---|---|---|---|
| pitch | nose leans toward belly/back | pitch, elevons together | yes (angle) |
| yaw (rudder) | nose leans left/right | yaw, differential thrust | yes (angle) |
| roll | spin about vertical | roll, elevons opposed | no (rate) |

The patch also limits the simulator build's quad-only `rMat` sign fix
(`imu.c`) to non-fixed-wing mixers, so the tailsitter's attitude reaches
Betaflight as a proper rotation.

In the sim, ch6 (AUX2) is a three-position mode switch, added by `load_cli.py`
(the real config has neither mode): low = PASSTHRU, middle = acro, high =
hover ANGLE.

PASSTHRU sends the sticks straight to the actuators, like CRRCSim. Stock
Betaflight only did that for the servos: it zeroes the PIDs in fixed-wing
PASSTHRU and the motor mix reads only PIDs, so rudder (differential thrust)
was dead -- as found on the real wing.
`SimITL/patches/betaflight/0002-passthru-drives-motor-mix.patch` feeds the motor
mix the same stick values (in PID units), still capped by `pidsum_limit_yaw`. `test_hover_angle.py` checks that the hover holds and
rejects 4 rad/s kicks on every axis.

Findings (sim only; hover realism is unvalidated):
- Hover authority is ample: ~7200 deg/s^2 pitch, ~4000 yaw (elevons),
  ~1800 roll at +-30 % differential thrust. Trim needs ~1.2 deg of elevon.
- The untrimmed hover pitching moment (~600 deg/s^2) is what makes the
  real config (`i_roll = i_pitch = 0`) feel unstable: hands-off, acro tips
  past 90 deg in 5 s, while ANGLE holds with a ~5.5 deg lean. With
  `i_roll = i_pitch = 40`, acro holds within 4 deg and ANGLE levels to 0.

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

**These findings predate the gyro-frame fix** (the SITL fed Betaflight pitch
and yaw rates with the wrong sign, and the output mapping had been flipped to
compensate, reversing the pitch and yaw sticks) and the as-built geometry
(motors at 60 mm, not 35.8 mm). Re-run before trusting them.

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
