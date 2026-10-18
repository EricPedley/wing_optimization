# Tailsitter LOS Flight Simulator — Plan

Goal: a line-of-sight practice simulator for *this* wing, specifically for the
part that is hard in real life — transitions into hover and holding a stable
hover — running the same Betaflight config as the real aircraft.

Status: planned, not started. Written to be handed to an implementing agent;
open decisions are resolved below, and the remaining to-dos are marked
**TODO (implementer)**.

Repo conventions: see `AGENTS.md` (run Python with `uv run`; set
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for pytest on machines with ROS).
`SimITL/` is a git submodule pointing at our SimITL fork, so commit
SimITL changes there first, then bump the pin here.

## Why not an existing sim

- **pr0p** (we have a BepInEx plugin for it): physics actually lives in SimITL,
  and SimITL's interface is hard-coded to 4 motors / no servos. A fixed wing
  would need new physics *and* Harmony patches into pr0p's closed `Quad`
  class, input, and camera. The only thing pr0p would contribute is scenery,
  which LOS flying barely needs.
- **CRRCSim** (installed at `~/programs/crrcsim`, flown 2026-10-17): its
  LaRCsim FDM computes control power from freestream dynamic pressure only, so
  elevons do nothing at zero airspeed; there is no slipstream-over-surface
  model; aero is linear stability derivatives with a stall patch, not valid at
  hover's ~90° AoA; all motors share one throttle, so no differential-thrust
  yaw. Its stock models are docile trainers and feel much easier than the real
  wing. Patching it means replacing the FDM, i.e. most of what we'd write
  anyway.

What we *do* take from CRRCSim: the LOS camera and the air-start launch
(below). It is open source; source is in `~/programs/crrcsim-src`.

## Architecture

```
 RadioMaster Pocket (USB HID joystick)
          │ raw axes
          ▼
 ┌──────────────── front end (C++, raylib + SDL2 input) ────────────────┐
 │ stick → rcData[8]   LOS camera   launch presets   crash → reset      │
 │ render wing mesh, elevons rotated about hinge axes                   │
 └───────────────┬──────────────────────────────────────▲───────────────┘
                 │ simitl_update(StateInput)            │ simitl_get_state()
                 ▼                                      │
 ┌──────────────── SimITL fork (libsimitl, in-process C API) ───────────┐
 │ Betaflight SITL (wing build, user's CLI dump)                        │
 │   → motorsPwm[2], servosPwm[2..3]                                    │
 │ TailsitterPhysics: 6-DOF body, strip aero ±180°, prop slipstream,    │
 │   servo dynamics, bench-calibrated motor/prop, battery               │
 └──────────────────────────────────────────────────────────────────────┘
```

SimITL already exposes a C API (`simitl_init / simitl_update /
simitl_get_state`, `SimITL/src/main.cpp`) that the game calls in-process, so
the front end links the library directly. The existing Python harness
(`multirotor/simitl_step_response.py`) keeps working for headless runs.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Renderer | Own minimal C++ front end (raylib) | Pose + mesh + ground + sky + shadow is all LOS needs; no engine to fight |
| Joystick input | SDL2 raw joystick axes | raylib only exposes mapped gamepads; the Pocket's HID descriptor isn't in the gamepad DB. SDL is what CRRCSim uses and it read the Pocket fine |
| Physics location | C++ inside the SimITL fork | Must step in lockstep with Betaflight's loop |
| Physics source of truth | `airfoil/airfoil_model.py` | Already models slipstream over elevons, diff-thrust yaw, inertia for this airframe |
| Flight controller | Betaflight SITL from SimITL with user's config | Same mixer/PIDs/rates as the real wing |
| Ground contact | **None.** Air start; touching ground = crash → reset | Real wing is always hand launched; CRRCSim's air start for gearless models was easy to manage |
| Camera | Port CRRCSim's LOS camera | User likes it; preserves real LOS geometry (reversed controls when inbound), which is the skill being practised |

## Betaflight side

Config: `~/Documents/BTFL_cli_backup_20261017_104010_BETAFPVG473_WING.txt`
(Betaflight 2026.6.0-alpha custom build, BETAFPVG473_WING, Acro).

Mixer, decoded:

- `mixer CUSTOMAIRPLANE`, `yaw_type = DIFF_THRUST`
- `mmix 0 1 0 0 +1`, `mmix 1 1 0 0 -1` → motor = throttle ± yaw
- `smix` source 0 = stabilized roll, 1 = stabilized pitch:
  - servo 2 (`SERVO_FLAPS` slot) = −roll − pitch
  - servo 3 (`SERVO_FLAPPERON_1` slot) = −roll + pitch
- Arm on AUX1 (channel 5) 1700–2100. `small_angle = 180`. I-term zero on all
  axes; `p_yaw 120`, `d_yaw 60`.

The vendored Betaflight (`SimITL/extern/betaflight`, 2025.12.0) already has
`CUSTOMAIRPLANE`, `yaw_type`/`YAW_TYPE_DIFF_THRUST` and `USE_WING` code.

Changes needed in the SimITL fork:

1. `extern/betaflightext/src/main/target/simITL/target.h`: define
   `USE_SERVOS` and `USE_WING` (neither is on today).
2. `target.c`: `servosPwm[]` is written by `pwmWriteServo` but `static`; export
   it like `motorsPwm` and read it in `src/sim/bf.cpp` next to the existing
   `motorsPwm[0..3]` reads (`bf.cpp:156`).
3. `src/network/packets.h`: add a vehicle type to `StateInit`, servo outputs
   (and position) to `StateOutput`, and tailsitter parameters. Keep the quad
   path working — the quad tuning tools depend on it.
4. Load the CLI dump into the SITL eeprom (paste via CLI over the SITL
   serial/TCP port, then `save`), and keep the resulting eeprom file in the
   repo as the sim's config.

Version (implementation detail): the real wing runs a custom
2026.6.0-alpha build, while SimITL vendors 2025.12.0. Start with the vendored
version. If the CLI dump doesn't apply cleanly (unknown settings), or wing
behaviour differs, move SimITL's Betaflight forward rather than editing the
config to fit.

## Airframe parameters (as-built)

Two sources, which disagree on purpose:

- **Airfoil body** (planform, thickness, camber, sweep): taken directly from
  the optimizer, `airfoil/optimum.json`.
- **Elevons and hinge/linkage**: the built aircraft does *not* follow the
  optimizer here. Use the Onshape document
  (`fc16e035b9272e1f16de9511`, part studio `Flying Wing`) as the source.

Onshape layout (checked over the API on 2026-10-17):

- Workspace `Main` (`8283d374293017844d6ea3cc`). The only branch.
- `Final Assembly` (`8e364e94f22d49c4a15a2f6a`): 10 part instances, **one
  half of the aircraft only**. One `wing`, one `control surface`, and one
  `motor mount` + prop; the mirroring happens in the part studio
  (`Mirror 1`). The extractor has to mirror across the centreline for the
  other side.
- Mates (read with `GET /assemblies/d/{did}/w/{wid}/e/{eid}` with
  `includeMateFeatures=true`):
  - `Revolute 1`: `control surface <1>` ↔ `fuselage <1>`, the **elevon hinge**.
    Mate connector z-axis (hinge direction) is ≈ (0.362, 0.932, 0) in the
    fuselage part frame, at ≈ (−121.4, −22.0, 1.3) mm.
  - `Revolute 2`: prop `Part 1 <3>` ↔ `motor mount <1>`, the **prop spin
    axis**. The connector is at (−29.5, −60.0, 3.0) mm on the motor mount,
    along −x of the mount frame.
  - The `matedCS` values are in each *part's* frame. Compose them with the
    instance transforms from `rootAssembly.occurrences` to get assembly-frame
    axes.
- Part studio `Flying Wing` (`b893633847e2c346141c4f25`) also has the
  `hinge holes` sketch and `hinge line normal` plane. Use them as a cross-check
  on the mate.

Plan: write `sim/as_built.py` to pull, over the existing `airfoil/onshape.py`
client (note: `_request` already prefixes `/api/v10`, so pass paths starting
at `/assemblies/...`):

1. Hinge axis per elevon from `Revolute 1` (mirrored), and prop position and
   axis per motor from `Revolute 2` (mirrored). The renderer uses the same
   axes.
2. Elevon chord, span, and inboard/outboard stations, from the
   `control surface` part's bounding box or geometry about the hinge axis,
   plus deflection limits.
3. **Mass, CG, and inertia tensor from Onshape's assembly mass properties.**
   This replaces the estimated `inertia()` in the model, as long as the part
   densities/masses in Onshape are right.
4. Servo-to-surface linkage ratio from `linkage/` with the as-built geometry.

Output: `sim/as_built.json` — the single parameter file the C++ physics loads.

Static thrust per motor (1002 21000 kv, 2× 0.9" 3-blade) is **assumed, not
measured**. Use the value `airfoil_model.py` uses (`THRUST_PER_MOTOR`) and
the `multirotor/` motor/prop model, and expose a thrust scale factor in
`as_built.json`. That way, if the sim's hover throttle looks wrong, it can be
corrected later without code changes. Decided: no bench test.

## Physics (TailsitterPhysics, C++ in the SimITL fork)

- 6-DOF rigid body, full inertia tensor from `as_built.json`.
- Strip theory along the span: each strip sees freestream + rotation-induced
  velocity + slipstream if it lies behind a prop (`washed_span_fraction`,
  `slipstream_velocity` with the same development and width factors as
  `airfoil_model.py`).
- Section aero valid for −180°…180° AoA: thin-airfoil Cl/Cm and the existing
  Cl_max correlation in the attached range, blended into Viterna/flat-plate
  post-stall coefficients. Hover lives at ~90°, so this is not optional.
- Elevon as a flap on each strip it spans, with the effectiveness and
  large-deflection efficiency from `airfoil_model.py`.
- Servo model: rate limit + first-order lag, travel limits.
- Motors/props: port the bench-calibrated model from `multirotor/`
  (`motor_model.py`, `prop_aero_model.py`, factor-graph params), including
  prop torque and rotor inertia; battery sag reuses SimITL's.
- Ground: none. Below ground plane = crash → reset to launch.

Reference check: a Python implementation of the same force/moment function
(JAX, built on `airfoil_model.py`) evaluates a grid of states (airspeed, AoA,
sideslip, throttle, elevon deflection). The C++ must match it to tolerance,
and that comparison is a test. That way the sim and the design model can't
drift apart without anyone noticing.

## Front end

- **Mesh.** Export from Onshape (glTF via the API, or the STEPs in
  `~/Downloads/Flying Wing/` converted with OCP/cadquery). Merge the body
  parts into one static mesh; keep each elevon separate (mirrored) and pivot
  it about its hinge axis from `as_built.json` using the servo position SimITL
  returns. Props can be a disc.
- **Input.** SDL2 joystick → `rcData[8]` in −1…1, with a one-time
  calibration screen (axis assignment, invert, endpoints) saved to a config
  file. AUX1 → arm.
- **Camera** (from `crrc_graphics.cpp:UpdateCamera`, `zoom.cpp`). The eye
  stays at the pilot position, about 1.7 m up; it does not translate.
  - Tracking: look-at follows the aircraft. Optional "sloppy" dead zone: the
    camera only turns once the aircraft leaves a cone of `sloppy × FOV`, and
    otherwise drifts toward it slowly.
  - Autozoom: `zoom_calc`. Beyond 21.5 m the effective distance grows only
    by `autozoom` (default 0.05) per metre, so the aircraft keeps a nearly
    constant on-screen size. Base FOV is 35°, adjustable from 5° to 65°.
  - Model-view inset: when the aircraft is small on screen, draw a magnified
    picture-in-picture of it looking straight from the pilot, so attitude
    stays readable.
- **Launch presets** (as in CRRCSim's `<launch>`: altitude, speed, pitch
  angle, offset from pilot), e.g.:
  - *Hand launch*: about 2 m up, a few m in front of the pilot, forward at
    launch speed.
  - *Hover start*: nose up at about 5 m with zero velocity, to drill hover
    without flying a transition first.
  - *Cruise start*: level at about 30 m, at cruise speed, to drill
    transitions into hover.
- **Visual cues** that matter for LOS: ground shadow, true scale and distance,
  distinct top/bottom colours, horizon and sun. Plain grass field is enough.

## Validation

1. **Unit:** C++ forces/moments match the Python reference grid.
2. **Open-loop:** trim speed, stall speed, and hover throttle in the sim match
   `airfoil_model.evaluate()` and the bench data.
3. **Closed-loop against flight. TODO (implementer):** find the Blackbox log
   of the most recent wing flight. The user says it's in `~/Downloads`; it
   wasn't found by name on 2026-10-17. Wing logs have board name
   `BETAFPVG473_WING`; the `*ASCENT82*` and `btfl_00*.bbl` files there are
   from other aircraft. That flight was hand launch → forward flight → belly
   landing (the big acceleration spike), with no hover, so it only covers
   cruise:
   - Replay its sticks through the sim and compare roll/pitch rate and yaw
     response in forward flight.
   - Ignore everything from the landing spike onwards.
   - If a hover-attempt log turns up later, the same comparison for hover
     (oscillation frequency/amplitude, yaw from differential thrust) is the
     important one.
4. **Pilot check:** does it feel as hard as the real wing?

## Milestones

1. **Betaflight wing SITL:** SimITL fork builds with servos; the user's CLI
   dump loads; headless sticks move servos with the right signs per the
   mixer above.
2. **`as_built.json`:** Onshape extraction (hinge axes, elevon geometry, mass
   properties) plus optimizer body.
3. **TailsitterPhysics:** C++ model plus the Python reference test; headless
   hover-hold and transition runs from Python.
4. **Front end:** mesh with animated elevons, SDL input with calibration,
   CRRCSim camera, launch presets, crash reset.
5. **Validation against flight logs;** tune any model terms the data
   disagrees with.

Later, if wanted: wind and gusts, sensor noise matched to the real gyro,
recording Blackbox from the SITL for side-by-side review.

## Open items

- **TODO (implementer):** locate the wing's Blackbox log and do validation
  step 3 (cruise only).
- **TODO (implementer):** sanity-check Onshape's mass properties against the
  real all-up weight and the `mass_items()` budget in `airfoil_model.py`
  before trusting its inertia. If parts lack materials, fall back to the
  model's `inertia()`.
- No hover-attempt log exists yet, so hover realism is unvalidated. Flag this
  in the sim's README rather than tuning by guesswork.
