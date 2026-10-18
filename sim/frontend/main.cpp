// Tailsitter LOS flight sim front end.
// raylib rendering + SDL2 joystick, driving the in-process SimITL library
// (Betaflight SITL + TailsitterPhysics) exactly like the headless runner.

#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>
#include <cmath>
#include <raylib.h>
#include <raymath.h>
#include <SDL2/SDL.h>

#include "simitl.h"
#include "sim/state.h"
#include "util/vector_math.h"
#include "json.h"
#include "wingcfg.h"
#include "wingmesh.h"
#include "input.h"
#include "loscam.h"
#include "skybox.h"
#include "terrain.h"
#include "scenery.h"
#include <rlgl.h>

using namespace SimITL;

// -- attitude helpers --------------------------------------------------------
// in.rotation rows-of-columns: columns are sim body axes in world coords.

// level flight, nose along +x world
static void levelAttitude(Vec3F rot[3]) {
  // aero level fwd: x_a=(1,0,0) y_a=(0,0,1) z_a=(0,-1,0)
  // sim axes: x_s=-y_a=(0,0,-1) y_s=-z_a=(0,1,0) z_s=x_a=(1,0,0)
  // columns => rows: [[0,0,1],[0,1,0],[-1,0,0]]
  rot[0] = { 0.0f, 0.0f, 1.0f};
  rot[1] = { 0.0f, 1.0f, 0.0f};
  rot[2] = {-1.0f, 0.0f, 0.0f};
}
// nose-up hover attitude (same as the headless runner's)
static void hoverAttitude(Vec3F rot[3]) {
  rot[0] = { 1.0f, 0.0f, 0.0f};
  rot[1] = { 0.0f, 0.0f, 1.0f};
  rot[2] = { 0.0f,-1.0f, 0.0f};
}

// motor output (0..1) above which a held aircraft is let go: idle is 0.055
// (motor_idle 5.5 %), so this is a few percent of real throttle
constexpr float kReleaseMotorOutput = 0.10f;

enum Preset { P_HAND = 0, P_HOVER, P_CRUISE, P_COUNT };
static const char* presetName[P_COUNT] = {"hand launch", "hover", "cruise"};

static void applyPreset(StateInput& in, int p) {
  in.position = {0, 0, 0};
  in.angularVelocity = {0, 0, 0};
  in.linearVelocity = {0, 0, 0};
  switch (p) {
  case P_HAND:
    in.position = {4.0f, 2.0f, 0.0f};
    in.linearVelocity = {9.0f, 0.0f, 0.0f};
    levelAttitude(in.rotation);
    break;
  case P_HOVER:
    in.position = {5.0f, 4.0f, 0.0f};
    hoverAttitude(in.rotation);
    break;
  case P_CRUISE:
    in.position = {-30.0f, 25.0f, 0.0f};
    in.linearVelocity = {15.0f, 0.0f, 0.0f};
    levelAttitude(in.rotation);
    break;
  }
}

// -- scene ------------------------------------------------------------------

static void drawGround() {
  DrawPlane({0, 0, 0}, {2000, 2000}, Color{52, 120, 52, 255});
  for (int i = -200; i <= 200; i += 5) {
    DrawLine3D({(float)i, 0.01f, -200}, {(float)i, 0.01f, 200},
               Color{45, 100, 45, 255});
    DrawLine3D({-200, 0.01f, (float)i}, {200, 0.01f, (float)i},
               Color{45, 100, 45, 255});
  }
}

static void drawShadow(const Vec3F& pos, float span, float groundY) {
  const float a = fminf(0.4f, 2.0f / fmaxf(pos.y - groundY, 0.3f));
  DrawCylinder({pos.x, groundY + 0.05f, pos.z}, span * 0.4f, span * 0.4f, 0.01f, 16,
               Fade(BLACK, a));
}

// "Windowed fullscreen": a fullscreen window at the desktop's own resolution,
// so the video mode never changes, but the window manager knows it is
// fullscreen (panels and docks stay behind it).  F11 goes back to a window.
static void toggleWindowedFullscreen() {
  if (!IsWindowFullscreen()) {
    const int mon = GetCurrentMonitor();
    SetWindowSize(GetMonitorWidth(mon), GetMonitorHeight(mon));
    ToggleFullscreen();
  } else {
    ToggleFullscreen();
    SetWindowSize(1280, 720);
  }
}

// -- main --------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string wingPath = "../as_built.json";
  std::string eeprom = "../wing.bin";   // sim/wing.bin, built by sim/load_cli.py
  std::string calPath = "calib.json";
  std::string assetDir = "../assets";
  // equirectangular photo sphere from the pilot spot; see skybox.h
  std::string skyPath = "../assets/sky/placeholder_equirect.png";
  float skyHeading = 0.0f;
  // low-poly ground mesh from sim/terrain.py (USGS 3DEP elevation)
  std::string terrainPath = "../assets/terrain/local.bin";
  int preset = P_HAND;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto val = [&](const char* n) {
      if (i + 1 >= argc) { fprintf(stderr, "--%s needs a value\n", n); exit(2); }
      return std::string(argv[++i]);
    };
    if (a == "--wing") wingPath = val("wing");
    else if (a == "--eeprom") eeprom = val("eeprom");
    else if (a == "--cal") calPath = val("cal");
    else if (a == "--preset")   // 1-based, same as the 1/2/3 keys
      preset = std::min(std::max(std::stoi(val("preset")) - 1, 0), P_COUNT - 1);
    else if (a == "--assets") assetDir = val("assets");
    else if (a == "--sky") skyPath = val("sky");
    else if (a == "--sky-heading") skyHeading = std::stof(val("sky-heading"));
    else if (a == "--no-sky") skyPath.clear();
    else if (a == "--terrain") terrainPath = val("terrain");
    else if (a == "--no-terrain") terrainPath.clear();
    else if (a == "--listaxes") {
      SDL_Init(SDL_INIT_JOYSTICK);
      const int n = SDL_NumJoysticks();
      printf("%d joystick(s)\n", n);
      for (int k = 0; k < n; k++) {
        SDL_Joystick* j = SDL_JoystickOpen(k);
        printf("%d: %s axes=%d\n", k, SDL_JoystickName(j),
               SDL_JoystickNumAxes(j));
        for (int i = 0; i < 300; i++) {
          SDL_JoystickUpdate();
          SDL_Delay(15);
          printf("  ");
          for (int ax = 0; ax < SDL_JoystickNumAxes(j); ax++)
            printf("[%d]=%+.2f ", ax, SDL_JoystickGetAxis(j, ax) / 32767.0f);
          printf("\r"); fflush(stdout);
        }
        printf("\n");
      }
      return 0;
    }
  }

  // ---- sim init -------------------------------------------------------------
  StateInit si{};
  loadWingConfigInto(wingPath, si);
  if (!eeprom.empty()) {
    // Betaflight writes its eeprom file on save; fly on a copy so the
    // verified config in the repo never changes underneath us.
    const std::string runtime = "wing_runtime.bin";
    std::error_code ec;
    std::filesystem::copy_file(eeprom, runtime,
        std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) {
      fprintf(stderr, "cannot read eeprom %s: %s\n", eeprom.c_str(),
              ec.message().c_str());
      return 1;
    }
    eeprom = runtime;
    std::fill(si.eepromName, si.eepromName + 512, 0);
    memcpy(si.eepromName, eeprom.c_str(), std::min(eeprom.size(), (size_t)511));
  }
  WingGeom wg = loadWingGeom(wingPath);
  simitl_init(si);

  StateInput in{};
  in.delta = 0.001f;
  in.vbat = 4.2f;

  // ---- input ------------------------------------------------------------------
  SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_GAMECONTROLLER);
  SDL_Joystick* joy = nullptr;
  if (SDL_NumJoysticks() > 0) {
    joy = SDL_JoystickOpen(0);
    printf("joystick: %s (%d axes)\n", SDL_JoystickName(joy),
           SDL_JoystickNumAxes(joy));
  } else {
    printf("no joystick - keyboard: arrows pitch/roll, W/S throttle, "
           "A/D yaw, space = ch5 switch\n");
  }
  // radio = receiver: channels pass straight through to Betaflight
  InputMap imap;
  if (imap.load(calPath)) printf("channel calibration: %s\n", calPath.c_str());
  KeyboardRadio kbRadio;

  // ---- window -----------------------------------------------------------------
  // windowed fullscreen from creation: size 0 = the desktop's resolution, so
  // no video-mode change (a toggle sent before the first map is ignored by
  // some window managers)
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT | FLAG_FULLSCREEN_MODE);
  InitWindow(0, 0, "wingsim - tailsitter LOS");
  SetTargetFPS(120);
  // the terrain mesh reaches ~1.7 km (diagonal); raylib's default far plane is 1 km
  rlSetClipPlanes(0.1, 5000.0);

  Camera3D cam{};
  cam.up = {0, 1, 0};
  cam.projection = CAMERA_PERSPECTIVE;
  LosCamera loscam;

  // ---- aircraft model -----------------------------------------------------------
  Terrain terrain;
  if (!terrainPath.empty() &&
      !terrain.load(terrainPath, siteLaunchHeading("../assets/terrain/site.json")))
    fprintf(stderr, "terrain %s not loaded; using flat ground\n", terrainPath.c_str());

  Runway runway;
  Scenery scenery;
  if (terrain.loaded) {
    runway.load("../assets/terrain/features.json", terrain);
    if (!scenery.load("../assets/terrain/scenery.bin"))
      fprintf(stderr, "no scenery.bin (houses, power lines, lights): "
                      "run `uv run python -m sim.scenery`\n");
  }

  Skybox sky;
  if (!skyPath.empty() && !sky.load(skyPath, skyHeading))
    fprintf(stderr, "sky image %s not loaded; using the plain ground\n",
            skyPath.c_str());

  AircraftModels aircraft;
  if (!loadAircraft(assetDir, wg, aircraft)) return 1;

  // props: blades + a blur disc that fades in with rpm, and the motor bells
  Model prop[2], propBlur[2], motor[2];
  for (int i = 0; i < 2; i++) {
    prop[i] = LoadModelFromMesh(makePropMesh(wg.propDia));
    propBlur[i] = LoadModelFromMesh(makePropDiscMesh(wg.propDia));
    motor[i] = LoadModelFromMesh(makeMotorMesh(0.0065f, 0.012f));
  }

  const Vector3 motPosR = aeroToSim(wg.cgX - wg.motorXAft, wg.motorY, wg.motorZ);
  const Vector3 motPosL = aeroToSim(wg.cgX - wg.motorXAft, -wg.motorY, wg.motorZ);

  RenderTexture2D pip = LoadRenderTexture(1, 1);   // sized on first use

  bool crashed = false;
  bool showHelp = true;

  // Launch hold: after a (re)launch the aircraft stays at the preset pose,
  // as if held in the hand, until Betaflight is armed and driving the motors
  // above idle.  Then it is let go with the preset's launch velocity.  This
  // watches Betaflight's outputs, not particular RC channels.
  bool held = true;
  Vec3F launchVel{0.0f, 0.0f, 0.0f};
  auto launch = [&]() {
    applyPreset(in, preset);
    in.position.y += terrain.heightAt(in.position.x, in.position.z);   // presets are AGL
    launchVel = in.linearVelocity;
    in.linearVelocity = {0.0f, 0.0f, 0.0f};
    in.angularVelocity = {0.0f, 0.0f, 0.0f};
    held = true;
    crashed = false;
    loscam.reset(cam, {in.position.x, in.position.y, in.position.z});
  };
  launch();
  float propAngle[2] = {0, 0};
  float leftover = 0.0f;

  while (!WindowShouldClose()) {
    const float dt = fminf(GetFrameTime(), 0.05f);

    // ---- input --------------------------------------------------------------
    // The radio goes to Betaflight exactly as a receiver would deliver it.
    // Arming, boot grace and throttle-low checks are Betaflight's own; a
    // crash only freezes the airframe, it does not touch the sticks.
    SDL_JoystickUpdate();
    float rc[RC_CHANNELS];
    if (joy) imap.fillRc(joy, rc);
    else     kbRadio.fillRc(rc, dt);
    if (IsKeyPressed(KEY_ONE))   { preset = P_HAND;   launch(); }
    if (IsKeyPressed(KEY_TWO))   { preset = P_HOVER;  launch(); }
    if (IsKeyPressed(KEY_THREE)) { preset = P_CRUISE; launch(); }
    if (IsKeyPressed(KEY_R))     launch();
    if (IsKeyPressed(KEY_H))     showHelp = !showHelp;
    if (IsKeyPressed(KEY_F11))   toggleWindowedFullscreen();
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))      loscam.zoomIn();
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) loscam.zoomOut();
    const float wheel = GetMouseWheelMove();
    if (wheel > 0) loscam.zoomIn();
    if (wheel < 0) loscam.zoomOut();
    for (int i = 0; i < RC_CHANNELS; i++) in.rcData[i] = rc[i];

    // ---- physics substeps ------------------------------------------------------
    {
      float acc = dt + leftover;
      const float h = in.delta;          // 1 ms
      int steps = 0;
      while (acc >= h && steps < 64) {
        simitl_update(in);
        const StateOutput o = simitl_get_state();
        if (held && o.armed &&
            fmaxf(o.motorOutput[0], o.motorOutput[1]) > kReleaseMotorOutput) {
          held = false;
          in.linearVelocity = launchVel;
        } else if (crashed || held) {
          // frozen (on the ground, or in the hand); the FC keeps running
          in.angularVelocity = {0.0f, 0.0f, 0.0f};
          in.linearVelocity = {0.0f, 0.0f, 0.0f};
        } else {
          const quat q{o.orientation.x, o.orientation.y,
                       o.orientation.z, o.orientation.w};
          copy(in.rotation, quat_to_mat3(q));
          in.angularVelocity = o.angularVelocity;
          in.linearVelocity  = o.linearVelocity;
          in.position.x += o.linearVelocity.x * h;
          in.position.y += o.linearVelocity.y * h;
          in.position.z += o.linearVelocity.z * h;
        }
        acc -= h;
        steps++;
      }
      leftover = acc;
    }
    const StateOutput out = simitl_get_state();
    for (int i = 0; i < 2; i++) {
      const int dir = (i == 0) ? 1 : -1;  // right +1 / left -1 (counter-rot.)
      propAngle[i] += out.motorRpm[i] * 2.0f * (float)M_PI / 60.0f * dt * dir;
    }
    const float groundY = terrain.heightAt(in.position.x, in.position.z);
    if (!held && in.position.y < groundY + 0.05f) crashed = true;

    // ---- model transforms --------------------------------------------------------
    const Matrix bodyM = bodyMatrix(in.rotation, in.position);
    aircraft.body.transform = bodyM;
    aircraft.elevon[0].transform =
        elevonTransform(bodyM, wg.hingePtL, wg.hingeAxL, out.elevonDeg[0]);
    aircraft.elevon[1].transform =
        elevonTransform(bodyM, wg.hingePtR, wg.hingeAxR, out.elevonDeg[1]);
    prop[0].transform = bodyChild(bodyM, motPosR, MatrixRotateZ(propAngle[0]));
    prop[1].transform = bodyChild(bodyM, motPosL, MatrixRotateZ(propAngle[1]));
    for (int i = 0; i < 2; i++) {
      propBlur[i].transform = prop[i].transform;
      motor[i].transform = bodyChild(bodyM, i == 0 ? motPosR : motPosL, MatrixIdentity());
    }

    loscam.update(cam, {in.position.x, in.position.y, in.position.z}, dt);

    auto drawAircraft = [&]() {
      DrawModel(aircraft.body, {0, 0, 0}, 1.0f, WHITE);
      DrawModel(aircraft.elevon[0], {0, 0, 0}, 1.0f, WHITE);
      DrawModel(aircraft.elevon[1], {0, 0, 0}, 1.0f, WHITE);
    };

    // ---- draw ----------------------------------------------------------------------
    BeginDrawing();
    ClearBackground(Color{135, 175, 210, 255});

    BeginMode3D(cam);
    sky.draw(cam);
    // ground: the terrain mesh if there is one; else a photo sphere already
    // shows the ground; else the flat plane
    if (terrain.loaded) { terrain.draw(); runway.draw(); scenery.draw(); }
    else if (!sky.loaded) drawGround();
    drawShadow(in.position, wg.span, groundY);
    drawAircraft();
    rlDisableBackfaceCulling();
    for (int i = 0; i < 2; i++) {
      DrawModel(motor[i], {0, 0, 0}, 1.0f, WHITE);
      // blades fade and a blur disc appears as the prop spins up; at full
      // rpm individual blades would only alias at the frame rate
      const float spin = fminf(1.0f, fabsf(out.motorRpm[i]) / 8000.0f);
      DrawModel(prop[i], {0, 0, 0}, 1.0f, Fade(WHITE, 1.0f - 0.75f * spin));
      if (spin > 0.05f)
        DrawModel(propBlur[i], {0, 0, 0}, 1.0f, Fade(WHITE, 0.35f * spin));
    }
    rlEnableBackfaceCulling();
    DrawCylinder({0, 0.85f, 0}, 0.25f, 0.2f, 1.7f, 12, DARKBLUE);
    EndMode3D();

    // CRRCSim model-view window: same eye, aimed straight at the aircraft,
    // FOV just fitting it, 15 % of the screen, translucent white backdrop.
    // (CRRCSim puts it top-left; top-right here keeps it off the HUD text.)
    const float dist = Vector3Length(Vector3Subtract(
        {in.position.x, in.position.y, in.position.z}, loscam.pilot));
    float insetFov = 0.0f;
    if (loscam.modelView(dist, 0.5f * wg.span, insetFov)) {
      const int iw = (int)(GetScreenWidth() * LosCamera::kModelViewSize);
      const int ih = (int)(GetScreenHeight() * LosCamera::kModelViewSize);
      if (pip.texture.width != iw || pip.texture.height != ih) {
        UnloadRenderTexture(pip);
        pip = LoadRenderTexture(iw, ih);
      }
      BeginTextureMode(pip);
      ClearBackground(Color{255, 255, 255, 128});
      Camera3D camPip = cam;
      camPip.target = {in.position.x, in.position.y, in.position.z};
      camPip.fovy = 2.0f * atanf(tanf(0.5f * insetFov * DEG2RAD)
                                 * (float)ih / (float)iw) * RAD2DEG;
      BeginMode3D(camPip);
      drawAircraft();
      EndMode3D();
      EndTextureMode();
      const float x = (float)(GetScreenWidth() - iw);
      DrawTextureRec(pip.texture, {0, 0, (float)iw, -(float)ih}, {x, 0.0f}, WHITE);
    }

    DrawFPS(10, 10);
    DrawText(TextFormat("preset: %s   (1=hand 2=hover 3=cruise, R reset)",
                        presetName[preset]),
             10, 34, 16, WHITE);
    DrawText(TextFormat("armed:%d  batt:%.2fV  alt:%.1f  spd:%.1f m/s",
                        (int)out.armed, out.batteryVoltage,
                        in.position.y,
                        Vector3Length({in.linearVelocity.x,
                                       in.linearVelocity.y,
                                       in.linearVelocity.z})),
             10, 54, 16, WHITE);
    DrawText(TextFormat("elevL %+5.1f deg  elevR %+5.1f deg",
                        out.elevonDeg[0], out.elevonDeg[1]),
             10, 74, 14, LIGHTGRAY);
    // what Betaflight receives, and why it won't arm (0 = armable)
    DrawText(TextFormat("ch1-8 %+.2f %+.2f %+.2f %+.2f | %+.2f %+.2f %+.2f %+.2f"
                        "   arming disabled 0x%08x",
                        in.rcData[0], in.rcData[1], in.rcData[2], in.rcData[3],
                        in.rcData[4], in.rcData[5], in.rcData[6], in.rcData[7],
                        (unsigned)out.armingDisableFlags),
             10, 92, 14, YELLOW);
    if (crashed)
      DrawText("CRASHED - press 1/2/3 or R to relaunch", 320, 200, 30, RED);
    else if (held)
      DrawText(out.armed ? "HELD - add throttle to launch"
                         : "HELD - arm, then add throttle to launch",
               320, 200, 26, YELLOW);
    if (showHelp) {
      int y = GetScreenHeight() - 90;
      DrawText("keys: 1/2/3 launch presets  R reset  H help  +/- zoom  F11 window", 10, y,
               14, WHITE); y += 18;
      DrawText(joy ? "joystick active"
                   : "keyboard: arrows pitch/roll, W/S thr, A/D yaw, SPACE ch5 switch",
               10, y, 14, WHITE);
    }
    EndDrawing();
  }

  simitl_stop();
  UnloadRenderTexture(pip);
  sky.unload();
  runway.unload();
  scenery.unload();
  terrain.unload();
  CloseWindow();
  if (joy) SDL_JoystickClose(joy);
  SDL_Quit();
  return 0;
}
