// Tailsitter LOS flight sim front end.
// raylib rendering + SDL2 joystick, driving the in-process SimITL library
// (Betaflight SITL + TailsitterPhysics) exactly like the headless runner.

#include <cstdio>
#include <cstring>
#include <string>
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

static void drawShadow(const Vec3F& pos, float span) {
  const float a = fminf(0.4f, 2.0f / fmaxf(pos.y, 0.3f));
  DrawCylinder({pos.x, 0.02f, pos.z}, span * 0.4f, span * 0.4f, 0.01f, 16,
               Fade(BLACK, a));
}

// -- main --------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string wingPath = "../as_built.json";
  std::string eeprom = "wing.bin";
  std::string calPath = "calib.json";
  int preset = P_HAND;
  float thrOverride = -999.0f;   // >=-1 forces throttle (testing without radio)
  bool autoArm = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto val = [&](const char* n) {
      if (i + 1 >= argc) { fprintf(stderr, "--%s needs a value\n", n); exit(2); }
      return std::string(argv[++i]);
    };
    if (a == "--wing") wingPath = val("wing");
    else if (a == "--eeprom") eeprom = val("eeprom");
    else if (a == "--cal") calPath = val("cal");
    else if (a == "--preset") preset = std::stoi(val("preset"));
    else if (a == "--thr") thrOverride = std::stof(val("thr"));
    else if (a == "--autoarm") autoArm = true;
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
    std::fill(si.eepromName, si.eepromName + 512, 0);
    memcpy(si.eepromName, eeprom.c_str(), std::min(eeprom.size(), (size_t)511));
  }
  WingGeom wg = loadWingGeom(wingPath);
  simitl_init(si);

  StateInput in{};
  in.delta = 0.001f;
  in.vbat = 4.2f;
  applyPreset(in, preset);

  // ---- input ------------------------------------------------------------------
  SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_GAMECONTROLLER);
  SDL_Joystick* joy = nullptr;
  if (SDL_NumJoysticks() > 0) {
    joy = SDL_JoystickOpen(0);
    printf("joystick: %s (%d axes)\n", SDL_JoystickName(joy),
           SDL_JoystickNumAxes(joy));
  } else {
    printf("no joystick - keyboard: arrows pitch/roll, W/S throttle, "
           "A/D yaw, space arm\n");
  }
  InputMap imap;
  if (!imap.load(calPath)) {
    // defaults: OpenTX order on the Pocket is Ail Ele Thr Rud on axes 0..3,
    // switches on 4+.  Throttle is unipolar: raw -1 = idle, +1 = full.
    imap.roll.axis = 0;  imap.roll.inv = 1.0f;
    imap.pitch.axis = 1; imap.pitch.inv = -1.0f;
    imap.throttle.axis = 2; imap.throttle.inv = 1.0f;
    imap.yaw.axis = 3;   imap.yaw.inv = 1.0f;
    imap.arm.axis = 4;
    imap.save(calPath);
  }

  // ---- window -----------------------------------------------------------------
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  InitWindow(1280, 720, "wingsim - tailsitter LOS");
  SetTargetFPS(120);

  Camera3D cam{};
  cam.up = {0, 1, 0};
  cam.projection = CAMERA_PERSPECTIVE;
  LosCamera loscam;

  // ---- aircraft model -----------------------------------------------------------
  Model wing;
  {
    const float semi = 0.5f * wg.span;
    wing.meshCount = 5;
    wing.meshes = (Mesh*)MemAlloc(5 * sizeof(Mesh));
    wing.meshes[0] = makeWingSection(wg, -wg.elevIn, wg.elevIn, 0.0f, 1.0f);
    wing.meshes[1] = makeWingSection(wg, wg.elevOut, semi, 0.0f, 1.0f);
    wing.meshes[2] = makeWingSection(wg, -semi, -wg.elevOut, 0.0f, 1.0f);
    wing.meshes[3] = makeWingSection(wg, wg.elevIn, wg.elevOut, 0.0f, wg.xHinge);
    wing.meshes[4] = makeWingSection(wg, -wg.elevOut, -wg.elevIn, 0.0f, wg.xHinge);
    wing.materialCount = 1;
    wing.materials = (Material*)MemAlloc(sizeof(Material));
    wing.materials[0] = LoadMaterialDefault();
    wing.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = RED;
    wing.meshMaterial = (int*)MemAlloc(5 * sizeof(int));
    for (int i = 0; i < 5; i++) wing.meshMaterial[i] = 0;
  }

  Model elev[2];
  elev[0] = LoadModelFromMesh(makeElevon(wg, -1, wg.hingePtL)); // left
  elev[1] = LoadModelFromMesh(makeElevon(wg, +1, wg.hingePtR)); // right
  elev[0].materials[0].maps[MATERIAL_MAP_DIFFUSE].color = ORANGE;
  elev[1].materials[0].maps[MATERIAL_MAP_DIFFUSE].color = ORANGE;

  Model prop[2];
  {
    const float r = 0.5f * wg.propDia;
    Mesh pm = GenMeshCylinder(r, 0.002f, 24);
    for (int i = 0; i < pm.vertexCount; i++) {   // axis y -> z (nose axis)
      const float y = pm.vertices[3*i + 1];
      pm.vertices[3*i + 1] = pm.vertices[3*i + 2];
      pm.vertices[3*i + 2] = -y;
    }
    prop[0] = LoadModelFromMesh(pm);
    prop[1] = LoadModelFromMesh(pm);
    prop[0].materials[0].maps[MATERIAL_MAP_DIFFUSE].color = Fade(DARKGRAY, 0.5f);
    prop[1].materials[0].maps[MATERIAL_MAP_DIFFUSE].color = Fade(DARKGRAY, 0.5f);
  }

  Model podL, podR;
  {
    podL = LoadModelFromMesh(GenMeshCylinder(0.006f, 0.02f, 12));
    podR = LoadModelFromMesh(GenMeshCylinder(0.006f, 0.02f, 12));
    podL.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = DARKGRAY;
    podR.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = DARKGRAY;
  }

  const Vector3 motPosR = aeroToSim(wg.cgX - wg.motorXAft, wg.motorY, wg.motorZ);
  const Vector3 motPosL = aeroToSim(wg.cgX - wg.motorXAft, -wg.motorY, wg.motorZ);

  RenderTexture2D pip = LoadRenderTexture(220, 160);

  bool crashed = false;
  bool showHelp = true;
  bool prevArmed = false;
  float propAngle[2] = {0, 0};
  float armBootDelay = 0.0f;

  while (!WindowShouldClose()) {
    const float dt = fminf(GetFrameTime(), 0.05f);
    armBootDelay += dt;

    // ---- input --------------------------------------------------------------
    SDL_JoystickUpdate();
    float rc[8] = {0};
    if (joy) imap.fillRc(joy, rc);
    keyboardRc(rc);
    // BF won't arm with throttle above idle: hold override low until armed
    if (thrOverride >= -1.0f) rc[2] = prevArmed ? thrOverride : -1.0f;
    if (autoArm) rc[4] = 1.0f;
    if (IsKeyPressed(KEY_ONE))   { applyPreset(in, P_HAND);   crashed = false; }
    if (IsKeyPressed(KEY_TWO))   { applyPreset(in, P_HOVER);  crashed = false; }
    if (IsKeyPressed(KEY_THREE)) { applyPreset(in, P_CRUISE); crashed = false; }
    if (IsKeyPressed(KEY_R))     { applyPreset(in, preset);   crashed = false; }
    if (IsKeyPressed(KEY_H))     showHelp = !showHelp;
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))      loscam.zoomIn();
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) loscam.zoomOut();
    const float wheel = GetMouseWheelMove();
    if (wheel > 0) loscam.zoomIn();
    if (wheel < 0) loscam.zoomOut();

    // keep arm low until BF boot settles (same grace window as headless runs)
    const float armSwitch = (armBootDelay < 5.5f) ? 0.0f : rc[4];

    // ---- physics substeps ------------------------------------------------------
    const StateOutput out = simitl_get_state();
    if (!crashed) {
      float acc = dt;
      const float h = in.delta;          // 1 ms
      while (acc >= h) {
        in.rcData[0] = rc[0];
        in.rcData[1] = rc[1];
        in.rcData[2] = rc[2];
        in.rcData[3] = rc[3];
        in.rcData[4] = armSwitch;
        simitl_update(in);
        const StateOutput o = simitl_get_state();

        const quat q{o.orientation.x, o.orientation.y,
                     o.orientation.z, o.orientation.w};
        const mat3 rot = quat_to_mat3(q);
        copy(in.rotation, rot);
        in.angularVelocity = o.angularVelocity;
        in.linearVelocity  = o.linearVelocity;
        in.position.x += o.linearVelocity.x * h;
        in.position.y += o.linearVelocity.y * h;
        in.position.z += o.linearVelocity.z * h;
        acc -= h;
      }
      for (int i = 0; i < 2; i++) {
        const int dir = (i == 0) ? 1 : -1;  // right +1 / left -1 (counter-rot.)
        propAngle[i] += out.motorRpm[i] * 2.0f * (float)M_PI / 60.0f * dt * dir;
      }
      if (in.position.y < 0.05f) crashed = true;
      prevArmed = out.armed != 0;
    }
    static float dbgT = 0.0f;
    dbgT += dt;
    if (dbgT > 1.0f) {
      dbgT = 0.0f;
      fprintf(stderr, "pos %.2f %.2f %.2f  vel %.2f %.2f %.2f  q %.2f %.2f %.2f %.2f  armed %d batt %.2f\n",
              in.position.x, in.position.y, in.position.z,
              out.linearVelocity.x, out.linearVelocity.y, out.linearVelocity.z,
              out.orientation.x, out.orientation.y, out.orientation.z,
              out.orientation.w, (int)out.armed, out.batteryVoltage);
    }

    // ---- model transforms --------------------------------------------------------
    const quat q{out.orientation.x, out.orientation.y,
                 out.orientation.z, out.orientation.w};
    const mat3 rotM = quat_to_mat3(q);
    const Matrix bodyM = {
      rotM[0][0], rotM[1][0], rotM[2][0], in.position.x,
      rotM[0][1], rotM[1][1], rotM[2][1], in.position.y,
      rotM[0][2], rotM[1][2], rotM[2][2], in.position.z,
      0, 0, 0, 1
    };
    wing.transform = bodyM;

    auto hingeM = [&](Vector3 pt, Vector3 ax, float deg) {
      const Matrix r = MatrixRotate(Vector3Normalize(ax),
                                    deg * (float)M_PI / 180.0f);
      return MatrixMultiply(
          MatrixMultiply(bodyM, MatrixTranslate(pt.x, pt.y, pt.z)),
          MatrixMultiply(r, MatrixTranslate(-pt.x, -pt.y, -pt.z)));
    };
    elev[0].transform = hingeM(wg.hingePtL, wg.hingeAxL, out.elevonDeg[0]);
    elev[1].transform = hingeM(wg.hingePtR, wg.hingeAxR, out.elevonDeg[1]);

    prop[0].transform = MatrixMultiply(bodyM,
        MatrixMultiply(MatrixTranslate(motPosR.x, motPosR.y, motPosR.z),
                       MatrixRotateZ(propAngle[0])));
    prop[1].transform = MatrixMultiply(bodyM,
        MatrixMultiply(MatrixTranslate(motPosL.x, motPosL.y, motPosL.z),
                       MatrixRotateZ(propAngle[1])));
    podR.transform = prop[0].transform;
    podL.transform = prop[1].transform;

    loscam.update(cam, {in.position.x, in.position.y, in.position.z}, dt);

    // ---- draw ----------------------------------------------------------------------
    BeginDrawing();
    ClearBackground(Color{135, 175, 210, 255});

    BeginMode3D(cam);
    drawGround();
    drawShadow(in.position, wg.span);
    DrawModel(wing, {0, 0, 0}, 1.0f, WHITE);
    DrawModel(elev[0], {0, 0, 0}, 1.0f, WHITE);
    DrawModel(elev[1], {0, 0, 0}, 1.0f, WHITE);
    DrawModel(prop[0], {0, 0, 0}, 1.0f, WHITE);
    DrawModel(prop[1], {0, 0, 0}, 1.0f, WHITE);
    DrawModel(podL, {0, 0, 0}, 1.0f, WHITE);
    DrawModel(podR, {0, 0, 0}, 1.0f, WHITE);
    DrawCylinder({0, 0.85f, 0}, 0.25f, 0.2f, 1.7f, 12, DARKBLUE);
    EndMode3D();

    const float dist = Vector3Length(Vector3Subtract(
        {in.position.x, in.position.y, in.position.z}, loscam.pilot));
    if (dist > 12.0f) {
      BeginTextureMode(pip);
      ClearBackground(Color{135, 175, 210, 255});
      Camera3D camPip = cam;
      camPip.fovy = fminf(4.0f, cam.fovy);
      camPip.target = {in.position.x, in.position.y, in.position.z};
      BeginMode3D(camPip);
      drawGround();
      DrawModel(wing, {0, 0, 0}, 1.0f, WHITE);
      DrawModel(elev[0], {0, 0, 0}, 1.0f, WHITE);
      DrawModel(elev[1], {0, 0, 0}, 1.0f, WHITE);
      EndMode3D();
      EndTextureMode();
      DrawTextureRec(pip.texture,
                     {0, 0, (float)pip.texture.width, -(float)pip.texture.height},
                     {10, GetScreenHeight() - 170.0f}, WHITE);
      DrawRectangleLines(10, GetScreenHeight() - 170, 220, 160, DARKGRAY);
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
    if (crashed)
      DrawText("CRASHED - press 1/2/3 to relaunch", 320, 200, 30, RED);
    if (armBootDelay < 5.5f)
      DrawText("FC booting...", 320, 240, 24, YELLOW);
    if (showHelp) {
      int y = GetScreenHeight() - 90;
      DrawText("keys: 1/2/3 launch presets  R reset  H help  +/- zoom", 10, y,
               14, WHITE); y += 18;
      DrawText(joy ? "joystick active"
                   : "keyboard: arrows pitch/roll, W/S thr, A/D yaw, SPACE arm",
               10, y, 14, WHITE);
    }
    EndDrawing();
  }

  simitl_stop();
  UnloadRenderTexture(pip);
  CloseWindow();
  if (joy) SDL_JoystickClose(joy);
  SDL_Quit();
  return 0;
}
