#pragma once
// SDL2 joystick input + per-channel calibration, keyboard fallback.
// Produces rcData[8] in -1..1 in AETR order: 0=roll 1=pitch 2=thr 3=yaw
// (matches the real aircraft's channel map), 4=arm, 5..7 aux.

#include <SDL2/SDL.h>
#include <raylib.h>
#include <string>
#include <cmath>
#include "json.h"

struct StickCal {
  int   axis = -1;    // SDL axis index, -1 = unbound
  float inv = 1.0f;   // -1 to invert
  float lo = -1.0f, hi = 1.0f, center = 0.0f;  // raw axis extents
};

struct InputMap {
  StickCal roll, pitch, throttle, yaw, arm;

  float read(SDL_Joystick* j, const StickCal& c) const {
    if (!j || c.axis < 0) return 0.0f;
    const float raw = SDL_JoystickGetAxis(j, c.axis) / 32767.0f;
    const float lo = fminf(c.lo, c.center), hi = fmaxf(c.hi, c.center);
    float v;
    if (raw < c.center)
      v = (raw - c.center) / fmaxf(c.center - lo, 1e-3f);
    else
      v = (raw - c.center) / fmaxf(hi - c.center, 1e-3f);
    return fminf(fmaxf(v * c.inv, -1.0f), 1.0f);
  }

  void fillRc(SDL_Joystick* j, float rc[8]) const {
    rc[0] = read(j, roll);
    rc[1] = read(j, pitch);
    rc[2] = read(j, throttle);
    rc[3] = read(j, yaw);
    rc[4] = read(j, arm);
    for (int i = 5; i < 8; i++) rc[i] = 0.0f;
  }

  static void dumpLive(SDL_Joystick* j) {
    if (!j) return;
    const int n = SDL_JoystickNumAxes(j);
    printf("axes:");
    for (int i = 0; i < n; i++)
      printf(" [%d]=%.2f", i, SDL_JoystickGetAxis(j, i) / 32767.0f);
    printf("\r");
    fflush(stdout);
  }

  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    fclose(f);
    auto j = json::parseFile(path);
    if (j.type != json::JsonObject::Object) return false;
    auto cal = [&](const char* name, StickCal& c) {
      if (!j.has(name)) return;
      auto v = j.get(name);
      c.axis = v.get("axis").asInt(-1);
      c.inv = (float)v.get("inv").asNumber(1.0);
      c.lo = (float)v.get("lo").asNumber(-1.0);
      c.hi = (float)v.get("hi").asNumber(1.0);
      c.center = (float)v.get("center").asNumber(0.0);
    };
    cal("roll", roll); cal("pitch", pitch); cal("throttle", throttle);
    cal("yaw", yaw);   cal("arm", arm);
    return true;
  }

  bool save(const std::string& path) const {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return false;
    auto w = [&](const char* name, const StickCal& c) {
      fprintf(f, "  \"%s\": {\"axis\": %d, \"inv\": %.1f, \"lo\": %.3f, "
                 "\"hi\": %.3f, \"center\": %.3f},\n",
              name, c.axis, c.inv, c.lo, c.hi, c.center);
    };
    fprintf(f, "{\n");
    w("roll", roll); w("pitch", pitch); w("throttle", throttle);
    w("yaw", yaw); w("arm", arm);
    fprintf(f, "  \"_note\": \"axis=SDL axis index, inv=-1 flips\"\n}\n");
    fclose(f);
    return true;
  }
};

// Guided wiggle calibration run inside the window: for `dur` seconds the user
// sweeps the stick; min/max/center per axis get recorded onto `cal.axis`.
struct CalibrateRun {
  float timeLeft = 0.0f;
  int   targetAxis = -1;          // axis under calibration
  StickCal* target = nullptr;
  float lo = 1e9f, hi = -1e9f;
  double sum = 0.0; int n = 0;

  void start(StickCal& c, SDL_Joystick* j) {
    target = &c; targetAxis = -1; timeLeft = 3.0f;
    lo = 1e9f; hi = -1e9f; sum = 0.0; n = 0;
    (void)j;
  }
  // returns true while calibrating; polls joystick
  bool step(SDL_Joystick* j, float dt) {
    if (!target || !j) return false;
    timeLeft -= dt;
    // find the axis with the largest excursion during the window
    const int nax = SDL_JoystickNumAxes(j);
    for (int i = 0; i < nax; i++) {
      const float a = SDL_JoystickGetAxis(j, i) / 32767.0f;
      if (targetAxis < 0 && fabsf(a) > 0.4f) targetAxis = i;
      if (i == targetAxis) {
        lo = fminf(lo, a); hi = fmaxf(hi, a); sum += a; n++;
      }
    }
    if (timeLeft <= 0.0f) {
      if (targetAxis >= 0 && hi - lo > 0.3f) {
        target->axis = targetAxis;
        target->lo = lo; target->hi = hi;
        target->center = (float)(sum / fmax(n, 1));
      }
      target = nullptr;
      return false;
    }
    return true;
  }
};

// keyboard fallback mapping
inline void keyboardRc(float rc[8]) {
  if (IsKeyDown(KEY_LEFT))  rc[0] -= 1.0f;
  if (IsKeyDown(KEY_RIGHT)) rc[0] += 1.0f;
  if (IsKeyDown(KEY_DOWN))  rc[1] += 1.0f;   // pitch back = up
  if (IsKeyDown(KEY_UP))    rc[1] -= 1.0f;
  if (IsKeyDown(KEY_W))     rc[2] += 0.02f;
  if (IsKeyDown(KEY_S))     rc[2] -= 0.02f;
  rc[2] = fminf(fmaxf(rc[2], -1.0f), 1.0f);
  if (IsKeyDown(KEY_A))     rc[3] -= 1.0f;
  if (IsKeyDown(KEY_D))     rc[3] += 1.0f;
  if (IsKeyPressed(KEY_SPACE)) rc[4] = (rc[4] > 0.5f) ? 0.0f : 1.0f;
}
