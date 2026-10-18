#pragma once
// Radio input -> rcData[8] in -1..1, passed to Betaflight untouched.
//
// A radio in USB-joystick mode (EdgeTX/OpenTX) reports its output channels
// 1..8 as joystick axes 0..7, i.e. the same values its receiver would deliver.
// So the sim treats the joystick as a receiver: channel i comes from axis i,
// and Betaflight's own `map` and `aux` settings (loaded from the aircraft's
// CLI dump) decide what each channel means, including arming.  Nothing here
// knows which channel is throttle or arm.
//
// calib.json can remap/invert/re-range a channel if the radio's sim model
// differs from the aircraft's; by default it is identity.

#include <SDL2/SDL.h>
#include <raylib.h>
#include <string>
#include <cmath>
#include <cstdio>
#include "json.h"

constexpr int RC_CHANNELS = 8;

struct ChannelCal {
  int   axis = -1;    // SDL axis index, -1 = unbound (reads centre)
  float inv = 1.0f;   // -1 to invert
  float lo = -1.0f, hi = 1.0f, center = 0.0f;  // raw axis extents
};

struct InputMap {
  ChannelCal ch[RC_CHANNELS];

  InputMap() {
    for (int i = 0; i < RC_CHANNELS; i++) ch[i].axis = i;
  }

  static float read(SDL_Joystick* j, const ChannelCal& c) {
    if (!j || c.axis < 0 || c.axis >= SDL_JoystickNumAxes(j)) return 0.0f;
    const float raw = SDL_JoystickGetAxis(j, c.axis) / 32767.0f;
    const float v = raw < c.center
        ? (raw - c.center) / fmaxf(c.center - c.lo, 1e-3f)
        : (raw - c.center) / fmaxf(c.hi - c.center, 1e-3f);
    return fminf(fmaxf(v * c.inv, -1.0f), 1.0f);
  }

  void fillRc(SDL_Joystick* j, float rc[RC_CHANNELS]) const {
    for (int i = 0; i < RC_CHANNELS; i++) rc[i] = read(j, ch[i]);
  }

  // calib.json: {"ch1": {"axis": 0, "inv": 1, "lo": -1, "hi": 1,
  // "center": 0}, ...}; missing channels keep the identity default.
  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    fclose(f);
    auto j = json::parseFile(path);
    if (j.type != json::JsonObject::Object) return false;
    for (int i = 0; i < RC_CHANNELS; i++) {
      const std::string key = "ch" + std::to_string(i + 1);
      if (!j.has(key.c_str())) continue;
      auto v = j.get(key.c_str());
      ch[i].axis   = v.get("axis").asInt(ch[i].axis);
      ch[i].inv    = (float)v.get("inv").asNumber(1.0);
      ch[i].lo     = (float)v.get("lo").asNumber(-1.0);
      ch[i].hi     = (float)v.get("hi").asNumber(1.0);
      ch[i].center = (float)v.get("center").asNumber(0.0);
    }
    return true;
  }
};

// No radio: the keyboard stands in for one, in the aircraft's AETR order with
// channel 5 as a two-position switch toggled by space.  Throttle is a held
// slider, like a real throttle stick.
struct KeyboardRadio {
  float throttle = -1.0f;
  bool  sw5 = false;

  void fillRc(float rc[RC_CHANNELS], float dt) {
    for (int i = 0; i < RC_CHANNELS; i++) rc[i] = 0.0f;
    if (IsKeyDown(KEY_LEFT))  rc[0] -= 1.0f;
    if (IsKeyDown(KEY_RIGHT)) rc[0] += 1.0f;
    if (IsKeyDown(KEY_UP))    rc[1] += 1.0f;   // stick forward
    if (IsKeyDown(KEY_DOWN))  rc[1] -= 1.0f;   // stick back
    if (IsKeyDown(KEY_W)) throttle += 1.0f * dt;
    if (IsKeyDown(KEY_S)) throttle -= 1.0f * dt;
    throttle = fminf(fmaxf(throttle, -1.0f), 1.0f);
    rc[2] = throttle;
    if (IsKeyDown(KEY_A)) rc[3] -= 1.0f;
    if (IsKeyDown(KEY_D)) rc[3] += 1.0f;
    if (IsKeyPressed(KEY_SPACE)) sw5 = !sw5;
    rc[4] = sw5 ? 1.0f : -1.0f;
  }
};
