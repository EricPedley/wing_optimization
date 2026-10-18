#pragma once
// Low-poly ground around the pilot, from sim/assets/terrain/local.bin
// (written by sim/terrain.py from USGS 3DEP elevation).
//
// Sim axes: +x = launch heading, +z = pilot's right, +y up; heights are
// metres above the pilot's ground.  Flat-shaded triangles with colours baked
// per face (slope, distance, height, sun), so it reads as low-poly.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>
#include "json.h"

struct Terrain {
  bool loaded = false;
  int nx = 0, nz = 0;
  float x0 = 0, z0 = 0, step = 1;
  std::vector<float> h;   // nz rows of nx
  Model model{};

  float at(int ix, int iz) const { return h[(size_t)iz * nx + ix]; }

  // bilinear height (m above the pilot's ground); 0 outside the grid
  float heightAt(float x, float z) const {
    if (!loaded) return 0.0f;
    const float fx = (x - x0) / step, fz = (z - z0) / step;
    if (fx < 0 || fz < 0 || fx >= nx - 1 || fz >= nz - 1) return 0.0f;
    const int ix = (int)fx, iz = (int)fz;
    const float tx = fx - ix, tz = fz - iz;
    return (at(ix, iz) * (1 - tx) + at(ix + 1, iz) * tx) * (1 - tz)
         + (at(ix, iz + 1) * (1 - tx) + at(ix + 1, iz + 1) * tx) * tz;
  }

  // sunBearingDeg/sunElevDeg are true; launchHeadingDeg places them in sim axes
  bool load(const std::string& path, float launchHeadingDeg,
            float sunBearingDeg = 225.0f, float sunElevDeg = 45.0f) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t dims[2];
    float geo[3];
    bool ok = fread(dims, 4, 2, f) == 2 && fread(geo, 4, 3, f) == 3;
    if (ok) {
      nx = dims[0]; nz = dims[1]; x0 = geo[0]; z0 = geo[1]; step = geo[2];
      h.resize((size_t)nx * nz);
      ok = fread(h.data(), 4, h.size(), f) == h.size();
    }
    fclose(f);
    if (!ok || nx < 2 || nz < 2) return false;

    const float rel = (sunBearingDeg - launchHeadingDeg) * DEG2RAD;
    const float el = sunElevDeg * DEG2RAD;
    const Vector3 sun{cosf(el) * cosf(rel), sinf(el), cosf(el) * sinf(rel)};

    const int tris = (nx - 1) * (nz - 1) * 2;
    Mesh m{};
    m.vertexCount = tris * 3;
    m.triangleCount = tris;
    m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
    m.normals = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
    m.colors = (unsigned char*)MemAlloc(m.vertexCount * 4);
    int v = 0;
    auto P = [&](int ix, int iz) {
      return Vector3{x0 + ix * step, at(ix, iz), z0 + iz * step};
    };
    auto face = [&](Vector3 a, Vector3 b, Vector3 c) {
      Vector3 n = Vector3Normalize(Vector3CrossProduct(Vector3Subtract(b, a),
                                                       Vector3Subtract(c, a)));
      if (n.y < 0) { n = Vector3Negate(n); std::swap(b, c); }
      const Vector3 mid = Vector3Scale(Vector3Add(Vector3Add(a, b), c), 1.0f / 3.0f);
      const float slope = acosf(fminf(n.y, 1.0f)) * RAD2DEG;
      const float dist = sqrtf(mid.x * mid.x + mid.z * mid.z);
      // same palette as sim/terrain.py: turf near the pilot, dry grass on the
      // flats, chaparral on slopes, browner up high
      const float flat = fmaxf(0.0f, fminf(1.0f, 1.0f - slope / 12.0f));
      const float near = fmaxf(0.0f, fminf(1.0f, 1.0f - dist / 400.0f));
      const float high = fmaxf(0.0f, fminf(0.6f, (mid.y - 80.0f) / 300.0f));
      float col[3];
      const float turf[3] = {96, 128, 62}, grass[3] = {150, 150, 95},
                  scrub[3] = {88, 96, 62}, dirt[3] = {140, 118, 90};
      const float shade = 0.45f + 0.65f * fmaxf(0.0f, Vector3DotProduct(n, sun));
      for (int k = 0; k < 3; k++) {
        const float flatC = near * turf[k] + (1 - near) * grass[k];
        const float base = flat * flatC + (1 - flat) * scrub[k];
        col[k] = fminf(255.0f, (base * (1 - high) + dirt[k] * high) * shade);
      }
      const Vector3 p[3] = {a, b, c};
      for (int k = 0; k < 3; k++, v++) {
        m.vertices[3 * v] = p[k].x; m.vertices[3 * v + 1] = p[k].y;
        m.vertices[3 * v + 2] = p[k].z;
        m.normals[3 * v] = n.x; m.normals[3 * v + 1] = n.y; m.normals[3 * v + 2] = n.z;
        m.colors[4 * v] = (unsigned char)col[0];
        m.colors[4 * v + 1] = (unsigned char)col[1];
        m.colors[4 * v + 2] = (unsigned char)col[2];
        m.colors[4 * v + 3] = 255;
      }
    };
    for (int iz = 0; iz < nz - 1; iz++)
      for (int ix = 0; ix < nx - 1; ix++) {
        face(P(ix, iz), P(ix + 1, iz + 1), P(ix + 1, iz));
        face(P(ix, iz), P(ix, iz + 1), P(ix + 1, iz + 1));
      }
    UploadMesh(&m, false);
    model = LoadModelFromMesh(m);
    loaded = true;
    return true;
  }

  void draw() const {
    if (loaded) DrawModel(model, {0, 0, 0}, 1.0f, WHITE);
  }

  void unload() {
    if (loaded) UnloadModel(model);
    loaded = false;
  }
};

// launch heading from sim/assets/terrain/site.json (0 if absent)
inline float siteLaunchHeading(const std::string& path) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return 0.0f;
  fclose(f);
  auto j = json::parseFile(path);
  return (float)j.get("launchHeadingDeg").asNumber(0.0);
}

// Dirt runway from sim/assets/terrain/features.json, draped on the terrain.
struct Runway {
  bool loaded = false;
  Model model{};

  bool load(const std::string& featuresPath, const Terrain& terrain) {
    FILE* f = fopen(featuresPath.c_str(), "r");
    if (!f) return false;
    fclose(f);
    auto j = json::parseFile(featuresPath);
    if (!j.has("runway")) return false;
    auto r = j.get("runway");
    const float cx = (float)r.get("x").asNumber(0.0);
    const float cz = (float)r.get("z").asNumber(0.0);
    const float hdg = (float)r.get("headingInSimDeg").asNumber(0.0) * DEG2RAD;
    const float len = (float)r.get("length").asNumber(30.0);
    const float wid = (float)r.get("width").asNumber(6.0);
    // bearing is clockwise from +x toward +z
    const Vector3 along{cosf(hdg), 0.0f, sinf(hdg)};
    const Vector3 across{-along.z, 0.0f, along.x};
    const int nl = (int)ceilf(len / 2.0f), nw = 2;
    const float lift = 0.15f;   // above the bilinear surface: no z-fighting

    Mesh m{};
    m.triangleCount = nl * nw * 2;
    m.vertexCount = m.triangleCount * 3;
    m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
    m.colors = (unsigned char*)MemAlloc(m.vertexCount * 4);
    auto P = [&](int il, int iw) {
      const float a = -0.5f * len + len * il / nl;
      const float b = -0.5f * wid + wid * iw / nw;
      const float x = cx + along.x * a + across.x * b;
      const float z = cz + along.z * a + across.z * b;
      return Vector3{x, terrain.heightAt(x, z) + lift, z};
    };
    int v = 0;
    auto tri = [&](Vector3 p0, Vector3 p1, Vector3 p2, unsigned char shade) {
      const Vector3 p[3] = {p0, p1, p2};
      for (int k = 0; k < 3; k++, v++) {
        m.vertices[3 * v] = p[k].x; m.vertices[3 * v + 1] = p[k].y;
        m.vertices[3 * v + 2] = p[k].z;
        m.colors[4 * v] = (unsigned char)(150 * shade / 255);
        m.colors[4 * v + 1] = (unsigned char)(118 * shade / 255);
        m.colors[4 * v + 2] = (unsigned char)(84 * shade / 255);
        m.colors[4 * v + 3] = 255;
      }
    };
    for (int il = 0; il < nl; il++)
      for (int iw = 0; iw < nw; iw++) {
        // slight checker so the strip reads as a surface, not a flat decal
        const unsigned char shade = ((il + iw) % 2) ? 235 : 255;
        tri(P(il, iw), P(il + 1, iw + 1), P(il + 1, iw), shade);
        tri(P(il, iw), P(il, iw + 1), P(il + 1, iw + 1), shade);
      }
    UploadMesh(&m, false);
    model = LoadModelFromMesh(m);
    loaded = true;
    return true;
  }

  void draw() const {
    if (!loaded) return;
    rlDisableBackfaceCulling();   // winding depends on the heading
    DrawModel(model, {0, 0, 0}, 1.0f, WHITE);
    rlEnableBackfaceCulling();
  }

  void unload() {
    if (loaded) UnloadModel(model);
    loaded = false;
  }
};
