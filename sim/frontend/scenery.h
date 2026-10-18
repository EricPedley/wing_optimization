#pragma once
// Baked OpenStreetMap scenery (houses, transmission towers and wires, light
// poles) from sim/assets/terrain/scenery.bin, written by sim/scenery.py.
// Triangles become one mesh; line segments (lattice steel, conductors) are
// drawn as GL lines, which stay one pixel wide at any distance -- like real
// wires seen from the ground.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <raylib.h>
#include <rlgl.h>

struct Scenery {
  struct Vert { float x, y, z; uint8_t r, g, b, a; };
  static_assert(sizeof(Vert) == 16, "packed as in scenery.bin");

  bool loaded = false;
  Model model{};
  std::vector<Vert> lines;

  bool load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    int32_t n[2];
    std::vector<Vert> tris;
    bool ok = fread(n, 4, 2, f) == 2 && n[0] >= 0 && n[1] >= 0;
    if (ok) {
      tris.resize(n[0]);
      lines.resize(n[1]);
      ok = fread(tris.data(), sizeof(Vert), tris.size(), f) == tris.size()
        && fread(lines.data(), sizeof(Vert), lines.size(), f) == lines.size();
    }
    fclose(f);
    if (!ok) return false;
    if (!tris.empty()) {
      Mesh m{};
      m.vertexCount = (int)tris.size();
      m.triangleCount = m.vertexCount / 3;
      m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
      m.colors = (unsigned char*)MemAlloc(m.vertexCount * 4);
      for (int i = 0; i < m.vertexCount; i++) {
        m.vertices[3 * i] = tris[i].x;
        m.vertices[3 * i + 1] = tris[i].y;
        m.vertices[3 * i + 2] = tris[i].z;
        m.colors[4 * i] = tris[i].r; m.colors[4 * i + 1] = tris[i].g;
        m.colors[4 * i + 2] = tris[i].b; m.colors[4 * i + 3] = tris[i].a;
      }
      UploadMesh(&m, false);
      model = LoadModelFromMesh(m);
    }
    loaded = true;
    return true;
  }

  void draw() const {
    if (!loaded) return;
    if (model.meshCount > 0) {
      rlDisableBackfaceCulling();   // footprint winding varies
      DrawModel(model, {0, 0, 0}, 1.0f, WHITE);
      rlEnableBackfaceCulling();
    }
    rlBegin(RL_LINES);
    for (const Vert& v : lines) {
      rlColor4ub(v.r, v.g, v.b, v.a);
      rlVertex3f(v.x, v.y, v.z);
    }
    rlEnd();
  }

  void unload() {
    if (loaded && model.meshCount > 0) UnloadModel(model);
    loaded = false;
  }
};
