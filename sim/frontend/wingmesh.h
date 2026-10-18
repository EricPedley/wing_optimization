#pragma once
// Aircraft geometry for the front end: the Onshape CAD meshes exported by
// `sim/as_built.py --fetch` (sim/assets/*.stl) and the as-built hinge/motor
// placement from as_built.json.
//
// Frames
//   CAD  : Onshape part studio, m, x fwd, y left, z up, origin root LE.
//          Only the right half (y < 0) exists; the left half is its mirror.
//   aero : x fwd from CG, y right, z down        = (cgX + x, -y, -z)
//   sim  : x port, y dorsal, z nose (TailsitterPhysics body frame)
//          = aeroToSim(aero) = (y_cad, z_cad, cgX + x_cad)
//
// raylib's MatrixMultiply(a, b) applies a FIRST, then b.  Everything that
// hangs off the body is composed with bodyChild() so the order is in one place.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <raylib.h>
#include <raymath.h>
#include "network/packets.h"
#include "json.h"

struct WingGeom {
  float span, cgX;
  Vector3 hingePtR, hingeAxR, hingePtL, hingeAxL;  // sim body frame
  float motorY, motorXAft, motorZ, propDia;
};

inline Vector3 aeroToSim(float xa, float ya, float za) {
  return Vector3{-ya, -za, xa};
}
inline Vector3 cadToSim(float x, float y, float z, float cgX) {
  return Vector3{y, z, cgX + x};
}
inline Vector3 jvec3(const json::JsonObject& v) {
  const Vec3F a = v.asVec3();
  return Vector3{a.x, a.y, a.z};
}

inline WingGeom loadWingGeom(const std::string& path) {
  auto j = json::parseFile(path);
  WingGeom g{};
  g.span      = (float)j.get("span").asNumber(0.33);
  g.cgX       = (float)j.get("cgX").asNumber(0.08);
  auto h = j.get("hinge");
  auto conv = [](const json::JsonObject& p) {
    const Vector3 a = jvec3(p);
    return aeroToSim(a.x, a.y, a.z);
  };
  g.hingePtR = conv(h.get("pointR")); g.hingeAxR = conv(h.get("axisR"));
  g.hingePtL = conv(h.get("pointL")); g.hingeAxL = conv(h.get("axisL"));
  g.motorY    = (float)j.get("motorY").asNumber(0.06);
  g.motorXAft = (float)j.get("motorXAft").asNumber(0.0);
  g.motorZ    = (float)j.get("motorZ").asNumber(0.0);
  g.propDia   = (float)j.get("propDiameter").asNumber(0.05);
  return g;
}

// --- transforms ------------------------------------------------------------------

// Body pose -> raylib matrix.  StateInput.rotation[i] is ROW i of the
// body-to-world rotation (SimITL's mat3 convention: quat_to_mat3 and xform
// work on rows), so body axis c in world is the column (rot[0][c], rot[1][c],
// rot[2][c]).  The raylib initialiser below is also written row by row.
inline Matrix bodyMatrix(const Vec3F rot[3], Vec3F pos) {
  return Matrix{
    rot[0].x, rot[0].y, rot[0].z, pos.x,
    rot[1].x, rot[1].y, rot[1].z, pos.y,
    rot[2].x, rot[2].y, rot[2].z, pos.z,
    0, 0, 0, 1
  };
}

// A part whose mesh is built around its own pivot: apply `local` about the
// pivot, move the pivot to `at` (body frame), then apply the body pose.
inline Matrix bodyChild(const Matrix& body, Vector3 at, const Matrix& local) {
  return MatrixMultiply(MatrixMultiply(local, MatrixTranslate(at.x, at.y, at.z)),
                        body);
}

// Elevon about its hinge.  Both hinge axes point to the aircraft's right, so
// +deg (right-hand rule) is trailing edge down on both sides -- the physics'
// sign convention.
inline Matrix elevonTransform(const Matrix& body, Vector3 hingePt,
                              Vector3 hingeAx, float deg) {
  return bodyChild(body, hingePt,
                   MatrixRotate(Vector3Normalize(hingeAx), deg * DEG2RAD));
}

// --- STL meshes -------------------------------------------------------------------

// Binary STL -> triangle soup (3 vertices per triangle), CAD frame.
inline bool readStl(const std::string& path, std::vector<Vector3>& tris) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char header[80];
  uint32_t n = 0;
  bool ok = fread(header, 1, 80, f) == 80 && fread(&n, 4, 1, f) == 1;
  tris.clear();
  tris.reserve(3 * (size_t)n);
  for (uint32_t i = 0; ok && i < n; i++) {
    float rec[12];
    uint16_t attr;
    ok = fread(rec, 4, 12, f) == 12 && fread(&attr, 2, 1, f) == 1;
    for (int k = 0; ok && k < 3; k++)
      tris.push_back(Vector3{rec[3 + 3*k], rec[4 + 3*k], rec[5 + 3*k]});
  }
  fclose(f);
  return ok;
}

// CAD triangles -> sim-frame raylib mesh, optionally mirrored to the left
// half, re-centred on `pivot` (sim frame).  Faces are flat-shaded into vertex
// colours with a fixed body-frame light, and the top (dorsal) and bottom get
// different colours so attitude reads at LOS distances.
inline Mesh buildMesh(const std::vector<Vector3>& cadTris, float cgX,
                      bool mirrorLeft, Vector3 pivot, Color top, Color bottom) {
  const int nv = (int)cadTris.size();
  Mesh m{};
  m.vertexCount = nv;
  m.triangleCount = nv / 3;
  m.vertices = (float*)MemAlloc(nv * 3 * sizeof(float));
  m.normals  = (float*)MemAlloc(nv * 3 * sizeof(float));
  m.colors   = (unsigned char*)MemAlloc(nv * 4);
  const Vector3 light = Vector3Normalize({0.3f, 1.0f, 0.5f});
  for (int t = 0; t < nv / 3; t++) {
    Vector3 p[3];
    for (int k = 0; k < 3; k++) {
      const Vector3 c = cadTris[3*t + k];
      p[k] = cadToSim(c.x, mirrorLeft ? -c.y : c.y, c.z, cgX);
    }
    // mirroring flips handedness: swap two vertices to keep outward winding
    if (mirrorLeft) { const Vector3 s = p[1]; p[1] = p[2]; p[2] = s; }
    const Vector3 nrm = Vector3Normalize(Vector3CrossProduct(
        Vector3Subtract(p[1], p[0]), Vector3Subtract(p[2], p[0])));
    const Color base = nrm.y >= 0.0f ? top : bottom;
    const float shade = 0.55f + 0.45f * fmaxf(0.0f, Vector3DotProduct(nrm, light));
    for (int k = 0; k < 3; k++) {
      const int i = 3*t + k;
      const Vector3 v = Vector3Subtract(p[k], pivot);
      m.vertices[3*i] = v.x; m.vertices[3*i+1] = v.y; m.vertices[3*i+2] = v.z;
      m.normals[3*i] = nrm.x; m.normals[3*i+1] = nrm.y; m.normals[3*i+2] = nrm.z;
      m.colors[4*i]   = (unsigned char)(base.r * shade);
      m.colors[4*i+1] = (unsigned char)(base.g * shade);
      m.colors[4*i+2] = (unsigned char)(base.b * shade);
      m.colors[4*i+3] = 255;
    }
  }
  UploadMesh(&m, false);
  return m;
}

// The whole airframe: body (both halves) plus one model per elevon, each
// elevon mesh centred on its hinge point.
struct AircraftModels {
  Model body;
  Model elevon[2];   // 0 = left, 1 = right (StateOutput.elevonDeg order)
};

inline bool loadAircraft(const std::string& assetDir, const WingGeom& g,
                         AircraftModels& out) {
  std::vector<Vector3> bodyTris, elevTris;
  if (!readStl(assetDir + "/body_right.stl", bodyTris) ||
      !readStl(assetDir + "/elevon_right.stl", elevTris)) {
    fprintf(stderr, "missing CAD meshes in %s "
                    "(run: uv run python -m sim.as_built --fetch)\n",
            assetDir.c_str());
    return false;
  }
  // all white, like the printed airframe; the baked sun shading gives form
  const Color top{238, 238, 234, 255}, bottom{232, 232, 228, 255};
  const Color elevTop = top, elevBottom = bottom;

  out.body = LoadModelFromMesh(buildMesh(bodyTris, g.cgX, false, {0, 0, 0},
                                         top, bottom));
  // second body half as an extra mesh on the same model
  Mesh left = buildMesh(bodyTris, g.cgX, true, {0, 0, 0}, top, bottom);
  out.body.meshCount = 2;
  out.body.meshes = (Mesh*)MemRealloc(out.body.meshes, 2 * sizeof(Mesh));
  out.body.meshes[1] = left;
  out.body.meshMaterial = (int*)MemRealloc(out.body.meshMaterial, 2 * sizeof(int));
  out.body.meshMaterial[1] = 0;

  out.elevon[0] = LoadModelFromMesh(buildMesh(elevTris, g.cgX, true,
                                              g.hingePtL, elevTop, elevBottom));
  out.elevon[1] = LoadModelFromMesh(buildMesh(elevTris, g.cgX, false,
                                              g.hingePtR, elevTop, elevBottom));
  return true;
}

// --- props ---------------------------------------------------------------------

// Mesh from flat triangle lists (positions + RGBA), uploaded once complete.
inline Mesh meshFromTris(const std::vector<Vector3>& v, const std::vector<Color>& c) {
  Mesh m{};
  m.vertexCount = (int)v.size();
  m.triangleCount = m.vertexCount / 3;
  m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
  m.colors = (unsigned char*)MemAlloc(m.vertexCount * 4);
  for (int i = 0; i < m.vertexCount; i++) {
    m.vertices[3 * i] = v[i].x; m.vertices[3 * i + 1] = v[i].y; m.vertices[3 * i + 2] = v[i].z;
    m.colors[4 * i] = c[i].r; m.colors[4 * i + 1] = c[i].g;
    m.colors[4 * i + 2] = c[i].b; m.colors[4 * i + 3] = c[i].a;
  }
  UploadMesh(&m, false);
  return m;
}

// Three-blade prop in its own frame: spin axis +z (the nose), blades in the
// x-y plane, each a tapered, twisted plate.  Draw with culling off.
inline Mesh makePropMesh(float diameter, int blades = 3) {
  std::vector<Vector3> v;
  std::vector<Color> c;
  const float R = 0.5f * diameter, r0 = 0.12f * R;
  const int ns = 6;                         // spanwise stations
  for (int k = 0; k < blades; k++) {
    const float th = 2.0f * PI * k / blades;
    const Vector3 rad{cosf(th), sinf(th), 0.0f}, tan{-sinf(th), cosf(th), 0.0f};
    auto P = [&](int i, float side) {
      const float t = (float)i / ns;
      const float r = r0 + (R - r0) * t;
      const float chord = (0.30f - 0.14f * t) * R;           // root -> tip taper
      const float pitch = (32.0f - 20.0f * t) * DEG2RAD;     // twist
      const float a = side * 0.5f * chord;
      return Vector3Add(Vector3Scale(rad, r),
                        Vector3Add(Vector3Scale(tan, a * cosf(pitch)),
                                   Vector3{0, 0, a * sinf(pitch)}));
    };
    for (int i = 0; i < ns; i++) {
      const Vector3 a = P(i, -1), b = P(i, 1), d = P(i + 1, 1), e = P(i + 1, -1);
      const Color col = {40, 40, 44, 255};
      v.insert(v.end(), {a, b, d, a, d, e});
      c.insert(c.end(), 6, col);
    }
  }
  return meshFromTris(v, c);
}

// Translucent disc for the blur of a spinning prop (same frame as the prop).
inline Mesh makePropDiscMesh(float diameter) {
  std::vector<Vector3> v;
  std::vector<Color> c;
  const float R = 0.5f * diameter;
  const int n = 24;
  for (int i = 0; i < n; i++) {
    const float a0 = 2.0f * PI * i / n, a1 = 2.0f * PI * (i + 1) / n;
    v.insert(v.end(), {{0, 0, 0}, {R * cosf(a0), R * sinf(a0), 0}, {R * cosf(a1), R * sinf(a1), 0}});
    c.insert(c.end(), 3, Color{40, 40, 44, 255});
  }
  return meshFromTris(v, c);
}

// Motor bell behind the prop: a short octagonal can along -z.
inline Mesh makeMotorMesh(float radius, float length) {
  std::vector<Vector3> v;
  std::vector<Color> c;
  const int n = 8;
  const Color side{120, 120, 125, 255}, cap{90, 90, 95, 255};
  for (int i = 0; i < n; i++) {
    const float a0 = 2.0f * PI * i / n, a1 = 2.0f * PI * (i + 1) / n;
    const Vector3 p0{radius * cosf(a0), radius * sinf(a0), 0}, p1{radius * cosf(a1), radius * sinf(a1), 0};
    const Vector3 q0{p0.x, p0.y, -length}, q1{p1.x, p1.y, -length};
    v.insert(v.end(), {p0, p1, q1, p0, q1, q0, {0, 0, 0}, p1, p0});
    c.insert(c.end(), 6, side);
    c.insert(c.end(), 3, cap);
  }
  return meshFromTris(v, c);
}
