#pragma once
// Procedural wing + elevon mesh built from the as_built.json parameters.
// Everything is constructed in the SIM body frame (x=port, y=dorsal, z=nose),
// same convention as TailsitterPhysics:
//   aero (x fwd, y right, z down) -> sim (x_s=-y_a, y_s=-z_a, z_s=x_a)

#include <vector>
#include <cmath>
#include <raylib.h>
#include <raymath.h>
#include "network/packets.h"
#include "json.h"

struct WingGeom {
  float span, rootChord, tipChord, rootHalf, sweepDeg;
  float tcRoot, tcTip;
  float camA1Root, camA2Root, camA1Tip, camA2Tip;
  float cgX;                 // m aft of root LE
  float xHinge;              // hinge line, fraction of chord
  float elevIn, elevOut;     // elevon spanwise stations |y|, m
  Vector3 hingePtR, hingeAxR, hingePtL, hingeAxL;  // sim body frame
  float motorY, motorXAft, motorZ, propDia;
};

inline Vector3 aeroToSim(float xa, float ya, float za) {
  return Vector3{-ya, -za, xa};
}
inline Vector3 jvec3(const json::JsonObject& v) {
  const Vec3F a = v.asVec3();
  return Vector3{a.x, a.y, a.z};
}

inline WingGeom loadWingGeom(const std::string& path) {
  auto j = json::parseFile(path);
  WingGeom g{};
  g.span      = (float)j.get("span").asNumber(0.33);
  g.rootChord = (float)j.get("rootChord").asNumber(0.15);
  g.tipChord  = (float)j.get("tipChord").asNumber(0.08);
  g.rootHalf  = (float)j.get("rootHalfWidth").asNumber(0.005);
  g.sweepDeg  = (float)j.get("leSweepDeg").asNumber(0.0);
  g.tcRoot    = (float)j.get("thicknessRatioRoot").asNumber(0.1);
  g.tcTip     = (float)j.get("thicknessRatioTip").asNumber(0.1);
  g.cgX       = (float)j.get("cgX").asNumber(0.08);
  g.xHinge    = (float)j.get("xHinge").asNumber(0.78);
  g.elevIn    = (float)j.get("elevonInboardY").asNumber(0.02);
  g.elevOut   = (float)j.get("elevonOutboardY").asNumber(0.15);
  if (j.has("camber")) {
    auto c = j.get("camber");
    g.camA1Root = (float)c.get("a1Root").asNumber(0.0);
    g.camA2Root = (float)c.get("a2Root").asNumber(0.0);
    g.camA1Tip  = (float)c.get("a1Tip").asNumber(g.camA1Root);
    g.camA2Tip  = (float)c.get("a2Tip").asNumber(g.camA2Root);
  }
  if (j.has("hinge")) {
    auto h = j.get("hinge");
    auto conv = [](const json::JsonObject& p) {
      Vector3 a = jvec3(p);
      return aeroToSim(a.x, a.y, a.z);
    };
    g.hingePtR = conv(h.get("pointR")); g.hingeAxR = conv(h.get("axisR"));
    g.hingePtL = conv(h.get("pointL")); g.hingeAxL = conv(h.get("axisL"));
  }
  g.motorY    = (float)j.get("motorY").asNumber(0.036);
  g.motorXAft = (float)j.get("motorXAft").asNumber(0.0);
  g.motorZ    = (float)j.get("motorZ").asNumber(0.0);
  g.propDia   = (float)j.get("propDiameter").asNumber(0.05);
  return g;
}

// planform helpers (same formulas as tailsitter.cpp / airfoil_model.py)
inline float wingLE(const WingGeom& g, float yAbs) {
  return tanf(g.sweepDeg * (float)M_PI / 180.0f)
         * fmaxf(yAbs - g.rootHalf, 0.0f);
}
inline float wingChord(const WingGeom& g, float yAbs) {
  const float semi = 0.5f * g.span;
  const float f0 = fminf(fmaxf(g.rootHalf / semi, 0.0f), 1.0f);
  const float t = fminf(fmaxf((yAbs / semi - f0) / fmaxf(1.0f - f0, 1e-9f),
                              0.0f), 1.0f);
  return g.rootChord + (g.tipChord - g.rootChord) * t;
}
inline float wingTC(const WingGeom& g, float yAbs) {
  const float semi = 0.5f * g.span;
  const float f0 = fminf(fmaxf(g.rootHalf / semi, 0.0f), 1.0f);
  const float t = fminf(fmaxf((yAbs / semi - f0) / fmaxf(1.0f - f0, 1e-9f),
                              0.0f), 1.0f);
  return g.tcRoot + (g.tcTip - g.tcRoot) * t;
}
// camber-line ordinate at chord fraction f (same expression as the physics'
// camber scan), multiplied by chord
inline float camberZ(const WingGeom& g, float yAbs, float f) {
  const float semi = 0.5f * g.span;
  const float f0 = fminf(fmaxf(g.rootHalf / semi, 0.0f), 1.0f);
  const float t = fminf(fmaxf((yAbs / semi - f0) / fmaxf(1.0f - f0, 1e-9f),
                              0.0f), 1.0f);
  const float a1 = g.camA1Root + (g.camA1Tip - g.camA1Root) * t;
  const float a2 = g.camA2Root + (g.camA2Tip - g.camA2Root) * t;
  const float ct = 1.0f - 2.0f * f;
  const float th = acosf(fminf(fmaxf(ct, -1.0f + 1e-9f), 1.0f - 1e-9f));
  const float c = cosf(th);
  return (a1 * sinf(th) * sinf(th) / 4.0f
        + a2 * (-c * c * c / 3.0f + c / 2.0f - 1.0f / 6.0f)
        + a2 * f / 3.0f) * wingChord(g, yAbs);
}

// build a triangle-soup mesh covering one span segment [y0, y1] from chord
// fraction fa to fb.  sim-frame vertices; top surface z_s positive.
struct MeshBuilder {
  std::vector<Vector3> v;
  std::vector<Vector3> n;
  std::vector<Vector2> uv;
  void tri(Vector3 a, Vector3 b, Vector3 c, Vector3 norm) {
    v.push_back(a); v.push_back(b); v.push_back(c);
    n.push_back(norm); n.push_back(norm); n.push_back(norm);
    uv.push_back({0,0}); uv.push_back({1,0}); uv.push_back({0,1});
  }
};

// chordwise profile point (aero coords): f in [0,1] chord, side +1 top / -1 bot
inline Vector3 profileAero(const WingGeom& g, float y, float f, int side) {
  const float c = wingChord(g, fabsf(y));
  const float xa = g.cgX - (wingLE(g, fabsf(y)) + f * c);
  const float th = wingTC(g, fabsf(y)) * c;
  // z_a down; camber raises the mid-line toward -z_a (top)
  const float zc = -camberZ(g, fabsf(y), f);
  const float half = 0.5f * th
      * sqrtf(fmaxf(0.0f, 1.0f - (2.0f * f - 1.0f) * (2.0f * f - 1.0f)));
  return Vector3{xa, y, zc + side * half};
}

// one span section mesh, chord [fa..fb]
inline Mesh makeWingSection(const WingGeom& g, float y0, float y1,
                            float fa, float fb, int ny = 3) {
  static const float fs[] = {0.0f, 0.5f, 1.0f};
  // chord fractions between fa..fb
  std::vector<float> chord;
  for (float f : fs) {
    float fr = fa + f * (fb - fa);
    chord.push_back(fr);
  }
  MeshBuilder mb;
  const int nx = (int)chord.size();
  auto V = [&](float y, int ix, int side) {
    Vector3 a = profileAero(g, y, chord[ix], side);
    return aeroToSim(a.x, a.y, a.z);
  };
  for (int iy = 0; iy < ny; iy++) {
    const float ya = y0 + (y1 - y0) * (float)iy / ny;
    const float yb = y0 + (y1 - y0) * (float)(iy + 1) / ny;
    for (int ix = 0; ix < nx - 1; ix++) {
      // top (+z_a is down; top surface faces -z_a => +y_s)
      mb.tri(V(ya, ix, +1), V(yb, ix, +1), V(yb, ix + 1, +1), {0, 1, 0});
      mb.tri(V(ya, ix, +1), V(yb, ix + 1, +1), V(ya, ix + 1, +1), {0, 1, 0});
      // bottom
      mb.tri(V(ya, ix + 1, -1), V(yb, ix + 1, -1), V(yb, ix, -1), {0, -1, 0});
      mb.tri(V(ya, ix + 1, -1), V(yb, ix, -1), V(ya, ix, -1), {0, -1, 0});
    }
    // leading-edge face at chord[0]
    mb.tri(V(ya, 0, -1), V(yb, 0, -1), V(yb, 0, +1), {0, 0, -1});
    mb.tri(V(ya, 0, -1), V(yb, 0, +1), V(ya, 0, +1), {0, 0, -1});
    // trailing-edge face at chord.back()
    mb.tri(V(ya, nx - 1, +1), V(yb, nx - 1, +1), V(yb, nx - 1, -1), {0, 0, 1});
    mb.tri(V(ya, nx - 1, +1), V(yb, nx - 1, -1), V(ya, nx - 1, -1), {0, 0, 1});
  }
  // tip caps at y0 / y1
  for (int ix = 0; ix < nx - 1; ix++) {
    const float ny0 = (y0 < 0) ? -1.0f : 0.0f;
    mb.tri(V(y0, ix, +1), V(y0, ix + 1, +1), V(y0, ix + 1, -1),
           Vector3Normalize(aeroToSim(0, -1, 0)));
    mb.tri(V(y0, ix, +1), V(y0, ix + 1, -1), V(y0, ix, -1),
           Vector3Normalize(aeroToSim(0, -1, 0)));
    mb.tri(V(y1, ix, +1), V(y1, ix + 1, -1), V(y1, ix + 1, +1),
           Vector3Normalize(aeroToSim(0, +1, 0)));
    mb.tri(V(y1, ix, +1), V(y1, ix, -1), V(y1, ix + 1, -1),
           Vector3Normalize(aeroToSim(0, +1, 0)));
    (void)ny0;
  }
  Mesh m{};
  m.vertexCount = (int)mb.v.size();
  m.triangleCount = m.vertexCount / 3;
  m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
  m.normals  = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
  m.texcoords = (float*)MemAlloc(m.vertexCount * 2 * sizeof(float));
  for (int i = 0; i < m.vertexCount; i++) {
    m.vertices[3*i] = mb.v[i].x; m.vertices[3*i+1] = mb.v[i].y;
    m.vertices[3*i+2] = mb.v[i].z;
    m.normals[3*i] = mb.n[i].x; m.normals[3*i+1] = mb.n[i].y;
    m.normals[3*i+2] = mb.n[i].z;
    m.texcoords[2*i] = mb.uv[i].x; m.texcoords[2*i+1] = mb.uv[i].y;
  }
  UploadMesh(&m, false);
  return m;
}

// elevon slab in a LOCAL frame whose origin is the hinge point; the mesh is
// rotated at draw time.  side: +1 right (aero y>0), -1 left.
inline Mesh makeElevon(const WingGeom& g, int side, Vector3 hingePtSim) {
  const float y0 = side > 0 ? g.elevIn : -g.elevOut;
  const float y1 = side > 0 ? g.elevOut : -g.elevIn;
  MeshBuilder mb;
  const int ny = 4, nx = 3;
  for (int iy = 0; iy < ny; iy++) {
    for (int ixc = 0; ixc < nx - 1; ixc++) {
      const float fa = g.xHinge + (1.0f - g.xHinge) * (float)ixc / (nx - 1);
      const float fb = g.xHinge + (1.0f - g.xHinge) * (float)(ixc + 1) / (nx - 1);
      const float ya = y0 + (y1 - y0) * (float)iy / ny;
      const float yb = y0 + (y1 - y0) * (float)(iy + 1) / ny;
      auto P = [&](float y, float f, int s) {
        Vector3 a = profileAero(g, y, f, s > 0 ? 1 : -1);
        // thinner elevon: shrink thickness toward zero
        Vector3 p = aeroToSim(a.x, a.y, a.z);
        return Vector3Subtract(p, hingePtSim);
      };
      mb.tri(P(ya, fa, +1), P(yb, fa, +1), P(yb, fb, +1), {0, 1, 0});
      mb.tri(P(ya, fa, +1), P(yb, fb, +1), P(ya, fb, +1), {0, 1, 0});
      mb.tri(P(ya, fb, -1), P(yb, fb, -1), P(yb, fa, -1), {0, -1, 0});
      mb.tri(P(ya, fb, -1), P(yb, fa, -1), P(ya, fa, -1), {0, -1, 0});
    }
    const float ya = y0 + (y1 - y0) * (float)iy / ny;
    const float yb = y0 + (y1 - y0) * (float)(iy + 1) / ny;
    auto P = [&](float y, float f, int s) {
      Vector3 a = profileAero(g, y, f, s > 0 ? 1 : -1);
      return Vector3Subtract(aeroToSim(a.x, a.y, a.z), hingePtSim);
    };
    mb.tri(P(ya, 1.0f, +1), P(yb, 1.0f, +1), P(yb, 1.0f, -1), {0, 0, 1});
    mb.tri(P(ya, 1.0f, +1), P(yb, 1.0f, -1), P(ya, 1.0f, -1), {0, 0, 1});
  }
  Mesh m{};
  m.vertexCount = (int)mb.v.size();
  m.triangleCount = m.vertexCount / 3;
  m.vertices = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
  m.normals  = (float*)MemAlloc(m.vertexCount * 3 * sizeof(float));
  m.texcoords = (float*)MemAlloc(m.vertexCount * 2 * sizeof(float));
  for (int i = 0; i < m.vertexCount; i++) {
    m.vertices[3*i] = mb.v[i].x; m.vertices[3*i+1] = mb.v[i].y;
    m.vertices[3*i+2] = mb.v[i].z;
    m.normals[3*i] = mb.n[i].x; m.normals[3*i+1] = mb.n[i].y;
    m.normals[3*i+2] = mb.n[i].z;
    m.texcoords[2*i] = mb.uv[i].x; m.texcoords[2*i+1] = mb.uv[i].y;
  }
  UploadMesh(&m, false);
  return m;
}
