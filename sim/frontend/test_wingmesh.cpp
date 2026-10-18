// Checks the render transforms against the physics conventions, without a
// window: positive elevon deflection must move the trailing edge down (toward
// -dorsal) on both sides, the hinge line must stay fixed, and body-child
// placement must apply the body pose last.
//
//   cmake --build build && ./build/test_wingmesh ../as_built.json

#include <cstdio>
#include <cmath>
#include "wingmesh.h"

static int failures = 0;
static void check(bool ok, const char* what) {
  printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) failures++;
}
static bool near(Vector3 a, Vector3 b, float tol = 1e-5f) {
  return Vector3Length(Vector3Subtract(a, b)) < tol;
}

int main(int argc, char** argv) {
  const WingGeom g = loadWingGeom(argc > 1 ? argv[1] : "../as_built.json");
  const Matrix I = MatrixIdentity();

  const Vector3 pt[2] = {g.hingePtL, g.hingePtR};
  const Vector3 ax[2] = {g.hingeAxL, g.hingeAxR};
  const char* side[2] = {"left", "right"};

  for (int s = 0; s < 2; s++) {
    // a trailing-edge point 30 mm aft of the hinge (sim z = nose)
    const Vector3 teLocal{0.0f, 0.0f, -0.03f};
    const Vector3 te0 = Vector3Transform(teLocal, elevonTransform(I, pt[s], ax[s], 0));
    const Vector3 teDown = Vector3Transform(teLocal, elevonTransform(I, pt[s], ax[s], 10));
    char msg[128];
    snprintf(msg, sizeof msg, "%s: zero deflection leaves the elevon in place", side[s]);
    check(near(te0, Vector3Add(pt[s], teLocal)), msg);
    snprintf(msg, sizeof msg, "%s: +10 deg moves the trailing edge down", side[s]);
    check(teDown.y < te0.y - 0.004f, msg);
    snprintf(msg, sizeof msg, "%s: hinge point stays on the hinge", side[s]);
    check(near(Vector3Transform({0, 0, 0}, elevonTransform(I, pt[s], ax[s], 25)), pt[s]), msg);
    // a mirror flips rotation sense, so TE-down-positive on both sides needs
    // both axes pointing to the aircraft's right (sim -x), not both outboard
    snprintf(msg, sizeof msg, "%s: hinge axis points to the aircraft's right", side[s]);
    check(ax[s].x < 0.0f, msg);
  }

  // StateInput.rotation is stored by rows.  simitl-wing's hover pose
  // (rows {1,0,0},{0,0,1},{0,-1,0}) puts the nose (sim +z) at world +y, the
  // port wingtip (sim +x) at world +x and the dorsal side (sim +y) at -z.
  const Vec3F hover[3] = {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
  const Vec3F pos{5.0f, 2.0f, -1.0f};
  const Matrix body = bodyMatrix(hover, pos);
  const Vector3 o = Vector3Transform({0, 0, 0}, body);
  check(near(o, {pos.x, pos.y, pos.z}), "body origin is at the position");
  check(near(Vector3Subtract(Vector3Transform({0, 0, 1}, body), o), {0, 1, 0}),
        "hover pose: nose points up");
  check(near(Vector3Subtract(Vector3Transform({1, 0, 0}, body), o), {1, 0, 0}),
        "hover pose: port wingtip points to world +x");
  check(near(Vector3Subtract(Vector3Transform({0, 1, 0}, body), o), {0, 0, -1}),
        "hover pose: dorsal side faces world -z");

  // hinge point -> R * p + t with R stored by rows; body pose applied last
  const Vector3 p = g.hingePtR;
  const Vector3 expect{hover[0].x * p.x + hover[0].y * p.y + hover[0].z * p.z + pos.x,
                       hover[1].x * p.x + hover[1].y * p.y + hover[1].z * p.z + pos.y,
                       hover[2].x * p.x + hover[2].y * p.y + hover[2].z * p.z + pos.z};
  check(near(Vector3Transform({0, 0, 0}, elevonTransform(body, p, g.hingeAxR, 15)), expect),
        "body pose is applied after the hinge rotation");

  printf(failures ? "%d failure(s)\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
