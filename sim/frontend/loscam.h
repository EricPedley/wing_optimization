#pragma once
// Line-of-sight camera, ported from CRRCSim (crrc_graphics.cpp UpdateCamera +
// zoom.cpp).  The eye stays at a fixed pilot position and only the look-at
// point moves; FOV shrinks with distance via zoom_calc so the aircraft keeps
// a near-constant on-screen size.

#include <cmath>
#include <raylib.h>
#include <raymath.h>

struct LosCamera {
  Vector3 pilot{0.0f, 1.7f, 0.0f};   // eye position, metres, y up
  Vector3 looking{4.0f, 1.7f, 0.0f}; // current look-at point
  float fovBase = 35.0f;             // deg, 5..65
  float autozoom = 0.05f;            // fraction per metre beyond normDist
  float sloppy = 0.25f;              // fraction of fov dead zone, 0 = track always
  float fovCurrent = 35.0f;

  // CRRCSim zoom.cpp zoom_calc
  float zoomCalc(float dist) {
    const float normDist = 21.5f;
    const float size = tanf(fovBase * (float)M_PI / 180.0f) * normDist;
    const float d = (dist > normDist)
        ? normDist + (dist - normDist) * autozoom : normDist;
    fovCurrent = atanf(size / d) * 180.0f / (float)M_PI;
    return fovCurrent;
  }

  // UpdateCamera sloppy branch: look-at only moves when the aircraft leaves
  // a cone of `sloppy * fov` around the current sight line.
  void update(Camera3D& cam, Vector3 planePos, float dt) {
    const float fovRad = fovCurrent * (float)M_PI / 180.0f;
    const float phimax = sloppy * fovRad;
    const float cmax = cosf(phimax) * cosf(phimax);

    Vector3 lookDir = Vector3Subtract(looking, pilot);
    Vector3 planeDir = Vector3Subtract(planePos, pilot);
    const float ll = Vector3Length(lookDir), pl = Vector3Length(planeDir);
    if (ll < 1e-3f || pl < 1e-3f) return;

    const float cosSqr =
        Vector3DotProduct(lookDir, planeDir)
        * Vector3DotProduct(lookDir, planeDir) / (ll * ll * pl * pl);
    const float cosAng = Vector3DotProduct(lookDir, planeDir) / (ll * pl);

    if (cosAng > 0 && cosSqr < cmax) {
      // outside the dead cone: swing the look direction toward the plane by
      // exactly phimax (right-angle-triangle construction from CRRCSim)
      const float k = (pl * pl) / Vector3DotProduct(planeDir, lookDir);
      lookDir = Vector3Scale(lookDir, k);
      Vector3 diff = Vector3Subtract(lookDir, planeDir);
      const float ddl = Vector3Length(diff);
      if (ddl > 1e-3f) {
        const float t = fabsf(tanf(acosf(sqrtf(cmax))));
        lookDir = Vector3Add(planeDir,
                             Vector3Scale(diff, pl * t / ddl));
        looking = Vector3Add(lookDir, pilot);
      }
    } else {
      // inside the cone: drift slowly toward the plane
      const Vector3 diff = Vector3Subtract(planePos, looking);
      looking = Vector3Add(looking, Vector3Scale(diff, 0.02f * dt));
    }

    const float dist = Vector3Length(planeDir);
    cam.fovy = zoomCalc(dist);
    cam.position = pilot;
    cam.target = looking;
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.projection = CAMERA_PERSPECTIVE;
  }

  void zoomIn()  { fovBase = fmaxf(5.0f, fovBase - 5.0f); }
  void zoomOut() { fovBase = fminf(65.0f, fovBase + 5.0f); }
};
