#pragma once
// Line-of-sight camera, ported from CRRCSim (crrc_graphics.cpp UpdateCamera +
// zoom.cpp + the model-view window).  The eye stays at a fixed pilot position
// and only the look-at point moves; FOV narrows with distance via zoom_calc.
//
// CRRCSim works in feet: zoom_calc's 21.5 normal distance is 21.5 ft, so
// distances are converted before using its constants.  Defaults match the
// user's ~/.crrcsim/crrcsim.xml: field_of_view 35, autozoom 0.05, sloppy 0,
// modelViewWindow auto.

#include <cmath>
#include <raylib.h>
#include <raymath.h>

struct LosCamera {
  Vector3 pilot{0.0f, 1.7f, 0.0f};   // eye position, metres, y up
  Vector3 looking{4.0f, 1.7f, 0.0f}; // current look-at point
  float fovBase = 35.0f;             // deg, HORIZONTAL (CRRCSim), 5..65
  float autozoom = 0.05f;            // fraction per metre beyond normDist
  float sloppy = 0.0f;               // fraction of fov dead zone, 0 = locked on
  float fovCurrent = 35.0f;

  static constexpr float kFtPerM = 3.28084f;

  // CRRCSim zoom.cpp zoom_calc (dist in metres, constants in feet)
  float zoomCalc(float distM) {
    const float dist = distM * kFtPerM;
    const float normDist = 21.5f;   // ft
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

    if (cosAng <= 0.0f) {
      // aircraft beside or behind the sight line (e.g. after a relaunch):
      // the triangle construction below is undefined here, so look at it
      looking = planePos;
    } else if (cosSqr < cmax) {
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

    // zoom_calc gives CRRCSim's horizontal FOV (plib setFOV(h, h/aspect));
    // raylib's fovy is vertical, so convert for the window's aspect ratio
    const float dist = Vector3Length(planeDir);
    const float hRad = zoomCalc(dist) * DEG2RAD;
    const float aspect = (float)GetScreenWidth() / fmaxf((float)GetScreenHeight(), 1.0f);
    cam.fovy = 2.0f * atanf(tanf(0.5f * hRad) / aspect) * RAD2DEG;
    cam.position = pilot;
    cam.target = looking;
    cam.up = {0.0f, 1.0f, 0.0f};
    cam.projection = CAMERA_PERSPECTIVE;
  }

  // CRRCSim model-view window (fUse = auto): when the aircraft, with a 30 %
  // margin, spans less than 15 % of the main field of view, show a close-up
  // whose horizontal FOV just fits it.  halfSize = half the aircraft's size.
  static constexpr float kModelViewSize = 0.15f;      // fraction of screen
  bool modelView(float distM, float halfSize, float& fovDeg) const {
    fovDeg = 2.0f * atanf(halfSize / fmaxf(distM, 0.01f)) * RAD2DEG * 1.3f;
    return fovCurrent * kModelViewSize / fovDeg > 1.0f;
  }

  // Aim straight at the aircraft, e.g. on (re)launch.
  void reset(Camera3D& cam, Vector3 planePos) {
    looking = planePos;
    update(cam, planePos, 0.0f);
  }

  void zoomIn()  { fovBase = fmaxf(5.0f, fovBase - 5.0f); }
  // CRRCSim stops at 65; up to 90 here so the view can match a photo-sphere
  // viewer (e.g. Google Maps shows 75) for comparing scenery scale
  void zoomOut() { fovBase = fminf(90.0f, fovBase + 5.0f); }
};
