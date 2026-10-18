#pragma once
// Equirectangular photo sphere as the scenery.
//
// The LOS eye never leaves the pilot spot, so a sphere photographed there is
// exactly right from the sim's viewpoint, ground included; the flat ground
// plane is then not drawn (the aircraft's shadow still is).
//
// Image format: 2:1 equirectangular (phone photo sphere / 360 camera / HDRI),
// top row zenith, bottom row nadir, longitude increasing to the right
// (clockwise seen from above), centre column = the panorama's 0 deg.
//
// headingDeg: the panorama heading (in the image's own degrees, measured from
// its centre column) that the pilot faces at launch, i.e. sim world +x.
//
// Drawn as a cube around the camera with a shader that turns each fragment's
// view direction into equirectangular (u, v) -- no seam or pole pinching.

#include <string>
#include <raylib.h>
#include <rlgl.h>

static const char* kSkyVs = R"(#version 330
in vec3 vertexPosition;
uniform mat4 mvp;
out vec3 dir;
void main() {
    dir = vertexPosition;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)";

static const char* kSkyFs = R"(#version 330
in vec3 dir;
uniform sampler2D texture0;
uniform float heading;   // radians, see skybox.h
out vec4 finalColor;
void main() {
    vec3 d = normalize(dir);
    // world azimuth from +x, clockwise seen from above (toward +z)
    float az = atan(d.z, d.x) + heading;
    float u = fract(az / 6.28318531 + 0.5);
    float v = acos(clamp(d.y, -1.0, 1.0)) / 3.14159265;
    finalColor = vec4(textureLod(texture0, vec2(u, v), 0.0).rgb, 1.0);
}
)";

struct Skybox {
  bool loaded = false;
  Texture2D tex{};
  Shader shader{};
  Model cube{};

  bool load(const std::string& path, float headingDeg) {
    if (!FileExists(path.c_str())) return false;
    tex = LoadTexture(path.c_str());
    if (tex.id == 0) return false;
    SetTextureFilter(tex, TEXTURE_FILTER_BILINEAR);
    SetTextureWrap(tex, TEXTURE_WRAP_REPEAT);
    shader = LoadShaderFromMemory(kSkyVs, kSkyFs);
    const float heading = headingDeg * DEG2RAD;
    SetShaderValue(shader, GetShaderLocation(shader, "heading"), &heading,
                   SHADER_UNIFORM_FLOAT);
    cube = LoadModelFromMesh(GenMeshCube(2.0f, 2.0f, 2.0f));
    cube.materials[0].shader = shader;
    cube.materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = tex;
    loaded = true;
    return true;
  }

  // Call first inside BeginMode3D: it is drawn around the eye at infinity
  // (no depth write), so everything else draws over it.
  void draw(const Camera3D& cam) const {
    if (!loaded) return;
    rlDisableBackfaceCulling();
    rlDisableDepthMask();
    DrawModel(cube, cam.position, 1.0f, WHITE);
    rlEnableDepthMask();
    rlEnableBackfaceCulling();
  }

  void unload() {
    if (!loaded) return;
    UnloadModel(cube);   // also unloads the shader set on its material
    UnloadTexture(tex);
    loaded = false;
  }
};
