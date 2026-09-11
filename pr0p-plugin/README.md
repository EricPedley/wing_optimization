# Pr0pCustomModel

BepInEx 6 plugin for **pr0p** (0.9.16, Unity 6000.2.12f1, Mono backend) that
registers a runtime-built `ModelConfig` so a quad JSON config can use a
custom 3D model loaded from a GLB file.

## What it does

- Harmony postfix on `QuadConfigLoader.Init()` loads
  `BepInEx/plugins/hq-51mm-micro.glb` (path configurable), builds a
  `GameObject` hierarchy, and registers it in
  `QuadConfigLoader.modelConfigs` under the key `hq-51mm-micro`.
- Quad configs whose `"model"` field equals that key get the custom model.
  `install.sh` patches `config/quad/hq-51mm-micro.json` accordingly.

## Model wiring

The GLB (`model/hq-51mm-micro.glb`, millimeters, Y-up, -Z forward) is parsed
by a dependency-free importer (`GlbImporter.cs` + `MiniJson.cs`): JSON/BIN
chunks, accessors, bufferViews, node TRS, glTF→Unity handedness (negate Z,
flip winding — this maps the GLB's -Z forward onto Unity's +Z forward).
The instantiated root is scaled 0.001 (mm → m) and left unrotated.

- Each `propN` node is re-parented under a `propN_pivot` GO placed at the
  prop mesh's bounds center (the mesh verts are not centered at the node
  origin, and `Motor.Update` spins its own transform about local Y — so
  Motor + Propeller + the prop-disk box collider all live on the pivot).
- `ModelConfig.motor1..4` are matched to the pivots by comparing pivot
  positions (root space, post-flip) to the quad config's `motorNPos` —
  the importer's Z-negation mirrors the GLB, so node order does not line
  up with physics motor order. `rotationDir` = the matched `motorNDir`.
- Forward convention: the game faces +Z (donor `vtx-slayer-3`
  cameraPosition.z > 0; `motor1Pos` z<0 is the rear-right motor,
  Betaflight order). The importer's Z-negation already makes the model
  +Z-forward, and the root is left unrotated — `Quad` overwrites
  `fpvCameraTransform.localRotation` with `Euler(-cameraAngle,0,0)`
  every frame, so a rotated root would point the FPV camera backwards.
- All other nodes get box colliders fitted to mesh bounds; every collider
  (incl. the camera sphere) is added to `ModelConfig.colliders`.
- `cameraPosition` = child transform at the `camera` node bounds center
  (identity rotation — the game drives camera tilt itself).
- `cameraCollider` = child GO with a `SphereCollider`. Unity 6.2's managed
  `SphereCollider` has no `radius` binding, so the GO is scaled ×14 to make
  the default 0.5-unit radius ≈ 7 mm in world space.
- Motor/prop internals are cloned from a donor modelConfig (default
  `vtx-slayer-3`, configurable): audio clips (new `AudioSource`s on each
  prop), scalar motor params via reflection, the prop material (for the
  `_RPM` spin shader — the extracted prop UVs match it), and the
  `ParticleBurnedEffect` / `ParticleImpactEffect` / `CollisionSoundEffect`
  prefab objects are `Instantiate`d under each prop so collision effects work
  with no NREs.
- Motor/Propeller components are added while the prop GO is inactive so
  `Motor.OnEnable` (which touches `soundIdle`) can't fire before wiring.
- The template root is parented under an *inactive* DontDestroyOnLoad
  holder at the origin. `Quad.LoadQuadModel`/`QuadPreview.LoadPreview`
  use `Instantiate(config, parent)` with no position reset, so the
  template's localPosition is copied into the clone — it must be zero
  (an earlier y=-10000 parking spot made clones spawn kilometres away
  and hid the menu preview). The inactive holder hides the template
  while `activeSelf` stays true, so clones are active under the quad.

## Install

```sh
./install.sh           # builds, installs BepInEx + plugin + GLB, patches the quad json
cd ~/programs/pr0p && ./run_bepinex.sh ./pr0p.x86_64
```

Then select the "HQ 51mm 1S Micro Freestyle" quad in-game. Log:
`~/programs/pr0p/BepInEx/LogOutput.log`.

## Config (BepInEx/config/dev.pr0p.custommodel.cfg)

| Key | Default | Meaning |
|---|---|---|
| `GlbPath` | `BepInEx/plugins/hq-51mm-micro.glb` | GLB to load |
| `ModelKey` | `hq-51mm-micro` | modelConfigs dict key |
| `Scale` | `0.001` | uniform root scale |
| `DonorModel` | `vtx-slayer-3` | donor modelConfig for materials/audio/effects |

## Caveats / known limitations

- Embedded GLB textures are not decoded (Unity 6.2 build lacks managed
  `ImageConversion.LoadImage`); `baseColorFactor` colors are used instead.
  The props use the donor prop material which carries its own texture.
- `AudioSource` in this Unity version exposes only clip/loop/volume/pitch —
  spatial audio settings can't be copied, so motor sounds are effectively
  2D defaults (they're on the local player's quad anyway).
- The camera sphere radius workaround relies on the default 0.5 radius.
- If the donor model key is missing, the first available modelConfig is used.
