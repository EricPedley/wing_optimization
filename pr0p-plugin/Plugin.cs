// Pr0pCustomModel — runtime GLB model loader for pr0p (Unity 6000.2, Mono).
//
// Injection method: a one-time Cecil patch adds a call to
//   Pr0pCustomModel.Loader.OnQuadConfigLoaderInit(this)
// at the end of QuadConfigLoader.Init() in pr0p_Data/Managed/Pr0Drone.dll
// (see Patcher.cs / install.sh). BepInEx can't bootstrap on this build —
// the game's trimmed corlib lacks Module.GetPEKind and Enumerable.Concat —
// and UnityDoorstop's entrypoint proved unreliable, so direct IL injection
// is used instead.
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Pr0Drone;
using UnityEngine;

namespace Pr0pCustomModel
{
    public static class Log
    {
        static string path;
        public static void Init()
        {
            try
            {
                var dir = Path.Combine(Log.GameDir(), "pr0p_mods");
                Directory.CreateDirectory(dir);
                path = Path.Combine(dir, "Pr0pCustomModel.log");
                File.WriteAllText(path, "");
            }
            catch { }
        }
        // Application.dataPath is "<game>/pr0p_Data"; this corlib lacks
        // Directory.GetParent, so strip the suffix manually.
        public static string GameDir()
        {
            var p = Application.dataPath;
            const string suffix = "pr0p_Data";
            if (p.EndsWith(suffix))
                p = p.Substring(0, p.Length - suffix.Length);
            return p.TrimEnd('/', '\\');
        }

        public static void Write(string msg)
        {
            try
            {
                if (path != null)
                    using (var w = new StreamWriter(path, append: true))
                        w.WriteLine($"[{DateTime.Now:HH:mm:ss}] {msg}");
            }
            catch { }
            try { Debug.Log("[Pr0pCustomModel] " + msg); } catch { }
        }
    }

    public static class Loader
    {
        public static string GlbPath;
        public static string ModelKey = "hq-51mm-micro";
        public static float ModelScale = 0.001f;
        public static string DonorModel = "vtx-slayer-3";
        public static bool AutoTest;
        public static QuadConfigLoader ConfigLoader;
        public static ModelConfig Template;
        public static float DonorCamZ;
        static bool initialized;

        // Called from the injected hook at the end of QuadConfigLoader.Init.
        public static void OnQuadConfigLoaderInit(QuadConfigLoader loader)
        {
            try
            {
                if (!initialized)
                {
                    initialized = true;
                    Log.Init();
                    var dir = Path.Combine(Log.GameDir(), "pr0p_mods");
                    GlbPath = Path.Combine(dir, "hq-51mm-micro.glb");
                    LoadConfig(Path.Combine(dir, "Pr0pCustomModel.ini"));
                    Log.Write($"init; glb={GlbPath} key={ModelKey} " +
                              $"scale={ModelScale} donor={DonorModel}");
                }
                Register(loader);
            }
            catch (Exception e)
            {
                Log.Write("OnQuadConfigLoaderInit failed: " + e);
            }
        }

        static void LoadConfig(string iniPath)
        {
            try
            {
                if (!File.Exists(iniPath)) return;
                foreach (var line in File.ReadAllText(iniPath).Split('\n'))
                {
                    var l = line.Trim();
                    if (l.Length == 0 || l.StartsWith("#") ||
                        l.StartsWith(";")) continue;
                    int eq = l.IndexOf('=');
                    if (eq < 0) continue;
                    var k = l.Substring(0, eq).Trim();
                    var v = l.Substring(eq + 1).Trim();
                    switch (k)
                    {
                        case "GlbPath": GlbPath = v; break;
                        case "ModelKey": ModelKey = v; break;
                        case "Scale":
                            ModelScale = float.Parse(v,
                                System.Globalization.CultureInfo
                                    .InvariantCulture);
                            break;
                        case "DonorModel": DonorModel = v; break;
                        case "AutoTest": AutoTest = v == "1" ||
                            v.Equals("true",
                                StringComparison.OrdinalIgnoreCase); break;
                    }
                }
            }
            catch (Exception e) { Log.Write("config read failed: " + e); }
        }

        static void Register(QuadConfigLoader loader)
        {
            try
            {
                var mc = ModelBuilder.Build(loader, GlbPath, ModelKey,
                    ModelScale, DonorModel);
                if (mc == null)
                {
                    Log.Write("model build failed");
                    return;
                }
                loader.modelConfigs[ModelKey] = mc;
                ConfigLoader = loader;
                Template = mc;
                Log.Write($"registered model '{ModelKey}' " +
                          $"({loader.modelConfigs.Count} models total)");
                // ReassignComponents already ran inside Init(); fix up any
                // quad configs that reference our key.
                int n = 0;
                foreach (var e in loader.quadConfigs.entries)
                {
                    if (e.value.model == ModelKey)
                    {
                        e.value.modelConfig = mc;
                        n++;
                        Log.Write($"quad '{e.key}' now uses model " +
                                  $"'{ModelKey}'");
                    }
                }
                if (n == 0)
                    Log.Write("WARNING: no quad config references model " +
                        "key '" + ModelKey + "' — set \"model\": \"" +
                        ModelKey + "\" in a quad JSON under config/quad/");
                if (AutoTest)
                {
                    var go = new GameObject("Pr0pCustomModelAutoTest");
                    UnityEngine.Object.DontDestroyOnLoad(go);
                    go.AddComponent<AutoTestRunner>();
                    Log.Write("autotest runner spawned");
                }
            }
            catch (Exception ex)
            {
                Log.Write("Register failed: " + ex);
            }
        }
    }

    static class ModelBuilder
    {
        // pr0p motor order: M1 front-right(+x,-z) M2 rear-right(+x,+z)
        // M3 front-left(-x,-z) M4 rear-left(-x,+z); the GLB generator used
        // the same ordering. Standard quad dirs: -1,1,1,-1.
        static readonly float[] MotorDirs = { -1f, 1f, 1f, -1f };

        // nodes that get a box collider fitted to mesh bounds
        static readonly string[] BoxNodes =
        {
            "frame_bottom", "frame_top",
            "standoff1", "standoff2", "standoff3", "standoff4",
            "arm1", "arm2", "arm3", "arm4",
            "motor1", "motor2", "motor3", "motor4",
            "camera", "battery", "antenna",
        };

        public static ModelConfig Build(QuadConfigLoader loader,
            string glbPath, string key, float scale, string donorKey)
        {
            if (!File.Exists(glbPath))
            {
                Log.Write("GLB not found: " + glbPath);
                return null;
            }

            // donor model supplies: prop material (_RPM shader), motor audio
            // clips/params, particle effect prefabs, collision sound prefab.
            ModelConfig donor = null;
            if (loader.modelConfigs.ContainsKey(donorKey))
                donor = loader.modelConfigs[donorKey];
            if (donor == null && loader.modelConfigs.entries.Count > 0)
                donor = loader.modelConfigs.entries[0].value;
            if (donor == null)
            {
                Log.Write("no donor ModelConfig available");
                return null;
            }
            var donorMotors = new Motor[4];
            var donorProps = new Propeller[4];
            var dm = new[] { donor.motor1, donor.motor2, donor.motor3,
                             donor.motor4 };
            for (int i = 0; i < 4; i++)
            {
                donorMotors[i] = dm[i].GetComponent<Motor>();
                donorProps[i] = donorMotors[i].prop;
                if (dm[i] != null)
                    Log.Write($"donor motor{i + 1} localPos=" +
                              $"{dm[i].transform.localPosition}");
            }
            if (donor.cameraPosition != null)
                Log.Write($"donor cameraPosition localPos=" +
                          $"{donor.cameraPosition.localPosition} " +
                          $"localRot={donor.cameraPosition.localEulerAngles}");
            var propRenderer = donorMotors[0].model
                .GetComponentInChildren<Renderer>();
            var propMat = propRenderer != null
                ? propRenderer.sharedMaterial : null;
            Log.Write($"donor '{donorKey}': propMat=" +
                      $"{(propMat != null ? propMat.name : "null")}");

            var res = GlbImporter.Load(glbPath);
            var root = res.root;
            root.name = key;
            // Park the template under an INACTIVE holder at the origin.
            // Quad.LoadQuadModel / QuadPreview.LoadPreview use
            // Instantiate(config, parent) with no position reset, so the
            // template's localPosition is copied verbatim into the clone —
            // it must be zero, or clones spawn kilometres away (which also
            // hid the menu preview). An inactive parent keeps the template
            // invisible while root.activeSelf stays true, so clones are
            // active when re-parented under the quad.
            var holder = new GameObject("Pr0pCustomModel.Templates");
            UnityEngine.Object.DontDestroyOnLoad(holder);
            holder.SetActive(false);
            root.transform.SetParent(holder.transform, false);
            root.transform.localPosition = Vector3.zero;
            root.transform.localRotation = Quaternion.identity;
            root.transform.localScale = Vector3.one * scale;

            var mc = root.AddComponent<ModelConfig>();
            var colliders = new List<Collider>();

            // static geometry: box colliders fitted to mesh bounds
            foreach (var name in BoxNodes)
            {
                if (!res.nodes.TryGetValue(name, out var go) ||
                    !res.meshes.TryGetValue(go, out var mesh))
                {
                    Log.Write("missing node/mesh: " + name);
                    continue;
                }
                var b = go.AddComponent<BoxCollider>();
                b.center = mesh.bounds.center;
                b.size = mesh.bounds.size;
                colliders.Add(b);
            }

            // props: a pivot GO is placed at the prop mesh's bounds center;
            // the prop mesh hangs below it. Motor.Update() rotates its own
            // transform about local Y, so the Motor (and the collider +
            // Propeller, which Quad.OnCollisionEnter looks up on the
            // collider's GO) must live on the pivot or the blades orbit
            // the node origin instead of spinning in place.
            var propPivots = new List<GameObject>();
            for (int i = 0; i < 4; i++)
            {
                string name = "prop" + (i + 1);
                if (!res.nodes.TryGetValue(name, out var go))
                {
                    Log.Write("missing prop node " + name);
                    return null;
                }
                propPivots.Add(BuildProp(go, res.meshes[go],
                    donorMotors[i], donorProps[i], propMat, colliders));
            }

            // camera anchor: child of root at the camera node's bounds center.
            // The game drives localRotation (camera angle), so keep identity.
            var camPos = new GameObject("CameraPosition");
            camPos.transform.SetParent(root.transform, false);
            if (res.nodes.TryGetValue("camera", out var camNode) &&
                res.meshes.TryGetValue(camNode, out var camMesh))
            {
                camPos.transform.localPosition =
                    camNode.transform.localPosition + camMesh.bounds.center;
            }
            mc.cameraPosition = camPos.transform;

            var camCol = new GameObject("CameraCollider");
            camCol.transform.SetParent(root.transform, false);
            camCol.transform.localPosition = camPos.transform.localPosition;
            // SphereCollider.radius has no managed binding in this Unity 6.2
            // build; default radius is 0.5 local units, so scale the GO to
            // get a ~7mm world-space radius under the 0.001-scaled root.
            var sc = camCol.AddComponent<SphereCollider>();
            camCol.transform.localScale = Vector3.one * 14f;
            colliders.Add(sc);
            mc.cameraCollider = camCol;

            mc.colliders = colliders;

            // Orientation: the GLB is authored -Z forward (mm, Y-up) and
            // the importer negates Z, so the geometry lands +Z-forward in
            // root space — the root must NOT be rotated. Quad sets
            // fpvCameraTransform.localRotation = Euler(-cameraAngle,0,0)
            // every frame, so any root flip would leave the FPV camera
            // facing into the frame. Just verify the camera anchor's z
            // sign matches the donor cameraPosition's (+0.041).
            if (donor.cameraPosition != null)
            {
                float donorCamZ = donor.cameraPosition.localPosition.z;
                Loader.DonorCamZ = donorCamZ;
                float ourCamZ = camPos.transform.localPosition.z;
                if (Mathf.Sign(donorCamZ) != Mathf.Sign(ourCamZ) &&
                    Mathf.Abs(donorCamZ) > 1e-4f)
                {
                    Log.Write($"WARNING: camZ sign mismatch: " +
                              $"donorCamZ={donorCamZ} ourCamZ={ourCamZ} " +
                              $"(root rotation left identity — importer " +
                              $"Z-flip should have fixed this)");
                }
                else
                {
                    Log.Write($"camZ OK: donorCamZ={donorCamZ} " +
                              $"ourCamZ={ourCamZ} (no root rotation)");
                }
            }

            // Assign mc.motor1..4 by matching each pivot's root-space xz
            // to the quad config's motorNPos (meters -> mm). The importer
            // mirrors Z for handedness, so GLB prop order does NOT line up
            // with the physics motor order — matching by position is the
            // reliable mapping. rotationDir comes from motorNDir of the
            // matched motor.
            QuadConfig qc = null;
            foreach (var e in loader.quadConfigs.entries)
                if (e.value.model == key) { qc = e.value; break; }
            var mpos = new[]
            {
                qc != null ? qc.motor1Pos : new Vector3(0.067f, 0, -0.0555f),
                qc != null ? qc.motor2Pos : new Vector3(0.067f, 0, 0.0555f),
                qc != null ? qc.motor3Pos : new Vector3(-0.067f, 0, -0.0555f),
                qc != null ? qc.motor4Pos : new Vector3(-0.067f, 0, 0.0555f),
            };
            var mdir = new[]
            {
                qc != null ? qc.motor1Dir : MotorDirs[0],
                qc != null ? qc.motor2Dir : MotorDirs[1],
                qc != null ? qc.motor3Dir : MotorDirs[2],
                qc != null ? qc.motor4Dir : MotorDirs[3],
            };
            var assigned = new GameObject[4];
            var used = new bool[propPivots.Count];
            for (int i = 0; i < 4; i++)
            {
                float best = float.MaxValue;
                int bestJ = -1;
                for (int j = 0; j < propPivots.Count; j++)
                {
                    if (used[j]) continue;
                    // pivot.localPosition is in root space (mm); the root
                    // is never rotated now, but apply its rotation anyway
                    // to be safe when converting to quad space
                    Vector3 p = root.transform.localRotation *
                        propPivots[j].transform.localPosition;
                    float d = (new Vector2(p.x, p.z) -
                        new Vector2(mpos[i].x * 1000f,
                                    mpos[i].z * 1000f)).magnitude;
                    if (d < best) { best = d; bestJ = j; }
                }
                used[bestJ] = true;
                assigned[i] = propPivots[bestJ];
                var mt = assigned[i].GetComponent<Motor>();
                if (mt != null) mt.rotationDir = mdir[i];
                Log.Write($"motor{i + 1} -> {propPivots[bestJ].name} " +
                          $"pos={propPivots[bestJ].transform.localPosition} " +
                          $"target={(mpos[i] * 1000f)} err={best:F1}mm " +
                          $"dir={mdir[i]}");
                if (best > 10f)
                    Log.Write($"WARNING: motor{i + 1} match off by " +
                              $"{best:F1}mm");
            }
            mc.motor1 = assigned[0];
            mc.motor2 = assigned[1];
            mc.motor3 = assigned[2];
            mc.motor4 = assigned[3];

            Log.Write($"model built: {res.nodes.Count} nodes, " +
                      $"{colliders.Count} colliders");
            return mc;
        }

        // Returns the pivot GameObject that carries Motor/Propeller and the
        // prop disk collider; the mesh node is re-parented underneath it.
        static GameObject BuildProp(GameObject go, Mesh mesh,
            Motor donorMotor, Propeller donorProp, Material propMat,
            List<Collider> colliders)
        {
            var parent = go.transform.parent;

            // pivot at the prop mesh's bounds center (world space, which
            // under the uniformly-scaled root is just a mm offset)
            var pivot = new GameObject(go.name + "_pivot");
            pivot.transform.SetParent(parent, false);
            Vector3 centerWorld =
                go.transform.TransformPoint(mesh.bounds.center);
            pivot.transform.position = centerWorld;
            go.transform.SetParent(pivot.transform, true);

            // thin box approximating the prop disk; the pivot origin is the
            // bounds center so the collider is centered at zero
            var bc = pivot.AddComponent<BoxCollider>();
            var size = mesh.bounds.size;
            bc.size = new Vector3(size.x,
                Mathf.Max(size.y, 1.5f), size.z); // mm units
            bc.center = Vector3.zero;
            colliders.Add(bc);

            // use the donor prop material so the _RPM spin shader works
            var rend = go.GetComponentInChildren<Renderer>();
            if (propMat != null && rend != null)
                rend.sharedMaterial = propMat;

            // add components while inactive so Motor.OnEnable doesn't fire
            // before its fields are wired
            bool wasActive = pivot.activeSelf;
            pivot.SetActive(false);
            go.SetActive(true);

            var prop = pivot.AddComponent<Propeller>();
            if (donorProp != null)
            {
                prop.soundPitch = donorProp.soundPitch;
                prop.durability = donorProp.durability;
                if (donorProp.impactEffect != null)
                {
                    var fx = UnityEngine.Object.Instantiate(
                        donorProp.impactEffect.gameObject, pivot.transform);
                    prop.impactEffect =
                        fx.GetComponent<ParticleImpactEffect>();
                }
                if (donorProp.soundEffect != null)
                {
                    var fx = UnityEngine.Object.Instantiate(
                        donorProp.soundEffect.gameObject, pivot.transform);
                    prop.soundEffect =
                        fx.GetComponent<CollisionSoundEffect>();
                }
            }

            var motor = pivot.AddComponent<Motor>();
            CopyMotorScalars(donorMotor, motor);
            motor.prop = prop;
            motor.model = pivot; // material pulled from child's renderer
            // rotationDir is assigned later from the matched motorNDir
            motor.soundIdle = CloneAudioSource(donorMotor.soundIdle, pivot);
            motor.soundMid = CloneAudioSource(donorMotor.soundMid, pivot);
            motor.soundHigh = CloneAudioSource(donorMotor.soundHigh, pivot);
            if (donorMotor.burnedEffect != null)
            {
                var fx = UnityEngine.Object.Instantiate(
                    donorMotor.burnedEffect.gameObject, pivot.transform);
                motor.burnedEffect =
                    fx.GetComponent<ParticleBurnedEffect>();
            }

            if (wasActive) pivot.SetActive(true);
            return pivot;
        }

        // scalar Motor fields safe to copy verbatim from the donor
        static readonly string[] MotorScalarFields =
        {
            "soundMaxRmp", "soundPitchFactor",
            "maxIdleVolume", "maxMidVolume", "maxHighVolume",
            "midPitchAdd", "highPitchAdd",
            "idleStart", "idleEnd", "midStart", "midPeak", "midEnd",
            "highStart", "highEnd",
            "maxTemp", "maxDamagePitch", "maxDamageOscPitch",
        };

        static void CopyMotorScalars(Motor src, Motor dst)
        {
            var t = typeof(Motor);
            foreach (var name in MotorScalarFields)
            {
                var f = t.GetField(name,
                    BindingFlags.Public | BindingFlags.Instance);
                if (f != null) f.SetValue(dst, f.GetValue(src));
            }
        }

        static AudioSource CloneAudioSource(AudioSource src, GameObject go)
        {
            var a = go.AddComponent<AudioSource>();
            if (src == null) return a;
            // Unity 6 managed AudioSource API only exposes these members;
            // playOnAwake defaults to true, which is what we want (the Motor
            // component drives volume/pitch every frame).
            a.clip = src.clip;
            a.loop = true;
            a.volume = 0f;
            a.pitch = src.pitch;
            return a;
        }
    }

    // Optional self-test (Pr0pCustomModel.ini: AutoTest=1): waits for the
    // main menu, starts a local time race with the quad config named
    // "HQ 51mm 1S Micro Freestyle", then verifies the model clone got instantiated.
    class AutoTestRunner : MonoBehaviour
    {
        IEnumerator Start()
        {
            TrackLoader tl = null;
            for (int i = 0; i < 120; i++)
            {
                tl = MonoBehaviourSingleton<TrackLoader>.nullableInstance;
                if (tl != null && !tl.loading &&
                    tl.sceneMode == SceneMode.MainMenu)
                    break;
                yield return new WaitForSeconds(1f);
            }
            if (tl == null)
            {
                Log.Write("autotest: TrackLoader never appeared");
                yield break;
            }
            Log.Write("autotest: in menu, starting local race");
            tl.SetMode(SceneMode.LocalTimeRace);
            tl.SetSceneName("GreenField");
            tl.SetQuadConfig("HQ 51mm 1S Micro Freestyle");
            tl.Load();
            yield return new WaitForSeconds(20f);

            var clone = GameObject.Find(Loader.ModelKey + "(Clone)");
            Log.Write("autotest: model clone found=" + (clone != null));
            if (clone != null)
            {
                var rends = clone.GetComponentsInChildren<Renderer>();
                int vis = 0;
                var bb = new Bounds();
                bool first = true;
                foreach (var r in rends)
                {
                    if (r.enabled) vis++;
                    if (r is MeshRenderer || r is MeshFilter)
                    {
                        if (first) { bb = r.bounds; first = false; }
                        else bb.Encapsulate(r.bounds);
                    }
                }
                Log.Write($"autotest: renderers={rends.Length} " +
                          $"enabled={vis} boundsSize={bb.size}");
                var motors = clone.GetComponentsInChildren<Motor>();
                Log.Write($"autotest: motors={motors.Length} " +
                          (motors.Length > 0
                              ? $"m1.model={motors[0].model?.name} " +
                                $"mat={motors[0].material?.name} " +
                                $"prop={motors[0].prop != null}"
                              : ""));
                var mc = clone.GetComponent<ModelConfig>();
                Log.Write("autotest: ModelConfig on clone=" +
                          (mc != null) + " camPos=" +
                          (mc != null && mc.cameraPosition != null));

                // --- fix 3: clone must sit at the parent origin (menu
                // preview uses the same Instantiate(config, parent) path)
                Log.Write($"autotest: clone localPos=" +
                          $"{clone.transform.localPosition} " +
                          $"PASS={clone.transform.localPosition.magnitude < 0.5f}");
                // simulate the QuadPreview path directly
                var tmp = new GameObject("PreviewSim");
                var sim = UnityEngine.Object.Instantiate(mc, tmp.transform);
                bool simPass =
                    sim.transform.localPosition.magnitude < 0.5f &&
                    sim.gameObject.activeInHierarchy;
                Log.Write($"autotest: preview clone localPos=" +
                          $"{sim.transform.localPosition} " +
                          $"active={sim.gameObject.activeInHierarchy} " +
                          $"PASS={simPass}");
                UnityEngine.Object.Destroy(tmp);

                // --- fix 2: pivot origin == prop mesh bounds center, and
                // bounds stay fixed while the motor spins the pivot
                var pivotT = clone.transform.Find("prop1_pivot");
                if (pivotT == null)
                {
                    foreach (Transform ch in clone.transform)
                        if (ch.name == "prop1_pivot") pivotT = ch;
                }
                if (pivotT != null)
                {
                    var pr = pivotT.GetComponentInChildren<Renderer>();
                    float pivotErr = (pivotT.position - pr.bounds.center)
                        .magnitude;
                    Log.Write($"autotest: pivot1 pos={pivotT.position} " +
                              $"meshCenter={pr.bounds.center} " +
                              $"err={pivotErr * 1000f:F2}mm " +
                              $"PASS={pivotErr < 0.002f}");
                    // force the motor to spin and watch the mesh bounds
                    var mot = pivotT.GetComponent<Motor>();
                    Vector3 c0 = pr.bounds.center;
                    Vector3 drift = Vector3.zero;
                    for (int f = 0; f < 30; f++)
                    {
                        mot.currentRpm = 9000f;
                        yield return null;
                        var d = pr.bounds.center - c0;
                        drift = new Vector3(
                            Mathf.Max(drift.x, Mathf.Abs(d.x)),
                            Mathf.Max(drift.y, Mathf.Abs(d.y)),
                            Mathf.Max(drift.z, Mathf.Abs(d.z)));
                    }
                    mot.currentRpm = 0f;
                    Log.Write($"autotest: spin bounds drift=" +
                              $"{drift * 1000f} mm " +
                              $"PASS={drift.magnitude < 0.002f}");
                }
                else
                {
                    Log.Write("autotest: FAIL prop1_pivot not found");
                }

                // --- fix 1: camera anchor must sit on the same side of the
                // root as the donor's (donorCamZ captured at build time),
                // AND the clone root must be unrotated — Quad overwrites
                // fpvCameraTransform.localRotation every frame, so the
                // anchor's forward is only +Z if the root rotation is
                // identity.
                float camZ = mc.cameraPosition.localPosition.z;
                // trimmed API has no Quaternion.Angle: angle to identity
                // is 2*acos(|w|)
                float qw = Mathf.Abs(clone.transform.localRotation.w);
                float rootRotErr = 2f * Mathf.Acos(
                    Mathf.Clamp(qw, -1f, 1f)) * 57.29578f;
                bool camPass = Mathf.Sign(Loader.DonorCamZ) ==
                               Mathf.Sign(camZ) && rootRotErr < 1f;
                Log.Write($"autotest: camZ donor={Loader.DonorCamZ} " +
                          $"ours={camZ} rootRotErr={rootRotErr:F2}deg " +
                          $"PASS={camPass}");

                // --- fix 3b: quad present in the selection-screen path
                var loader2 = Loader.ConfigLoader;
                bool inList = false;
                if (loader2 != null)
                {
                    foreach (var e in loader2.quadConfigs.entries)
                        if (e.value.model == Loader.ModelKey &&
                            e.value.modelConfig == Loader.Template)
                            inList = true;
                }
                Log.Write("autotest: quad in browser list path PASS=" +
                          inList);
            }
        }
    }
}
