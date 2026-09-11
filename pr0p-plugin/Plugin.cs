// Pr0pCustomModel — doorstop-loaded plugin for pr0p (Unity 6000.2, Mono).
//
// Originally designed as a BepInEx plugin, but neither BepInEx 5 nor 6 can
// bootstrap on this build: the game's trimmed corlib is missing
// System.Reflection.Module.GetPEKind and System.Linq.Enumerable.Concat,
// which BepInEx's preloader/chainloader require. So this dll is loaded
// DIRECTLY by UnityDoorstop: libdoorstop calls Doorstop.Entrypoint.Start(),
// which waits for Pr0Drone.dll to load, Harmony-patches
// QuadConfigLoader.Init, builds a ModelConfig from a GLB file and registers
// it in QuadConfigLoader.modelConfigs so a quad JSON "model" field can
// reference it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using HarmonyLib;
using Pr0Drone;
using UnityEngine;

// Doorstop invokes `static void Doorstop.Entrypoint.Start()`.
namespace Doorstop
{
    public static class Entrypoint
    {
        public static void Start()
        {
            try { File.WriteAllText("/tmp/pr0p_doorstop_trace.log",
                "entrypoint hit " + DateTime.Now + "\n"); } catch { }
            Trace("Start: domain=" +
                  AppDomain.CurrentDomain.FriendlyName +
                  " asm=" + Assembly.GetExecutingAssembly().Location);
            try
            {
                Pr0pCustomModel.Loader.Init();
                Trace("Init returned");
            }
            catch (Exception e)
            {
                Trace("entrypoint failed: " + e);
                Pr0pCustomModel.Log.Write("entrypoint failed: " + e);
            }
        }

        public static void Trace(string msg)
        {
            try
            {
                File.AppendAllText(
                    Path.Combine(
                        Path.GetDirectoryName(
                            Assembly.GetExecutingAssembly().Location),
                        "Pr0pCustomModel.trace.log"),
                    $"[{DateTime.Now:HH:mm:ss.fff}] {msg}\n");
            }
            catch { }
        }
    }
}

namespace Pr0pCustomModel
{
    public static class Log
    {
        static string path;
        public static void Init(string pluginDir)
        {
            path = Path.Combine(pluginDir, "Pr0pCustomModel.log");
            File.WriteAllText(path, "");
        }
        public static void Write(string msg)
        {
            try
            {
                if (path != null)
                    File.AppendAllText(path,
                        $"[{DateTime.Now:HH:mm:ss}] {msg}\n");
            }
            catch { }
            try { Debug.Log("[Pr0pCustomModel] " + msg); } catch { }
        }
    }

    public static class Loader
    {
        public static string GlbPath;
        public static string ModelKey = "hq-51mm-whoop";
        public static float ModelScale = 0.001f;
        public static string DonorModel = "vtx-slayer-3";
        static bool registered;

        public static void Init()
        {
            Doorstop.Entrypoint.Trace("Loader.Init enter");
            string pluginDir = Path.GetDirectoryName(
                Assembly.GetExecutingAssembly().Location);
            Log.Init(pluginDir);
            GlbPath = Path.Combine(pluginDir, "hq-51mm-whoop.glb");
            LoadConfig(Path.Combine(pluginDir, "Pr0pCustomModel.ini"));
            Log.Write($"init; glb={GlbPath} key={ModelKey} " +
                      $"scale={ModelScale} donor={DonorModel}");

            // resolve dependencies (e.g. 0Harmony.dll) from the plugin dir
            AppDomain.CurrentDomain.AssemblyResolve += (s, e) =>
            {
                var f = Path.Combine(pluginDir,
                    new AssemblyName(e.Name).Name + ".dll");
                return File.Exists(f) ? Assembly.LoadFile(f) : null;
            };

            // Pr0Drone.dll may already be loaded or load later — handle both.
            foreach (var a in AppDomain.CurrentDomain.GetAssemblies())
                if (a.GetName().Name == "Pr0Drone")
                {
                    Patch();
                    return;
                }
            AppDomain.CurrentDomain.AssemblyLoad += (s, e) =>
            {
                if (e.LoadedAssembly.GetName().Name == "Pr0Drone")
                    Patch();
            };
        }

        static void LoadConfig(string iniPath)
        {
            try
            {
                if (!File.Exists(iniPath)) return;
                foreach (var line in File.ReadAllLines(iniPath))
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
                    }
                }
            }
            catch (Exception e) { Log.Write("config read failed: " + e); }
        }

        static void Patch()
        {
            try
            {
                var h = new Harmony("dev.pr0p.custommodel");
                h.PatchAll(typeof(Loader).Assembly);
                Log.Write("harmony patches applied");
            }
            catch (Exception e) { Log.Write("patch failed: " + e); }
        }

        internal static void Register(QuadConfigLoader loader)
        {
            if (registered) return;
            registered = true;
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
            }
            catch (Exception ex)
            {
                Log.Write("Register failed: " + ex);
            }
        }
    }

    [HarmonyPatch(typeof(QuadConfigLoader), "Init")]
    static class InitPatch
    {
        static void Postfix(QuadConfigLoader __instance)
            => Loader.Register(__instance);
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
            }
            var propRenderer = donorMotors[0].model
                .GetComponentInChildren<Renderer>();
            var propMat = propRenderer != null
                ? propRenderer.sharedMaterial : null;
            Log.Write($"donor '{donorKey}': propMat=" +
                      $"{(propMat != null ? propMat.name : "null")}");

            var res = GlbImporter.Load(glbPath);
            var root = res.root;
            root.name = key;
            UnityEngine.Object.DontDestroyOnLoad(root);
            root.transform.localScale = Vector3.one * scale;
            // park the template far below the world so it isn't visible;
            // Quad.LoadQuadModel Instantiates it under the quad transform.
            root.transform.position = new Vector3(0f, -10000f, 0f);

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

            // props: collider + Propeller + Motor (spinning node)
            var propGOs = new GameObject[4];
            for (int i = 0; i < 4; i++)
            {
                string name = "prop" + (i + 1);
                if (!res.nodes.TryGetValue(name, out var go))
                {
                    Log.Write("missing prop node " + name);
                    return null;
                }
                propGOs[i] = go;
                BuildProp(go, res.meshes[go], donorMotors[i],
                          donorProps[i], propMat, MotorDirs[i], colliders);
            }
            mc.motor1 = propGOs[0];
            mc.motor2 = propGOs[1];
            mc.motor3 = propGOs[2];
            mc.motor4 = propGOs[3];

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
            Log.Write($"model built: {res.nodes.Count} nodes, " +
                      $"{colliders.Count} colliders");
            return mc;
        }

        static void BuildProp(GameObject go, Mesh mesh, Motor donorMotor,
            Propeller donorProp, Material propMat, float dir,
            List<Collider> colliders)
        {
            // thin box approximating the prop disk
            var bc = go.AddComponent<BoxCollider>();
            var size = mesh.bounds.size;
            bc.size = new Vector3(size.x,
                Mathf.Max(size.y, 1.5f), size.z); // mm units
            bc.center = mesh.bounds.center;
            colliders.Add(bc);

            // use the donor prop material so the _RPM spin shader works
            var rend = go.GetComponentInChildren<Renderer>();
            if (propMat != null && rend != null)
                rend.sharedMaterial = propMat;

            // add components while inactive so Motor.OnEnable doesn't fire
            // before its fields are wired
            bool wasActive = go.activeSelf;
            go.SetActive(false);

            var prop = go.AddComponent<Propeller>();
            if (donorProp != null)
            {
                prop.soundPitch = donorProp.soundPitch;
                prop.durability = donorProp.durability;
                if (donorProp.impactEffect != null)
                {
                    var fx = UnityEngine.Object.Instantiate(
                        donorProp.impactEffect.gameObject, go.transform);
                    prop.impactEffect =
                        fx.GetComponent<ParticleImpactEffect>();
                }
                if (donorProp.soundEffect != null)
                {
                    var fx = UnityEngine.Object.Instantiate(
                        donorProp.soundEffect.gameObject, go.transform);
                    prop.soundEffect =
                        fx.GetComponent<CollisionSoundEffect>();
                }
            }

            var motor = go.AddComponent<Motor>();
            CopyMotorScalars(donorMotor, motor);
            motor.prop = prop;
            motor.model = go; // material pulled from this GO's renderer
            motor.rotationDir = dir;
            motor.soundIdle = CloneAudioSource(donorMotor.soundIdle, go);
            motor.soundMid = CloneAudioSource(donorMotor.soundMid, go);
            motor.soundHigh = CloneAudioSource(donorMotor.soundHigh, go);
            if (donorMotor.burnedEffect != null)
            {
                var fx = UnityEngine.Object.Instantiate(
                    donorMotor.burnedEffect.gameObject, go.transform);
                motor.burnedEffect =
                    fx.GetComponent<ParticleBurnedEffect>();
            }

            if (wasActive) go.SetActive(true);
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
}
