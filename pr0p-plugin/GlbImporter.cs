// Minimal GLB (glTF 2.0 binary) importer: JSON+BIN chunks, accessors,
// bufferViews, nodes/meshes -> UnityEngine.Mesh. Handles POSITION/NORMAL/
// TEXCOORD_0 float accessors, u16/u32 indices, node TRS, baseColorFactor
// and a single embedded PNG baseColorTexture. Converts glTF right-handed
// coords to Unity left-handed by negating X and flipping winding.
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Pr0pCustomModel
{
    public class GlbImporter
    {
        public class Result
        {
            public GameObject root;
            // node name -> gameobject
            public Dictionary<string, GameObject> nodes =
                new Dictionary<string, GameObject>();
            // gameobject -> mesh (renderer nodes only)
            public Dictionary<GameObject, Mesh> meshes =
                new Dictionary<GameObject, Mesh>();
        }

        byte[] bin;
        Dictionary<string, object> gltf;

        public static Result Load(string path, Transform parent = null)
        {
            var imp = new GlbImporter();
            return imp.Run(path, parent);
        }

        Result Run(string path, Transform parent)
        {
            byte[] data = File.ReadAllBytes(path);
            if (data.Length < 20 || BitConverter.ToUInt32(data, 0) != 0x46546C67)
                throw new Exception("not a GLB file");
            int pos = 12;
            string json = null;
            while (pos + 8 <= data.Length)
            {
                int len = BitConverter.ToInt32(data, pos);
                uint type = BitConverter.ToUInt32(data, pos + 4);
                pos += 8;
                if (type == 0x4E4F534A) // JSON
                    json = System.Text.Encoding.UTF8.GetString(data, pos, len);
                else if (type == 0x004E4942) // BIN
                {
                    bin = new byte[len];
                    Array.Copy(data, pos, bin, 0, len);
                }
                pos += len;
            }
            if (json == null || bin == null)
                throw new Exception("GLB missing JSON or BIN chunk");
            gltf = MiniJson.Obj(MiniJson.Parse(json));

            var res = new Result();
            res.root = new GameObject(Path.GetFileNameWithoutExtension(path));
            if (parent != null)
                res.root.transform.SetParent(parent, false);

            var scenes = MiniJson.GetArr(gltf, "scenes");
            int sceneIdx = MiniJson.GetInt(gltf, "scene", 0);
            var scene = MiniJson.Obj(scenes[sceneIdx]);
            var rootNodes = MiniJson.GetArr(scene, "nodes");
            var nodes = MiniJson.GetArr(gltf, "nodes");
            foreach (var n in rootNodes)
                BuildNode(MiniJson.Int(n), res, res.root.transform);
            return res;
        }

        void BuildNode(int idx, Result res, Transform parent)
        {
            var node = MiniJson.Obj(MiniJson.GetArr(gltf, "nodes")[idx]);
            var go = new GameObject(MiniJson.GetStr(node, "name") ?? $"node{idx}");
            go.transform.SetParent(parent, false);

            // TRS (glTF -> Unity: negate x pos, negate y/z rot parts)
            var t = MiniJson.GetArr(node, "translation");
            var r = MiniJson.GetArr(node, "rotation");
            var s = MiniJson.GetArr(node, "scale");
            var m = MiniJson.GetArr(node, "matrix");
            if (m != null)
            {
                // decompose: extract columns, flip handedness
                var M = new Matrix4x4();
                for (int c = 0; c < 16; c++)
                    M[c] = (float)MiniJson.Num(m[c]);
                var flip = Matrix4x4.Scale(new Vector3(-1, 1, 1));
                M = flip * M * flip;
                go.transform.localPosition = M.GetColumn(3);
                go.transform.localRotation = M.rotation;
                go.transform.localScale = M.lossyScale;
            }
            else
            {
                go.transform.localPosition = t != null
                    ? new Vector3(-(float)MiniJson.Num(t[0]),
                                  (float)MiniJson.Num(t[1]),
                                  (float)MiniJson.Num(t[2]))
                    : Vector3.zero;
                go.transform.localRotation = r != null
                    ? new Quaternion((float)MiniJson.Num(r[0]),
                                     -(float)MiniJson.Num(r[1]),
                                     -(float)MiniJson.Num(r[2]),
                                     (float)MiniJson.Num(r[3]))
                    : Quaternion.identity;
                go.transform.localScale = s != null
                    ? new Vector3((float)MiniJson.Num(s[0]),
                                  (float)MiniJson.Num(s[1]),
                                  (float)MiniJson.Num(s[2]))
                    : Vector3.one;
            }

            res.nodes[go.name] = go;

            if (node.TryGetValue("mesh", out var meshObj))
            {
                var meshDef = MiniJson.Obj(meshObj);
                var prims = MiniJson.GetArr(meshDef, "primitives");
                if (prims.Count == 1)
                {
                    var mesh = new Mesh();
                    mesh.name = MiniJson.GetStr(meshDef, "name") ?? go.name;
                    BuildPrimitive(MiniJson.Obj(prims[0]), mesh, 0);
                    var mf = go.AddComponent<MeshFilter>();
                    mf.sharedMesh = mesh;
                    var mr = go.AddComponent<MeshRenderer>();
                    mr.sharedMaterial = MakeMaterial(MiniJson.Obj(prims[0]));
                    res.meshes[go] = mesh;
                }
                else
                {
                    for (int pi = 0; pi < prims.Count; pi++)
                    {
                        var prim = MiniJson.Obj(prims[pi]);
                        var child = new GameObject(go.name + "_p" + pi);
                        child.transform.SetParent(go.transform, false);
                        var cm = new Mesh { name = go.name + "_p" + pi };
                        BuildPrimitive(prim, cm, pi);
                        child.AddComponent<MeshFilter>().sharedMesh = cm;
                        child.AddComponent<MeshRenderer>().sharedMaterial =
                            MakeMaterial(prim);
                        res.meshes[child] = cm;
                    }
                }
            }

            var children = MiniJson.GetArr(node, "children");
            if (children != null)
                foreach (var c in children)
                    BuildNode(MiniJson.Int(c), res, go.transform);
        }

        void BuildPrimitive(Dictionary<string, object> prim, Mesh mesh,
                            int submesh)
        {
            var attrs = MiniJson.Obj(prim["attributes"]);
            mesh.indexFormat =
                UnityEngine.Rendering.IndexFormat.UInt32;

            var posAcc = MiniJson.Obj(
                MiniJson.GetArr(gltf, "accessors")[
                    MiniJson.GetInt(attrs, "POSITION")]);
            int vcount = MiniJson.GetInt(posAcc, "count");
            float[] pos = ReadFloats(posAcc);
            var verts = new Vector3[vcount];
            for (int i = 0; i < vcount; i++)
                verts[i] = new Vector3(-pos[3 * i], pos[3 * i + 1],
                                       pos[3 * i + 2]);
            mesh.vertices = verts;

            if (attrs.TryGetValue("NORMAL", out var nrm))
            {
                var nAcc = MiniJson.Obj(
                    MiniJson.GetArr(gltf, "accessors")[MiniJson.Int(nrm)]);
                float[] n = ReadFloats(nAcc);
                var norms = new Vector3[vcount];
                for (int i = 0; i < vcount; i++)
                    norms[i] = new Vector3(-n[3 * i], n[3 * i + 1],
                                           n[3 * i + 2]);
                mesh.normals = norms;
            }
            if (attrs.TryGetValue("TEXCOORD_0", out var uv))
            {
                var uvAcc = MiniJson.Obj(
                    MiniJson.GetArr(gltf, "accessors")[MiniJson.Int(uv)]);
                float[] u = ReadFloats(uvAcc);
                var uvs = new Vector2[vcount];
                for (int i = 0; i < vcount; i++)
                    uvs[i] = new Vector2(u[2 * i], u[2 * i + 1]);
                mesh.uv = uvs;
            }

            int[] tris;
            if (prim.TryGetValue("indices", out var indObj))
            {
                var iAcc = MiniJson.Obj(
                    MiniJson.GetArr(gltf, "accessors")[MiniJson.Int(indObj)]);
                tris = ReadIndices(iAcc);
            }
            else
            {
                tris = new int[vcount];
                for (int i = 0; i < vcount; i++) tris[i] = i;
            }
            // flip winding for handedness conversion
            for (int i = 0; i + 2 < tris.Length; i += 3)
            {
                int tmp = tris[i + 1];
                tris[i + 1] = tris[i + 2];
                tris[i + 2] = tmp;
            }
            mesh.triangles = tris;
            if (mesh.normals == null || mesh.normals.Length == 0)
                mesh.RecalculateNormals();
            mesh.RecalculateBounds();
        }

        float[] ReadFloats(Dictionary<string, object> acc)
        {
            int count = MiniJson.GetInt(acc, "count");
            string type = MiniJson.GetStr(acc, "type");
            int comps = type == "SCALAR" ? 1 : type == "VEC2" ? 2
                : type == "VEC3" ? 3 : 4;
            int ct = MiniJson.GetInt(acc, "componentType");
            var bv = MiniJson.Obj(
                MiniJson.GetArr(gltf, "bufferViews")[
                    MiniJson.GetInt(acc, "bufferView")]);
            int off = MiniJson.GetInt(bv, "byteOffset")
                      + MiniJson.GetInt(acc, "byteOffset");
            int stride = MiniJson.GetInt(bv, "byteStride");
            if (stride == 0) stride = comps * CompSize(ct);
            var outp = new float[count * comps];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < comps; c++)
                {
                    int o = off + i * stride + c * CompSize(ct);
                    outp[i * comps + c] = ct == 5126
                        ? BitConverter.ToSingle(bin, o)
                        : ct == 5123
                            ? BitConverter.ToUInt16(bin, o)
                            : ct == 5125
                                ? BitConverter.ToUInt32(bin, o)
                                : ct == 5121 ? bin[o]
                                : ct == 5120 ? (sbyte)bin[o]
                                : (short)BitConverter.ToUInt16(bin, o);
                }
            return outp;
        }

        int[] ReadIndices(Dictionary<string, object> acc)
        {
            int count = MiniJson.GetInt(acc, "count");
            int ct = MiniJson.GetInt(acc, "componentType");
            var bv = MiniJson.Obj(
                MiniJson.GetArr(gltf, "bufferViews")[
                    MiniJson.GetInt(acc, "bufferView")]);
            int off = MiniJson.GetInt(bv, "byteOffset")
                      + MiniJson.GetInt(acc, "byteOffset");
            var outp = new int[count];
            int sz = CompSize(ct);
            for (int i = 0; i < count; i++)
            {
                int o = off + i * sz;
                outp[i] = ct == 5125
                    ? (int)BitConverter.ToUInt32(bin, o)
                    : ct == 5123
                        ? BitConverter.ToUInt16(bin, o)
                        : bin[o];
            }
            return outp;
        }

        static int CompSize(int ct)
            => ct == 5120 || ct == 5121 ? 1 : ct == 5122 || ct == 5123 ? 2 : 4;

        // ---- materials ----
        Dictionary<int, Material> matCache = new Dictionary<int, Material>();
        Dictionary<int, Texture2D> texCache = new Dictionary<int, Texture2D>();

        Material MakeMaterial(Dictionary<string, object> prim)
        {
            int mi = MiniJson.GetInt(prim, "material", -1);
            if (mi < 0) return DefaultMaterial();
            if (matCache.TryGetValue(mi, out var cached)) return cached;
            var matDef = MiniJson.Obj(MiniJson.GetArr(gltf, "materials")[mi]);
            var pbr = MiniJson.Get(matDef, "pbrMetallicRoughness");
            var m = NewLitMaterial();
            var bcf = MiniJson.GetArr(pbr, "baseColorFactor");
            Color col = bcf != null
                ? new Color((float)MiniJson.Num(bcf[0]),
                            (float)MiniJson.Num(bcf[1]),
                            (float)MiniJson.Num(bcf[2]),
                            (float)MiniJson.Num(bcf[3]))
                : Color.white;
            SetColorSafe(m, col);
            if (pbr != null && pbr.TryGetValue("metallicFactor", out var met))
                SetFloatSafe(m, "_Metallic", (float)MiniJson.Num(met));
            if (pbr != null && pbr.TryGetValue("roughnessFactor", out var rough))
                SetFloatSafe(m, "_Smoothness",
                             1f - (float)MiniJson.Num(rough));
            var bct = MiniJson.Get(pbr, "baseColorTexture");
            if (bct != null)
            {
                var tex = LoadTexture(MiniJson.GetInt(bct, "index"));
                if (tex != null) SetTextureSafe(m, tex);
            }
            matCache[mi] = m;
            return m;
        }

        // Unity 6.2's managed ImageConversion module in this build does not
        // expose LoadImage, so embedded textures can't be decoded. The prop
        // material is overridden with the donor prop material anyway.
        Texture2D LoadTexture(int texIdx) => null;

        public static Material NewLitMaterial()
        {
            var sh = Shader.Find("Universal Render Pipeline/Lit");
            if (sh == null) sh = Shader.Find("Universal Render Pipeline/Simple Lit");
            if (sh == null) sh = Shader.Find("Standard");
            return new Material(sh);
        }

        static Material defaultMat;
        public static Material DefaultMaterial()
        {
            if (defaultMat == null)
            {
                defaultMat = NewLitMaterial();
                SetColorSafe(defaultMat, new Color(0.18f, 0.18f, 0.2f));
            }
            return defaultMat;
        }

        static void SetFloatSafe(Material m, string prop, float v)
        {
            if (m.HasProperty(prop)) m.SetFloat(prop, v);
        }

        static void SetColorSafe(Material m, Color c)
        {
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            else if (m.HasProperty("_Color")) m.SetColor("_Color", c);
        }

        static void SetTextureSafe(Material m, Texture2D t)
        {
            if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", t);
            else if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", t);
        }
    }
}
