// Injects a call to Pr0pCustomModel.Loader.OnQuadConfigLoaderInit(this)
// at the end of Pr0Drone.QuadConfigLoader.Init() in Pr0Drone.dll.
// Usage: patcher <path-to-Pr0Drone.dll>
using System;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

class Resolver : IAssemblyResolver
{
    readonly string dir;
    internal AssemblyDefinition self;
    public Resolver(string dir) { this.dir = dir; }
    public AssemblyDefinition Resolve(AssemblyNameReference name)
        => Resolve(name, new ReaderParameters());
    public AssemblyDefinition Resolve(AssemblyNameReference name,
                                      ReaderParameters p)
    {
        var f = Path.Combine(dir, name.Name + ".dll");
        if (File.Exists(f))
            return AssemblyDefinition.ReadAssembly(f,
                new ReaderParameters { AssemblyResolver = this });
        return self;
    }
    public void Dispose() { }
}

class P
{
    static int Main(string[] args)
    {
        var path = args[0];
        var pluginPath = args[1];
        var dir = Path.GetDirectoryName(path);
        var resolver = new Resolver(dir);
        var asm = AssemblyDefinition.ReadAssembly(path,
            new ReaderParameters { AssemblyResolver = resolver,
                                   InMemory = true });
        resolver.self = asm;

        var type = asm.MainModule.Types
            .First(t => t.Name == "QuadConfigLoader");
        var init = type.Methods.First(m => m.Name == "Init");

        // already patched?
        var pluginRef = new AssemblyNameReference("Pr0pCustomModel",
                                                  new Version(1, 0, 0, 0));
        if (!asm.MainModule.AssemblyReferences.Any(
                r => r.Name == "Pr0pCustomModel"))
            asm.MainModule.AssemblyReferences.Add(pluginRef);
        var pluginAsm = AssemblyDefinition.ReadAssembly(pluginPath,
            new ReaderParameters { AssemblyResolver = resolver });
        var loaderType = pluginAsm.MainModule.Types
            .First(t => t.FullName == "Pr0pCustomModel.Loader");
        var hook = loaderType.Methods
            .First(m => m.Name == "OnQuadConfigLoaderInit");
        var hookRef = asm.MainModule.ImportReference(hook);

        if (init.Body.Instructions.Any(i =>
                i.Operand is MethodReference mr &&
                mr.Name == "OnQuadConfigLoaderInit"))
        {
            Console.WriteLine("already patched");
            return 0;
        }

        var il = init.Body.GetILProcessor();
        // find last real ret (there may be an early ret inside `if`)
        var ret = init.Body.Instructions.Last(i => i.OpCode == OpCodes.Ret);
        il.InsertBefore(ret, Instruction.Create(OpCodes.Ldarg_0));
        il.InsertBefore(ret, Instruction.Create(OpCodes.Call, hookRef));
        asm.Write(path + ".patched");
        asm.Dispose();
        pluginAsm.Dispose();
        File.Copy(path + ".patched", path, true);
        File.Delete(path + ".patched");
        Console.WriteLine("patched " + path);
        return 0;
    }
}
