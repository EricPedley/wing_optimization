#!/bin/sh
# Build + install Pr0pCustomModel into ~/programs/pr0p.
#
# Injection: patches pr0p_Data/Managed/Pr0Drone.dll (backup kept as
# Pr0Drone.dll.orig) so that QuadConfigLoader.Init() calls
# Pr0pCustomModel.Loader.OnQuadConfigLoaderInit(this). The plugin dll goes
# into pr0p_Data/Managed/ so the Mono runtime resolves it normally.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GAME=${PR0P_DIR:-$HOME/programs/pr0p}
GLB=${GLB:-$HERE/../model/hq-51mm-micro.glb}
QUAD_JSON="$HOME/.config/unity3d/sigsegowl/pr0p/config/quad/hq-51mm-micro.json"
MANAGED="$GAME/pr0p_Data/Managed"
MODDIR="$GAME/pr0p_mods"

cd "$HERE"
dotnet build -c Release
dotnet build patcher -c Release

cp bin/Release/net472/Pr0pCustomModel.dll "$MANAGED/"
mkdir -p "$MODDIR"
cp "$GLB" "$MODDIR/hq-51mm-micro.glb"

# clean up earlier doorstop/bepinex attempts
rm -rf "$GAME/BepInEx" "$GAME/run_bepinex.sh" "$GAME/libdoorstop.so" \
       "$GAME/run_plugin.sh" "$GAME/.doorstop_version" \
       "$MODDIR/0Harmony.dll" "$MODDIR/HarmonyXInterop.dll" \
       "$MODDIR/MonoMod.RuntimeDetour.dll" "$MODDIR/MonoMod.Utils.dll" \
       "$MODDIR/m.dll" "$GAME"/preloader_*.log 2>/dev/null || true

if [ ! -f "$MANAGED/Pr0Drone.dll.orig" ]; then
    cp "$MANAGED/Pr0Drone.dll" "$MANAGED/Pr0Drone.dll.orig"
fi
dotnet patcher/bin/Release/net8.0/Patcher.dll \
    "$MANAGED/Pr0Drone.dll" \
    "$MANAGED/Pr0pCustomModel.dll"

# point the quad config at the custom model key
if [ -f "$QUAD_JSON" ]; then
    sed -i 's/"model": *"[^"]*"/"model": "hq-51mm-micro"/' "$QUAD_JSON"
    echo "updated $QUAD_JSON"
fi

echo "done. Launch the game normally:  $GAME/pr0p.x86_64"
echo "log: $MODDIR/Pr0pCustomModel.log"
echo "uninstall: mv $MANAGED/Pr0Drone.dll.orig $MANAGED/Pr0Drone.dll && rm $MANAGED/Pr0pCustomModel.dll"
