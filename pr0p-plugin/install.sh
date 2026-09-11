#!/bin/sh
# Build + install Pr0pCustomModel into ~/programs/pr0p.
#
# NOTE: this is NOT a BepInEx install. BepInEx 5/6 cannot bootstrap on this
# build (the game's trimmed corlib lacks Module.GetPEKind and
# Enumerable.Concat). Instead we use UnityDoorstop (libdoorstop.so, taken
# from the BepInEx 5.4.23.4 package) to load Pr0pCustomModel.dll directly:
# it exposes Doorstop.Entrypoint.Start().
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GAME=${PR0P_DIR:-$HOME/programs/pr0p}
DOORSTOP_ZIP_URL="https://github.com/BepInEx/BepInEx/releases/download/v5.4.23.4/BepInEx_linux_x64_5.4.23.4.zip"
GLB=${GLB:-$HERE/../model/hq-51mm-whoop.glb}
QUAD_JSON="$HOME/.config/unity3d/sigsegowl/pr0p/config/quad/hq-51mm-whoop.json"
MODDIR="$GAME/pr0p_mods"

cd "$HERE"
dotnet build -c Release

# remove a previous BepInEx install if present (it cannot run on this build)
rm -rf "$GAME/BepInEx"

if [ ! -f "$MODDIR/0Harmony.dll" ]; then
    echo "fetching doorstop + harmonyx"
    tmp=$(mktemp -d)
    curl -sL -o "$tmp/be.zip" "$DOORSTOP_ZIP_URL"
    unzip -o -q "$tmp/be.zip" -d "$tmp/be"
    mkdir -p "$MODDIR"
    cp "$tmp/be/libdoorstop.so" "$GAME/"
    cp "$tmp/be/BepInEx/core/0Harmony.dll" \
       "$tmp/be/BepInEx/core/HarmonyXInterop.dll" \
       "$tmp/be/BepInEx/core/MonoMod.RuntimeDetour.dll" \
       "$tmp/be/BepInEx/core/MonoMod.Utils.dll" \
       "$MODDIR/" 2>/dev/null || true
    rm -rf "$tmp"
fi

cp bin/Release/net472/Pr0pCustomModel.dll "$MODDIR/"
cp "$GLB" "$MODDIR/hq-51mm-whoop.glb"

# launcher script
cat > "$GAME/run_plugin.sh" <<EOF
#!/bin/sh
cd "\$(dirname "\$0")"
export DOORSTOP_ENABLED=1
export DOORSTOP_TARGET_ASSEMBLY="$MODDIR/Pr0pCustomModel.dll"
export LD_LIBRARY_PATH="$GAME:\$LD_LIBRARY_PATH"
export LD_PRELOAD="libdoorstop.so\${LD_PRELOAD:+:\$LD_PRELOAD}"
exec ./pr0p.x86_64 "\$@"
EOF
chmod +x "$GAME/run_plugin.sh"

# point the quad config at the custom model key
if [ -f "$QUAD_JSON" ]; then
    sed -i 's/"model": *"[^"]*"/"model": "hq-51mm-whoop"/' "$QUAD_JSON"
    echo "updated $QUAD_JSON"
fi

echo "done. Launch with:  $GAME/run_plugin.sh"
echo "log: $MODDIR/Pr0pCustomModel.log"
