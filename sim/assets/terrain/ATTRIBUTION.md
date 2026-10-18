# Terrain data

`tiles/` holds elevation tiles from the AWS Open Data "Terrain Tiles"
dataset (Terrarium encoding), fetched by `sim/terrain.py`:
https://registry.opendata.aws/terrain-tiles/

For this site (San Diego, CA) the source is the USGS 3D Elevation Program
(3DEP), a public-domain US Government dataset. The dataset's attribution
requirements: https://github.com/tilezen/joerd/blob/master/docs/attribution.md

`local.bin`, `site.json` and `../sky/placeholder_equirect.png` are derived from
these tiles.

# Map data

`osm.json` is OpenStreetMap data fetched from the Overpass API by
`sim/scenery.py`: © OpenStreetMap contributors, available under the Open
Database License (https://www.openstreetmap.org/copyright). `scenery.bin` (the
houses, transmission towers/wires and light poles) is derived from it.
