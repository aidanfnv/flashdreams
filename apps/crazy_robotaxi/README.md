# Crazy Robotaxi

Crazy Robotaxi is a standalone game built on `omnidreams-game-engine` and the
OmniDreams inference session.

Launch the native game:

```bash
flashdreams-run crazy-robotaxi
```

Select the bundled performance manifest:

```bash
flashdreams-run crazy-robotaxi \
  --world-model-manifest example_world_model_perf.yaml
```

`flashdreams-run` reserves `--manifest` for its launch-manifest format. Use
`--world-model-manifest` for the OmniDreams model manifest. Runner
booleans use explicit `True` / `False` values because the shared FlashDreams
CLI disables implicit boolean flag conversion.

The dedicated entry point exposes the application options:

```bash
crazy-robotaxi --help
```

Use `--stream-mjpeg HOST:PORT` with either entry point to run the browser HUD
instead of opening a local Vulkan window.

## Configuration files

The standalone game keeps its portable configuration in three independent,
strict YAML documents:

- `*.robotaxi.yaml` describes one map, including topology, geometry, profiles,
  compiler settings, spawns, and visual seed variants.
- `default_renderer.yaml` describes primary-camera and BEV rendering.
- `default_game.yaml` describes rules, scoring, controls, taxi dimensions, and
  arcade physics.

The packaged renderer and game files are used when no path is supplied. Select
edited copies independently:

```bash
flashdreams-run crazy-robotaxi \
  --renderer-config /path/to/renderer.yaml \
  --game-config /path/to/game.yaml
```

Existing explicit CLI tuning flags override YAML values. The world-model
manifest remains separate because it configures inference rather than the map,
renderer, or game. All three YAML formats reject missing and unknown fields.

## Node-graph game maps

Crazy Robotaxi maps use schema version 1 of the engine's node-graph format.
Roads are topological edges; intersections, road joints, cul-de-sacs, parking
lots, and driveways are explicitly posed graph nodes. Road joints provide
minimal tangent-continuous degree-two bends and own lane-count or lane-width
tapers between otherwise uniform road edges. Intersections have at least three
road arms and can apply the same taper to each inferred through-road pair
independently. Node footprints follow the bearings and centerline tangents of
the roads connected to them.

The complete, authoritative format is documented in
[`../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md`](../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md).
The bundled `minimal_loop.robotaxi.yaml` is a compact working example, while
`boulevard_district.robotaxi.yaml` recreates the original scene's surface-street
layout at its source scale, including the curved arterial split, neighborhood
grid, eastern commercial loops, cul-de-sacs, and parking lots. The elevated
highway and its on-ramps in the northwest are intentionally omitted.

Select a map with either entry point:

```bash
flashdreams-run crazy-robotaxi \
  --map /path/to/city.robotaxi.yaml
```

Validate a map or produce top-down and spawn-camera previews without loading a
model:

```bash
crazy-robotaxi-map validate /path/to/city.robotaxi.yaml
crazy-robotaxi-map preview /path/to/city.robotaxi.yaml --output city.svg
crazy-robotaxi-map preview-spawn /path/to/city.robotaxi.yaml \
  --spawn taxi_start --output taxi_start.png
```

The engine validates the topology, compiles roads and inferred parking-access
geometry, creates curb colliders, and derives directed navigation lanes. That
lane graph powers fare reachability and future NPC routing; it does not limit
where the player's taxi may physically drive. Parking-lot fare points are
sampled anywhere inside the authored polygon, while their route distance stops
at the connected driveway or intersection.

The engine's private ClipGT archive is cached under
`$FLASHDREAMS_CACHE_DIR/omnidreams-game-engine/game-maps/`. The cache key includes the YAML,
resolved geometry, compiler implementation, and referenced seed assets, so
map or compiler edits rebuild it at the next load. The archive is runtime
output, not an authoring format. Pass `--force-map-recompile` when launching
the game to rebuild each selected map once for that process even if its cache
entry is valid. The standalone equivalent is:

```bash
crazy-robotaxi-map compile /path/to/city.robotaxi.yaml --force-map-recompile
```

Parking-lot surfaces are emitted as green `ROI_POLYGON_ROADNET_MASK` regions.
Parking-space dividers remain intentionally omitted because ClipGT would encode
them as ordinary lane lines, which can condition the model to produce bike or
turn lanes. Lots do not contain inferred navigation aisles or turnarounds.

Each visual variant may omit `image`. In that case, map compilation and scene
selection use a deterministic synthetic view projected from the spawn through
the runtime front camera. It aligns roads, boundaries, curbs, and markings but
does not synthesize buildings, vegetation, traffic, or other scenery. Authors
can inspect that fallback with `preview-spawn` before choosing or generating a
checked-in image.

The bundled maps reuse the existing OmniDreams seed image. Their authored
geometry is not expected to match that image exactly, so the first generated
frames may visibly adjust toward the selected semantic map. Static prompts
describe visual setting and atmosphere rather than duplicating map topology;
the BEV remains the source of truth as maps change.
