# Crazy Robotaxi

Crazy Robotaxi is a standalone game built on `omnidreams-game-engine` and the
legacy OmniDreams inference session. It does not import or modify the
Interactive Drive demo.

Launch the native game:

```bash
flashdreams-run crazy-robotaxi
```

Select the bundled performance manifest and load the scene immediately:

```bash
flashdreams-run crazy-robotaxi \
  --world-model-manifest example_world_model_perf.yaml \
  --auto-start True
```

`flashdreams-run` reserves `--manifest` for its launch-manifest format. Use
`--world-model-manifest` for the legacy OmniDreams model manifest, or use the
dedicated `crazy-robotaxi` executable below. Runner booleans use explicit
`True` / `False` values because the shared FlashDreams CLI disables implicit
boolean flag conversion.

The dedicated entry point exposes the complete legacy option surface:

```bash
crazy-robotaxi --help
```

Use `--stream-mjpeg HOST:PORT` with either entry point to run the browser HUD
instead of opening a local Vulkan window.

## Semantic game maps

Crazy Robotaxi maps are authored as versioned `.robotaxi.yaml` files. The game
engine validates named road profiles, snaps road and intersection ports,
generates directed navigation lanes and curb colliders, and transparently
compiles the result into the private ClipGT archive consumed by OmniDreams.
The generated archive is cached under
`$FLASHDREAMS_CACHE_DIR/crazy-robotaxi/maps/`; it is not an authoring format.

The bundled `minimal_loop.robotaxi.yaml` map is used by default. Select another
map with either entry point:

```bash
flashdreams-run crazy-robotaxi \
  --scene /path/to/city.robotaxi.yaml \
  --auto-start True
```

Validate a map or produce a top-down SVG showing its lanes, connection ports,
and curbs without loading a model:

```bash
crazy-robotaxi-map validate /path/to/city.robotaxi.yaml
crazy-robotaxi-map preview /path/to/city.robotaxi.yaml --output city.svg
```

A map contains one anchored element; every other element attaches a named port
to an existing `element.port`. Additional `connections` close loops and are
validated for position, heading, and road-profile compatibility. Unconnected
road ports receive curb end caps automatically.

Minimal authoring shape:

```yaml
schema_version: 1
id: my-map
name: My Map
profiles:
  neighborhood:
    lane_width_m: 3.6
    lanes: [backward, forward]
    speed_limit_mps: 13.4
    curb: true
    lane_marking: {style: DASHED_SINGLE, color: WHITE}
elements:
  - id: main
    type: road_segment
    profile: neighborhood
    geometry: {kind: straight, length_m: 50}
    pose: {x_m: 0, y_m: 0, heading_deg: 0}
  - id: corner
    type: road_segment
    profile: neighborhood
    geometry: {kind: arc, radius_m: 15, sweep_deg: 90}
    attach: {port: start, to: main.end}
spawns:
  - id: taxi_start
    element: main
    lane: 1
    distance_m: 5
    variants:
      default:
        image: seed.png
        prompt: A forward-facing taxi on a neighborhood road.
```

Exactly one element uses `pose`; attached element transforms are derived from
their ports. Positive arc sweeps turn left and negative sweeps turn right.
Seed paths are relative to the YAML file. The compiler cache key includes the
YAML, every referenced seed, and the compiler version, so edits rebuild the
private archive automatically on the next load.

The current schema supports straight and constant-radius curved road segments,
T intersections, four-way intersections, flat ground, and per-spawn visual
variants. Driveways, parking lots and openings, boulevards, elevation, and
freeform splines are planned extensions.

The bundled WIP map reuses the existing OmniDreams seed image. Its geometry is
not expected to match that image, so the first generated frames may visibly
adjust toward the semantic map.
