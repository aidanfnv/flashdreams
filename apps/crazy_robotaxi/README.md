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
engine validates named road profiles, snaps element ports,
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
    curb_offset_m: 0.6
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

`lane_width_m` controls routing and lane rails. `curb_offset_m` adds paved
roadside clearance beyond the outer lane rail on each side before the physical
curb; it defaults to zero. Profiles with more than two lanes may provide
`divider_markings`, ordered from the leftmost adjacent lane pair to the
rightmost, to distinguish white same-direction dividers from the yellow
opposing-direction centerline.

A `boulevard` uses the same straight or constant-radius geometry as a
`road_segment` but remains explicit in the semantic map. Intersections can
override individual port profiles with `geometry.port_profiles`; this lets a
four-lane boulevard connect to narrower neighborhood streets without discarding
profile compatibility checks.

```yaml
profiles:
  boulevard:
    lane_width_m: 3.6
    curb_offset_m: 0.6
    lanes: [backward, backward, forward, forward]
    speed_limit_mps: 15.6
    curb: true
    lane_marking: {style: DASHED_SINGLE, color: WHITE}
    divider_markings:
      - {style: DASHED_SINGLE, color: WHITE}
      - {style: SOLID_GROUP, color: YELLOW}
      - {style: DASHED_SINGLE, color: WHITE}
elements:
  - id: boulevard_crossing
    type: intersection
    profile: boulevard
    geometry:
      kind: four_way
      port_profiles: {north: neighborhood, south: neighborhood}
```

Parking destinations use three explicit elements. A `parking_lot_opening`
transitions from its `geometry.road_profile` to its access `profile`; connect
its `road` and `access` ports to the public road and driveway respectively. A
straight `driveway` carries the access profile to a rectangular `parking_lot`,
whose `width_m` and `depth_m` define its enclosed surface. The lot compiler
cuts a curb gap only at a connected `entrance`, creates a two-way access aisle,
links the aisle directions with an internal turnaround route, and emits painted
parking-space dividers on both sides of the aisle. `parking_space_width_m`
controls the bay spacing and defaults to 2.7 meters. In the compiled BEV, the
entire lot surface is emitted as a green `ROI_POLYGON_ROADNET_MASK`, matching
the original map's representation of unlaned drivable areas; it is not emitted
as an intersection.

```yaml
profiles:
  parking_access:
    lane_width_m: 3.2
    curb_offset_m: 0.4
    lanes: [backward, forward]
    speed_limit_mps: 5.5
    curb: true
    lane_marking: {style: VIRTUAL, color: WHITE}
elements:
  - id: lot_opening
    type: parking_lot_opening
    profile: parking_access
    geometry: {length_m: 5, road_profile: neighborhood}
    attach: {port: road, to: main.end}
  - id: lot_driveway
    type: driveway
    profile: parking_access
    geometry: {kind: straight, length_m: 10}
    attach: {port: start, to: lot_opening.access}
  - id: neighborhood_lot
    type: parking_lot
    profile: parking_access
    geometry: {width_m: 26, depth_m: 32, parking_space_width_m: 2.7}
    attach: {port: entrance, to: lot_driveway.end}
```

Exactly one element uses `pose`; attached element transforms are derived from
their ports. Positive arc sweeps turn left and negative sweeps turn right.
Seed paths are relative to the YAML file. The compiler cache key includes the
YAML, every referenced seed, and the compiler version, so edits rebuild the
private archive automatically on the next load.

The current schema supports straight and constant-radius curved road segments,
boulevards, mixed-profile T and four-way intersections, driveways,
profile-transitioning parking-lot openings, marked and bounded parking lots,
flat ground, and per-spawn visual variants. Elevation and freeform splines are
planned extensions.

The bundled maps reuse the existing OmniDreams seed image. The minimal loop is
the default; `boulevard_district.robotaxi.yaml` is a clean, game-oriented
reinterpretation of the original Quiet Suburban Boulevard around its starting
area, with connected neighborhood blocks and routed parking lots. Incidental
legacy roadnet masks are omitted rather than treated as parking destinations.
The authored geometry is not expected to match the seed exactly, so the first
generated frames may visibly adjust toward the selected semantic map.
