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

## Configuration files

The standalone game keeps its portable configuration in three independent,
strict YAML documents:

- `*.robotaxi.yaml` describes one map, including compiler geometry and paint.
- `default_renderer.yaml` describes primary-camera and BEV rendering.
- `default_game.yaml` describes rules, scoring, controls, taxi dimensions, and
  arcade physics.

The packaged renderer and game files are used when no path is supplied. Select
edited copies independently:

```bash
flashdreams-run crazy-robotaxi \
  --renderer-config /path/to/renderer.yaml \
  --game-config /path/to/game.yaml \
  --auto-start True
```

The dedicated executable accepts the same two arguments. Existing explicit CLI
tuning flags override YAML values. The world-model manifest remains separate
because it configures inference rather than the map, renderer, or game.

All three YAML formats reject missing and unknown fields. The packaged files are
complete reference configurations intended to be copied and edited.

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
compiler:
  sample_spacing_m: 2.0
  ground_margin_m: 20.0
  intersection_connector_samples: 8
  parking_lot:
    turnaround_width_multiplier: 0.75
    turnaround_min_depth_m: 5.0
    turnaround_control_inset_m: 0.75
profiles:
  neighborhood:
    lane_width_m: 3.6
    curb_offset_m: 0.6
    lanes: [backward, forward]
    speed_limit_mps: 13.4
    curb: true
    lane_marking: {style: DASHED_SINGLE, color: WHITE}
    divider_markings:
      - {style: SOLID_GROUP, color: YELLOW}
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
connections: []
spawns:
  - id: taxi_start
    element: main
    lane: 1
    distance_m: 5
    variants:
      default:
        image: seed.png
        prompt: A forward-facing view from a taxi in a quiet neighborhood at daylight.
```

`lane_width_m` controls routing and lane rails. `curb_offset_m` adds paved
roadside clearance beyond the outer lane rail on each side before the physical
curb. Every profile provides one `divider_markings` entry per adjacent lane
pair, ordered from the leftmost adjacent lane pair to the
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

Road segments and boulevards also support cubic Bézier centerlines. The start
point is the element-local origin; the three authored control points are the
two Bézier controls followed by the endpoint. Port headings come from the
curve's endpoint tangents, so curves attach and close through the same named
port mechanism as straight and circular roads.

```yaml
- id: arterial_sweep
  type: boulevard
  profile: boulevard
  geometry:
    kind: cubic_bezier
    control_points:
      - {x_m: 30, y_m: 0}
      - {x_m: 65, y_m: -20}
      - {x_m: 80, y_m: -50}
  attach: {port: start, to: gateway.southeast}
```

Use a `freeform` intersection for skewed or multi-angle junctions. Its surface,
connector center, and ports use local coordinates. Each port must lie on a
surface edge, point perpendicular and outward from that edge, and have enough
edge length for its profile's full paved width. Connected ports cut matching
curb openings; unconnected edges remain physically bounded.

```yaml
- id: gateway
  type: intersection
  profile: boulevard
  geometry:
    kind: freeform
    surface:
      - {x_m: 0, y_m: -10}
      - {x_m: 20, y_m: -10}
      - {x_m: 20, y_m: 10}
      - {x_m: 0, y_m: 10}
    connector_center: {x_m: 10, y_m: 0}
    ports:
      west: {x_m: 0, y_m: 0, heading_deg: 180, profile: boulevard}
      east: {x_m: 20, y_m: 0, heading_deg: 0, profile: boulevard}
      north: {x_m: 10, y_m: 10, heading_deg: 90, profile: neighborhood}
```

A `cul_de_sac` closes an otherwise unfinished road with a curb-bounded circular
turnaround. `radius_m` controls the bulb and `neck_length_m` carries the curb
back to its single `entrance` port. The compiler creates a navigable U-turn
inside the bulb, but its lane rails are virtual: the ClipGT BEV contains the
road surface and curb boundary without a visible centerline or lane dividers.

```yaml
- id: neighborhood_turnaround
  type: cul_de_sac
  profile: neighborhood
  geometry: {radius_m: 10, neck_length_m: 10}
  attach: {port: entrance, to: unfinished_street.end}
```

Parking destinations use three explicit elements. A `parking_lot_opening`
transitions from its `geometry.road_profile` to its access `profile`; connect
its `road` and `access` ports to the public road and driveway respectively. A
straight `driveway` carries the access profile to a rectangular `parking_lot`,
whose `width_m` and `depth_m` define its enclosed surface. The lot compiler
cuts a curb gap only at a connected `entrance`, creates a two-way access aisle,
and links the aisle directions with an internal turnaround route. In the
compiled BEV, the entire lot surface is emitted as a green
`ROI_POLYGON_ROADNET_MASK`. Parking-space dividers are intentionally omitted:
ClipGT would encode them as ordinary lane lines, which can condition the model
to produce bike lanes or turn lanes. The lot is not emitted as an intersection.

```yaml
profiles:
  parking_access:
    lane_width_m: 3.2
    curb_offset_m: 0.4
    lanes: [backward, forward]
    speed_limit_mps: 5.5
    curb: true
    lane_marking: {style: VIRTUAL, color: WHITE}
    divider_markings:
      - {style: VIRTUAL, color: WHITE}
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
    geometry: {width_m: 26, depth_m: 32}
    attach: {port: entrance, to: lot_driveway.end}
```

Exactly one element uses `pose`; attached element transforms are derived from
their ports. Positive arc sweeps turn left and negative sweeps turn right.
Seed paths are relative to the YAML file. The compiler cache key includes the
YAML, every referenced seed, and the compiler version, so edits rebuild the
private archive automatically on the next load.

The current schema supports straight, constant-radius, and cubic Bézier road
segments and boulevards; regular and freeform mixed-profile intersections;
driveways; profile-transitioning parking-lot openings; bounded parking lots;
curb-bounded unmarked cul-de-sacs; flat ground; and per-spawn visual variants.
Elevation remains a planned extension.

The bundled maps reuse the existing OmniDreams seed image. The minimal loop is
the default; `boulevard_district.robotaxi.yaml` is a clean, game-oriented
reinterpretation of the original Quiet Suburban Boulevard around its starting
area and its full eastern district, with connected neighborhood blocks, an
irregular eastern gateway, two northern return districts, a southern commercial
grid, routed parking lots, and cul-de-sacs replacing the original map's cut-off
roads. The western grade-separated highway and ramps are deliberately omitted:
flat overlapping roads are ambiguous in both the current schema and the world
model's BEV conditioning. Incidental legacy roadnet masks are omitted rather
than treated as parking destinations.
Static prompts describe visual setting and atmosphere rather than map topology;
the BEV conditioning remains the source of truth as maps change. The authored
geometry is not expected to match the seed exactly, so the first generated
frames may visibly adjust toward the selected semantic map.
