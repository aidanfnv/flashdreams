# Local map-frame authoring

This workflow lives on the fork branch
`dev/aidanf/crazy-robotaxi/astra-map-authoring`, based on the cleaned map branch
at `aa251a65`. It is a local authoring tool, not a CI test or runtime component.

The `*_artwork.png` files are the original image-generation outputs. The
`*_initial_frame.prompt.txt` files beside the map YAMLs record their generation
prompts and compositing history. The compositor does not invoke image generation.

From the repository root, using an environment with Crazy Robotaxi, the
Omnidreams game engine, Ludus renderer, OpenCV, Pillow, NumPy, and Shapely:

```bash
python apps/crazy_robotaxi/tools/compose_map_frames.py boulevard_district
```

Other supported map names are `flashdreams_raceway`, `pacific_breakwater`,
`harbor_market`, and `mesa_gardens`.

The script overwrites the map's `*_initial_frame.png`. It projects road surfaces,
curbs, and runtime-loaded lane markings from the compiled spawn camera, reuses
the generated asphalt texture, warps scenery to fit the road edges, and resizes
and blends the hood. Its registration landmarks are specific to the saved
artwork; replacing that artwork can require adjusting the landmarks.

It also writes `*_alignment.png` and `*_spawn_geometry.png` beside the map.
These local previews are ignored by Git and excluded from the runtime package.
The source artwork and generation notes are also excluded from that package.

The assertions check static paint alignment only. They do not establish that
Omnidreams respects vehicle motion. Inspect the resulting frame and validate
motion separately before promoting an updated runtime frame to the map branch.
