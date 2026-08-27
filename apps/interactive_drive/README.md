<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Drive

A separate long-running native-v2 driving demo with its own
`InteractiveDriveUILoop`. The HUD contains scene and variant selection,
driving telemetry, steering-wheel and pedal sprites, post-processing controls,
and a BEV minimap. Dear ImGui builds the immediate-mode HUD and SlangPy renders
it with GPU textures; the application does not use CSS.

ClipGT2V supplies reusable scene simulation and rendering components, while the
Interactive Drive application owns its session and HUD behavior. Its world-model
binding is supplied by an integration adapter.

## Usage

Install the application, then start it with no application arguments:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams python integrations_v2/omnidreams/impl/omnidreams_singleview/tools/sync_thirdparty.py sync
uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc
```

Use `interactive-drive-omnidreams-perf` instead for the native-accelerated,
performance-tuned configuration.

The default scene downloads on first use from the gated
`nvidia/omni-dreams-scenes` Hugging Face dataset. Application arguments are
optional and follow the `--` separator:

| Argument | Description |
| --- | --- |
| `--scene PATH` | Use a local USDZ scene instead of downloading the default scene. |
| `--backend {raster,world_model}` | Render HD-map conditioning directly or run the world model. Default: `world_model`. |
| `--prompt TEXT` | Override the prompt stored in the selected scene variant. |
| `--camera NAME` | Select a camera from the scene. Default: `camera_front_wide_120fov`. |
| `--variant NAME` | Select the scene's initial-frame and prompt variant. Default: `default`. |
| `--total-blocks N` | Stop after this many generated blocks; `0` runs until the session is stopped. Default: `0`. |
| `--fps N` | Set the application frame rate. Default: `30`. |
| `--width N` | Set the output width. Default: `1280` (`1168` for the perf app). |
| `--height N` | Set the output height. Default: `704` (`640` for the perf app). |
| `--view {rgb,hdmap,physx}` | Select the initial RGB, HD-map conditioning, or PhysX collider view. Default: `rgb`. |
| `--game-mode` | Enable the speed limit and collisions with scene actors and static map geometry. |
| `--postprocess-preset NAME` | Start with a registered video post-processing preset enabled. Default: none. |
| `--world-model-profile` | Enable synchronized world-model profiling. |
| `--world-model-device DEVICE` | Select the model device. Default: `cuda:0`. |
| `--world-model-seed N` | Pin the seed used for each rollout. |
| `--world-model-debug-condition-frame-dir PATH` | Override first-chunk condition frames for debugging. |

For example, use a local scene, select its rain variant, override its prompt,
and enable RTX super resolution:

```bash
uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc -- \
    --scene scene.usdz --variant rain --prompt "A rainy night drive" \
    --game-mode --postprocess-preset rtx-super-resolution
```

The HUD view button cycles through **RGB → HDMAP → PHYSX**. The number keys
select those views directly: press `1` for RGB, `2` for the HD map, or `3` for
PhysX.

The preset starts enabled and the HUD's **Post-processing** checkbox can toggle
it between generated chunks. Run
`uv run flashdreams-run-v2 interactive-drive-omnidreams -- --help` to see the presets
registered in the current environment. The built-in `rtx-*` presets require
the optional NVIDIA VFX dependency, installable with
`uv pip install 'flashdreams[rtx-postprocess]'`, and supported RTX hardware.

The downloaded default scene is
`scenes/clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz`. Pass
`-- --scene scene.usdz` to use a local scene instead.

## Tests

```bash
uv run --no-sync pytest apps/interactive_drive -m ci_cpu -v
```
