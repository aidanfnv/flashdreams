<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ClipGT2V application

A native v2 scene-driving rollout with SlangPy prompt/view controls and keyboard, gamepad, and wheel input.

## Usage

Install the application, then start it with no application arguments:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run flashdreams-run-v2 clipgt2v-omnidreams --mode webrtc
```

Use `clipgt2v-omnidreams-perf` instead for the native-accelerated,
performance-tuned configuration.

The default scene downloads on first use from the gated
`nvidia/omni-dreams-scenes` Hugging Face dataset. Application arguments are
optional and follow the `--` separator:

| Argument | Description |
| --- | --- |
| `--scene PATH` | Use a local USDZ scene instead of downloading the default scene. |
| `--prompt TEXT` | Override the prompt stored in the selected scene variant. |
| `--camera NAME` | Select a camera from the scene. Default: `camera_front_wide_120fov`. |
| `--variant NAME` | Select the scene's initial-frame and prompt variant. Default: `default`. |
| `--total-blocks N` | Stop after this many generated blocks; `0` runs until the session is stopped. Default: `60`. |
| `--fps N` | Set the application frame rate. Default: `30`. |
| `--width N` | Set the output width. Default: `1280` (`1168` for the perf app). |
| `--height N` | Set the output height. Default: `704` (`640` for the perf app). |
| `--view {rgb,hdmap,physx}` | Select the initial RGB, HD-map conditioning, or PhysX collider view. Default: `rgb`. |
| `--no-ui` | Disable prompt/view controls and present model output directly. |
| `--game-mode` | Enable the speed limit and collisions with scene actors and static map geometry. |
| `--postprocess-preset NAME` | Start with a registered video post-processing preset enabled. Default: none. |
| `--world-model-profile` | Enable synchronized world-model profiling. |
| `--world-model-device DEVICE` | Select the model device. Default: `cuda:0`. |
| `--world-model-seed N` | Pin the seed used for each rollout. |
| `--world-model-debug-condition-frame-dir PATH` | Override first-chunk condition frames for debugging. |

For example, use a local scene, render its HD-map conditioning view, and run until the
session is stopped:

```bash
uv run flashdreams-run-v2 clipgt2v-omnidreams --mode webrtc -- \
    --scene scene.usdz --view hdmap --game-mode --no-ui \
    --total-blocks 0
```

With `--no-ui`, keyboard, gamepad, and wheel events still reach the model loop;
only the SlangPy controls and HUD composition are skipped.

Press `1`, `2`, or `3` while driving to select the RGB, HD-map, or PhysX view.

Run `uv run flashdreams-run-v2 clipgt2v-omnidreams -- --help` to list the registered
post-processing presets and all current application arguments. Scene,
simulation, and backend work stay on the model loop.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API. The SlangPy `step_ui` callback exposes the complete [SlangPy UI API](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui).
