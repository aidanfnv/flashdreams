# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote camera-controlled sessions using the existing Cam2V model loop."""

import math
from dataclasses import replace
from pathlib import Path

from model_service.flashdreams_backend import (
    T2VBackend,
    T2VSession,
    UnsupportedApplication,
)

CAMERA_KEYS = {
    "forward": "w",
    "backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "strafe_left": "q",
    "strafe_right": "e",
    "look_up": "i",
    "look_down": "k",
}
"""Semantic wire actions translated into the existing Cam2V input contract."""

CAMERA_INPUTS = {name: {"type": "boolean", "default": False} for name in CAMERA_KEYS}
"""Complete control snapshots; omitted actions are released."""


def _conditioning(values, *, fov, world_scale):
    """Build camera calibration in output pixels for a user-supplied image."""
    import torch
    from cam2v import Cam2VConditioning

    width, height = values["pixel_width"], values["pixel_height"]
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return Cam2VConditioning(
        prompt=values["prompt"],
        first_frame_path=Path(values["image_path"]),
        base_intrinsics=torch.tensor([focal, focal, width / 2, height / 2]),
        world_scale=world_scale,
    )


class Cam2VBackend(T2VBackend):
    """Adapt shared Cam2V applications to image uploads and semantic controls."""

    def _validate_application(self, application):
        """Require a shared camera application and identify synthetic pipelines."""
        from cam2v import Cam2VApplication

        if not isinstance(application, Cam2VApplication):
            raise UnsupportedApplication("Expected a Cam2V application")
        self._dummy = not hasattr(application.pipeline_config, "diffusion_model")

    def __init__(self, model, factory, *, device="cuda", resident=None):
        """Advertise image and camera settings without preparing model weights."""
        super().__init__(model, factory, device=device, resident=resident)
        self._input_image = "required"
        self._schema.update(
            {
                "horizontal_fov": {
                    "type": "number",
                    "default": 90.0,
                    "minimum": 10,
                    "maximum": 150,
                },
                "world_scale": {
                    "type": "number",
                    "default": 1.0,
                    "minimum": 0.001,
                    "maximum": 100,
                },
            }
        )

    def catalog(self):
        """Describe the camera snapshot contract alongside generation settings."""
        entries = super().catalog()
        entries[0].update(
            interaction="camera",
            inputs=CAMERA_INPUTS,
            stub=self._dummy,
            control_timing="next_block",
            control_clock="model",
        )
        return entries

    def _extra_cache_key(self, parameters):
        """Invalidate prepared conditioning when scene inputs change."""
        return tuple(parameters[k] for k in ("prompt", "horizontal_fov", "world_scale"))

    def _prepare_application(self, application, parameters, arguments):
        """Bind uploaded scene inputs to the shared conditioning interface."""
        from functools import partial

        # Calibration is explicit in the remote project, with no server-local
        # example assets or pose replay. Retain the adapter's generation and
        # coordinate-system conventions.
        application.defaults = replace(
            application.defaults,
            input_resolver=partial(
                _conditioning,
                fov=parameters["horizontal_fov"],
                world_scale=parameters["world_scale"],
            ),
        )
        # The dummy has no diffusion seed or compilation options.
        if self._dummy:
            arguments = list(arguments)
            for flag in ("--compile", "--no-compile"):
                if flag in arguments:
                    arguments.remove(flag)
            if "--seed" in arguments:
                index = arguments.index("--seed")
                del arguments[index : index + 2]
        application.init(arguments)

    def _wrap_session(self, model, parameters, session):
        """Return the camera-aware RGB session wrapper."""
        return Cam2VSession(model, parameters, session, self._resident)


class Cam2VSession(T2VSession):
    """Apply the newest control snapshot when a block needs generation."""

    def __init__(self, *args):
        """Keep replies within one block so controls remain attributable."""
        super().__init__(*args)
        self._input_schema = CAMERA_INPUTS
        self._one_block_per_reply = True

    def _events(self):
        """Replace held actions at the start of the next model-time window."""
        from flashdreams.runtime_v2.user_input_events import UserInputEvents

        state = self._session.model_loop.state
        start = state.input_timeline.next_window_start_s
        # Replace state at the next model-time boundary. Wall-clock inference
        # delays never create an accumulated movement jump or an event backlog.
        state.keyboard_track.release_all(timestamp_s=start)
        for name, key in CAMERA_KEYS.items():
            if self._inputs[name]:
                state.keyboard_track.on_edge(
                    timestamp_s=start, action="keydown", key=key
                )
        self._block_metadata = {
            "block": self._step,
            "applied_inputs": dict(self._inputs),
            "control_time_s": start,
        }
        return UserInputEvents([])
