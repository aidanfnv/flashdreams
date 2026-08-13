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

"""Physical-state prompt composition for Crazy Robotaxi."""

from __future__ import annotations

import math

import numpy as np
from omnidreams.interactive_drive.types import TextPromptUpdate, TrajectoryChunk

_REVERSE_ENTER_SPEED_MPS = -0.25
"""Signed speed that activates reverse conditioning despite zero-speed noise."""

_REVERSE_EXIT_SPEED_MPS = -0.05
"""Signed speed above which an entire chunk exits reverse conditioning."""

_COLLISION_FUTURE_CHUNKS = 6
"""Number of post-impact chunks that retain collision conditioning."""

_REVERSE_PROMPT = (
    "The camera vehicle is reversing backward; the scene moves consistently "
    "with backward vehicle motion."
)
"""Stable prompt modifier for authoritative reverse motion."""

_FALLBACK_COLLISION_PROMPT = (
    "A complete, solid vehicle remains immediately against the dashcam vehicle "
    "at close-range contact, filling the same nearby area of the view as it "
    "moves from the impact."
)
"""Collision prompt used when the struck actor cannot be spatially resolved."""


class WorldConsistencyPromptController:
    """Compose stable prompts from chunk-level physical state."""

    def __init__(self, base_prompt: str) -> None:
        self._base_prompt = base_prompt.strip()
        if not self._base_prompt:
            raise ValueError("Crazy Robotaxi requires a non-empty scene prompt")
        self._reverse_active = False
        self._collision_future_chunks = 0
        self._collision_prompt: str | None = None

    def update(self, trajectory: TrajectoryChunk) -> TextPromptUpdate:
        """Return the prompt synchronized to ``trajectory`` physical state."""
        speeds = tuple(state.speed_mps for state in trajectory.vehicle_states)
        if self._reverse_active:
            if all(speed >= _REVERSE_EXIT_SPEED_MPS for speed in speeds):
                self._reverse_active = False
        elif any(speed <= _REVERSE_ENTER_SPEED_MPS for speed in speeds):
            self._reverse_active = True

        collision_active = trajectory.actor_collision_detected
        if collision_active:
            self._collision_future_chunks = _COLLISION_FUTURE_CHUNKS
            self._collision_prompt = _collision_contact_prompt(trajectory)
        elif self._collision_future_chunks > 0:
            collision_active = True
            self._collision_future_chunks -= 1
        else:
            self._collision_prompt = None

        modifiers: list[tuple[str, str]] = []
        if self._reverse_active:
            modifiers.append(("reverse", _REVERSE_PROMPT))
        if collision_active:
            modifiers.append(
                ("collision", self._collision_prompt or _FALLBACK_COLLISION_PROMPT)
            )

        prompt_parts = [self._base_prompt, *(text for _, text in modifiers)]
        return TextPromptUpdate(
            prompt=" ".join(prompt_parts),
            active_modifiers=tuple(label for label, _ in modifiers),
        )


def _collision_contact_prompt(trajectory: TrajectoryChunk) -> str:
    """Describe the closest struck actor relative to the ego at impact."""
    frame_index = trajectory.actor_collision_frame_index
    if frame_index is None:
        return _FALLBACK_COLLISION_PROMPT
    struck_ids = set(trajectory.actor_collision_entity_ids)
    candidates = tuple(
        actor
        for actor in trajectory.dynamic_actors
        if not struck_ids or actor.entity_id in struck_ids
    )
    if not candidates:
        return _FALLBACK_COLLISION_PROMPT

    ego = trajectory.vehicle_states[frame_index]
    ego_xy = np.asarray([ego.x_m, ego.y_m], dtype=np.float32)
    actor = min(
        candidates,
        key=lambda candidate: float(
            np.linalg.norm(candidate.translations_world[frame_index, :2] - ego_xy)
        ),
    )
    delta_xy = actor.translations_world[frame_index, :2] - ego_xy
    forward_xy = np.asarray(
        [math.cos(ego.yaw_rad), math.sin(ego.yaw_rad)], dtype=np.float32
    )
    left_xy = np.asarray([-forward_xy[1], forward_xy[0]], dtype=np.float32)
    longitudinal_m = float(np.dot(delta_xy, forward_xy))
    lateral_m = float(np.dot(delta_xy, left_xy))
    actor_label = _actor_prompt_label(actor.object_type)

    if abs(longitudinal_m) >= abs(lateral_m):
        if longitudinal_m >= 0.0:
            location = "immediately ahead of"
            contact = "at bumper-to-bumper contact"
        else:
            location = "immediately behind"
            contact = "at close-range contact"
    elif lateral_m >= 0.0:
        location = "immediately beside the left side of"
        contact = "at close-range side contact"
    else:
        location = "immediately beside the right side of"
        contact = "at close-range side contact"

    return (
        f"A complete, solid {actor_label} remains {location} the dashcam vehicle "
        f"{contact}, filling the same nearby area of the view as it moves from "
        "the impact."
    )


def _actor_prompt_label(object_type: str) -> str:
    """Return a natural-language vehicle label from scene semantics."""
    normalized = object_type.casefold()
    for label in ("car", "truck", "bus", "van", "motorcycle"):
        if label in normalized:
            return label
    return "vehicle"
