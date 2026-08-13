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

_COLLISION_PROMPT = (
    "Vehicles remain solid, distinct, and physically coherent through contact; "
    "struck vehicles keep their shape and follow the collision motion."
)
"""Stable prompt modifier for authoritative vehicle collisions."""


class WorldConsistencyPromptController:
    """Compose stable prompts from chunk-level physical state."""

    def __init__(self, base_prompt: str) -> None:
        self._base_prompt = base_prompt.strip()
        if not self._base_prompt:
            raise ValueError("Crazy Robotaxi requires a non-empty scene prompt")
        self._reverse_active = False
        self._collision_future_chunks = 0

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
        elif self._collision_future_chunks > 0:
            collision_active = True
            self._collision_future_chunks -= 1

        modifiers: list[tuple[str, str]] = []
        if self._reverse_active:
            modifiers.append(("reverse", _REVERSE_PROMPT))
        if collision_active:
            modifiers.append(("collision", _COLLISION_PROMPT))

        prompt_parts = [self._base_prompt, *(text for _, text in modifiers)]
        return TextPromptUpdate(
            prompt=" ".join(prompt_parts),
            active_modifiers=tuple(label for label, _ in modifiers),
        )
