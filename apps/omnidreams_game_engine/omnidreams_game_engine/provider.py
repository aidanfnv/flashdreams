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

"""FlashDreams model-input provider for interactive OmniDreams games."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from flashdreams.runtime import (
    InferenceInput,
    StepRequirements,
)
from flashdreams.runtime._utils import freeze_mapping

from .alignment import CausalStateAligner
from .application import GameApplication
from .ludus import LudusGameScene
from .scenario import OmnidreamsGameScenario
from .simulation import ArcadeVehicleSimulator
from .types import DriverCommand, DynamicActorTrajectory, EngineFrame, SceneDefinition

APPLICATION_FRAMES_METADATA_KEY = "application_frames"
"""Generated-result metadata key carrying synchronized presentation frames."""

GAME_ACTION_EVENT = "game_action"
"""Raw application event used for restart and high-score name entry."""


@dataclass(frozen=True, kw_only=True, slots=True)
class PreparedGameStep:
    """Model conditioning and synchronized presentation state for one step."""

    inference_input: InferenceInput
    result_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "result_metadata", freeze_mapping(self.result_metadata)
        )


class GameSceneRuntime(Protocol):
    """Scene operations required by :class:`OmnidreamsGameInputProvider`."""

    def definition(self) -> SceneDefinition: ...

    def render(
        self,
        *,
        vehicles: Sequence[Any],
        timestamps_us: np.ndarray,
        dynamic_actors: Sequence[DynamicActorTrajectory] = (),
    ) -> torch.Tensor: ...

    def close(self) -> None: ...


SceneFactory = Callable[[OmnidreamsGameScenario, torch.device], GameSceneRuntime]


class OmnidreamsGameInputProvider:
    """Advance simulation and prepare synchronized OmniDreams HD-map inputs."""

    def __init__(
        self,
        *,
        scenario: OmnidreamsGameScenario,
        device: str | torch.device,
        application: GameApplication,
        simulator: ArcadeVehicleSimulator | None = None,
        scene_factory: SceneFactory | None = None,
    ) -> None:
        self._scenario = scenario
        self._device = torch.device(device)
        self._application = application
        self._simulator = simulator or ArcadeVehicleSimulator()
        self._scene_factory = scene_factory or _default_scene_factory
        self._scene_runtime: GameSceneRuntime | None = None
        self._scene: SceneDefinition | None = None
        self._next_timestamp_us = 0
        self._aligner = CausalStateAligner()
        self._closed = False

    def prepare_initial_input(self) -> InferenceInput:
        """Load the scene and return session-global OmniDreams inputs."""
        self._require_open()
        scene = self._ensure_scene()
        frame = torch.from_numpy(scene.first_frame_rgb).permute(2, 0, 1)
        frame = frame.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        first_frame = frame.to(device=self._device, dtype=torch.bfloat16) / 127.5 - 1.0
        return InferenceInput(
            global_conditioning={
                "scenario": self._scenario.model,
                "prompt": [[scene.prompt]],
                "first_frame": first_frame,
            },
            metadata={
                "view_names": (scene.camera_name,),
                "scene_id": scene.scene_id,
            },
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        command: DriverCommand,
    ) -> PreparedGameStep:
        """Advance one model chunk and attach aligned application state."""
        self._require_open()
        self._ensure_scene()
        if request.step_index >= self._scenario.model.total_blocks:
            raise RuntimeError(
                "OmniDreams game scenario reached its block limit before the "
                "application session stopped."
            )
        if command.reset:
            self.reset()
            raise RuntimeError(
                "Application-host reset is not available; reset must be filtered "
                "before preparing a game step."
            )

        frame_count = request.input_frame_count
        dt_s = 1.0 / float(self._scenario.model.fps)
        timestamps = self._consume_timestamps(frame_count)
        engine_frames: list[EngineFrame] = []
        vehicles = []
        dynamic_actors: dict[str, list[DynamicActorTrajectory]] = {}
        for timestamp_us in timestamps:
            vehicle = self._simulator.step(command, dt_s)
            update = self._application.advance(
                vehicle=vehicle,
                command=command,
                timestamp_us=int(timestamp_us),
                dt_s=dt_s,
            )
            vehicles.append(vehicle)
            engine_frames.append(
                EngineFrame(
                    timestamp_us=int(timestamp_us),
                    vehicle=vehicle,
                    command=command,
                    application=update.presentation,
                )
            )
            for actor in update.dynamic_actors:
                dynamic_actors.setdefault(actor.entity_id, []).append(actor)

        hdmap = self._require_scene_runtime().render(
            vehicles=vehicles,
            timestamps_us=timestamps,
            dynamic_actors=tuple(
                _merge_actor_samples(samples) for samples in dynamic_actors.values()
            ),
        )
        aligned = self._aligner.align(engine_frames)
        return PreparedGameStep(
            inference_input=InferenceInput(
                step={"hdmap": hdmap},
                metadata={
                    "frame_timestamps_us": tuple(int(value) for value in timestamps),
                },
            ),
            result_metadata={
                APPLICATION_FRAMES_METADATA_KEY: tuple(
                    frame.as_dict() for frame in aligned
                )
            },
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Reset simulation, application, device, and alignment state together."""
        del inputs
        self._require_open()
        scene = self._ensure_scene()
        vehicle = self._simulator.reset(scene)
        initial = self._application.reset(scene, vehicle)
        self._aligner.reset(
            EngineFrame(
                timestamp_us=scene.initial_timestamp_us,
                vehicle=vehicle,
                command=DriverCommand(),
                application=initial.presentation,
            )
        )
        self._next_timestamp_us = scene.initial_timestamp_us

    def close(self) -> None:
        """Release renderer resources idempotently."""
        if self._closed:
            return
        self._closed = True
        if self._scene_runtime is not None:
            self._scene_runtime.close()
        self._scene_runtime = None
        self._scene = None

    def _ensure_scene(self) -> SceneDefinition:
        if self._scene is None:
            runtime = self._scene_factory(self._scenario, self._device)
            scene = runtime.definition()
            self._scene_runtime = runtime
            self._scene = scene
            self._simulator.reset(scene)
            initial = self._application.reset(scene, scene.initial_vehicle)
            self._next_timestamp_us = scene.initial_timestamp_us
            self._aligner.reset(
                EngineFrame(
                    timestamp_us=scene.initial_timestamp_us,
                    vehicle=scene.initial_vehicle,
                    command=DriverCommand(),
                    application=initial.presentation,
                )
            )
        return self._scene

    def _consume_timestamps(self, count: int) -> np.ndarray:
        step_us = int(round(1_000_000 / float(self._scenario.model.fps)))
        values = self._next_timestamp_us + np.arange(count, dtype=np.int64) * step_us
        self._next_timestamp_us += count * step_us
        return values

    def _require_scene_runtime(self) -> GameSceneRuntime:
        self._ensure_scene()
        if self._scene_runtime is None:
            raise RuntimeError("OmniDreams game scene runtime is unavailable.")
        return self._scene_runtime

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams game input provider is closed.")


def _default_scene_factory(
    scenario: OmnidreamsGameScenario, device: torch.device
) -> GameSceneRuntime:
    return LudusGameScene(scenario, device=device)


def _merge_actor_samples(
    samples: Sequence[DynamicActorTrajectory],
) -> DynamicActorTrajectory:
    """Merge per-frame samples into one renderer trajectory."""
    first = samples[0]
    return DynamicActorTrajectory(
        entity_id=first.entity_id,
        object_type=first.object_type,
        timestamps_us=np.concatenate([sample.timestamps_us for sample in samples]),
        translations_world=np.concatenate(
            [sample.translations_world for sample in samples]
        ),
        orientations_xyzw=np.concatenate(
            [sample.orientations_xyzw for sample in samples]
        ),
        dimensions_lwh=first.dimensions_lwh,
        is_simulated=first.is_simulated,
    )


__all__ = [
    "APPLICATION_FRAMES_METADATA_KEY",
    "GAME_ACTION_EVENT",
    "GameSceneRuntime",
    "OmnidreamsGameInputProvider",
    "PreparedGameStep",
    "SceneFactory",
]
