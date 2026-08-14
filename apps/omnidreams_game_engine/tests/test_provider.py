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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omnidreams.demo.spec import OmnidreamsLudusReplayScenario
from omnidreams_game_engine import (
    DriverCommand,
    DynamicActorTrajectory,
    GameFrameUpdate,
    SceneDefinition,
    VehicleState,
)
from omnidreams_game_engine.provider import (
    APPLICATION_FRAMES_METADATA_KEY,
    OmnidreamsGameInputProvider,
)
from omnidreams_game_engine.scenario import OmnidreamsGameScenario

from flashdreams.runtime.types import StepRequirements

pytestmark = pytest.mark.ci_cpu


class _SceneRuntime:
    def __init__(self) -> None:
        self.actors: tuple[DynamicActorTrajectory, ...] = ()

    def definition(self) -> SceneDefinition:
        return SceneDefinition(
            scene_id="scene",
            scene_path=Path("scene.usdz"),
            camera_name="front",
            prompt="drive",
            first_frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            route_world=np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32),
            initial_vehicle=VehicleState(x_m=0, y_m=0),
            initial_timestamp_us=10,
        )

    def render(
        self,
        *,
        vehicles: Any,
        timestamps_us: np.ndarray,
        dynamic_actors: Any = (),
    ) -> torch.Tensor:
        self.actors = tuple(dynamic_actors)
        return torch.zeros((1, 1, len(vehicles), 3, 4, 4))

    def close(self) -> None:
        return None


class _Application:
    def reset(self, scene: SceneDefinition, vehicle: VehicleState) -> GameFrameUpdate:
        return GameFrameUpdate(presentation={"index": -1})

    def advance(
        self,
        *,
        vehicle: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> GameFrameUpdate:
        del vehicle, command, dt_s
        actor = DynamicActorTrajectory(
            entity_id="passenger",
            object_type="Pedestrian",
            timestamps_us=np.array([timestamp_us], dtype=np.int64),
            translations_world=np.zeros((1, 3), dtype=np.float32),
            orientations_xyzw=np.array([[0, 0, 0, 1]], dtype=np.float32),
            dimensions_lwh=np.ones(3, dtype=np.float32),
        )
        return GameFrameUpdate(
            presentation={"index": timestamp_us}, dynamic_actors=(actor,)
        )

    def handle_action(self, action: str, value: object | None = None) -> None:
        del action, value


def test_provider_merges_actor_samples_and_emits_causally_aligned_metadata() -> None:
    model = OmnidreamsLudusReplayScenario(
        keyboard_events=(), scene_path=Path("scene.usdz"), total_blocks=10, fps=30
    )
    game_scenario = OmnidreamsGameScenario(model=model)
    scene_runtime = _SceneRuntime()
    provider = OmnidreamsGameInputProvider(
        scenario=game_scenario,
        device="cpu",
        application=_Application(),
        scene_factory=lambda scenario, device: scene_runtime,
    )
    step = provider.prepare_step(
        request=StepRequirements(step_index=0, input_frame_count=2),
        command=DriverCommand(),
    )
    frames = cast(
        Sequence[Mapping[str, Any]],
        step.result_metadata[APPLICATION_FRAMES_METADATA_KEY],
    )
    assert [frame["application"]["index"] for frame in frames] == [-1, 10]
    assert len(scene_runtime.actors) == 1
    assert scene_runtime.actors[0].timestamps_us.tolist() == [10, 33343]
