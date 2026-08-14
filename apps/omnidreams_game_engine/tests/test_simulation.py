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

from pathlib import Path

import numpy as np
import pytest
from omnidreams_game_engine import (
    ArcadeVehicleSimulator,
    DriverCommand,
    SceneDefinition,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def _scene() -> SceneDefinition:
    return SceneDefinition(
        scene_id="test",
        scene_path=Path("test.usdz"),
        camera_name="front",
        prompt="drive",
        first_frame_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        route_world=np.array([[0, 0, 0], [100, 0, 0], [100, 100, 0]], dtype=np.float32),
        initial_vehicle=VehicleState(x_m=0.0, y_m=0.0),
        initial_timestamp_us=0,
    )


def test_arcade_vehicle_supports_throttle_steering_braking_and_reverse() -> None:
    simulator = ArcadeVehicleSimulator()
    simulator.reset(_scene())

    forward = simulator.step(DriverCommand(throttle=1.0, steer=1.0), 0.5)
    braking = simulator.step(DriverCommand(brake=1.0), 0.5)
    reverse = simulator.step(DriverCommand(throttle=1.0, reverse=True), 0.5)

    assert forward.speed_mps > 0
    assert forward.yaw_rad > 0
    assert braking.speed_mps == 0
    assert reverse.speed_mps < 0


def test_reset_is_deterministic_and_boundary_is_bounded() -> None:
    simulator = ArcadeVehicleSimulator()
    expected = simulator.reset(_scene())
    for _ in range(100):
        simulator.step(DriverCommand(throttle=1.0), 0.5)
    assert simulator.state.x_m <= 108.0
    assert simulator.reset(_scene()) == expected
