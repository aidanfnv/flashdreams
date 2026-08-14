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

"""Reusable simulation and conditioning contracts for OmniDreams games."""

from .alignment import CausalStateAligner
from .application import GameApplication, GameFrameUpdate
from .input import (
    GAME_DRIVER_COMMAND,
    GAMEPAD_STATE_EVENT,
    AnalogDriverCommandConverter,
    EvdevWheelReader,
    KeyboardDriverCommandConverter,
    WheelProfile,
    analog_state_event,
    game_user_input_schema,
    normalize_axis,
)
from .simulation import ArcadeVehicleConfig, ArcadeVehicleSimulator
from .types import (
    DriverCommand,
    DynamicActorTrajectory,
    EngineFrame,
    SceneDefinition,
    VehicleState,
)

__all__ = [
    "GAME_DRIVER_COMMAND",
    "GAMEPAD_STATE_EVENT",
    "AnalogDriverCommandConverter",
    "ArcadeVehicleConfig",
    "ArcadeVehicleSimulator",
    "CausalStateAligner",
    "DriverCommand",
    "DynamicActorTrajectory",
    "EngineFrame",
    "EvdevWheelReader",
    "GameApplication",
    "GameFrameUpdate",
    "KeyboardDriverCommandConverter",
    "SceneDefinition",
    "VehicleState",
    "WheelProfile",
    "analog_state_event",
    "game_user_input_schema",
    "normalize_axis",
]
