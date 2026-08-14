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

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from crazy_robotaxi.game import CrazyRobotaxiGame, TaxiGameConfig
from crazy_robotaxi.high_scores import HighScoreStore
from omnidreams_game_engine import DriverCommand, SceneDefinition, VehicleState

pytestmark = pytest.mark.ci_cpu


def _scene() -> SceneDefinition:
    return SceneDefinition(
        scene_id="taxi-test",
        scene_path=Path("test.usdz"),
        camera_name="front",
        prompt="drive",
        first_frame_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        route_world=np.array(
            [[0, 0, 0], [50, 0, 0], [100, 0, 0], [100, 50, 0]],
            dtype=np.float32,
        ),
        initial_vehicle=VehicleState(x_m=0.0, y_m=0.0),
        initial_timestamp_us=0,
    )


def _at(target: object) -> VehicleState:
    xyz = cast(Sequence[object], target)
    return VehicleState(
        x_m=float(str(xyz[0])),
        y_m=float(str(xyz[1])),
        z_m=float(str(xyz[2])),
    )


def test_pickup_passenger_disappears_then_dropoff_scores(tmp_path: Path) -> None:
    game = CrazyRobotaxiGame(
        TaxiGameConfig(high_scores_path=tmp_path / "scores.csv", seed=1)
    )
    initial = game.reset(_scene(), _scene().initial_vehicle)
    assert len(initial.dynamic_actors) == 1
    pickup = initial.presentation["target_xyz_m"]
    after_pickup = game.advance(
        vehicle=_at(pickup), command=DriverCommand(), timestamp_us=1, dt_s=0.1
    )
    assert after_pickup.presentation["phase"] == "dropoff"
    assert after_pickup.presentation["event"] == "pickup_complete"
    assert after_pickup.dynamic_actors == ()

    dropoff = after_pickup.presentation["target_xyz_m"]
    completed = game.advance(
        vehicle=_at(dropoff), command=DriverCommand(), timestamp_us=2, dt_s=0.1
    )
    assert completed.presentation["phase"] == "pickup"
    assert int(completed.presentation["score"]) >= 500
    assert len(completed.dynamic_actors) == 1


def test_game_over_name_entry_persists_and_displays_board(tmp_path: Path) -> None:
    store = HighScoreStore(tmp_path / "scores.csv")
    game = CrazyRobotaxiGame(
        TaxiGameConfig(game_time_s=0.1, high_scores_path=store.path),
        high_scores=store,
    )
    game.reset(_scene(), _scene().initial_vehicle)
    game._score = 1234
    expired = game.advance(
        vehicle=_scene().initial_vehicle,
        command=DriverCommand(),
        timestamp_us=1,
        dt_s=0.2,
    )
    assert expired.presentation["session_state"] == "awaiting_name"
    game.handle_action("submit_name", "bad,name")
    invalid = game.advance(
        vehicle=_scene().initial_vehicle,
        command=DriverCommand(),
        timestamp_us=2,
        dt_s=0.1,
    )
    assert invalid.presentation["session_state"] == "awaiting_name"
    assert invalid.presentation["name_error"]
    game.handle_action("submit_name", "TAXI")
    board = game.advance(
        vehicle=_scene().initial_vehicle,
        command=DriverCommand(),
        timestamp_us=3,
        dt_s=0.1,
    )
    assert board.presentation["session_state"] == "leaderboard"
    assert board.presentation["leaderboard"][0]["name"] == "TAXI"
