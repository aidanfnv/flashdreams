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

"""Crazy Robotaxi rules implemented against the standalone game-engine API."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from omnidreams_game_engine import (
    DriverCommand,
    DynamicActorTrajectory,
    GameFrameUpdate,
    SceneDefinition,
    VehicleState,
)

from .high_scores import HighScoreEntry, HighScoreStore

TaxiPhase = Literal["pickup", "dropoff"]
TaxiSessionState = Literal["playing", "awaiting_name", "leaderboard"]


@dataclass(frozen=True, slots=True)
class TaxiGameConfig:
    """Tunable game rules independent from rendering and model execution."""

    seed: int = 42
    game_time_s: float = 60.0
    pickup_radius_m: float = 6.0
    dropoff_radius_m: float = 7.0
    pickup_min_distance_m: float = 25.0
    target_spacing_m: float = 45.0
    base_fare_points: int = 500
    bonus_points_per_second: int = 100
    dropoff_time_bonus_s: float = 30.0
    fare_speed_mps: float = 10.0
    fare_grace_s: float = 10.0
    min_fare_time_s: float = 15.0
    max_fare_time_s: float = 50.0
    high_scores_path: Path | None = None


class CrazyRobotaxiGame:
    """Deterministic pickup/dropoff game with passengers and high scores."""

    def __init__(
        self,
        config: TaxiGameConfig | None = None,
        *,
        high_scores: HighScoreStore | None = None,
    ) -> None:
        self.config = config or TaxiGameConfig()
        self._scores = high_scores or HighScoreStore(self.config.high_scores_path)
        self._scene: SceneDefinition | None = None
        self._rng = np.random.default_rng(self.config.seed)
        self._targets = np.empty((0, 3), dtype=np.float32)
        self._target_index = 0
        self._phase: TaxiPhase = "pickup"
        self._score = 0
        self._global_remaining_s = self.config.game_time_s
        self._fare_remaining_s: float | None = None
        self._session_state: TaxiSessionState = "playing"
        self._leaderboard: tuple[HighScoreEntry, ...] = ()
        self._rank: int | None = None
        self._event: str | None = None
        self._name_error: str | None = None

    def reset(self, scene: SceneDefinition, vehicle: VehicleState) -> GameFrameUpdate:
        """Start a fresh game for ``scene``."""
        self._scene = scene
        seed_digest = hashlib.sha256(scene.scene_id.encode()).digest()
        scene_seed = int.from_bytes(seed_digest[:8], "little") ^ self.config.seed
        self._rng = np.random.default_rng(scene_seed)
        self._targets = _sample_targets(
            scene.route_world,
            spacing_m=self.config.target_spacing_m,
            origin_xy=np.array([vehicle.x_m, vehicle.y_m], dtype=np.float32),
            minimum_distance_m=self.config.pickup_min_distance_m,
        )
        self._target_index = int(self._rng.integers(0, len(self._targets)))
        self._phase = "pickup"
        self._score = 0
        self._global_remaining_s = self.config.game_time_s
        self._fare_remaining_s = None
        self._session_state = "playing"
        self._leaderboard = self._scores.read()
        self._rank = None
        self._event = None
        self._name_error = None
        return self._update(vehicle, timestamp_us=scene.initial_timestamp_us)

    def advance(
        self,
        *,
        vehicle: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> GameFrameUpdate:
        """Advance timers, target transitions, scoring, and passenger state."""
        del command
        if self._scene is None:
            raise RuntimeError("CrazyRobotaxiGame.reset must be called first")
        if self._session_state != "playing":
            return self._update(vehicle, timestamp_us=timestamp_us)

        self._global_remaining_s = max(0.0, self._global_remaining_s - dt_s)
        if self._fare_remaining_s is not None:
            self._fare_remaining_s = max(0.0, self._fare_remaining_s - dt_s)
        if self._global_remaining_s <= 0.0:
            self._finish_game()
            return self._update(vehicle, timestamp_us=timestamp_us)

        distance = self._distance_to_target(vehicle)
        if self._phase == "pickup" and distance <= self.config.pickup_radius_m:
            self._phase = "dropoff"
            self._target_index = self._next_distant_target(vehicle)
            trip_distance = self._distance_to_target(vehicle)
            self._fare_remaining_s = min(
                self.config.max_fare_time_s,
                max(
                    self.config.min_fare_time_s,
                    trip_distance / self.config.fare_speed_mps
                    + self.config.fare_grace_s,
                ),
            )
            self._event = "pickup_complete"
        elif self._phase == "dropoff":
            if distance <= self.config.dropoff_radius_m:
                bonus = int(self._fare_remaining_s or 0.0)
                self._score += (
                    self.config.base_fare_points
                    + bonus * self.config.bonus_points_per_second
                )
                self._global_remaining_s += self.config.dropoff_time_bonus_s
                self._phase = "pickup"
                self._fare_remaining_s = None
                self._target_index = self._next_distant_target(vehicle)
                self._event = "fare_complete"
            elif self._fare_remaining_s is not None and self._fare_remaining_s <= 0.0:
                self._phase = "pickup"
                self._fare_remaining_s = None
                self._target_index = self._next_distant_target(vehicle)
                self._event = "time_expired"
        return self._update(vehicle, timestamp_us=timestamp_us)

    def handle_action(self, action: str, value: object | None = None) -> None:
        """Handle transport-neutral UI actions."""
        if action == "submit_name" and self._session_state == "awaiting_name":
            try:
                entry, self._leaderboard = self._scores.record(
                    str(value or ""), self._score
                )
            except ValueError as exc:
                self._name_error = str(exc)
                return
            self._rank = (
                None
                if entry is None
                else next(
                    index
                    for index, candidate in enumerate(self._leaderboard, start=1)
                    if candidate == entry
                )
            )
            self._session_state = "leaderboard"
            self._name_error = None
        elif action == "skip_name" and self._session_state == "awaiting_name":
            self._leaderboard = self._scores.read()
            self._session_state = "leaderboard"

    def _finish_game(self) -> None:
        self._leaderboard = self._scores.read()
        self._rank = self._scores.qualifying_rank(self._score)
        self._session_state = (
            "awaiting_name" if self._rank is not None else "leaderboard"
        )
        self._event = "game_over"

    def _next_distant_target(self, vehicle: VehicleState) -> int:
        distances = np.linalg.norm(
            self._targets[:, :2]
            - np.array([vehicle.x_m, vehicle.y_m], dtype=np.float32),
            axis=1,
        )
        eligible = np.flatnonzero(distances >= self.config.pickup_min_distance_m)
        if len(eligible) == 0:
            return int(np.argmax(distances))
        choices = eligible[eligible != self._target_index]
        if len(choices) == 0:
            choices = eligible
        return int(self._rng.choice(choices))

    def _distance_to_target(self, vehicle: VehicleState) -> float:
        target = self._targets[self._target_index]
        return float(math.hypot(target[0] - vehicle.x_m, target[1] - vehicle.y_m))

    def _update(self, vehicle: VehicleState, *, timestamp_us: int) -> GameFrameUpdate:
        target = self._targets[self._target_index]
        bearing = math.atan2(target[1] - vehicle.y_m, target[0] - vehicle.x_m)
        relative_bearing = (bearing - vehicle.yaw_rad + math.pi) % (
            2 * math.pi
        ) - math.pi
        presentation = {
            "phase": self._phase,
            "session_state": self._session_state,
            "target_xyz_m": target.tolist(),
            "distance_m": self._distance_to_target(vehicle),
            "relative_bearing_rad": relative_bearing,
            "target_radius_m": (
                self.config.pickup_radius_m
                if self._phase == "pickup"
                else self.config.dropoff_radius_m
            ),
            "score": self._score,
            "global_remaining_time_s": self._global_remaining_s,
            "fare_remaining_time_s": self._fare_remaining_s,
            "event": self._event,
            "high_score_rank": self._rank,
            "leaderboard": [entry.as_dict() for entry in self._leaderboard],
            "name_error": self._name_error,
        }
        passengers = (
            (_passenger(target, timestamp_us),)
            if self._phase == "pickup" and self._session_state == "playing"
            else ()
        )
        self._event = None
        return GameFrameUpdate(presentation=presentation, dynamic_actors=passengers)


def _sample_targets(
    route_xyz: np.ndarray,
    *,
    spacing_m: float,
    origin_xy: np.ndarray,
    minimum_distance_m: float,
) -> np.ndarray:
    """Select well-spaced target candidates from a route polyline."""
    route = np.asarray(route_xyz, dtype=np.float32)
    if len(route) == 0:
        route = np.array([[origin_xy[0] + 40.0, origin_xy[1], 0.0]], dtype=np.float32)
    selected = [route[0]]
    accumulated = 0.0
    for previous, point in zip(route[:-1], route[1:], strict=True):
        accumulated += float(np.linalg.norm(point[:2] - previous[:2]))
        if accumulated >= spacing_m:
            selected.append(point)
            accumulated = 0.0
    targets = np.asarray(selected, dtype=np.float32)
    distant = targets[
        np.linalg.norm(targets[:, :2] - origin_xy, axis=1) >= minimum_distance_m
    ]
    if len(distant) >= 2:
        return distant
    fallback = route[:: max(1, len(route) // 8)]
    return fallback if len(fallback) >= 2 else np.vstack((route[0], route[-1]))


def _passenger(target_xyz: np.ndarray, timestamp_us: int) -> DynamicActorTrajectory:
    """Build one stationary pedestrian actor for the active pickup."""
    digest = hashlib.sha256(
        np.asarray(target_xyz, dtype=np.float32).tobytes()
    ).hexdigest()[:16]
    center = np.asarray(target_xyz, dtype=np.float32) + np.array(
        [0.0, 0.0, 0.9], dtype=np.float32
    )
    return DynamicActorTrajectory(
        entity_id=f"taxi-passenger-{digest}",
        object_type="Pedestrian",
        timestamps_us=np.array([timestamp_us], dtype=np.int64),
        translations_world=center[None, :],
        orientations_xyzw=np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        dimensions_lwh=np.array([0.6, 0.6, 1.8], dtype=np.float32),
        is_simulated=True,
    )


__all__ = ["CrazyRobotaxiGame", "TaxiGameConfig", "TaxiPhase", "TaxiSessionState"]
