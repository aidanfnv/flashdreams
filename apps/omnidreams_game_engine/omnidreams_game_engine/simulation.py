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

"""Deterministic arcade vehicle integration for OmniDreams games."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import DriverCommand, SceneDefinition, VehicleState


@dataclass(frozen=True, kw_only=True, slots=True)
class ArcadeVehicleConfig:
    """Tuning constants for the reusable planar vehicle model."""

    acceleration_mps2: float = 13.0
    brake_mps2: float = 20.0
    reverse_acceleration_mps2: float = 8.0
    coast_drag_mps2: float = 2.5
    max_forward_speed_mps: float = 32.0
    max_reverse_speed_mps: float = 12.0
    steering_rate_per_s: float = 7.0
    steering_return_per_s: float = 5.0
    yaw_rate_rad_per_s: float = 1.45
    handbrake_yaw_multiplier: float = 1.8
    boundary_margin_m: float = 8.0


class ArcadeVehicleSimulator:
    """Advance authoritative vehicle state with bounded arcade handling."""

    def __init__(self, config: ArcadeVehicleConfig = ArcadeVehicleConfig()) -> None:
        self.config = config
        self._state = VehicleState(x_m=0.0, y_m=0.0)
        self._bounds: tuple[float, float, float, float] | None = None

    @property
    def state(self) -> VehicleState:
        """Return the current authoritative state."""
        return self._state

    def reset(self, scene: SceneDefinition) -> VehicleState:
        """Reset the vehicle and derive a safe play-area enclosure."""
        self._state = scene.initial_vehicle
        points = scene.route_world[:, :2]
        margin = self.config.boundary_margin_m
        self._bounds = (
            float(points[:, 0].min() - margin),
            float(points[:, 0].max() + margin),
            float(points[:, 1].min() - margin),
            float(points[:, 1].max() + margin),
        )
        return self._state

    def step(self, command: DriverCommand, dt_s: float) -> VehicleState:
        """Advance one fixed-duration simulation frame."""
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive.")
        cfg = self.config
        state = self._state
        target_steer = command.steer
        rate = (
            cfg.steering_rate_per_s
            if abs(target_steer) > abs(state.steering)
            else cfg.steering_return_per_s
        )
        steering = _approach(state.steering, target_steer, rate * dt_s)

        speed = state.speed_mps
        explicit_reverse = command.reverse
        if command.throttle > 0.0:
            direction = -1.0 if explicit_reverse else 1.0
            acceleration = (
                cfg.reverse_acceleration_mps2
                if explicit_reverse
                else cfg.acceleration_mps2
            )
            speed += direction * acceleration * command.throttle * dt_s
        if command.brake > 0.0:
            if speed > 0.25:
                speed = max(0.0, speed - cfg.brake_mps2 * command.brake * dt_s)
            else:
                speed -= cfg.reverse_acceleration_mps2 * command.brake * dt_s
        if command.throttle == 0.0 and command.brake == 0.0:
            speed = _approach(speed, 0.0, cfg.coast_drag_mps2 * dt_s)
        speed = max(-cfg.max_reverse_speed_mps, min(cfg.max_forward_speed_mps, speed))

        speed_fraction = min(1.0, abs(speed) / 12.0)
        direction = 1.0 if speed >= 0.0 else -1.0
        handbrake = cfg.handbrake_yaw_multiplier if command.handbrake else 1.0
        yaw = _normalize_angle(
            state.yaw_rad
            + steering
            * cfg.yaw_rate_rad_per_s
            * speed_fraction
            * direction
            * handbrake
            * dt_s
        )
        x_m = state.x_m + math.cos(yaw) * speed * dt_s
        y_m = state.y_m + math.sin(yaw) * speed * dt_s
        x_m, y_m, speed = self._resolve_bounds(x_m, y_m, speed)
        self._state = VehicleState(
            x_m=x_m,
            y_m=y_m,
            z_m=state.z_m,
            yaw_rad=yaw,
            speed_mps=speed,
            steering=steering,
        )
        return self._state

    def _resolve_bounds(
        self, x_m: float, y_m: float, speed_mps: float
    ) -> tuple[float, float, float]:
        if self._bounds is None:
            return x_m, y_m, speed_mps
        min_x, max_x, min_y, max_y = self._bounds
        clamped_x = float(np.clip(x_m, min_x, max_x))
        clamped_y = float(np.clip(y_m, min_y, max_y))
        if clamped_x != x_m or clamped_y != y_m:
            speed_mps *= -0.2
        return clamped_x, clamped_y, speed_mps


def _approach(value: float, target: float, amount: float) -> float:
    if value < target:
        return min(target, value + amount)
    return max(target, value - amount)


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["ArcadeVehicleConfig", "ArcadeVehicleSimulator"]
