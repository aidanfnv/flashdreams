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

"""Device-independent simulation and presentation data for OmniDreams games."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True, slots=True)
class DriverCommand:
    """Normalized level-triggered driving intent."""

    throttle: float = 0.0
    """Forward acceleration in ``[0, 1]``."""

    brake: float = 0.0
    """Brake or reverse-pedal pressure in ``[0, 1]``."""

    steer: float = 0.0
    """Steering in ``[-1, 1]`` with positive values turning left."""

    handbrake: bool = False
    """Whether the arcade handbrake is active."""

    reverse: bool = False
    """Whether reverse gear is explicitly selected."""

    reset: bool = False
    """Whether the current game session should restart."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "throttle", _clamp(self.throttle, 0.0, 1.0))
        object.__setattr__(self, "brake", _clamp(self.brake, 0.0, 1.0))
        object.__setattr__(self, "steer", _clamp(self.steer, -1.0, 1.0))

    def as_payload(self) -> dict[str, float | bool]:
        """Return a JSON-safe canonical modality payload."""
        return {
            "throttle": self.throttle,
            "brake": self.brake,
            "steer": self.steer,
            "handbrake": self.handbrake,
            "reverse": self.reverse,
            "reset": self.reset,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DriverCommand":
        """Build a command from one canonical modality payload."""
        return cls(
            throttle=float(payload.get("throttle", 0.0)),
            brake=float(payload.get("brake", 0.0)),
            steer=float(payload.get("steer", 0.0)),
            handbrake=bool(payload.get("handbrake", False)),
            reverse=bool(payload.get("reverse", False)),
            reset=bool(payload.get("reset", False)),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class VehicleState:
    """Authoritative planar vehicle state for one simulation frame."""

    x_m: float
    """World-space X position in metres."""

    y_m: float
    """World-space Y position in metres."""

    z_m: float = 0.0
    """World-space ground height in metres."""

    yaw_rad: float = 0.0
    """FLU heading around the world Z axis."""

    speed_mps: float = 0.0
    """Signed longitudinal speed in metres per second."""

    steering: float = 0.0
    """Smoothed steering state in ``[-1, 1]``."""

    def rig_to_world(self) -> npt.NDArray[np.float32]:
        """Return the FLU rig-to-world transform."""
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        return np.asarray(
            [
                [cosine, -sine, 0.0, self.x_m],
                [sine, cosine, 0.0, self.y_m],
                [0.0, 0.0, 1.0, self.z_m],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-safe state payload."""
        return {
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "z_m": float(self.z_m),
            "yaw_rad": float(self.yaw_rad),
            "speed_mps": float(self.speed_mps),
            "steering": float(self.steering),
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class DynamicActorTrajectory:
    """Renderer-compatible dynamic actor trajectory."""

    entity_id: str
    object_type: str
    timestamps_us: npt.NDArray[np.int64]
    translations_world: npt.NDArray[np.float32]
    orientations_xyzw: npt.NDArray[np.float32]
    dimensions_lwh: npt.NDArray[np.float32]
    is_simulated: bool = True


@dataclass(frozen=True, kw_only=True, slots=True)
class EngineFrame:
    """State synchronized with one HD-map conditioning frame."""

    timestamp_us: int
    vehicle: VehicleState
    command: DriverCommand
    application: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe presentation payload."""
        return {
            "timestamp_us": int(self.timestamp_us),
            "vehicle": self.vehicle.as_dict(),
            "command": self.command.as_payload(),
            "application": dict(self.application),
        }


@dataclass(frozen=True, kw_only=True, slots=True)
class SceneDefinition:
    """Runtime-ready scene facts shared by simulation and conditioning."""

    scene_id: str
    scene_path: Path
    camera_name: str
    prompt: str
    first_frame_rgb: npt.NDArray[np.uint8]
    route_world: npt.NDArray[np.float32]
    initial_vehicle: VehicleState
    initial_timestamp_us: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        route = np.asarray(self.route_world, dtype=np.float32)
        if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] < 2:
            raise ValueError(
                "SceneDefinition.route_world must have shape [N>=2, D>=2]."
            )
        frame = np.asarray(self.first_frame_rgb, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError("SceneDefinition.first_frame_rgb must be HWC RGB data.")
        object.__setattr__(self, "route_world", np.ascontiguousarray(route))
        object.__setattr__(
            self, "first_frame_rgb", np.ascontiguousarray(frame[..., :3])
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Driver command values must be finite.")
    return max(lower, min(upper, numeric))
