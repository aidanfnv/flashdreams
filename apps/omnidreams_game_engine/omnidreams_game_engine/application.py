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

"""Application hooks for reusable OmniDreams game simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .types import DriverCommand, DynamicActorTrajectory, SceneDefinition, VehicleState


@dataclass(frozen=True, kw_only=True, slots=True)
class GameFrameUpdate:
    """Application-authored state and actors for one simulation frame."""

    presentation: Mapping[str, Any] = field(default_factory=dict)
    """JSON-safe state synchronized with the generated frame."""

    dynamic_actors: Sequence[DynamicActorTrajectory] = ()
    """Actors injected into the HD-map conditioning scene."""

    stop_session: bool = False
    """Whether the application has reached a terminal state."""


@runtime_checkable
class GameApplication(Protocol):
    """Game policy advanced by the reusable simulation engine."""

    def reset(self, scene: SceneDefinition, vehicle: VehicleState) -> GameFrameUpdate:
        """Reset application state and return the initial presentation state."""
        ...

    def advance(
        self,
        *,
        vehicle: VehicleState,
        command: DriverCommand,
        timestamp_us: int,
        dt_s: float,
    ) -> GameFrameUpdate:
        """Advance game policy for one simulated frame."""
        ...

    def handle_action(self, action: str, value: object | None = None) -> None:
        """Handle an application action such as high-score name submission."""
        ...
