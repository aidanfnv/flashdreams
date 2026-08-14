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

"""Scenario configuration for interactive OmniDreams games."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omnidreams.demo.spec import OmnidreamsLudusReplayScenario


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsGameScenario:
    """Model and scene settings for one interactive game rollout."""

    model: OmnidreamsLudusReplayScenario
    """OmniDreams model scenario used to initialize the inference session."""

    scene_path: Path | None = None
    """Optional local USDZ scene path overriding model discovery."""

    wheel_profile: Path | None = None
    """Optional native wheel profile used by the local-window frontend."""


__all__ = ["OmnidreamsGameScenario"]
