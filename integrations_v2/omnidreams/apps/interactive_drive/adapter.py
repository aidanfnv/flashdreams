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

"""OmniDreams binding for the reusable interactive-drive application."""

from __future__ import annotations

from clipgt2v import ClipGT2VApplicationDefaults
from interactive_drive import InteractiveDriveApplication
from omnidreams.config import (
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)
from omnidreams.impl.pipeline import OmnidreamsPipelineConfig

from flashdreams.api_v2.application import IApplication

from .hooks import (
    create_omnidreams_application_hooks,
    resolve_default_interactive_drive_scene,
)

OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS = ClipGT2VApplicationDefaults(
    title="Interactive Drive",
    slug="interactive-drive",
    backend="world_model",
    total_blocks=0,
    fps=30,
    width=1280,
    height=704,
)
OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS = ClipGT2VApplicationDefaults(
    title="Interactive Drive (Perf)",
    slug="interactive-drive-perf",
    backend="world_model",
    total_blocks=0,
    fps=30,
    width=1168,
    height=640,
)


def _create_app(
    defaults: ClipGT2VApplicationDefaults,
    pipeline_config: OmnidreamsPipelineConfig,
) -> IApplication:
    return InteractiveDriveApplication(
        defaults=defaults,
        hooks=create_omnidreams_application_hooks(pipeline_config),
        default_scene_resolver=resolve_default_interactive_drive_scene,
    )


def create_app() -> IApplication:
    """Create Interactive Drive with the regular OmniDreams config."""
    return _create_app(
        OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
        OMNIDREAMS_PIPELINE_CONFIG,
    )


def create_perf_app() -> IApplication:
    """Create Interactive Drive with the performance OmniDreams config."""
    return _create_app(
        OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
        OMNIDREAMS_PERF_PIPELINE_CONFIG,
    )


__all__ = ["create_app", "create_perf_app"]
