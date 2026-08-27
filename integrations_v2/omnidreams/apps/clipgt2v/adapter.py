# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable scene-driving application."""

from __future__ import annotations

from clipgt2v import ClipGT2VApplication, ClipGT2VApplicationDefaults
from omnidreams.config import (
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication

OMNIDREAMS_APPLICATION_DEFAULTS = ClipGT2VApplicationDefaults(
    title="OmniDreams",
    slug="clipgt2v",
    backend="world_model",
    total_blocks=60,
    fps=30,
    width=1280,
    height=704,
    pipeline_config=OMNIDREAMS_PIPELINE_CONFIG,
)
OMNIDREAMS_PERF_APPLICATION_DEFAULTS = ClipGT2VApplicationDefaults(
    title="OmniDreams (Perf)",
    slug="clipgt2v-perf",
    backend="world_model",
    total_blocks=60,
    fps=30,
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_PERF_PIPELINE_CONFIG,
)


def create_app() -> IApplication:
    """Create the regular OmniDreams scene-driving application."""
    return ClipGT2VApplication(
        defaults=OMNIDREAMS_APPLICATION_DEFAULTS,
    )


def create_perf_app() -> IApplication:
    """Create the performance-tuned OmniDreams scene-driving application."""
    return ClipGT2VApplication(
        defaults=OMNIDREAMS_PERF_APPLICATION_DEFAULTS,
    )


__all__ = ["create_app", "create_perf_app"]
