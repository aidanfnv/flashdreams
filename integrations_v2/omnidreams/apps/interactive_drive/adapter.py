# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams binding for the reusable interactive-drive application."""

from __future__ import annotations

from clipgt2v import ClipGT2VApplicationDefaults
from interactive_drive import InteractiveDriveApplication
from omnidreams.config import (
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication

OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS = ClipGT2VApplicationDefaults(
    title="Interactive Drive",
    slug="interactive-drive",
    total_blocks=0,
    fps=30,
    width=1280,
    height=704,
    pipeline_config=OMNIDREAMS_PIPELINE_CONFIG,
)
OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS = ClipGT2VApplicationDefaults(
    title="Interactive Drive (Perf)",
    slug="interactive-drive-perf",
    total_blocks=0,
    fps=30,
    width=1168,
    height=640,
    pipeline_config=OMNIDREAMS_PERF_PIPELINE_CONFIG,
)


def create_app() -> IApplication:
    """Create Interactive Drive with the regular OmniDreams config."""
    return InteractiveDriveApplication(
        defaults=OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
    )


def create_perf_app() -> IApplication:
    """Create Interactive Drive with the performance OmniDreams config."""
    return InteractiveDriveApplication(
        defaults=OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
    )


__all__ = ["create_app", "create_perf_app"]
