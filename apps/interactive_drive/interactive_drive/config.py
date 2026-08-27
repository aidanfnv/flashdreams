# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive Drive application configuration."""

from dataclasses import dataclass

from clipgt2v.app import ClipGT2VConfig


@dataclass(frozen=True, slots=True)
class InteractiveDriveConfig(ClipGT2VConfig):
    """Parsed command-line configuration owned by Interactive Drive."""
