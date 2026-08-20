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

"""Crazy Robotaxi gameplay and runtime application."""

from .app import (
    CrazyRobotaxiSessionConfig,
    CrazyRobotaxiStepResult,
    CrazyRobotaxiV1ApplicationAdapter,
    CrazyRobotaxiV1SessionAdapter,
    CrazyRobotaxiV2Application,
    CrazyRobotaxiV2Session,
    create_app,
    create_v2_app,
)
from .game import CrazyRobotaxiGame, TaxiGameConfig, TaxiPhase, TaxiSessionState
from .high_scores import HighScoreEntry, HighScoreStore

__all__ = [
    "CrazyRobotaxiStepResult",
    "CrazyRobotaxiSessionConfig",
    "CrazyRobotaxiV1ApplicationAdapter",
    "CrazyRobotaxiV1SessionAdapter",
    "CrazyRobotaxiV2Application",
    "CrazyRobotaxiV2Session",
    "CrazyRobotaxiGame",
    "HighScoreEntry",
    "HighScoreStore",
    "TaxiGameConfig",
    "TaxiPhase",
    "TaxiSessionState",
    "create_app",
    "create_v2_app",
]
