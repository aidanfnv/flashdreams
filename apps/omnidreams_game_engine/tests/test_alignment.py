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

from __future__ import annotations

import pytest
from omnidreams_game_engine import (
    CausalStateAligner,
    DriverCommand,
    EngineFrame,
    VehicleState,
)

pytestmark = pytest.mark.ci_cpu


def _frame(index: int) -> EngineFrame:
    return EngineFrame(
        timestamp_us=index,
        vehicle=VehicleState(x_m=float(index), y_m=0.0),
        command=DriverCommand(),
    )


def test_alignment_delays_state_across_chunk_boundaries() -> None:
    aligner = CausalStateAligner()
    aligner.reset(_frame(0))
    first = aligner.align((_frame(1), _frame(2)))
    second = aligner.align((_frame(3),))
    assert [frame.timestamp_us for frame in first] == [0, 1]
    assert [frame.timestamp_us for frame in second] == [2]
