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
from omnidreams_game_engine.input import (
    GAME_DRIVER_COMMAND,
    AnalogDriverCommandConverter,
    AxisCalibration,
    KeyboardDriverCommandConverter,
    analog_state_event,
    normalize_axis,
)
from omnidreams_game_engine.input import game_user_input_schema

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import InputCanonicalizer, UserInputEvent, UserInputs

pytestmark = pytest.mark.ci_cpu


def test_keyboard_is_level_triggered_and_supports_handbrake() -> None:
    canonicalizer = InputCanonicalizer((KeyboardDriverCommandConverter(),))
    window = TimeWindow(start_s=0.0, end_s=1.0)
    inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.1, event_type="key_down", payload={"key": "w"}
            ),
            UserInputEvent(
                timestamp_s=0.2, event_type="key_down", payload={"key": "space"}
            ),
        )
    )
    first = canonicalizer.canonicalize(
        inputs, window=window, source_schema=game_user_input_schema()
    )
    quiet = canonicalizer.canonicalize(
        UserInputs(), window=window, source_schema=game_user_input_schema()
    )
    assert first.values[GAME_DRIVER_COMMAND.name]["throttle"] == 1.0
    assert quiet.values[GAME_DRIVER_COMMAND.name]["handbrake"] is True


def test_connected_analog_device_has_priority_then_yields_to_keyboard() -> None:
    canonicalizer = InputCanonicalizer(
        (AnalogDriverCommandConverter(priority=100), KeyboardDriverCommandConverter())
    )
    window = TimeWindow(start_s=0.0, end_s=1.0)
    events = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.1, event_type="key_down", payload={"key": "w"}
            ),
            analog_state_event(timestamp_s=0.2, steer=-0.5, throttle=0.25, brake=0.0),
        )
    )
    analog = canonicalizer.canonicalize(
        events, window=window, source_schema=game_user_input_schema()
    )
    assert analog.values[GAME_DRIVER_COMMAND.name]["throttle"] == 0.25
    disconnected = canonicalizer.canonicalize(
        UserInputs(
            events=(
                analog_state_event(
                    timestamp_s=0.3, steer=0, throttle=0, brake=0, connected=False
                ),
            )
        ),
        window=window,
        source_schema=game_user_input_schema(),
    )
    assert disconnected.values[GAME_DRIVER_COMMAND.name]["throttle"] == 1.0


def test_axis_calibration_supports_center_deadzone_and_inversion() -> None:
    centered = AxisCalibration(minimum=0, maximum=1000, center=500, deadzone=0.1)
    pedal = AxisCalibration(minimum=0, maximum=255, inverted=True)
    assert normalize_axis(530, centered, centered=True) == 0.0
    assert normalize_axis(1000, centered, centered=True) == pytest.approx(1.0)
    assert normalize_axis(0, pedal, centered=False) == pytest.approx(1.0)
