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
    AnalogDriverCommandConverter,
    AxisCalibration,
    DriverCommandEventData,
    KeyboardDriverCommandConverter,
    analog_state_event,
    keyboard_key_event,
    normalize_axis,
)

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


def test_keyboard_is_level_triggered_and_supports_handbrake() -> None:
    converter = KeyboardDriverCommandConverter()
    inputs = UserInputEvents(
        [
            keyboard_key_event(timestamp_us=100_000, key="w", pressed=True),
            keyboard_key_event(timestamp_us=200_000, key="space", pressed=True),
        ]
    )
    first = converter.convert(inputs)
    quiet = converter.convert(UserInputEvents([]))
    assert isinstance(first, DriverCommandEventData)
    assert first.command.throttle == 1.0
    assert quiet.command.handbrake is True


def test_connected_analog_device_has_priority_then_yields_to_keyboard() -> None:
    analog_converter = AnalogDriverCommandConverter()
    keyboard_converter = KeyboardDriverCommandConverter()
    events = UserInputEvents(
        [
            keyboard_key_event(timestamp_us=100_000, key="w", pressed=True),
            analog_state_event(
                timestamp_us=200_000,
                steer=-0.5,
                throttle=0.25,
                brake=0.0,
            ),
        ]
    )
    keyboard = keyboard_converter.convert(events)
    analog = analog_converter.convert(events)
    assert analog is not None
    assert analog.command.throttle == 0.25
    disconnected = analog_converter.convert(
        UserInputEvents(
            [
                analog_state_event(
                    timestamp_us=300_000,
                    steer=0,
                    throttle=0,
                    brake=0,
                    connected=False,
                )
            ]
        )
    )
    assert disconnected is None
    assert keyboard.command.throttle == 1.0


def test_axis_calibration_supports_center_deadzone_and_inversion() -> None:
    centered = AxisCalibration(minimum=0, maximum=1000, center=500, deadzone=0.1)
    pedal = AxisCalibration(minimum=0, maximum=255, inverted=True)
    assert normalize_axis(530, centered, centered=True) == 0.0
    assert normalize_axis(1000, centered, centered=True) == pytest.approx(1.0)
    assert normalize_axis(0, pedal, centered=False) == pytest.approx(1.0)
