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

"""Canonical keyboard and analog driving input for OmniDreams games."""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    CanonicalModality,
    DeviceConverterSchema,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.keyboard import KeyboardState, normalize_key

from .types import DriverCommand

GAMEPAD_STATE_EVENT = "gamepad_state"
"""Raw level-state event emitted by native HID and browser Gamepad sources."""

GAME_DRIVER_COMMAND = CanonicalModality(
    name="game_driver_command",
    payload_fields=frozenset(
        {"throttle", "brake", "steer", "handbrake", "reverse", "reset"}
    ),
    description="Normalized arcade driving intent shared by every input device.",
)

_GAMEPAD_FIELDS = frozenset(
    {"connected", "throttle", "brake", "steer", "handbrake", "reverse", "reset"}
)


def game_user_input_schema() -> UserInputSchema:
    """Describe raw keyboard and analog events understood by engine converters."""
    return UserInputSchema(
        capabilities=(
            UserInputCapability(
                event_type="key_down",
                input_modality="keyboard",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_type="key_up",
                input_modality="keyboard",
                payload_fields=frozenset({"key"}),
            ),
            UserInputCapability(
                event_type=GAMEPAD_STATE_EVENT,
                input_modality="analog-driving",
                payload_fields=_GAMEPAD_FIELDS,
            ),
        ),
        description="Keyboard and wheel/gamepad driving controls.",
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class AxisCalibration:
    """Calibration for one absolute wheel or pedal axis."""

    minimum: float
    maximum: float
    center: float | None = None
    inverted: bool = False
    deadzone: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Axis calibration bounds must be finite.")
        if self.maximum <= self.minimum:
            raise ValueError("Axis calibration maximum must exceed minimum.")
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("Axis calibration deadzone must be in [0, 1).")


@dataclass(frozen=True, kw_only=True, slots=True)
class WheelProfile:
    """Data-only native or browser wheel calibration profile."""

    name: str
    steer: AxisCalibration
    throttle: AxisCalibration
    brake: AxisCalibration
    device_path: Path | None = None
    axis_codes: Mapping[str, int] = field(default_factory=dict)
    button_codes: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "WheelProfile":
        """Load a wheel profile from a YAML file."""
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Wheel profile must contain a YAML mapping.")
        axes = payload.get("axes")
        if not isinstance(axes, Mapping):
            raise ValueError("Wheel profile requires an 'axes' mapping.")
        return cls(
            name=str(payload.get("name", path.stem)),
            device_path=(
                None
                if payload.get("device_path") is None
                else Path(str(payload["device_path"]))
            ),
            steer=_axis_from_mapping(axes.get("steer"), centered=True),
            throttle=_axis_from_mapping(axes.get("throttle"), centered=False),
            brake=_axis_from_mapping(axes.get("brake"), centered=False),
            axis_codes=_integer_mapping(payload.get("axis_codes", {})),
            button_codes=_integer_mapping(payload.get("button_codes", {})),
        )


class KeyboardDriverCommandConverter:
    """Convert keyboard edges into the shared arcade driving modality."""

    _BINDINGS = {
        "throttle": frozenset({"w", "up"}),
        "brake": frozenset({"s", "down"}),
        "left": frozenset({"a", "left"}),
        "right": frozenset({"d", "right"}),
        "handbrake": frozenset({"space"}),
        "reverse": frozenset({"r"}),
        "reset": frozenset({"escape"}),
    }

    def __init__(self, *, priority: int = 0) -> None:
        keys = frozenset(key for values in self._BINDINGS.values() for key in values)
        self._supported_keys = keys
        self._state = KeyboardState(supported_keys=keys)
        self._schema = DeviceConverterSchema(
            name="game-keyboard-driver-command",
            produces=GAME_DRIVER_COMMAND,
            device_kind="keyboard",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type="key_down", payload_fields=frozenset({"key"})
                ),
                UserInputCapability(
                    event_type="key_up", payload_fields=frozenset({"key"})
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        """Return the converter compatibility declaration."""
        return self._schema

    def reset(self) -> None:
        """Clear held-key state at a rollout boundary."""
        self._state = KeyboardState(supported_keys=self._supported_keys)

    def convert(
        self, user_inputs: UserInputs, window: TimeWindow
    ) -> Mapping[str, Any] | None:
        """Convert keyboard events into one level-triggered command."""
        del window
        for event in user_inputs.events:
            if event.event_type not in {"key_down", "key_up"}:
                continue
            key = event.payload.get("key")
            if isinstance(key, str):
                self._state.apply_event(
                    event="keydown" if event.event_type == "key_down" else "keyup",
                    key=key,
                )
        pressed = {normalize_key(key) for key in self._state.snapshot()}

        def held(action: str) -> bool:
            return bool(self._BINDINGS[action] & pressed)

        command = DriverCommand(
            throttle=1.0 if held("throttle") else 0.0,
            brake=1.0 if held("brake") else 0.0,
            steer=(1.0 if held("left") else 0.0) - (1.0 if held("right") else 0.0),
            handbrake=held("handbrake"),
            reverse=held("reverse"),
            reset=held("reset"),
        )
        return GAME_DRIVER_COMMAND.value(command.as_payload())


class AnalogDriverCommandConverter:
    """Convert native-wheel or browser-gamepad state into driving intent."""

    def __init__(self, *, priority: int = 100) -> None:
        self._connected = False
        self._command = DriverCommand()
        self._schema = DeviceConverterSchema(
            name="game-analog-driver-command",
            produces=GAME_DRIVER_COMMAND,
            device_kind="wheel",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type=GAMEPAD_STATE_EVENT,
                    payload_fields=_GAMEPAD_FIELDS,
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        """Return the converter compatibility declaration."""
        return self._schema

    def reset(self) -> None:
        """Drop retained analog device state."""
        self._connected = False
        self._command = DriverCommand()

    def convert(
        self, user_inputs: UserInputs, window: TimeWindow
    ) -> Mapping[str, Any] | None:
        """Return the latest analog command, yielding to keyboard when disconnected."""
        del window
        for event in user_inputs.events:
            if event.event_type != GAMEPAD_STATE_EVENT:
                continue
            self._connected = bool(event.payload.get("connected", True))
            if self._connected:
                self._command = DriverCommand.from_payload(event.payload)
            else:
                self._command = DriverCommand()
        if not self._connected:
            return None
        return GAME_DRIVER_COMMAND.value(self._command.as_payload())


class EvdevWheelReader:
    """Read one Linux evdev wheel and emit canonicalizable analog events."""

    _EVENT_STRUCT = struct.Struct("llHHi")
    _EV_KEY = 0x01
    _EV_ABS = 0x03

    def __init__(self, profile: WheelProfile) -> None:
        if profile.device_path is None:
            raise ValueError("Native wheel profiles require device_path.")
        self.profile = profile
        self._device_path = profile.device_path
        self._handle: BinaryIO | None = None
        self._axes: dict[str, float] = {
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
        }
        self._buttons: dict[str, bool] = {}

    def open(self) -> None:
        """Open the configured evdev node without blocking reads."""
        self.close()
        self._handle = self._device_path.open("rb", buffering=0)

    def close(self) -> None:
        """Close the evdev node idempotently."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def sample(self, *, timestamp_s: float | None = None) -> UserInputEvent:
        """Drain available reports and return one normalized state event."""
        if self._handle is None:
            raise RuntimeError(
                "EvdevWheelReader.open() must be called before sample()."
            )
        import os

        os.set_blocking(self._handle.fileno(), False)
        connected = True
        while True:
            try:
                record = self._handle.read(self._EVENT_STRUCT.size)
            except BlockingIOError:
                break
            except OSError:
                connected = False
                break
            if not record:
                connected = False
                break
            if len(record) != self._EVENT_STRUCT.size:
                break
            _seconds, _microseconds, event_type, code, value = (
                self._EVENT_STRUCT.unpack(record)
            )
            if event_type == self._EV_ABS:
                for name, axis_code in self.profile.axis_codes.items():
                    if code == axis_code:
                        calibration = getattr(self.profile, name)
                        self._axes[name] = normalize_axis(
                            value, calibration, centered=name == "steer"
                        )
            elif event_type == self._EV_KEY:
                for name, button_code in self.profile.button_codes.items():
                    if code == button_code:
                        self._buttons[name] = bool(value)
        return analog_state_event(
            timestamp_s=time.monotonic() if timestamp_s is None else timestamp_s,
            steer=self._axes["steer"],
            throttle=self._axes["throttle"],
            brake=self._axes["brake"],
            handbrake=self._buttons.get("handbrake", False),
            reverse=self._buttons.get("reverse", False),
            reset=self._buttons.get("reset", False),
            connected=connected,
            source="evdev",
        )


def analog_state_event(
    *,
    timestamp_s: float,
    steer: float,
    throttle: float,
    brake: float,
    handbrake: bool = False,
    reverse: bool = False,
    reset: bool = False,
    connected: bool = True,
    source: str = "gamepad",
) -> UserInputEvent:
    """Build the raw event shared by native and browser analog devices."""
    command = DriverCommand(
        steer=steer,
        throttle=throttle,
        brake=brake,
        handbrake=handbrake,
        reverse=reverse,
        reset=reset,
    )
    return UserInputEvent(
        timestamp_s=timestamp_s,
        event_type=GAMEPAD_STATE_EVENT,
        payload={"connected": connected, **command.as_payload()},
        source=source,
    )


def normalize_axis(
    value: float,
    calibration: AxisCalibration,
    *,
    centered: bool,
) -> float:
    """Normalize a raw axis to ``[-1, 1]`` or ``[0, 1]``."""
    raw = float(value)
    if centered:
        center = (
            (calibration.minimum + calibration.maximum) * 0.5
            if calibration.center is None
            else calibration.center
        )
        span = max(center - calibration.minimum, calibration.maximum - center)
        normalized = 0.0 if span <= 0.0 else (raw - center) / span
        if calibration.inverted:
            normalized = -normalized
        magnitude = abs(normalized)
        if magnitude <= calibration.deadzone:
            return 0.0
        normalized = math.copysign(
            (magnitude - calibration.deadzone) / (1.0 - calibration.deadzone),
            normalized,
        )
        return max(-1.0, min(1.0, normalized))
    normalized = (raw - calibration.minimum) / (
        calibration.maximum - calibration.minimum
    )
    if calibration.inverted:
        normalized = 1.0 - normalized
    if normalized <= calibration.deadzone:
        return 0.0
    normalized = (normalized - calibration.deadzone) / (1.0 - calibration.deadzone)
    return max(0.0, min(1.0, normalized))


def _axis_from_mapping(value: object, *, centered: bool) -> AxisCalibration:
    if not isinstance(value, Mapping):
        raise ValueError("Every wheel axis requires a calibration mapping.")
    fields = {str(name): field_value for name, field_value in value.items()}
    return AxisCalibration(
        minimum=float(str(fields["minimum"])),
        maximum=float(str(fields["maximum"])),
        center=(
            float(str(fields["center"]))
            if centered and fields.get("center") is not None
            else None
        ),
        inverted=bool(fields.get("inverted", False)),
        deadzone=float(str(fields.get("deadzone", 0.0))),
    )


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("Wheel code bindings must be a mapping.")
    return {str(name): int(str(code)) for name, code in value.items()}


__all__ = [
    "GAMEPAD_STATE_EVENT",
    "GAME_DRIVER_COMMAND",
    "AnalogDriverCommandConverter",
    "AxisCalibration",
    "EvdevWheelReader",
    "KeyboardDriverCommandConverter",
    "WheelProfile",
    "analog_state_event",
    "game_user_input_schema",
    "normalize_axis",
]
