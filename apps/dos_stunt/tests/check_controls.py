#!/usr/bin/env python3
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

"""Check native keyboard decoding, DOS input, M, and rebound controls."""

from dataclasses import replace
from functools import partial
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from crazy_robotaxi.controls import (
    ControlsConfig,
    InputBinding,
    keyboard_drive_key,
    keyboard_driver_command,
)
from crazy_robotaxi.ui import TaxiHudState
from dos_stunt.dos_output import DosOutput
from dos_stunt.ethernet_transport import ETHERTYPE, GUEST_MAC, HOST_MAC
from dos_stunt.keyboard import KEY_BYTES, pressed_keys
from omnidreams_game_engine.input import DriverInput

from dos_stunt.stream_client_window import input_event_from_message
from flashdreams.runtime_v2.user_input_event import CloseUserInputEvent
from flashdreams.runtime_v2.user_input_events import UserInputEvents


def packet(sequence: int, *scans: int) -> bytes:
    snapshot = bytearray(KEY_BYTES)
    for scan in scans:
        snapshot[scan >> 3] |= 1 << (scan & 7)
    return (
        HOST_MAC
        + GUEST_MAC
        + ETHERTYPE
        + b"FD\x01\x06"
        + sequence.to_bytes(2, "little")
        + snapshot
    ).ljust(60, b"\0")


def check_controls() -> None:
    output = DosOutput()
    defaults = ControlsConfig()
    controls = replace(
        defaults,
        keyboard=replace(
            defaults.keyboard, drive_forward=(InputBinding("key", "i"), None)
        ),
    )
    driver = DriverInput(
        keyboard_command=partial(keyboard_driver_command, controls.keyboard),
        key_normalizer=partial(keyboard_drive_key, controls.keyboard),
    )
    ui = TaxiHudState(320, 200, None, controls=controls)
    ui._menu_stage = "game"

    def send(sequence, *scans):
        output._on_packet(packet(sequence, *scans))
        events = UserInputEvents(
            [
                input_event_from_message(message, timestamp=np.uint64(0))
                for message in output.get_input_messages()
            ]
        )
        driver.apply(events)
        ui.consume_input_events(events)
        return events

    send(0xFFFF, 0x17, 0x1E, 0x32)  # I (rebound throttle), A, M.
    assert driver.command().throttle == 1.0 and driver.command().steer == 1.0
    assert ui.show_hdmap
    send(0, 0x17, 0x1E, 0x32)  # Held M must not toggle again; sequence wraps.
    assert ui.show_hdmap
    send(0, 0x11)  # Duplicate snapshots cannot change input.
    assert driver.command().throttle == 1.0
    send(1)
    assert driver.command().throttle == 0.0 and driver.command().steer == 0.0
    send(2, 0x11)  # W is no longer bound to throttle.
    assert driver.command().throttle == 0.0
    send(3, 0x32)
    assert not ui.show_hdmap
    send(4, 0xC8, 0x3C, 0x2A, 0x36)  # Up, F2, both Shift keys.
    assert output.pressed_keys == {"up", "f2", "Shift"}
    send(5, 0x36)
    assert output.pressed_keys == {"Shift"}
    output.last_control_at -= 1
    messages = output.get_input_messages()
    assert messages == [{"type": "keyboard", "key": "Shift", "pressed": False}]
    output._on_packet(packet(6, 0x01))
    events = [
        input_event_from_message(m, timestamp=np.uint64(0))
        for m in output.get_input_messages()
    ]
    assert events[0].key == "escape"
    assert not any(isinstance(e, CloseUserInputEvent) for e in events)
    assert pressed_keys(packet(7, 0x48)[20:52]) == {"kp_8"}
    print(
        "DOS input: M toggles HD map; rebound I drives; arrows/modifiers/releases checked"
    )


def check_native_decoder() -> None:
    with tempfile.TemporaryDirectory(prefix="dos-keyboard-") as temporary:
        executable = Path(temporary) / "check-keyboard"
        subprocess.run(
            [
                "cc",
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(Path(__file__).with_name("check_keyboard.c")),
                "-o",
                str(executable),
            ],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    check_native_decoder()
    check_controls()
