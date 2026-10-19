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

"""Decode PC scan-set-1 snapshots into application-independent key names."""

KEY_BYTES = 32
"""128 ordinary key positions followed by 128 E0-prefixed positions."""

_SCAN_KEYS = {
    0x01: "escape",
    0x0E: "backspace",
    0x0F: "tab",
    0x1C: "enter",
    0x1D: "Control",
    0x2A: "Shift",
    0x36: "Shift",
    0x38: "Alt",
    0x39: "space",
    0x3A: "caps_lock",
    0x45: "num_lock",
    0x46: "scroll_lock",
    0x37: "kp_multiply",
    0x47: "kp_7",
    0x48: "kp_8",
    0x49: "kp_9",
    0x4A: "kp_subtract",
    0x4B: "kp_4",
    0x4C: "kp_5",
    0x4D: "kp_6",
    0x4E: "kp_add",
    0x4F: "kp_1",
    0x50: "kp_2",
    0x51: "kp_3",
    0x52: "kp_0",
    0x53: "kp_decimal",
    0x56: "\\",
    0x57: "f11",
    0x58: "f12",
    0x9C: "kp_enter",
    0x9D: "Control",
    0xB5: "kp_divide",
    0xB7: "print_screen",
    0xB8: "Alt",
    0xC7: "home",
    0xC8: "up",
    0xC9: "page_up",
    0xCB: "left",
    0xCD: "right",
    0xCF: "end",
    0xD0: "down",
    0xD1: "page_down",
    0xD2: "insert",
    0xD3: "delete",
    0xDB: "Meta",
    0xDC: "Meta",
    0xDD: "menu",
    0xFF: "pause",
}
for _start, _characters in (
    (0x02, "1234567890-="),
    (0x10, "qwertyuiop[]"),
    (0x1E, "asdfghjkl;'`"),
    (0x2B, "\\zxcvbnm,./"),
):
    _SCAN_KEYS.update(enumerate(_characters, _start))
_SCAN_KEYS.update((0x3B + i, f"f{i + 1}") for i in range(10))


def pressed_keys(snapshot: bytes) -> set[str]:
    """Decode a full keyboard bitmap using standard US PC key names.

    Different physical modifier keys share one logical name. Releasing one
    Shift key therefore leaves Shift pressed while the other remains down.
    """
    if len(snapshot) != KEY_BYTES:
        raise ValueError("Expected a 32-byte DOS keyboard snapshot")
    return {
        key
        for scan, key in _SCAN_KEYS.items()
        if snapshot[scan >> 3] & (1 << (scan & 7))
    }
