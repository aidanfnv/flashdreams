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

"""DOS conversion, protocol errors, and keyboard checks without a model."""

from unittest.mock import Mock

import numpy as np
import pytest
from dos_stunt.dos_output import DosOutput, rgb24_to_vga, stabilize_vga_palette
from dos_stunt.ethernet_transport import FRAME_SIZE, PALETTE_SIZE, send_frame

from .check_controls import check_controls, check_native_decoder
from .check_server import check_startup_error

pytestmark = pytest.mark.ci_cpu


def test_startup_error() -> None:
    """A bad frame must cancel pending DOS startup and reach the producer."""
    check_startup_error()


def test_generic_keyboard_controls() -> None:
    """Forward full keyboard state through normal and rebound app controls."""
    check_controls()


def test_native_keyboard_decoder() -> None:
    """Exercise the native scan decoder used by the DOS interrupt handler."""
    check_native_decoder()


@pytest.mark.parametrize(
    ("contrast", "dark", "light"),
    [(1.0, 73, 214), (1.5, 37, 250), (3.0, 0, 255)],
)
def test_contrast_before_rgb332(contrast: float, dark: int, light: int) -> None:
    """Queue contrast-adjusted VGA indices without changing the RGB source."""
    frame = np.empty((200, 320, 3), dtype=np.uint8)
    frame[:100] = (64, 80, 96)
    frame[100:] = (192, 176, 160)
    original = frame.copy()
    output = DosOutput(contrast=contrast, palette="rgb332")
    output.sock = Mock()

    output.write(frame)

    assert output.pending_pixels == bytes((dark,)) * 32000 + bytes((light,)) * 32000
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("contrast", [-1.0, float("nan"), float("inf")])
def test_invalid_contrast(contrast: float) -> None:
    """Reject invalid contrast factors before opening a DOS connection."""
    with pytest.raises(ValueError, match="contrast must be finite and nonnegative"):
        DosOutput(contrast=contrast)


def test_adaptive_palette_preserves_subtle_colors() -> None:
    """Preserve muted green shades that collapse in the fixed RGB332 cube."""
    red = np.tile(np.arange(48, 208, dtype=np.uint8), (200, 2))
    frame = np.stack((red, red + 12, red - 12), axis=-1)
    original = frame.copy()
    output = DosOutput()
    output.sock = Mock()
    output.write(frame)
    encoded = output.pending_pixels
    assert encoded is not None and len(encoded) == FRAME_SIZE + PALETTE_SIZE
    colors = np.frombuffer(encoded[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    assert colors.max() <= 63
    indices = np.frombuffer(encoded[:FRAME_SIZE], dtype=np.uint8).reshape(200, 320)
    restored = colors[indices].astype(np.float64) * 255 / 63
    assert np.max(np.abs(restored - frame)) <= 255 / 126

    fixed = np.frombuffer(rgb24_to_vga(frame, palette="rgb332"), dtype=np.uint8)
    assert len(np.unique(indices)) == 160
    assert len(np.unique(fixed)) < 30
    np.testing.assert_array_equal(frame, original)


def test_palette_permutation_preserves_colors_and_indices() -> None:
    """Restore every slot when all 256 colors return in a different order."""
    values = np.arange(256, dtype=np.uint8)
    palette = np.stack((values >> 5, (values >> 2) & 7, values & 3), axis=1)
    indices = bytes(range(256)) * 250
    previous = indices + palette.tobytes()
    order = np.roll(values, 67)
    mapping = np.empty(256, dtype=np.uint8)
    mapping[order] = values
    current = (
        mapping[np.frombuffer(indices, dtype=np.uint8)].tobytes()
        + palette[order].tobytes()
    )
    assert stabilize_vga_palette(current, previous) == previous


def _palette_scene(background):
    frame = np.full((200, 320, 3), background, dtype=np.uint8)
    frame[10:20, 10:20] = (255, 0, 0)
    frame[10:20, 30:40] = (255, 0, 255)
    return frame


def test_adaptive_palette_keeps_ui_colors_when_background_changes() -> None:
    """Keep rare red/magenta slots while preserving the exact quantized image."""
    previous = rgb24_to_vga(_palette_scene((0, 0, 0)))
    current = rgb24_to_vga(_palette_scene((240, 240, 240)))
    stable = stabilize_vga_palette(current, previous)
    old_indices = np.frombuffer(previous[:FRAME_SIZE], dtype=np.uint8).reshape(200, 320)
    before = np.frombuffer(current[:FRAME_SIZE], dtype=np.uint8).reshape(200, 320)
    after = np.frombuffer(stable[:FRAME_SIZE], dtype=np.uint8).reshape(200, 320)
    old_palette = np.frombuffer(previous[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    new_palette = np.frombuffer(current[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    stable_palette = np.frombuffer(stable[FRAME_SIZE:], dtype=np.uint8).reshape(256, 3)
    for x in (10, 30):
        assert before[10, x] != old_indices[10, x]
        assert after[10, x] == old_indices[10, x]
        np.testing.assert_array_equal(
            old_palette[after[10, x]], stable_palette[after[10, x]]
        )
    np.testing.assert_array_equal(stable_palette[after], new_palette[before])
    assert stabilize_vga_palette(current, None) is current
    fixed = bytes(FRAME_SIZE)
    assert stabilize_vga_palette(fixed, previous) is fixed


def test_palette_matching_uses_last_sent_frame(monkeypatch) -> None:
    """Match against the displayed palette even when an intermediate frame is dropped."""
    output = DosOutput(max_frames=2)
    output.sock = Mock()
    output.sent = 1
    output.last_pixels = rgb24_to_vga(_palette_scene((0, 0, 0)))
    output.write(_palette_scene((0, 240, 240)))
    dropped = output.pending_pixels
    output.write(_palette_scene((240, 240, 240)))
    current = output.pending_pixels
    expected = stabilize_vga_palette(current, output.last_pixels)
    assert expected != stabilize_vga_palette(current, dropped)
    sent = []

    def deliver(sock, sequence, pixels, *args, **kwargs):
        assert sequence == 1 and pixels == expected
        assert output.inflight_pixels == pixels
        sent.append(pixels)
        return 0

    monkeypatch.setattr("dos_stunt.dos_output.send_frame", deliver)
    output._send_frames()
    assert output.sender_error is None
    assert output.sent == 2 and output.dropped == 1
    assert output.last_pixels == expected and sent == [expected]


@pytest.mark.parametrize("palette", ["adaptive", "rgb332"])
def test_frame_transfer_includes_palette(monkeypatch, palette: str) -> None:
    """Transmit every byte, including the final palette chunk, with matching ACKs."""
    frame = rgb24_to_vga(
        np.full((200, 320, 3), (48, 96, 144), dtype=np.uint8), palette=palette
    )
    link = Mock()

    def acknowledge(sock, acknowledged, on_packet):
        packet = sock.send.call_args.args[0]
        acknowledged.add((int.from_bytes(packet[4:6], "little"), packet[6]))

    monkeypatch.setattr("dos_stunt.ethernet_transport.read_acks", acknowledge)
    assert send_frame(link, 65535, frame, set(), max_attempts=1) == 0
    packets = [call.args[0] for call in link.send.call_args_list]
    count = 47 if palette == "adaptive" else 46
    assert len(packets) == count
    assert b"".join(packet[12:] for packet in packets) == frame
    for chunk, packet in enumerate(packets):
        assert packet[:6] == b"FD\x01\x01\xff\xff"
        assert packet[6:8] == bytes((chunk, count))
        assert int.from_bytes(packet[8:10], "little") == chunk * 1400
        assert int.from_bytes(packet[10:12], "little") == len(packet) - 12
