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

"""Length-prefixed RGB frame and input messages for external output services."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

PROTOCOL = "flashdreams-frame-stream"
VERSION = 1
MAX_HEADER_BYTES = 64 * 1024
"""Maximum JSON envelope size."""
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
"""Maximum RGB image size; larger images require a revised protocol limit."""
_PREFIX = struct.Struct("!II")
"""Network-order JSON and binary payload lengths."""


def send_message(
    sock: socket.socket, header: Mapping[str, Any], payload: bytes = b""
) -> None:
    """Write a JSON envelope followed by an optional binary payload.

    One thread must own writes to a socket, or callers must serialize them.

    Args:
        sock: Connected blocking stream socket.
        header: JSON-compatible message envelope.
        payload: Binary image data, empty for control messages.

    Raises:
        ValueError: Either part exceeds the protocol's size limit.
    """
    encoded = json.dumps(dict(header), separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    if not 0 < len(encoded) <= MAX_HEADER_BYTES or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Frame-stream message exceeds the size limit")
    sock.sendall(_PREFIX.pack(len(encoded), len(payload)) + encoded)
    if payload:
        sock.sendall(payload)


def receive_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    """Read one bounded message, handling fragmented stream reads.

    Args:
        sock: Connected blocking stream socket.

    Returns:
        The JSON envelope and binary payload.

    Raises:
        EOFError: The peer disconnected, possibly during a message.
        ValueError: The envelope is malformed or exceeds the size limit.
    """
    header_size, payload_size = _PREFIX.unpack(_receive_exact(sock, _PREFIX.size))
    if not 0 < header_size <= MAX_HEADER_BYTES or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError("Invalid frame-stream message size")
    header = json.loads(_receive_exact(sock, header_size))
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ValueError("Frame-stream envelope requires a message type")
    return header, _receive_exact(sock, payload_size)


def hello_message() -> dict[str, Any]:
    """Describe the protocol version."""
    return {
        "type": "hello",
        "protocol": PROTOCOL,
        "version": VERSION,
    }


def check_hello(header: Mapping[str, Any], payload: bytes) -> None:
    """Reject an incompatible greeting before accepting frames or events."""
    if (
        payload
        or header.get("type") != "hello"
        or header.get("protocol") != PROTOCOL
        or type(header.get("version")) is not int
        or header["version"] != VERSION
    ):
        raise ValueError("Incompatible frame-stream greeting")


def frame_shape(header: Mapping[str, Any], payload: bytes) -> tuple[int, int]:
    """Validate one RGB24 image and return its width and height."""
    width, height = header.get("width"), header.get("height")
    if (
        header.get("type") != "frame"
        or header.get("format") != "rgb24"
        or type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or len(payload) != width * height * 3
    ):
        raise ValueError("Invalid RGB24 frame envelope or payload")
    return width, height


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    position = 0
    while position < size:
        count = sock.recv_into(view[position:])
        if count == 0:
            raise EOFError("Frame-stream peer disconnected")
        position += count
    return bytes(data)
