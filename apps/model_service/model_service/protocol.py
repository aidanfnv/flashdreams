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

"""Bounded request/response framing for the FlashDreams model service."""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from typing import Any

PROTOCOL = "flashdreams-model-service"
"""Protocol name negotiated by the initial hello request."""
VERSION = 1
"""Prototype command and payload contract version."""
MAX_HEADER = 16 * 1024
"""Maximum UTF-8 JSON envelope size in bytes."""
MAX_PAYLOAD = 12 * 1024 * 1024
"""Maximum RGB payload size in bytes."""
_PREFIX = struct.Struct("!4sII")
"""FDM1 magic followed by network-order header and payload lengths."""


def send_message(
    connection: socket.socket, header: Mapping[str, Any], payload: bytes = b""
) -> None:
    """Write one metadata envelope and optional raw RGB payload."""
    encoded = json.dumps(
        dict(header), ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if not 0 < len(encoded) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError("Model-service message exceeds size limits")
    connection.sendall(_PREFIX.pack(b"FDM1", len(encoded), len(payload)) + encoded)
    if payload:
        connection.sendall(payload)


def receive_message(connection: socket.socket) -> tuple[dict[str, Any], bytes]:
    """Read a complete bounded message, including fragmented TCP arrivals.

    Raises:
        EOFError: The connection closes before a message is complete.
        ValueError: Framing or JSON metadata is invalid.
    """
    magic, header_size, payload_size = _PREFIX.unpack(
        _read_exact(connection, _PREFIX.size)
    )
    if magic != b"FDM1":
        raise ValueError("Invalid model-service magic")
    if not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError("Model-service message exceeds size limits")
    header = json.loads(
        _read_exact(connection, header_size).decode("utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(header, dict):
        raise ValueError("Model-service metadata must be an object")
    return header, _read_exact(connection, payload_size)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON number: {value}")


def _read_exact(connection: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise EOFError("Connection closed during model-service message")
        data.extend(chunk)
    return bytes(data)
