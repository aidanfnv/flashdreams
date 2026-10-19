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

"""Cancellable model-service requests using the existing RGB protocol."""

import argparse
import re
import socket
import sys
import threading

from model_service.protocol import PROTOCOL, VERSION, receive_message, send_message


def server_endpoint(value: str) -> tuple[str, str, int]:
    """Parse a named IPv4 or hostname endpoint from the command line."""
    name, separator, address = value.partition("=")
    host, colon, port = address.rpartition(":")
    if (
        not separator
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,62}", name)
        or not colon
        or not host
        or any(c.isspace() or c in ":/=" for c in host)
        or not port.isascii()
        or not port.isdecimal()
        or not 1 <= int(port) <= 65535
    ):
        raise argparse.ArgumentTypeError("Expected NAME=HOST:PORT (port 1..65535)")
    return name, host, int(port)


class RequestCancelled(Exception):
    """The local window requested shutdown while a remote request was pending."""


class _RequestSocket:
    """Poll a socket without losing partial protocol headers or payloads."""

    def __init__(self, connection: socket.socket, stop: threading.Event | None):
        self.connection = connection
        self.stop = stop

    def _check_stop(self) -> None:
        if self.stop is not None and self.stop.is_set():
            raise RequestCancelled()

    def recv(self, size: int) -> bytes:
        while True:
            self._check_stop()
            try:
                return self.connection.recv(size)
            except socket.timeout:
                continue

    def sendall(self, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            self._check_stop()
            try:
                sent = self.connection.send(remaining)
            except socket.timeout:
                continue
            if sent == 0:
                raise ConnectionError("The remote service disconnected")
            remaining = remaining[sent:]


class ModelConnection:
    """Own one sequential model-service connection, with no inference deadline."""

    def __init__(self, host: str, port: int):
        self.socket = socket.create_connection((host, port), timeout=10)
        self.socket.settimeout(0.2)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._request_id = 0
        self.broken = False
        self.session_open = False
        try:
            hello, payload = self.exchange(
                "hello", protocol=PROTOCOL, version=VERSION, status_messages=True
            )
            maximum = hello.get("max_frames_per_step")
            if (
                payload
                or hello.get("type") != "hello"
                or hello.get("protocol") != PROTOCOL
                or hello.get("version") != VERSION
                or hello.get("frame_format") != "rgb24"
                or hello.get("end_of_stream") != "finished"
                or type(maximum) is not int
                or not 1 <= maximum <= 4
            ):
                raise ValueError("Incompatible remote model service")
            self.max_frames = maximum
            selection = hello.get("device_selection", False)
            if type(selection) is not bool:
                raise ValueError("Invalid remote device-selection capability")
            self.device_selection = selection
            pagination = hello.get("catalog_pagination", False)
            if type(pagination) is not bool:
                raise ValueError("Invalid remote catalog-pagination capability")
            self.catalog_pagination = pagination
        except BaseException:
            self.close()
            raise

    def exchange(
        self,
        operation: str,
        payload: bytes = b"",
        *,
        stop: threading.Event | None = None,
        **fields,
    ) -> tuple[dict, bytes]:
        """Send one request and consume its progress messages and final reply.

        Args:
            operation: Model-service operation to request.
            payload: Optional packed RGB input.
            stop: Local session shutdown event, polled during socket I/O.
            **fields: Operation-specific JSON fields.

        Raises:
            RequestCancelled: The local session stopped during the request.
            ValueError: The server rejected the request or sent an invalid reply.
        """
        self._request_id += 1
        connection = _RequestSocket(self.socket, stop)
        try:
            send_message(
                connection,
                {"id": self._request_id, "op": operation, **fields},
                payload,
            )
            while True:
                header, result = receive_message(connection)
                if header.get("id") != self._request_id:
                    raise ValueError("Remote response request ID does not match")
                kind = header.get("type")
                if kind == "status":
                    if result:
                        raise ValueError("Remote status unexpectedly contains pixels")
                    if header.get("phase") in ("loading", "reusing", "preparing"):
                        print(
                            f"Remote service: {header['phase']}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                if kind == "error":
                    raise ValueError(f"Remote service: {header.get('message')}")
                return header, result
        except BaseException:
            # A cancelled/failed read may have consumed part of a message.
            self.broken = True
            raise

    def models(self, *, stop: threading.Event | None = None) -> list[dict]:
        """Discover every model, using bounded catalog pages when supported."""
        models = []
        identifiers = set()
        while True:
            fields = {"offset": len(models)} if self.catalog_pagination else {}
            header, payload = self.exchange("models", stop=stop, **fields)
            entries = header.get("models")
            if (
                header.get("type") != "models"
                or payload
                or not isinstance(entries, list)
            ):
                raise ValueError("Invalid remote model catalog")
            for entry in entries:
                identifier = entry.get("id") if isinstance(entry, dict) else None
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier in identifiers
                ):
                    raise ValueError("Invalid or duplicate remote model ID")
                identifiers.add(identifier)
                models.append(entry)
            if self.catalog_pagination and "next_offset" not in header:
                raise ValueError("Missing remote model catalog continuation")
            offset = header.get("next_offset")
            if offset is None:
                return models
            if (
                not self.catalog_pagination
                or not entries
                or type(offset) is not int
                or offset != len(models)
            ):
                raise ValueError("Invalid remote model catalog continuation")

    def devices(self) -> dict:
        """Read and validate the server's device names, memory, and job availability."""
        if not self.device_selection:
            raise ValueError("The remote service does not support device discovery")
        header, payload = self.exchange("devices")
        entries = header.get("devices")
        if (
            header.get("type") != "devices"
            or payload
            or not isinstance(entries, list)
            or not entries
        ):
            raise ValueError("Invalid remote device catalog")
        identifiers = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid remote device entry")
            identifier, memory = entry.get("id"), entry.get("total_memory")
            if (
                not isinstance(identifier, str)
                or not identifier
                or not isinstance(entry.get("name"), str)
                or not isinstance(entry.get("type"), str)
                or type(entry.get("busy")) is not bool
                or (memory is not None and (type(memory) is not int or memory < 0))
            ):
                raise ValueError("Invalid remote device entry")
            identifiers.append(identifier)
        if (
            len(set(identifiers)) != len(identifiers)
            or header.get("default_device") not in identifiers
        ):
            raise ValueError("Invalid remote default or duplicate device ID")
        return header

    def close_session(self) -> None:
        """Release a healthy remote job, or disconnect after an interrupted request."""
        if self.broken:
            self.close()
        elif self.session_open:
            try:
                header, payload = self.exchange("close")
                if header.get("type") != "closed" or payload:
                    raise ValueError("Invalid remote close response")
            finally:
                self.session_open = False

    def close(self) -> None:
        """Disconnect and let the service release any remaining job."""
        self.socket.close()
        self.session_open = False
