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

"""Shared TCP model-service lifecycle for synthetic and FlashDreams backends."""

from __future__ import annotations

import json
import logging
import math
import socket
import socketserver
from typing import Any

from model_service.protocol import (
    MAX_HEADER,
    PROTOCOL,
    VERSION,
    receive_message,
    send_message,
)

MAX_FRAMES = 4
"""Maximum frames returned by one request, bounding output allocation."""


def _parameters(values: Any, schema: dict[str, dict]) -> dict[str, Any]:
    if not isinstance(values, dict) or values.keys() - schema.keys():
        raise ValueError("Unknown or invalid parameters")
    result = {}
    for name, spec in schema.items():
        value = values.get(name, spec["default"])
        kind = spec["type"]
        if kind in ("integer", "number"):
            if (
                (
                    type(value) is not int
                    if kind == "integer"
                    else type(value) not in (int, float)
                )
                or not spec["minimum"] <= value <= spec["maximum"]
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{name} must be {kind} in {spec['minimum']}..{spec['maximum']}"
                )
        elif kind == "boolean":
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
        elif kind == "string":
            if not isinstance(value, str) or len(value) > spec["max_length"]:
                raise ValueError(
                    f"{name} must be a string of at most {spec['max_length']} characters"
                )
        else:
            raise ValueError(f"Unsupported setting type: {kind}")
        result[name] = value
    return result


def serve_connection(connection: socket.socket, backend: Any, capture=None) -> None:
    """Serve one isolated session and release it on close, failure, or disconnect."""
    ready = False
    wants_status = False
    session = None
    logger = logging.getLogger(__name__)
    try:
        while True:
            try:
                request, payload = receive_message(connection)
            except EOFError:
                return
            except ValueError as error:
                send_message(
                    connection,
                    {
                        "id": 0,
                        "type": "error",
                        "code": "protocol",
                        "message": str(error),
                    },
                )
                return
            request_id = request.get("id", 0)
            operation = request.get("op")
            logger.debug("Request %s: %s", request_id, operation)

            def status(phase, **fields):
                if wants_status:
                    send_message(
                        connection,
                        {"id": request_id, "type": "status", "phase": phase, **fields},
                    )

            try:
                if type(request_id) is not int or not 1 <= request_id <= 0xFFFFFFFF:
                    request_id = 0
                    raise ValueError("id must be an integer in 1..4294967295")
                if not ready:
                    if (
                        operation != "hello"
                        or payload
                        or request.get("protocol") != PROTOCOL
                        or type(request.get("version")) is not int
                        or request["version"] != VERSION
                        or type(request.get("status_messages", False)) is not bool
                    ):
                        raise ValueError("Expected compatible model-service hello")
                    response = {
                        "type": "hello",
                        "protocol": PROTOCOL,
                        "version": VERSION,
                        "max_frames_per_step": MAX_FRAMES,
                        "frame_format": "rgb24",
                        "end_of_stream": "finished",
                        "status_messages": True,
                        "catalog_pagination": True,
                        "device_selection": callable(getattr(backend, "devices", None)),
                    }
                    wants_status = request.get("status_messages", False)
                    ready = True
                elif operation == "models":
                    if payload:
                        raise ValueError("models does not accept a payload")
                    response = {"type": "models", "models": backend.catalog()}
                    if "offset" in request:
                        offset = request["offset"]
                        entries = response["models"]
                        if type(offset) is not int or not 0 <= offset <= len(entries):
                            raise ValueError("Invalid model catalog offset")
                        response = {"type": "models", "models": [], "next_offset": None}
                        for index in range(offset, len(entries)):
                            page = {
                                "id": request_id,
                                "type": "models",
                                "models": [*response["models"], entries[index]],
                                "next_offset": index + 1
                                if index + 1 < len(entries)
                                else None,
                            }
                            if (
                                len(json.dumps(page, separators=(",", ":")).encode())
                                > MAX_HEADER
                            ):
                                if not response["models"]:
                                    raise ValueError(
                                        "A model catalog entry exceeds the protocol limit"
                                    )
                                break
                            response = page
                    if (
                        len(
                            json.dumps(
                                {"id": request_id, **response}, separators=(",", ":")
                            ).encode()
                        )
                        > MAX_HEADER
                    ):
                        raise ValueError(
                            "Model catalog exceeds the protocol limit; use a client supporting catalog pagination"
                        )
                elif operation == "devices":
                    if payload:
                        raise ValueError("devices does not accept a payload")
                    if not callable(getattr(backend, "devices", None)):
                        raise ValueError(
                            "This service does not support device discovery"
                        )
                    response = {"type": "devices", **backend.devices()}
                elif operation == "open":
                    if session is not None:
                        raise ValueError(
                            "Close the current session before opening another"
                        )
                    options = {"status": status} if wants_status else {}
                    if "device" in request:
                        if not callable(getattr(backend, "devices", None)):
                            raise ValueError(
                                "This service does not support device selection"
                            )
                        if (
                            not isinstance(request["device"], str)
                            or not request["device"]
                        ):
                            raise ValueError("device must be a nonempty string")
                        options["device"] = request["device"]
                    session = backend.open(
                        request.get("model"),
                        request.get("parameters", {}),
                        payload,
                        **options,
                    )
                    response = {
                        "type": "session",
                        "model": session.model,
                        "parameters": session.parameters,
                        **(
                            {"device": session.device}
                            if hasattr(session, "device")
                            else {}
                        ),
                    }
                elif operation == "close":
                    if payload:
                        raise ValueError("close does not accept a payload")
                    if session is not None:
                        session.close()
                        session = None
                    response = {"type": "closed"}
                elif operation in ("step", "reset"):
                    if session is None:
                        raise ValueError("Open a model session first")
                    if payload:
                        raise ValueError(f"{operation} does not accept a payload")
                    if operation == "step":
                        response, result = session.step(
                            request.get("count", 1),
                            request.get("inputs", {}),
                            **({"status": status} if wants_status else {}),
                        )
                        if capture is not None and response["type"] == "frames":
                            capture(response, result)
                        logger.debug(
                            "Sending request %s: %s RGB bytes", request_id, len(result)
                        )
                        if response["type"] == "frames":
                            status(
                                "transferring",
                                first_frame=response["first_frame"],
                                frame_count=response["frame_count"],
                                bytes=len(result),
                            )
                        send_message(connection, {"id": request_id, **response}, result)
                        logger.debug("Sent request %s", request_id)
                        continue
                    session.reset()
                    response = {"type": "reset"}
                else:
                    raise ValueError("Unknown operation")
            except ValueError as error:
                response = {"type": "error", "code": "request", "message": str(error)}
            except ConnectionError:
                # The peer went away; the outer finally releases its session.
                raise
            except Exception as error:
                logger.exception("Model request %s failed", request_id)
                if session is not None:
                    session.close()
                    session = None
                response = {"type": "error", "code": "execution", "message": str(error)}
            send_message(connection, {"id": request_id, **response})
            if not ready:
                return
    finally:
        if session is not None:
            session.close()


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            serve_connection(
                self.request,
                self.server.backend,
                self._capture if self.server.capture_frames else None,
            )
        except (ConnectionError, OSError):
            pass

    def _capture(self, metadata, pixels):
        self.server.last_frames = metadata, pixels


class ModelServer(socketserver.ThreadingTCPServer):
    """Listen for model clients, delegating execution to a supplied backend."""

    allow_reuse_address = True
    daemon_threads = True
    # ponytail: One thread per client; bound the worker count before scaling up.

    def __init__(self, address, backend, handler=_Handler, *, capture_frames=False):
        self.backend = backend
        self.capture_frames = capture_frames
        self.last_frames = None
        super().__init__(address, handler)

    def server_close(self) -> None:
        """Close the listener and release weights when the active job finishes."""
        super().server_close()
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()
