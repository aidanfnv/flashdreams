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

"""Bidirectional RGB presentation through an external TCP output service."""

from __future__ import annotations

import math
import queue
import socket
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from dos_stunt.frame_stream import (
    check_hello,
    hello_message,
    receive_message,
    send_message,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    FocusUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    MouseUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor


def input_event_from_message(
    message: Mapping[str, Any], *, timestamp: np.uint64
) -> UserInputEvent:
    """Validate a stream control message and convert it to a runtime event.

    Args:
        message: Keyboard, mouse, focus, reset, or close envelope.
        timestamp: Local input time in microseconds since the window opened.

    Raises:
        ValueError: The message type or its fields are invalid.
    """
    kind = message.get("type")
    if kind == "keyboard":
        key, pressed = message.get("key"), message.get("pressed")
        if not isinstance(key, str) or not key or not isinstance(pressed, bool):
            raise ValueError("Keyboard input requires a key and boolean pressed value")
        return KeyboardUserInputEvent(
            timestamp=timestamp,
            key=key,
            state=KeyboardInputState.PRESSED
            if pressed
            else KeyboardInputState.RELEASED,
        )
    if kind == "close":
        return CloseUserInputEvent(timestamp=timestamp)
    if kind == "reset":
        return ResetUserInputEvent(timestamp=timestamp)
    if kind == "focus" and isinstance(message.get("focused"), bool):
        return FocusUserInputEvent(timestamp=timestamp, focused=message["focused"])
    if kind == "mouse":
        action = message.get("action", "move")
        button, pressed = message.get("button", 0), message.get("pressed", False)
        values = {
            name: message.get(name, 0.0) for name in ("x", "y", "wheel_x", "wheel_y")
        }
        if (
            action not in ("move", "button", "wheel")
            or type(button) is not int
            or button < 0
            or not isinstance(pressed, bool)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values.values()
            )
        ):
            raise ValueError("Invalid mouse input fields")
        return MouseUserInputEvent(
            timestamp=timestamp, action=action, button=button, pressed=pressed, **values
        )
    raise ValueError(f"Unsupported stream input type: {kind!r}")


class StreamClientWindow(IClientWindow):
    """Send presented RGB images to one external output service.

    A sender finishes its current message while the UI replaces a single pending
    image. A separate reader keeps input responsive while output is stalled.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 9386) -> None:
        """Connect to the output service and negotiate the stream protocol.

        Args:
            host: Service hostname or address.
            port: Service's TCP listener port.
        """
        self.host, self.port = host, port
        self._sock = socket.create_connection((host, port), timeout=5)
        try:
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_message(self._sock, hello_message())
            greeting, payload = receive_message(self._sock)
            check_hello(greeting, payload)
            self._sock.settimeout(None)
        except BaseException:
            self._sock.close()
            raise
        self._condition = threading.Condition()
        self._pending: tuple[dict[str, Any], bytes] | None = None
        self._events: queue.SimpleQueue[UserInputEvent] = queue.SimpleQueue()
        self._session_desc: SessionDesc | None = None
        self._epoch_ns: int | None = None
        self._stop = False
        self._closed = False
        self._peer_closed = False
        self._failure: Exception | None = None
        self.seen = 0
        self.sent = 0
        self.dropped = 0
        self._sender = threading.Thread(
            target=self._send_frames, name="frame-stream-sender", daemon=True
        )
        self._reader = threading.Thread(
            target=self._read_input, name="frame-stream-input", daemon=True
        )
        self._sender.start()
        self._reader.start()

    def open(self, session_desc: SessionDesc) -> None:
        """Use the current output layout while preserving the connected service."""
        if self._closed:
            raise RuntimeError("Frame stream is closed")
        self._session_desc = session_desc
        if self._epoch_ns is None:
            self._epoch_ns = time.monotonic_ns()

    def write(self, result: StepResult) -> None:
        """Queue the newest RGB image."""
        if self._session_desc is None:
            raise RuntimeError("StreamClientWindow.open() must run before write()")
        # Preserve the presented dimensions; an output service owns its resizing.
        shape = result.read_output().shape
        frames = (
            result_to_rgb24_tensor(result, self._session_desc, (shape[-1], shape[-2]))
            .cpu()
            .numpy()
        )
        for frame in frames:
            header = {
                "type": "frame",
                "format": "rgb24",
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
            pixels = frame.tobytes()
            with self._condition:
                if self._stop or self._peer_closed:
                    return
                self.seen += 1
                if self._pending is not None:
                    self.dropped += 1
                self._pending = header, pixels
                self._condition.notify_all()

    def get_user_input_events(self) -> UserInputEvents:
        """Drain validated reverse-channel events without waiting for frames."""
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return UserInputEvents(events)

    def close(self) -> None:
        """Finish or cancel the current transfer and close the external stream."""
        if self._closed:
            return
        self._closed = True
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
        self._sender.join(timeout=2)
        try:
            if not self._sender.is_alive() and not self._peer_closed:
                self._sock.settimeout(2)
                send_message(self._sock, {"type": "close"})
        except OSError:
            pass
        finally:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sender.join()
            self._reader.join()
            self._sock.close()
        if self._failure is not None:
            raise RuntimeError("External frame stream failed") from self._failure

    def _send_frames(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None or self._stop
                    )
                    if self._stop:
                        return
                    message = self._pending
                    self._pending = None
                assert message is not None
                send_message(self._sock, *message)
                self.sent += 1
        except Exception as error:
            if not self._closed and not self._peer_closed:
                self._failure = error
                self._signal_close()

    def _read_input(self) -> None:
        try:
            while True:
                message, payload = receive_message(self._sock)
                if payload:
                    raise ValueError("Input messages cannot contain image payloads")
                if message.get("type") == "error":
                    raise RuntimeError(
                        str(message.get("message", "Output service failed"))
                    )
                timestamp = np.uint64(
                    max(
                        0,
                        (time.monotonic_ns() - (self._epoch_ns or time.monotonic_ns()))
                        // 1000,
                    )
                )
                event = input_event_from_message(message, timestamp=timestamp)
                self._events.put(event)
                if isinstance(event, CloseUserInputEvent):
                    self._peer_closed = True
                    return
        except (EOFError, OSError):
            if not self._closed:
                self._signal_close()
        except Exception as error:
            if not self._closed:
                self._failure = error
                self._signal_close()

    def _signal_close(self) -> None:
        self._peer_closed = True
        self._events.put(CloseUserInputEvent(timestamp=np.uint64(0)))
