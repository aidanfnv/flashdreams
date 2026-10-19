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

"""CPU checks for framed image delivery and independent reverse input."""

import queue
import socket
import struct
import threading
import time

import pytest
import torch

from dos_stunt import stream_client_window
from dos_stunt.frame_stream import (
    MAX_PAYLOAD_BYTES,
    check_hello,
    frame_shape,
    hello_message,
    receive_message,
    send_message,
)
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from dos_stunt.stream_client_window import StreamClientWindow
from flashdreams.runtime_v2.user_input_event import (
    KeyboardUserInputEvent,
    MouseUserInputEvent,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


def test_stream_keeps_latest_frame_and_input_when_output_is_stalled(
    monkeypatch,
) -> None:
    gate = threading.Event()
    sending = threading.Event()
    observed = queue.Queue()
    failures = []
    real_send = stream_client_window.send_message

    def delayed_send(sock, header, payload=b""):
        if header["type"] == "frame" and not sending.is_set():
            sending.set()
            assert gate.wait(5)
        real_send(sock, header, payload)

    monkeypatch.setattr(stream_client_window, "send_message", delayed_send)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def service():
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5)
                    check_hello(*receive_message(connection))
                    send_message(connection, hello_message())
                    assert sending.wait(5)
                    send_message(
                        connection, {"type": "keyboard", "key": "w", "pressed": True}
                    )
                    send_message(
                        connection,
                        {"type": "mouse", "action": "move", "x": 0.5, "y": 0.25},
                    )
                    while True:
                        header, pixels = receive_message(connection)
                        if header["type"] == "close":
                            break
                        assert frame_shape(header, pixels) == (4, 2)
                        observed.put((header, pixels))
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=service, daemon=True)
        worker.start()
        window = StreamClientWindow(port=listener.getsockname()[1])
        try:
            desc = SessionDesc(video_width=4, video_height=2)
            window.open(desc)
            epoch = window._epoch_ns
            window.open(desc)
            assert window._epoch_ns == epoch

            def write(value):
                window.write(
                    StepResult(
                        step_index=value,
                        output=torch.full((1, 3, 2, 4), value, dtype=torch.uint8),
                        frame_count=1,
                        output_layout=VideoTensorLayout.tchw,
                        metrics=None,
                    )
                )

            write(1)
            assert sending.wait(5)
            write(2)
            write(3)
            assert window.dropped == 1
            events = []
            deadline = time.monotonic() + 5
            while len(events) < 2:
                events.extend(window.get_user_input_events().get_events())
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert (
                isinstance(events[0], KeyboardUserInputEvent) and events[0].key == "w"
            )
            assert isinstance(events[1], MouseUserInputEvent) and events[1].x == 0.5
            gate.set()
            for value in (1, 3):
                header, pixels = observed.get(timeout=5)
                assert "presentation_data" not in header
                assert pixels == bytes([value]) * 24
        finally:
            gate.set()
            window.close()
            worker.join(timeout=5)
        assert not worker.is_alive() and not failures


def test_stream_rejects_oversized_envelope_before_reading_payload() -> None:
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("!II", 1, MAX_PAYLOAD_BYTES + 1))
        with pytest.raises(ValueError, match="size"):
            receive_message(receiver)


def test_stream_detects_disconnect_mid_message() -> None:
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(struct.pack("!II", 50, 0) + b"{")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            receive_message(receiver)
