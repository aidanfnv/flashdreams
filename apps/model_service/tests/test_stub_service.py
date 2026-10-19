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

"""Native-client workflows, session isolation, and bounded TCP framing."""

import json
from io import BytesIO
import socket
import struct
import threading
from types import SimpleNamespace

import pytest
from model_service.protocol import (
    MAX_HEADER,
    MAX_PAYLOAD,
    PROTOCOL,
    VERSION,
    receive_message,
    send_message,
)
from model_service.stub_service import StubServer, _Handler
from model_service.service import serve_connection

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("failed_write", [3, 5])
def test_disconnect_during_status_or_payload_releases_session(failed_write):
    """Release a departed peer's session without attempting an execution-error reply."""
    closed = []
    session = SimpleNamespace(
        model="test",
        parameters={},
        step=lambda count, inputs, **kwargs: (
            {"type": "frames", "first_frame": 0, "frame_count": 1},
            b"RGB",
        ),
        close=lambda: closed.append(True),
    )
    backend = SimpleNamespace(open=lambda *args, **kwargs: session)
    incoming = bytearray()
    for header in (
        {
            "id": 1,
            "op": "hello",
            "protocol": PROTOCOL,
            "version": VERSION,
            "status_messages": True,
        },
        {"id": 2, "op": "open", "model": "test"},
        {"id": 3, "op": "step"},
    ):
        encoded = json.dumps(header).encode()
        incoming.extend(struct.pack("!4sII", b"FDM1", len(encoded), 0) + encoded)

    class Peer:
        def __init__(self):
            self.input = BytesIO(incoming)
            self.writes = 0

        def recv(self, length):
            return self.input.read(length)

        def sendall(self, data):
            self.writes += 1
            if self.writes == failed_write:
                raise BrokenPipeError("peer disconnected")

    peer = Peer()
    with pytest.raises(BrokenPipeError, match="peer disconnected"):
        serve_connection(peer, backend)
    assert closed == [True] and peer.writes == failed_write


def test_model_service_workflow_and_session_isolation():
    """Discover, generate, control, reset, upload an image, and recover request errors."""
    with StubServer(("127.0.0.1", 0), _Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=5) as client:

                def request(request_id, operation, payload=b"", **fields):
                    send_message(
                        client, {"id": request_id, "op": operation, **fields}, payload
                    )
                    reply, result = receive_message(client)
                    assert reply["id"] == request_id
                    return reply, result

                response, _ = request(1, "hello", protocol=PROTOCOL, version=VERSION)
                assert (
                    response["type"] == "hello" and response["max_frames_per_step"] == 4
                )
                response, _ = request(2, "models")
                models = {model["id"]: model for model in response["models"]}
                assert models.keys() == {"stub-gradient", "stub-image"}
                assert all(model["stub"] for model in models.values())
                assert (
                    models["stub-gradient"]["parameters"]["prompt"]["type"] == "string"
                )
                response, _ = request(3, "step")
                assert response["type"] == "error"
                parameters = {
                    "width": 5,
                    "height": 2,
                    "prompt": "neon café",
                    "seed": 17,
                    "fps": 24,
                }
                response, _ = request(
                    4, "open", model="stub-gradient", parameters=parameters
                )
                assert response["parameters"] == parameters
                response, _ = request(5, "step", count=True)
                assert response["type"] == "error"
                response, _ = request(6, "step", inputs={"unknown": 2})
                assert response["type"] == "error"
                response, original = request(7, "step", count=2, inputs={"pan": 0})
                assert response["first_frame"] == 0 and response["frame_count"] == 2
                assert len(original) == 5 * 2 * 3 * 2 and original[:30] == original[30:]
                response, changed = request(8, "step", count=2, inputs={"pan": 2})
                assert response["first_frame"] == 2 and changed[:30] != changed[30:]

                with socket.create_connection(
                    server.server_address, timeout=5
                ) as other:
                    send_message(
                        other,
                        {
                            "id": 1,
                            "op": "hello",
                            "protocol": PROTOCOL,
                            "version": VERSION,
                        },
                    )
                    receive_message(other)
                    send_message(
                        other,
                        {
                            "id": 2,
                            "op": "open",
                            "model": "stub-gradient",
                            "parameters": parameters,
                        },
                    )
                    receive_message(other)
                    send_message(
                        other, {"id": 3, "op": "step", "count": 2, "inputs": {"pan": 0}}
                    )
                    metadata, fresh = receive_message(other)
                    assert metadata["first_frame"] == 0 and fresh == original

                request(9, "reset")
                response, repeated = request(10, "step", count=2, inputs={"pan": 0})
                assert response["first_frame"] == 0 and repeated == original
                request(11, "close")
                parameters = {"width": 3, "height": 2}
                response, _ = request(
                    12, "open", b"short", model="stub-image", parameters=parameters
                )
                assert response["type"] == "error"
                image = bytes(range(18))
                response, _ = request(
                    13, "open", image, model="stub-image", parameters=parameters
                )
                assert response["type"] == "session"
                response, frames = request(14, "step", count=2, inputs={"pan": -1})
                assert frames[:18] == image
                assert (
                    frames[18:] == image[6:9] + image[:6] + image[15:18] + image[9:15]
                )
                response, _ = request(15, "step", count=5)
                assert response["type"] == "error"
                response, _ = request(16, "step", count=1)
                assert response["first_frame"] == 2
                request(17, "close")
        finally:
            server.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()


@pytest.mark.parametrize(
    "prefix",
    [
        struct.pack("!4sII", b"BAD!", 2, 0),
        struct.pack("!4sII", b"FDM1", MAX_HEADER + 1, 0),
        struct.pack("!4sII", b"FDM1", 2, MAX_PAYLOAD + 1),
    ],
)
def test_invalid_framing_rejected_before_reading_body(prefix):
    """Reject invalid or oversized framing without waiting for its declared body."""
    sender, receiver = socket.socketpair()
    with sender, receiver:
        receiver.settimeout(1)
        sender.sendall(prefix)
        with pytest.raises(ValueError):
            receive_message(receiver)


def test_fragmented_messages_and_truncated_payload():
    """Reassemble tiny TCP reads and detect an incomplete binary upload."""
    sender, receiver = socket.socketpair()
    metadata = json.dumps({"id": 1, "op": "open"}).encode()
    message = struct.pack("!4sII", b"FDM1", len(metadata), 8) + metadata + b"rgb-data"

    class Fragmented:
        def recv(self, length):
            return receiver.recv(min(length, 2))

    with sender, receiver:
        sender.sendall(message)
        header, payload = receive_message(Fragmented())
        assert header == {"id": 1, "op": "open"} and payload == b"rgb-data"
        sender.sendall(message[:-2])
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            receive_message(receiver)


def test_oversized_catalog_returns_actionable_error():
    """Keep the connection usable when installed model metadata exceeds framing."""
    incoming = BytesIO()

    class Stream:
        def __init__(self):
            self.out = BytesIO()

        def recv(self, size):
            return incoming.read(size)

        def sendall(self, data):
            self.out.write(data)

    peer = Stream()
    for header in (
        {"id": 1, "op": "hello", "protocol": PROTOCOL, "version": VERSION},
        {"id": 2, "op": "models"},
        {"id": 3, "op": "close"},
    ):
        send_message(peer, header)
    incoming.write(peer.out.getvalue())
    incoming.seek(0)
    peer.out = BytesIO()
    serve_connection(
        peer, SimpleNamespace(catalog=lambda: [{"name": "x" * MAX_HEADER}])
    )
    incoming = BytesIO(peer.out.getvalue())
    receive_message(peer)
    error, _ = receive_message(peer)
    assert error["type"] == "error" and "catalog pagination" in error["message"]
    assert receive_message(peer)[0]["type"] == "closed"
