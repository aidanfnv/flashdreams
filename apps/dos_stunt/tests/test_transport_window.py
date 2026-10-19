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

"""Recover lost/reordered chunks and ACKs with a bounded sender window."""

from collections import Counter

import pytest
from dos_stunt.dos_output import DosOutput
from dos_stunt.ethernet_transport import (
    ETHERTYPE,
    FRAME_SIZE,
    GUEST_MAC,
    HOST_MAC,
    send_frame,
)

from .check_controls import packet

pytestmark = pytest.mark.ci_cpu


def test_window_recovers_loss_and_dispatches_input(monkeypatch) -> None:
    """Recover lost image/ACK packets, including the last ACK, without a backlog."""
    acknowledged = set()
    controls = []
    clock = [0.0]
    sequence = 65535

    class Peer:
        def __init__(self):
            self.pending = []
            self.attempts = Counter()
            self.sent = []
            self.image = bytearray(FRAME_SIZE)
            self.accepted = set()

        def send(self, payload):
            chunk = payload[6]
            self.sent.append(chunk)
            self.attempts[chunk] += 1
            base = next(i for i in range(46) if (sequence, i) not in acknowledged)
            assert base <= chunk < base + 4
            assert payload[:6] == b"FD\x01\x07\xff\xff"
            if chunk == 1 and self.attempts[chunk] == 1:
                return  # Lose one image packet.
            offset = int.from_bytes(payload[8:10], "little")
            self.image[offset : offset + len(payload) - 12] = payload[12:]
            self.accepted.add(chunk)
            if chunk in (2, 45) and self.attempts[chunk] == 1:
                return  # Lose a middle ACK and the final ACK.
            self.pending.append(
                HOST_MAC
                + GUEST_MAC
                + ETHERTYPE
                + b"FD\x01\x02"
                + sequence.to_bytes(2, "little")
                + bytes((chunk,))
            )
            if chunk == 3:
                self.pending.append(packet(0, 0x32))

        def recv(self, size):
            if not self.pending:
                raise BlockingIOError
            return self.pending.pop()  # ACKs arrive in reverse order.

    peer = Peer()

    def wait(readers, writers, errors, timeout):
        if not peer.pending:
            clock[0] += timeout

    monkeypatch.setattr("dos_stunt.ethernet_transport.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("dos_stunt.ethernet_transport.select.select", wait)
    pixels = bytes(range(256)) * 250
    retries = send_frame(
        peer,
        sequence,
        pixels,
        acknowledged,
        controls.append,
        window=4,
        direct_vga=True,
    )
    assert peer.sent[:4] == [0, 1, 2, 3]
    assert peer.image == pixels and len(peer.accepted) == 46
    assert retries == 3 and peer.attempts == Counter(
        {**dict.fromkeys(range(46), 1), 1: 2, 2: 2, 45: 2}
    )
    assert controls == [packet(0, 0x32)]


def test_retry_window_cancels(monkeypatch) -> None:
    """Cancellation stops unlimited retries even when the receiver sends nothing."""

    class SilentPeer:
        def send(self, payload):
            pass

        def recv(self, size):
            raise BlockingIOError

    checks = 0

    def cancel():
        nonlocal checks
        checks += 1
        if checks == 8:
            raise InterruptedError

    monkeypatch.setattr(
        "dos_stunt.ethernet_transport.select.select", lambda *args: None
    )
    with pytest.raises(InterruptedError):
        send_frame(
            SilentPeer(),
            0,
            bytes(FRAME_SIZE),
            set(),
            window=4,
            max_attempts=None,
            check_active=cancel,
        )


@pytest.mark.parametrize("capacity", [0, 1, 4, 255])
def test_receiver_capacity_and_legacy_fallback(capacity) -> None:
    """Use advertised queue capacity; older clients retain one-packet pacing."""
    output = DosOutput()
    ready = bytearray(packet(0))
    ready[52] = capacity
    output._on_packet(ready)
    assert output.receiver_window == (capacity if 1 <= capacity <= 4 else 1)
    direct = DosOutput(palette="rgb332", direct_vga=True)
    with pytest.raises(RuntimeError, match="rebuild DOSRX.EXE"):
        direct._on_packet(ready)
    ready[53] = 1
    direct._on_packet(ready)
    assert direct.control_packets == 1
