#!/usr/bin/env python3
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

"""Check that a stalled DOS receiver can resume after the old ACK deadline."""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from pathlib import Path

from dos_stunt import run_demo
from dos_stunt.ethernet_transport import ETHERTYPE, GUEST_MAC, HOST_MAC

with tempfile.TemporaryDirectory(prefix="dos-transport-") as temporary:
    host_path = Path(temporary) / "host.sock"
    guest_path = Path(temporary) / "guest.sock"
    host = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    guest = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    host.bind(str(host_path))
    host.setblocking(False)
    host = run_demo.EthernetLink(host, destination=str(guest_path))
    guest.bind(str(guest_path))
    received = bytearray(run_demo.FRAME_SIZE)
    receiver_error: list[Exception] = []

    def receive() -> None:
        try:
            next_chunk = 0
            while next_chunk < 46:
                packet = guest.recv(2048)
                sequence = int.from_bytes(packet[18:20], "little")
                chunk = packet[20]
                if chunk == 6 and next_chunk == 6:
                    time.sleep(3.5)
                if chunk == next_chunk:
                    offset = int.from_bytes(packet[22:24], "little")
                    length = int.from_bytes(packet[24:26], "little")
                    received[offset : offset + length] = packet[26 : 26 + length]
                    next_chunk += 1
                ack = (
                    HOST_MAC
                    + GUEST_MAC
                    + ETHERTYPE
                    + b"FD\x01\x02"
                    + sequence.to_bytes(2, "little")
                    + bytes((chunk,))
                )
                guest.sendto(ack, str(host_path))
        except Exception as error:
            receiver_error.append(error)

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    pixels = bytes(range(256)) * 250
    acknowledged: set[tuple[int, int]] = set()
    deadline = time.monotonic() + 10

    def check_receiver() -> None:
        if receiver_error:
            raise receiver_error[0]
        if time.monotonic() > deadline:
            raise TimeoutError("receiver did not recover")

    retries = run_demo.send_frame(
        host,
        4617,
        pixels,
        acknowledged,
        max_attempts=None,
        check_active=check_receiver,
    )
    receiver.join(timeout=2)
    assert not receiver.is_alive() and not receiver_error
    assert received == pixels
    assert retries >= 20

    checks = 0

    def cancel() -> None:
        global checks
        checks += 1
        if checks == 3:
            raise InterruptedError

    try:
        run_demo.send_frame(
            host, 4618, pixels, set(), max_attempts=None, check_active=cancel
        )
    except InterruptedError:
        pass
    else:
        raise AssertionError("sender did not cancel")
    guest.close()
    host.close()

print(f"stalled frame resumed after {retries} retries; cancellation checked")
