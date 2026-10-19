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

"""Ethernet transport for the DOS VGA client."""

from __future__ import annotations

import errno
import fcntl
import os
import select
import socket
import time
from collections.abc import Callable

HOST_MAC = bytes.fromhex("02 46 44 4f 53 01")
GUEST_MAC = bytes.fromhex("ac de 48 88 99 ab")
BROADCAST_MAC = b"\xff" * 6
ETHERTYPE = bytes.fromhex("88 b5")
PACKET_FD_ENV = "DOS_BRIDGE_PACKET_FD"


class EthernetLink:
    """Send complete Ethernet packets over a raw or Unix datagram socket."""

    def __init__(
        self,
        sock: socket.socket,
        *,
        destination: str | None = None,
        host_mac: bytes = HOST_MAC,
        guest_mac: bytes | None = GUEST_MAC,
    ) -> None:
        self.sock = sock
        self.destination = destination
        self.host_mac = host_mac
        self.guest_mac = guest_mac

    def fileno(self) -> int:
        """Return the descriptor used by ``select``."""
        return self.sock.fileno()

    def send(self, payload: bytes) -> None:
        """Wrap one protocol payload in an Ethernet header and send it."""
        if self.guest_mac is None:
            raise RuntimeError("DOS receiver has not sent its ready packet")
        packet = (self.guest_mac + self.host_mac + ETHERTYPE + payload).ljust(60, b"\0")
        if self.destination is None:
            self.sock.send(packet)
        else:
            self.sock.sendto(packet, self.destination)

    def recv(self, size: int) -> bytes:
        """Read the next packet addressed to this host from the DOS peer."""
        while True:
            packet = self.sock.recv(size)
            if (
                len(packet) >= 21
                and packet[12:14] == ETHERTYPE
                and packet[:6] in (self.host_mac, BROADCAST_MAC)
                and packet[6:12] != self.host_mac
                and not packet[6] & 1
                and packet[6:12] != bytes(6)
                and (self.guest_mac is None or packet[6:12] == self.guest_mac)
            ):
                return packet

    def close(self) -> None:
        """Close the underlying socket."""
        self.sock.close()


def open_packet_socket(interface: str) -> socket.socket:
    """Open a raw socket for this protocol on one Ethernet interface."""
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x88B5))
    try:
        sock.bind((interface, 0x88B5))
    except BaseException:
        sock.close()
        raise
    return sock


def physical_link(interface: str) -> EthernetLink:
    """Open or adopt an inherited raw socket and validate its interface."""
    inherited = os.environ.pop(PACKET_FD_ENV, None)
    if inherited is not None:
        sock = socket.socket(fileno=int(inherited))
    else:
        try:
            sock = open_packet_socket(interface)
        except PermissionError as error:
            raise PermissionError(
                "Opening Ethernet needs CAP_NET_RAW; use flashdreams-dos-ethernet "
                "to open the socket before running FlashDreams as your user"
            ) from error
    try:
        address = sock.getsockname()
        if (
            sock.family != socket.AF_PACKET
            or sock.type != socket.SOCK_RAW
            or address[0] != interface
            or address[1] != 0x88B5
            or address[3] != 1
            or len(address[4]) != 6
        ):
            raise ValueError(
                "Packet socket does not match the requested Ethernet interface"
            )
        # Keep the transport outside the low descriptor range closed during
        # model initialization, while staying below select()'s descriptor limit.
        fd = fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD_CLOEXEC, 512)
        sock.close()
        sock = socket.socket(fileno=fd)
        sock.setblocking(False)
        return EthernetLink(sock, host_mac=address[4], guest_mac=None)
    except BaseException:
        sock.close()
        raise


FRAME_SIZE = 320 * 200
PALETTE_SIZE = 256 * 3
CHUNK_SIZE = 1400


def send_chunk(
    sock: EthernetLink,
    sequence: int,
    chunk: int,
    pixels: bytes,
    *,
    direct_vga: bool = False,
) -> None:
    """Send one indexed chunk of a VGA frame."""
    count = (len(pixels) + CHUNK_SIZE - 1) // CHUNK_SIZE
    offset = chunk * CHUNK_SIZE
    payload = pixels[offset : offset + CHUNK_SIZE]
    header = (
        b"FD\x01"
        + bytes((7 if direct_vga else 1,))
        + sequence.to_bytes(2, "little")
        + bytes((chunk, count))
        + offset.to_bytes(2, "little")
        + len(payload).to_bytes(2, "little")
    )
    sock.send(header + payload)


def read_acks(
    sock: EthernetLink,
    acknowledged: set[tuple[int, int]],
    on_packet: Callable[[bytes], None] | None = None,
) -> None:
    """Collect queued acknowledgments and dispatch reverse-direction packets."""
    while True:
        try:
            packet = sock.recv(2048)
        except BlockingIOError:
            return
        if (
            len(packet) >= 21
            and packet[12:14] == ETHERTYPE
            and packet[14:18] == b"FD\x01\x02"
        ):
            acknowledged.add((int.from_bytes(packet[18:20], "little"), packet[20]))
        elif on_packet is not None:
            on_packet(packet)


def send_frame(
    sock: EthernetLink,
    sequence: int,
    pixels: bytes,
    acknowledged: set[tuple[int, int]],
    on_packet: Callable[[bytes], None] | None = None,
    *,
    max_attempts: int | None = 20,
    check_active: Callable[[], None] | None = None,
    on_retry: Callable[[int, int, int], None] | None = None,
    window: int = 1,
    direct_vga: bool = False,
) -> int:
    """Send pixel indices and an optional appended VGA palette, returning retries.

    ``max_attempts=None`` retries until ``check_active`` cancels or fails.
    At most ``window`` chunks await ACKs; only missing chunks are retried.
    Keyboard packets are dispatched while awaiting acknowledgments.
    """
    if len(pixels) not in (FRAME_SIZE, FRAME_SIZE + PALETTE_SIZE):
        raise ValueError(
            "Expected 64000 pixel indices with an optional 768-byte palette"
        )
    if direct_vga and len(pixels) != FRAME_SIZE:
        raise ValueError("Direct VGA writes require a fixed RGB332 palette")
    if not 1 <= window <= 4:
        raise ValueError("window must be between 1 and 4")
    retries = 0
    count = (len(pixels) + CHUNK_SIZE - 1) // CHUNK_SIZE
    attempts = [0] * count
    deadlines = [0.0] * count
    base = 0
    while base < count:
        if check_active is not None:
            check_active()
        if attempts[base]:
            read_acks(sock, acknowledged, on_packet)
        while base < count and (sequence, base) in acknowledged:
            base += 1
        if base == count:
            break
        pending = range(base, min(base + window, count))
        for chunk in pending:
            if (sequence, chunk) in acknowledged or time.monotonic() < deadlines[chunk]:
                continue
            attempt = attempts[chunk]
            if max_attempts is not None and attempt >= max_attempts:
                raise RuntimeError(f"no ACK for frame {sequence}, chunk {chunk}")
            if check_active is not None:
                check_active()
            if attempt:
                retries += 1
                if on_retry is not None:
                    on_retry(sequence, chunk, attempt)
            try:
                send_chunk(sock, sequence, chunk, pixels, direct_vga=direct_vga)
            except OSError as error:
                # A paused guest can fill its Unix datagram receive queue.
                if error.errno not in (errno.EAGAIN, errno.ENOBUFS):
                    raise
            attempts[chunk] += 1
            deadlines[chunk] = time.monotonic() + (0.15 if attempt < 20 else 1.0)
        read_acks(sock, acknowledged, on_packet)
        if (sequence, base) not in acknowledged:
            deadline = min(
                deadlines[chunk]
                for chunk in pending
                if (sequence, chunk) not in acknowledged
            )
            select.select([sock], [], [], max(0, deadline - time.monotonic()))
    return retries


def send_stop(
    sock: EthernetLink,
    sequence: int,
    acknowledged: set[tuple[int, int]],
    on_packet: Callable[[bytes], None] | None = None,
) -> None:
    """Ask the DOS receiver to finish the stream."""
    header = b"FD\x01\x03" + sequence.to_bytes(2, "little") + b"\xff"
    for _ in range(20):
        sock.send(header)
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            read_acks(sock, acknowledged, on_packet)
            if (sequence, 255) in acknowledged:
                return
            select.select([sock], [], [], max(0, deadline - time.monotonic()))
    raise RuntimeError("DOS receiver did not acknowledge stop packet")
