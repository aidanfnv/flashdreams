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

"""Exercise the DOSBox-X NE2000 client with synthetic VGA frames."""

from __future__ import annotations

import argparse
import os
import select
import socket
import subprocess
import sys
import time

from dos_stunt import CLIENT_DIR, DOSBOX_CONFIG, DOSBOX_PATH, ROOT
from dos_stunt.ethernet_transport import (
    FRAME_SIZE,
    EthernetLink,
    physical_link,
    read_acks,
    send_frame,
    send_stop,
)

HERE = CLIENT_DIR
HOST_SOCKET = HERE / "host.sock"
GUEST_SOCKET = HERE / "guest.sock"
TARGET_FRAMES = 30


def make_frame(sequence: int) -> bytes:
    frame = bytearray(FRAME_SIZE)
    blue = (sequence // 5) & 3
    bar = (sequence * 9) % 320
    for y in range(200):
        green = (y * 8 // 200) << 2
        row = y * 320
        for x in range(320):
            frame[row + x] = (x * 8 // 320 << 5) | green | blue
        for x in range(bar, min(bar + 8, 320)):
            frame[row + x] = 255
    return bytes(frame)


def verify_screen(pixels: bytes, stats: dict[str, str]) -> int:
    screen = (HERE / "FRAME.RAW").read_bytes()
    if len(screen) != FRAME_SIZE or screen != pixels:
        raise RuntimeError("DOS VGA framebuffer differs from the Linux frame sent")
    verified = FRAME_SIZE
    return verified


def run_physical(interface: str, fps: float) -> int:
    sock = physical_link(interface)
    acknowledged: set[tuple[int, int]] = set()

    def on_packet(packet: bytes) -> None:
        if packet[14:18] == b"FD\x01\x06" and sock.guest_mac is None:
            sock.guest_mac = packet[6:12]
            print(f"DOS receiver: {sock.guest_mac.hex(':')}", flush=True)

    try:
        print(
            f"Waiting on {interface}; start DOSRX.EXE /continuous on the DOS machine",
            flush=True,
        )
        while sock.guest_mac is None:
            read_acks(sock, acknowledged, on_packet)
            if sock.guest_mac is None:
                select.select([sock], [], [], 0.1)
        started = time.monotonic()
        retries = 0
        for sequence in range(TARGET_FRAMES):
            acknowledged.clear()
            retries += send_frame(
                sock,
                sequence,
                make_frame(sequence),
                acknowledged,
                on_packet,
            )
            if fps:
                time.sleep(max(0, started + (sequence + 1) / fps - time.monotonic()))
        elapsed = time.monotonic() - started
        send_stop(sock, TARGET_FRAMES, acknowledged, on_packet)
        print(
            f"sent_frames={TARGET_FRAMES} retries={retries} "
            f"display_fps={TARGET_FRAMES / elapsed:.2f}; "
            "FRAME.RAW and STATS.TXT are on the DOS machine",
            flush=True,
        )
        return 0
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visible", action="store_true", help="show the DOSBox-X VGA window"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="target frame rate; 0 sends as fast as ACKs allow (default: 10)",
    )
    parser.add_argument(
        "--interface",
        help="send to a physical DOS machine on this Linux Ethernet interface",
    )
    args = parser.parse_args()
    if args.fps < 0:
        parser.error("--fps must be zero or positive")
    if args.interface:
        if args.visible:
            parser.error(
                "--visible applies to DOSBox-X; use --interface for the physical display"
            )
        return run_physical(args.interface, args.fps)
    for name in ("FRAME.RAW", "WORLD.RAW", "STATS.TXT", "DRIVER.TXT", "CLIENT.TXT"):
        (HERE / name).unlink(missing_ok=True)
    HOST_SOCKET.unlink(missing_ok=True)
    GUEST_SOCKET.unlink(missing_ok=True)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(HOST_SOCKET))
    sock.setblocking(False)
    sock = EthernetLink(sock, destination=str(GUEST_SOCKET))
    env = os.environ.copy()
    env.update(
        XDG_CONFIG_HOME=str(ROOT / ".local-build/dosbox-config"),
        DOSBOX_X_ETHERNET_LOCAL=str(GUEST_SOCKET),
        DOSBOX_X_ETHERNET_PEER=str(HOST_SOCKET),
    )
    if args.visible:
        if env.get("DISPLAY"):
            env["SDL_VIDEODRIVER"] = "x11"
        elif env.get("WAYLAND_DISPLAY"):
            env["SDL_VIDEODRIVER"] = "wayland"
        else:
            parser.error("--visible needs an X11 or Wayland desktop session")
    else:
        env["SDL_VIDEODRIVER"] = "dummy"
    command = [str(DOSBOX_PATH)]
    if not args.visible:
        command.append("-silent")
    command.extend(
        [
            "-fastlaunch",
            "-nopromptfolder",
            "-conf",
            str(DOSBOX_CONFIG),
            "-c",
            "mount c .",
            "-c",
            "c:",
            "-c",
            r"z:\system\ne2000.com 0x60 10 0x300 > driver.txt",
            "-c",
            "dosrx.exe /hold > client.txt"
            if args.visible
            else "dosrx.exe > client.txt",
            "-c",
            "exit",
        ]
    )
    sent: dict[int, bytes] = {}
    acknowledged: set[tuple[int, int]] = set()
    retries = 0
    started = time.monotonic()
    try:
        with (HERE / "dosbox-run.log").open("w") as output:
            process = subprocess.Popen(
                command, cwd=HERE, env=env, stdout=output, stderr=subprocess.STDOUT
            )
            while not GUEST_SOCKET.exists():
                if process.poll() is not None:
                    raise RuntimeError(
                        "DOSBox-X exited before opening the emulated Ethernet socket"
                    )
                time.sleep(0.01)
            transmitting_at = time.monotonic()
            next_at = time.monotonic()
            sequence = 0
            while sequence < TARGET_FRAMES and time.monotonic() - started < 18:
                if process.poll() is not None:
                    break
                pixels = make_frame(sequence)
                sent[sequence] = pixels
                retries += send_frame(
                    sock,
                    sequence,
                    pixels,
                    acknowledged,
                )
                sequence += 1
                if args.fps:
                    next_at += 1.0 / args.fps
                    time.sleep(max(0, next_at - time.monotonic()))
            transmit_seconds = time.monotonic() - transmitting_at
            if sequence == TARGET_FRAMES and process.poll() is None:
                send_stop(sock, sequence, acknowledged)
            if args.visible and sequence == TARGET_FRAMES:
                print(
                    "DOSBox-X is holding the final frame. Focus its window and press any key to close it.",
                    flush=True,
                )
            process.wait()
    finally:
        sock.close()
        HOST_SOCKET.unlink(missing_ok=True)

    stats_path = HERE / "STATS.TXT"
    if not stats_path.exists():
        raise RuntimeError("DOS client did not write STATS.TXT; inspect dosbox-run.log")
    stats = dict(item.split("=") for item in stats_path.read_text().strip().split())
    last_sequence = int(stats["last_sequence"])
    verified = verify_screen(sent[last_sequence], stats)
    print(stats_path.read_text().strip())
    pixel_mbps = len(sent) * FRAME_SIZE * 8 / transmit_seconds / 1_000_000
    data_acks = sum(chunk < 46 for _, chunk in acknowledged)
    print(
        f"acknowledged_chunks={data_acks} retries={retries} sent_frames={len(sent)} world_bytes_verified={FRAME_SIZE} vga_bytes_verified={verified} transmit_seconds={transmit_seconds:.2f} pixel_mbps={pixel_mbps:.2f}"
    )
    return 0 if int(stats["frames"]) == TARGET_FRAMES else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"DOS demo failed: {error}", file=sys.stderr)
        sys.exit(1)
