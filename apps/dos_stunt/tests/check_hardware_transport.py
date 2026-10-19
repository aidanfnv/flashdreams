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

"""Check real Linux packet sockets against DOSBox-X in an isolated namespace.

The check creates an isolated veth pair named fdtest0/fdtest1. The relay
replaces the physical cable and card but preserves Ethernet frames.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from dos_stunt import run_demo
from dos_stunt.ethernet_transport import open_packet_socket


def exercise(directory: Path, packet_driver: Path | None, flashdreams: bool) -> None:
    raw = open_packet_socket("fdtest1")
    raw.setblocking(False)
    guest_mac = raw.getsockname()[4]
    relay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    relay_path = directory / "relay.sock"
    guest_path = directory / "dos.sock"
    relay.bind(str(relay_path))
    relay.setblocking(False)
    stop = threading.Event()
    relay_errors: list[Exception] = []

    def forward() -> None:
        try:
            while not stop.is_set():
                ready, _, _ = select.select([raw, relay], [], [], 0.05)
                for endpoint in ready:
                    if endpoint is raw:
                        packet = raw.recv(2048)
                        if packet[6:12] != guest_mac:
                            relay.sendto(packet, str(guest_path))
                    else:
                        raw.send(relay.recv(2048))
        except Exception as error:
            relay_errors.append(error)

    config = directory / "dosbox.conf"
    config.write_text(
        "[dosbox]\nquit warning=false\n[cpu]\ncore=normal\ncputype=286\n"
        "[ne2000]\nne2000=true\nnicbase=300\nnicirq=3\n"
        f"macaddr={guest_mac.hex(':')}\nbackend=unix\n"
    )
    command = [
        str(run_demo.ROOT / ".local-build/dosbox-x-prefix/bin/dosbox-x"),
        "-silent",
        "-fastlaunch",
        "-nopromptfolder",
        "-conf",
        str(config),
        "-c",
        "mount c .",
        "-c",
        "c:",
    ]
    if packet_driver is not None:
        command += [
            "-c",
            f'mount d "{packet_driver.parent}"',
            "-c",
            f"d:\\{packet_driver.name} 0x60 3 0x300 > driver.txt",
        ]
    else:
        command += ["-c", r"z:\system\ne2000.com 0x60 3 0x300 > driver.txt"]
    command += ["-c", "dosrx.exe /continuous > client.txt", "-c", "exit"]
    env = os.environ.copy()
    env.update(
        SDL_VIDEODRIVER="dummy",
        XDG_CONFIG_HOME=str(directory / "config"),
        DOSBOX_X_ETHERNET_LOCAL=str(guest_path),
        DOSBOX_X_ETHERNET_PEER=str(relay_path),
    )
    log = (directory / "dosbox.log").open("w")
    process = subprocess.Popen(
        command, cwd=run_demo.HERE, env=env, stdout=log, stderr=subprocess.STDOUT
    )
    worker = threading.Thread(target=forward, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while not guest_path.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError((directory / "dosbox.log").read_text())
            time.sleep(0.01)
        if flashdreams:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("check_server.py")),
                    "--interface",
                    "fdtest0",
                ],
                cwd=run_demo.ROOT,
                timeout=40,
            )
            assert result.returncode == 0
            expected_frames = 12
        else:
            run_demo.TARGET_FRAMES = 5

            def expired(signum, frame):
                raise TimeoutError("raw Ethernet sender did not finish")

            old_handler = signal.signal(signal.SIGALRM, expired)
            signal.alarm(20)
            try:
                assert run_demo.run_physical("fdtest0", 0) == 0
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            expected_frames = 5
        process.wait(timeout=10)
        assert process.returncode == 0
        assert not relay_errors, relay_errors
        stats = dict(
            item.split("=")
            for item in (run_demo.HERE / "STATS.TXT").read_text().split()
        )
        assert int(stats["frames"]) == expected_frames, stats
        screen = (run_demo.HERE / "FRAME.RAW").read_bytes()
        assert len(screen) == run_demo.FRAME_SIZE
        if not flashdreams:
            assert screen == run_demo.make_frame(expected_frames - 1)
        print(
            f"raw Ethernet, 286, IRQ 3: {expected_frames} DOS frames; "
            f"driver={packet_driver.name if packet_driver else 'DOSBox-X built-in'}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        stop.set()
        worker.join(timeout=2)
        raw.close()
        relay.close()
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-driver", type=Path)
    parser.add_argument(
        "--flashdreams",
        action="store_true",
        help="also exercise the FlashDreams physical mode",
    )
    parser.add_argument("--parent-netns", help=argparse.SUPPRESS)
    args = parser.parse_args()
    namespace = os.readlink("/proc/self/ns/net")
    if args.parent_netns is None:
        subprocess.run(
            [
                "unshare",
                "--user",
                "--map-root-user",
                "--net",
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--parent-netns",
                namespace,
            ],
            check=True,
        )
        return
    if namespace == args.parent_netns:
        parser.error("the check requires an isolated network namespace")
    for command in (
        ["ip", "link", "set", "lo", "up"],
        ["ip", "link", "add", "fdtest0", "type", "veth", "peer", "name", "fdtest1"],
        ["ip", "link", "set", "fdtest0", "up"],
        ["ip", "link", "set", "fdtest1", "up"],
    ):
        subprocess.run(command, check=True)
    driver = args.packet_driver.resolve() if args.packet_driver else None
    if driver is not None and not driver.is_file():
        parser.error("packet driver file does not exist")
    with tempfile.TemporaryDirectory(prefix="dos-raw-") as temporary:
        exercise(Path(temporary), driver, False)
    if args.flashdreams:
        with tempfile.TemporaryDirectory(prefix="dos-raw-") as temporary:
            exercise(Path(temporary), driver, True)


if __name__ == "__main__":
    main()
