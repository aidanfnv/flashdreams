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

"""Check the standalone DOS server through the normal FlashDreams CLI."""

from __future__ import annotations

import argparse
import socket
import subprocess
import threading
from pathlib import Path
import sys

from dos_stunt import CLIENT_DIR, ROOT
from dos_stunt.dos_server import (
    check_hello,
    hello_message,
    receive_message,
    send_message,
    serve_connection,
)

HERE = CLIENT_DIR
PYTHON = Path(sys.executable)


def check_startup_error() -> None:
    """Verify a malformed frame is reported while DOS startup is waiting."""

    class WaitingOutput:
        sender_error = None

        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()

        def write(self, frame):
            self.started.set()
            assert self.stopped.wait(5)
            raise InterruptedError("Startup cancelled")

        def get_input_messages(self):
            return []

        def request_stop(self):
            self.stopped.set()

        def close(self):
            pass

    output = WaitingOutput()
    errors = []
    server, producer = socket.socketpair()
    with server, producer:
        producer.settimeout(5)

        def serve():
            try:
                serve_connection(server, output)
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        send_message(producer, hello_message())
        check_hello(*receive_message(producer))
        send_message(
            producer,
            {"type": "frame", "format": "rgb24", "width": 1, "height": 1},
            b"\0\0\0",
        )
        assert output.started.wait(5)
        send_message(
            producer, {"type": "frame", "format": "rgb24", "width": -1, "height": 1}
        )
        header, payload = receive_message(producer)
        assert (
            header["type"] == "error" and "RGB24" in header["message"] and not payload
        )
        worker.join(timeout=5)
        assert (
            not worker.is_alive()
            and len(errors) == 1
            and isinstance(errors[0], ValueError)
        )
    print("DOS startup cancellation preserves protocol error reporting")


def check(
    *,
    capped: bool,
    interface: str | None = None,
    client_dir: Path = HERE,
    palette: str = "adaptive",
    direct_vga: bool = False,
) -> None:
    command = [
        str(PYTHON.with_name("flashdreams-dos-server")),
        "--port",
        "0",
        "--client-dir",
        str(client_dir),
        "--palette",
        palette,
    ]
    if capped:
        command += ["--max-frames", "12"]
    if direct_vga:
        command += ["--direct-vga"]
    if interface is not None:
        command += ["--interface", interface]
    server = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert server.stdout is not None
    try:
        first_line = server.stdout.readline().strip()
        if not first_line.startswith("DOS server listening at "):
            log, _ = server.communicate(timeout=5)
            raise RuntimeError(first_line + "\n" + log)
        port = first_line.rsplit(":", 1)[1]
        result = subprocess.run(
            [
                str(PYTHON.with_name("flashdreams-run-v2")),
                "color_fade.app",
                "--mode",
                "stream",
                "--stream-port",
                port,
                "--layout",
                "bcthw",
                "--",
                "--seconds",
                "3",
                "--frames-per-step",
                "8",
            ],
            cwd=ROOT / "flashdreams",
            timeout=30,
        )
        assert result.returncode == 0, "FlashDreams producer failed"
        log, _ = server.communicate(timeout=15)
        print(log, end="")
        assert server.returncode == 0
        if interface is None:
            stats = dict(
                item.split("=")
                for item in (client_dir / "STATS.TXT").read_text().split()
            )
            assert int(stats["frames"]) == 12 if capped else int(stats["frames"]) > 0
            assert "world bytes verified=64000" in log
            if palette == "adaptive":
                assert "VGA palette bytes verified=768" in log
        print(
            f"Standalone DOS server checked: source image unchanged, optional frame cap={capped}"
        )
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface",
        help="Isolated virtual Ethernet interface for the hardware check.",
    )
    args = parser.parse_args()
    check_startup_error()
    check(capped=True, interface=args.interface)
    if args.interface is None:
        check(capped=False)
