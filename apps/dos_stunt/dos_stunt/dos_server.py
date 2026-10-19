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

"""Receive a general FlashDreams frame stream and serve a DOS VGA client."""

from __future__ import annotations

import argparse
import math
import os
import socket
import threading
from pathlib import Path

import numpy as np
from dos_stunt import CLIENT_DIR, DOSBOX_CONFIG, DOSBOX_PATH
from dos_stunt.dos_output import DosOutput

# The public wire helpers use only the standard library; no model runtime loads.
from dos_stunt.frame_stream import (
    check_hello,
    frame_shape,
    hello_message,
    receive_message,
    send_message,
)


def serve_connection(connection: socket.socket, output: DosOutput) -> None:
    """Forward one producer's images and return DOS input over the same stream."""
    connection.settimeout(5)
    greeting, payload = receive_message(connection)
    check_hello(greeting, payload)
    send_message(connection, hello_message())
    connection.settimeout(None)
    condition = threading.Condition()
    stop = threading.Event()
    write_lock = threading.Lock()
    pending = None
    failure: list[Exception] = []
    dropped = 0

    def send(header):
        with write_lock:
            send_message(connection, header)

    def finish():
        stop.set()
        output.request_stop()
        with condition:
            condition.notify_all()

    def receive():
        nonlocal pending, dropped
        try:
            while not stop.is_set():
                header, pixels = receive_message(connection)
                if header["type"] == "close" and not pixels:
                    return
                width, height = frame_shape(header, pixels)
                with condition:
                    if pending is not None:
                        dropped += 1
                    pending = width, height, pixels
                    condition.notify_all()
        except (EOFError, OSError):
            pass
        except Exception as error:
            failure.append(error)
        finally:
            finish()

    def forward_input():
        try:
            while not stop.wait(0.01):
                if output.sender_error is not None:
                    send({"type": "error", "message": str(output.sender_error)})
                    return
                for message in output.get_input_messages():
                    send(message)
                    if message["type"] == "close":
                        return
        except OSError:
            finish()

    reader = threading.Thread(target=receive, name="dos-server-frames", daemon=True)
    controls = threading.Thread(
        target=forward_input, name="dos-server-input", daemon=True
    )
    reader.start()
    controls.start()
    try:
        while True:
            with condition:
                condition.wait_for(lambda: pending is not None or stop.is_set())
                if stop.is_set():
                    break
                frame = pending
                pending = None
            assert frame is not None
            width, height, pixels = frame
            try:
                output.write(
                    np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3),
                )
            except InterruptedError:
                if not stop.is_set():
                    raise
                break
        if failure:
            raise failure[0]
    except Exception as error:
        try:
            send({"type": "error", "message": str(error)})
        except OSError:
            pass
        raise
    finally:
        finish()
        try:
            output.close()
        finally:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            reader.join()
            controls.join()
        print(f"DOS server replaced {dropped} pending RGB frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default="127.0.0.1", help="Frame-stream listener address."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9386,
        help="Frame-stream TCP port (0 selects a free port).",
    )
    parser.add_argument(
        "--interface", help="Dedicated Ethernet port for a physical DOS client."
    )
    parser.add_argument("--visible", action="store_true", help="Show DOSBox-X.")
    parser.add_argument(
        "--max-frames", type=int, help="Optional limit on completed DOS frames."
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="RGB contrast before VGA quantization (1.0 unchanged; try 1.4).",
    )
    parser.add_argument(
        "--palette",
        choices=("adaptive", "rgb332"),
        default="adaptive",
        help="VGA palette: colors selected per frame (default), or fixed RGB332.",
    )
    parser.add_argument(
        "--direct-vga",
        action="store_true",
        help="Write RGB332 chunks directly to VGA (faster, may show partial frames).",
    )
    parser.add_argument(
        "--packet-window",
        type=int,
        choices=range(1, 5),
        default=4,
        help="Maximum chunks awaiting ACKs (default 4; 1 waits after every chunk).",
    )
    parser.add_argument(
        "--dosbox-path",
        type=Path,
        default=DOSBOX_PATH,
    )
    parser.add_argument("--client-dir", type=Path, default=CLIENT_DIR)
    parser.add_argument("--dosbox-config", type=Path, default=DOSBOX_CONFIG)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not math.isfinite(args.contrast) or args.contrast < 0:
        parser.error("--contrast must be finite and nonnegative")
    if args.direct_vga and args.palette != "rgb332":
        parser.error("--direct-vga requires --palette rgb332")
    if args.interface and args.visible:
        parser.error("--visible requires DOSBox-X")
    if not args.interface:
        if not args.dosbox_path.is_file() or not os.access(args.dosbox_path, os.X_OK):
            parser.error("--dosbox-path must name an executable DOSBox-X file")
        if not (args.client_dir / "DOSRX.EXE").is_file():
            parser.error("--client-dir must contain DOSRX.EXE")
        if not args.dosbox_config.is_file():
            parser.error("--dosbox-config must name a configuration file")
        if args.visible and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            parser.error("--visible needs an X11 or Wayland desktop session")
    output = DosOutput(
        visible=args.visible,
        max_frames=args.max_frames,
        contrast=args.contrast,
        palette=args.palette,
        direct_vga=args.direct_vga,
        packet_window=args.packet_window,
        interface=args.interface,
        dosbox_path=args.dosbox_path,
        client_dir=args.client_dir,
        dosbox_config=args.dosbox_config,
    )
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.host, args.port))
            listener.listen(1)
            print(
                f"DOS server listening at {args.host}:{listener.getsockname()[1]}",
                flush=True,
            )
            # ponytail: one producer per invocation; add an accept loop for reuse.
            try:
                connection, _ = listener.accept()
                with connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    serve_connection(connection, output)
            finally:
                output.close()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
