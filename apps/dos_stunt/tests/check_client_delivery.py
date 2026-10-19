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

"""Check VGA delivery, loss recovery and throughput on a headless 8 MHz 286."""

from __future__ import annotations

import argparse
import hashlib
import json
import select
import shutil
import statistics
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from dos_stunt import CLIENT_DIR, DOSBOX_PATH
from dos_stunt.dos_output import DosOutput, rgb24_to_vga
from dos_stunt.ethernet_transport import (
    CHUNK_SIZE,
    FRAME_SIZE,
    read_acks,
    send_chunk,
    send_frame,
)
from dos_stunt.run_demo import make_frame
import numpy as np


@contextmanager
def receiver(executable: Path, *, direct: bool = False, palette: str = "rgb332"):
    """Use a private writable DOS drive so checks cannot disturb a visible run."""
    with tempfile.TemporaryDirectory(prefix="dos-delivery-") as temporary:
        directory = Path(temporary)
        shutil.copy2(executable, directory / "DOSRX.EXE")
        config = directory / "cpu.conf"
        config.write_text("[cpu]\ncore=normal\ncputype=286\ncycles=fixed 750\n")
        output = DosOutput(
            client_dir=directory,
            dosbox_config=config,
            dosbox_path=DOSBOX_PATH,
            direct_vga=direct,
            palette=palette,
        )
        try:
            output._start_emulator()
            yield output, directory
        finally:
            output.close()


def transfer(output, sequence, pixels, *, window, direct=False):
    output.acknowledged.clear()
    output.inflight_pixels = pixels
    retries = send_frame(
        output.sock,
        sequence,
        pixels,
        output.acknowledged,
        output._on_packet,
        window=window,
        direct_vga=direct,
        check_active=output._check_sender,
    )
    output.sent += 1
    output.seen += 1
    output.last_pixels = pixels
    output.inflight_pixels = None
    output.retries += retries
    return retries


def check_partial(executable, *, first_frame):
    """Verify cancelled out-of-order direct chunks before and after a full frame."""
    with receiver(executable, direct=True) as (output, directory):
        previous = bytes(FRAME_SIZE)
        if not first_frame:
            previous = make_frame(0)
            transfer(output, 0, previous, window=4, direct=True)
        pixels = make_frame(17)
        output.inflight_pixels = pixels
        output.acknowledged.clear()
        sequence = output.sent
        chunks = (3, 0, 1, 5)
        for chunk in chunks:
            send_chunk(output.sock, sequence, chunk, pixels, direct_vga=True)
        deadline = time.monotonic() + 5
        while not all((sequence, chunk) in output.acknowledged for chunk in chunks):
            assert time.monotonic() < deadline, "partial-frame ACK timed out"
            output._check_sender()
            read_acks(output.sock, output.acknowledged, output._on_packet)
            select.select([output.sock], [], [], 0.01)
        output.close()
        expected = bytearray(previous)
        for chunk in chunks:
            offset = chunk * CHUNK_SIZE
            expected[offset : offset + CHUNK_SIZE] = pixels[
                offset : offset + CHUNK_SIZE
            ]
        assert (directory / "FRAME.RAW").read_bytes() == expected
        stats = dict(
            item.split("=") for item in (directory / "STATS.TXT").read_text().split()
        )
        assert int(stats["frames"]) == (0 if first_frame else 1)
        assert bytes.fromhex(stats["chunk_mask"]) == b"\x2b\0\0\0\0\0"
    print(f"Partial direct frame verified; first_frame={first_frame}", flush=True)


def check_loss(executable, *, direct):
    """Lose the first image chunk and final ACK against the actual DOS receiver."""
    with receiver(executable, direct=direct) as (output, directory):
        original_send, original_recv = output.sock.send, output.sock.recv
        dropped = set()

        def send(payload):
            if (
                payload[:4] in (b"FD\x01\x01", b"FD\x01\x07")
                and payload[6] == 0
                and "image" not in dropped
            ):
                dropped.add("image")
                return
            original_send(payload)

        def recv(size):
            while True:
                packet = original_recv(size)
                if (
                    packet[14:18] == b"FD\x01\x02"
                    and packet[20] == 45
                    and "ack" not in dropped
                ):
                    dropped.add("ack")
                    continue
                return packet

        output.sock.send, output.sock.recv = send, recv
        pixels = make_frame(9)
        assert transfer(output, 65535, pixels, window=4, direct=direct) == 2
        pixels = make_frame(10)
        assert transfer(output, 0, pixels, window=4, direct=direct) == 0
        output.close()
        stats = dict(
            item.split("=") for item in (directory / "STATS.TXT").read_text().split()
        )
        assert int(stats["frames"]) == 2 and int(stats["accepted"]) == 92
        assert int(stats["last_sequence"]) == 0
        assert (directory / "FRAME.RAW").read_bytes() == pixels
        assert output.control_packets > 1
    print(f"First-chunk/final-ACK loss recovered; direct={direct}", flush=True)


def benchmark(label, executable, *, window, direct, frames):
    """Measure only ACK-confirmed delivery, excluding startup and five warmups."""
    elapsed = []
    with receiver(executable, direct=direct) as (output, directory):
        for sequence, pixels in enumerate(frames):
            started = time.perf_counter()
            transfer(output, sequence, pixels, window=window, direct=direct)
            elapsed.append(time.perf_counter() - started)
        output.close()
        stats = dict(
            item.split("=") for item in (directory / "STATS.TXT").read_text().split()
        )
        assert int(stats["frames"]) == len(frames)
        assert (directory / "FRAME.RAW").read_bytes() == frames[-1]
        assert int(stats["key_changes"]) == 0 and output.retries == 0
    measured = elapsed[5:]
    row = {
        "label": label,
        "window": window,
        "direct": direct,
        "fps": len(measured) / sum(measured),
        "median_ms": statistics.median(measured) * 1000,
        "p90_ms": sorted(measured)[int((len(measured) - 1) * 0.9)] * 1000,
        "stats": stats,
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, help="Previous DOSRX.EXE for comparison."
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--frames", type=int, default=10, help="Measured frames after five warmups."
    )
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/dos-client-delivery.json")
    )
    args = parser.parse_args()
    if args.runs < 1 or args.frames < 1:
        parser.error("--runs and --frames must be positive")
    executable = CLIENT_DIR / "DOSRX.EXE"
    check_partial(executable, first_frame=True)
    check_partial(executable, first_frame=False)
    check_loss(executable, direct=False)
    check_loss(executable, direct=True)
    with receiver(executable, palette="adaptive") as (output, _):
        image = np.arange(200 * 320 * 3, dtype=np.uint8).reshape(200, 320, 3)
        transfer(output, 0, rgb24_to_vga(image), window=4)
    configurations = [
        ("buffered", executable, 1, False),
        ("direct", executable, 1, True),
        ("queued", executable, 4, False),
        ("queued-direct", executable, 4, True),
    ]
    if args.baseline:
        configurations.insert(0, ("baseline", args.baseline, 1, False))
    frames = [make_frame(i) for i in range(5 + args.frames)]
    results = []
    for run in range(args.runs):
        for label, binary, window, direct in (
            configurations if run % 2 == 0 else reversed(configurations)
        ):
            results.append(
                benchmark(label, binary, window=window, direct=direct, frames=frames)
            )
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cpu": "286",
        "core": "normal",
        "cycles": 750,
        "backend": "unix",
        "palette": "rgb332",
        "warmup_frames": 5,
        "measured_frames": args.frames,
        "runs": results,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for label, *_ in configurations:
        rates = [row["fps"] for row in results if row["label"] == label]
        print(f"{label}: mean {statistics.mean(rates):.3f} fps", flush=True)


if __name__ == "__main__":
    main()
