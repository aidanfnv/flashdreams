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

"""Diagnostic model-service client that writes received RGB frames as PPM files."""

from __future__ import annotations

import argparse
import itertools
import socket
from pathlib import Path

from model_service.protocol import PROTOCOL, VERSION, receive_message, send_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9390)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--prompt", default="FlashDreams on DOS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pan", type=int, default=1)
    parser.add_argument(
        "--initial-rgb",
        type=Path,
        help="Packed RGB24 input, at the requested width/height.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or not 1 <= args.port <= 65535:
        parser.error("--frames must be positive and --port must be in 1..65535")
    identifiers = itertools.count(1)
    try:
        image = args.initial_rgb.read_bytes() if args.initial_rgb else b""
        with socket.create_connection((args.host, args.port), timeout=10) as connection:

            def exchange(operation, payload=b"", **fields):
                request_id = next(identifiers)
                send_message(
                    connection, {"id": request_id, "op": operation, **fields}, payload
                )
                reply, result = receive_message(connection)
                if reply.get("id") != request_id:
                    raise ValueError("Response request ID does not match")
                if reply.get("type") == "error":
                    raise ValueError(reply["message"])
                return reply, result

            hello, _ = exchange("hello", protocol=PROTOCOL, version=VERSION)
            catalog, _ = exchange("models")
            print(
                "Available models: "
                + ", ".join(model["id"] for model in catalog["models"])
            )
            parameters = {"width": args.width, "height": args.height}
            model = "stub-image" if args.initial_rgb else "stub-gradient"
            if not args.initial_rgb:
                parameters.update(prompt=args.prompt, seed=args.seed)
            exchange("open", image, model=model, parameters=parameters)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            frame_index = 0
            while frame_index < args.frames:
                count = min(hello["max_frames_per_step"], args.frames - frame_index)
                metadata, frames = exchange(
                    "step", count=count, inputs={"pan": args.pan}
                )
                size = args.width * args.height * 3
                if (
                    metadata.get("type") != "frames"
                    or metadata.get("format") != "rgb24"
                    or metadata.get("width") != args.width
                    or metadata.get("height") != args.height
                    or metadata.get("first_frame") != frame_index
                    or metadata.get("frame_count") != count
                    or len(frames) != size * count
                ):
                    raise ValueError("Invalid RGB frame response")
                for offset in range(count):
                    path = args.output_dir / f"frame-{frame_index:05}.ppm"
                    path.write_bytes(
                        f"P6\n{args.width} {args.height}\n255\n".encode("ascii")
                        + frames[offset * size : (offset + 1) * size]
                    )
                    print(f"Saved {path}")
                    frame_index += 1
            exchange("close")
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"Model probe failed: {error}\n")


if __name__ == "__main__":
    main()
