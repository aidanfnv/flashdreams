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

"""Client-paced synthetic model service without FlashDreams or GPU imports."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import Any

from model_service.service import MAX_FRAMES, ModelServer, _Handler, _parameters
from model_service.devices import DevicePool

_COMMON_PARAMETERS = {
    "width": {"type": "integer", "default": 320, "minimum": 1, "maximum": 1024},
    "height": {"type": "integer", "default": 240, "minimum": 1, "maximum": 768},
    "fps": {"type": "integer", "default": 30, "minimum": 1, "maximum": 120},
}
"""Output parameters understood by both stub models."""
MODELS = {
    "stub-gradient": {
        "name": "Moving color pattern",
        "stub": True,
        "input_image": "none",
        "parameters": {
            **_COMMON_PARAMETERS,
            "prompt": {"type": "string", "default": "", "max_length": 2048},
            "seed": {
                "type": "integer",
                "default": 0,
                "minimum": 0,
                "maximum": 0xFFFFFFFF,
            },
        },
    },
    "stub-image": {
        "name": "Animated input image",
        "stub": True,
        "input_image": "required_rgb24",
        "parameters": _COMMON_PARAMETERS,
    },
}
"""Discoverable synthetic models and their settings schemas."""
_INPUTS = {
    "pan": {"type": "integer", "default": 1, "minimum": -16, "maximum": 16},
}
"""Per-frame horizontal motion, demonstrating interactive model inputs."""


@dataclass
class StubSession:
    """Connection-local model state advanced only by explicit step requests."""

    model: str
    parameters: dict[str, Any]
    image: bytes
    frame_index: int = 0
    phase: int = 0

    def step(self, count: int, inputs: dict, *, status=None) -> tuple[dict, bytes]:
        """Generate raw RGB frames and report their simulation timeline."""
        if type(count) is not int or not 1 <= count <= MAX_FRAMES:
            raise ValueError(f"count must be an integer in 1..{MAX_FRAMES}")
        pan = _parameters(inputs, _INPUTS)["pan"]
        width, height = self.parameters["width"], self.parameters["height"]
        first = self.frame_index
        if status is not None and first == 0:
            status("generating", block=first + 1)
        payload = bytearray()
        for _ in range(count):
            if self.model == "stub-image":
                offset = (self.phase % width) * 3
                stride = width * 3
                for row in range(height):
                    source = self.image[row * stride : (row + 1) * stride]
                    payload.extend(source[offset:] + source[:offset])
            else:
                bias = (
                    int.from_bytes(
                        hashlib.sha256(
                            self.parameters["prompt"].encode("utf-8")
                        ).digest()[:4],
                        "little",
                    )
                    ^ self.parameters["seed"]
                )
                for y in range(height):
                    green = (y * 255 // max(1, height - 1) + (bias >> 8)) & 255
                    for x in range(width):
                        payload.extend(
                            (
                                (x * 255 // max(1, width - 1) + self.phase + bias)
                                & 255,
                                green,
                                ((x + y + self.phase) * 3 + (bias >> 16)) & 255,
                            )
                        )
            self.phase = (self.phase + pan) % width
            self.frame_index += 1
        return {
            "type": "frames",
            "format": "rgb24",
            "width": width,
            "height": height,
            "first_frame": first,
            "frame_count": count,
            "fps": self.parameters["fps"],
        }, bytes(payload)

    def reset(self) -> None:
        self.frame_index = self.phase = 0

    def close(self) -> None:
        return


class StubBackend:
    """Synthetic models using the shared model-service transport."""

    def catalog(self) -> list[dict]:
        return [
            {"id": name, **details, "inputs": _INPUTS}
            for name, details in MODELS.items()
        ]

    def open(self, model, values, payload, *, status=None) -> StubSession:
        if not isinstance(model, str) or model not in MODELS:
            raise ValueError("Unknown model")
        parameters = _parameters(values, MODELS[model]["parameters"])
        expected = (
            parameters["width"] * parameters["height"] * 3
            if model == "stub-image"
            else 0
        )
        if len(payload) != expected:
            raise ValueError(f"{model} requires {expected} RGB payload bytes")
        return StubSession(model, parameters, payload)


class StubServer(ModelServer):
    """Run the shared server with the deterministic synthetic backend."""

    def __init__(self, address, handler=_Handler, *, devices=None):
        backend = StubBackend()
        if devices is not None:
            backend = DevicePool(
                {name: StubBackend() for name in devices},
                [
                    {
                        "id": name,
                        "name": "Simulated inference device",
                        "type": "stub",
                        "total_memory": None,
                    }
                    for name in devices
                ],
            )
        super().__init__(address, backend, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--device",
        action="append",
        help="Simulated device ID; repeat to expose several.",
    )
    parser.add_argument(
        "--port", type=int, default=9390, help="TCP port; 0 chooses a free port."
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in 0..65535")
    if args.device and (
        any(not name.strip() for name in args.device)
        or len(set(args.device)) != len(args.device)
    ):
        parser.error("--device IDs must be nonempty and unique")
    with StubServer(
        (args.host, args.port), _Handler, devices=args.device or ["stub:0", "stub:1"]
    ) as server:
        print(
            f"Model stub listening at {server.server_address[0]}:{server.server_address[1]}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
