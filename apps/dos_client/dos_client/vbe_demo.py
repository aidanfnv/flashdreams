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

"""DOSBox-X launcher and framebuffer check for the native VBE display probe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _verify(directory: Path, source: bytes, bpp: int | None, output: Path) -> None:
    metadata = dict(
        item.split("=") for item in (directory / "MODE.TXT").read_text().split()
    )
    mode = {key: int(value) for key, value in metadata.items()}
    if (
        (mode["width"], mode["height"]) != (640, 480)
        or mode["version"] < 0x200
        or mode["bpp"] not in (24, 32)
        or (bpp is not None and mode["bpp"] != bpp)
    ):
        raise ValueError(f"Unexpected video mode: {mode}")
    pixel_size = mode["bpp"] // 8
    pitch = mode["pitch"]
    if pitch < 640 * pixel_size:
        raise ValueError("Invalid framebuffer pitch")
    actual = (directory / "VBE.RAW").read_bytes()
    expected = bytearray(pitch * 480)
    preview = bytearray()
    for index, (red, green, blue) in enumerate(
        zip(source[0::3], source[1::3], source[2::3])
    ):
        packed = red << mode["red"] | green << mode["green"] | blue << mode["blue"]
        offset = index // 640 * pitch + index % 640 * pixel_size
        expected[offset : offset + pixel_size] = packed.to_bytes(pixel_size, "little")
        readback = int.from_bytes(actual[offset : offset + pixel_size], "little")
        preview.extend(
            (readback >> mode[channel]) & 255 for channel in ("red", "green", "blue")
        )
    if actual != expected:
        raise ValueError("DOS framebuffer differs from the input RGB frame")
    output.mkdir(parents=True, exist_ok=True)
    if directory.resolve() != output.resolve():
        for name in ("MODE.TXT", "VBE.RAW", "CLIENT.TXT", "ERROR.TXT", "DOSBOX.LOG"):
            shutil.copy2(directory / name, output / name)
    path = output / "frame.ppm"
    path.write_bytes(b"P6\n640 480\n255\n" + preview)
    print(f"DOS VBE check passed: 640x480, {mode['bpp']} bpp, pitch {pitch}; {path}")


def main() -> None:
    """Launch the visible probe, or verify its device readback without a window."""
    app = Path(__file__).resolve().parents[1]
    workspace = app.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Headless RGB framebuffer check."
    )
    parser.add_argument("--bpp", type=int, choices=(24, 32))
    parser.add_argument(
        "--rgb", type=Path, help="640x480 packed RGB24 file to display."
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory for check artifacts."
    )
    parser.add_argument(
        "--dosbox-config", type=Path, help="Additional DOSBox-X settings."
    )
    parser.add_argument(
        "--dosbox-path",
        type=Path,
        default=workspace / ".local-build/dosbox-x-prefix/bin/dosbox-x",
    )
    parser.add_argument(
        "--client-path",
        type=Path,
        default=app / "client/build/VBEDEMO.EXE",
    )
    args = parser.parse_args()
    if args.output_dir and not args.check:
        parser.error("--output-dir requires --check")
    try:
        if not args.client_path.is_file():
            raise ValueError(
                "Build the native probe first: bash apps/dos_client/client/build.sh"
            )
        if not args.dosbox_path.is_file():
            raise ValueError(f"DOSBox-X executable does not exist: {args.dosbox_path}")
        source = args.rgb.read_bytes() if args.rgb else None
        if args.check and source is None:
            from model_service.stub_service import StubSession

            session = StubSession(
                "stub-gradient",
                {
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "prompt": "DOS VBE framebuffer check",
                    "seed": 42,
                },
                b"",
            )
            _, source = session.step(1, {"pan": 0})
        if source is not None and len(source) != 640 * 480 * 3:
            raise ValueError("--rgb must contain exactly 640x480 packed RGB24 pixels")
        with tempfile.TemporaryDirectory(prefix="dos-vbe-") as temporary:
            directory = Path(temporary)
            shutil.copy2(args.client_path, directory / "VBEDEMO.EXE")
            guest = "VBEDEMO.EXE"
            if args.bpp:
                guest += f" --bpp {args.bpp}"
            if source is not None:
                (directory / "IMAGE.RGB").write_bytes(source)
                guest += " --rgb IMAGE.RGB"
            if args.check:
                guest += " --check"
            command = [
                str(args.dosbox_path.resolve()),
                "-fastlaunch",
                "-nopromptfolder",
                "-conf",
                str(app / "client/dosbox-vbe.conf"),
            ]
            if args.dosbox_config:
                command.extend(["-conf", str(args.dosbox_config.resolve())])
            if args.check:
                command.append("-silent")
            command.extend(
                [
                    "-c",
                    "mount c .",
                    "-c",
                    "c:",
                    "-c",
                    guest + " > CLIENT.TXT",
                    "-c",
                    "exit",
                ]
            )
            env = os.environ.copy()
            env["XDG_CONFIG_HOME"] = str(directory / "config")
            env["SDL_VIDEODRIVER"] = (
                "dummy" if args.check else ("x11" if env.get("DISPLAY") else "wayland")
            )
            env["SDL_AUDIODRIVER"] = "dummy"
            with (directory / "DOSBOX.LOG").open("w") as log:
                subprocess.run(
                    command,
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=30 if args.check else None,
                )
            client_log = (directory / "CLIENT.TXT").read_text()
            print(client_log.strip())
            if not client_log.startswith("VBE "):
                error = directory / "ERROR.TXT"
                raise ValueError(
                    error.read_text().strip()
                    if error.is_file()
                    else "Native display probe failed"
                )
            if args.check:
                output = (
                    args.output_dir or app / f"client/build/check-{args.bpp or 'auto'}"
                )
                _verify(directory, source, args.bpp, output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"VBE demo failed: {error}\n")


if __name__ == "__main__":
    main()
