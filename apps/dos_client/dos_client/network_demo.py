# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check native DOS model output against a private service in DOSBox-X."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import shutil
import subprocess
import sys
import threading
from importlib.metadata import entry_points
from fractions import Fraction
from pathlib import Path

from dos_client.emulator import launch
from dos_client.vbe_demo import _verify


def _check_settings(
    args, model: str, settings: list[str], catalog: list[dict]
) -> tuple[dict, dict]:
    """Interpret requested catalog settings independently of the native parser."""
    from model_service.service import _parameters

    entry = next(item for item in catalog if item["id"] == model)
    schema, input_schema = entry["parameters"], entry["inputs"]
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name, spec in (schema | input_schema).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type={
                "integer": int,
                "number": float,
                "string": str,
                "boolean": json.loads,
            }[spec["type"]],
            default=argparse.SUPPRESS,
        )
    parsed = vars(parser.parse_args(settings))
    parameters = {name: value for name, value in parsed.items() if name in schema}
    for name, value in (("prompt", args.prompt), ("seed", args.seed)):
        if value is not None:
            if name in parameters or name not in schema:
                raise ValueError(f"Duplicate or unsupported --{name}")
            parameters[name] = value
    if args.prompt_file:
        if "prompt" in parameters or "prompt" not in schema:
            raise ValueError("Duplicate or unsupported --prompt-file")
        parameters["prompt"] = args.prompt_file.read_bytes().decode("utf-8")
    inputs = {name: value for name, value in parsed.items() if name in input_schema}
    return _parameters(parameters, schema), _parameters(inputs, input_schema)


def _display_rgb(source: bytes, width: int, height: int) -> bytes:
    """Fit a source frame into 640×480 using nearest pixels and black borders."""
    draw_width, draw_height = 640, 480
    if width * 480 > height * 640:
        draw_height = max(1, height * 640 // width)
    else:
        draw_width = max(1, width * 480 // height)
    left, top = (640 - draw_width) // 2, (480 - draw_height) // 2
    result = bytearray(640 * 480 * 3)
    columns = [x * width // draw_width for x in range(draw_width)]
    for y in range(draw_height):
        row = y * height // draw_height * width * 3
        pixels = b"".join(source[row + x * 3 : row + x * 3 + 3] for x in columns)
        offset = ((y + top) * 640 + left) * 3
        result[offset : offset + len(pixels)] = pixels
    return bytes(result)


def _verify_mp4(
    path: Path, source: bytes, width: int, height: int, fps: int, frame_count: int = 4
) -> None:
    """Decode the DOS-written file and check its timing, dimensions, and last frame."""
    metadata = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    )
    streams = metadata["streams"]
    if len(streams) != 1:
        raise ValueError("Expected one MP4 video stream")
    video = streams[0]
    if (
        video["codec_name"] != "h264"
        or (video["width"], video["height"]) != (width, height)
        or video["pix_fmt"] != "yuv420p"
        or int(video["nb_read_frames"]) != frame_count
        or abs(Fraction(video["avg_frame_rate"]) - fps) > 0.02
    ):
        raise ValueError(f"Invalid native MP4 metadata: {video}")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    size = width * height * 3
    if len(decoded) != frame_count * size:
        raise ValueError(f"Native MP4 did not decode to {frame_count} complete frames")
    # Compare to an independent 4:2:0 reference: tiny synthetic patterns can
    # lose substantial chroma detail even before H.264 quantization.
    reference = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-vf",
            "format=yuv420p,format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        input=source,
        capture_output=True,
        check=True,
        timeout=120,
    ).stdout
    if len(reference) != size:
        raise ValueError("Invalid reference image for MP4 verification")
    difference = sum(abs(a - b) for a, b in zip(decoded[-size:], reference)) / size
    if difference > 12:
        raise ValueError(
            f"Native MP4 differs from the 4:2:0 reference (mean error {difference:.2f})"
        )
    print(
        f"DOS MP4 check passed: {width}x{height}, {fps} fps, {frame_count} frames; mean RGB error vs 4:2:0 reference {difference:.2f}; {path}"
    )


def main() -> None:
    """Check native frames or catalog operations against a private service."""
    app = Path(__file__).resolve().parents[1]
    workspace = app.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", help="Remote model ID; defaults to stub-gradient."
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument(
        "--check-frames",
        type=int,
        default=4,
        help="Diagnostic frame count (default 4).",
    )
    parser.add_argument(
        "--mode", choices=("native-window", "mp4"), default="native-window"
    )
    parser.add_argument(
        "--output-path", type=Path, help="MP4 filename inside the check drive."
    )
    parser.add_argument("--ip", help="DOS static IPv4; omit for DHCP.")
    parser.add_argument("--netmask", default="255.255.255.0")
    parser.add_argument("--gateway", default="0.0.0.0")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-rgb", type=Path, help="Packed RGB24 input image.")
    parser.add_argument(
        "--timeout", type=int, help="Network inactivity limit for this check."
    )
    parser.add_argument("--bpp", type=int, choices=(24, 32))
    parser.add_argument(
        "--output-dir", type=Path, help="Persistent DOS drive for check artifacts."
    )
    parser.add_argument("--dosbox-config", type=Path)
    parser.add_argument(
        "--dosbox-path",
        type=Path,
        default=workspace / ".local-build/dosbox-x-prefix/bin/dosbox-x",
    )
    parser.add_argument(
        "--client-path", type=Path, default=app / "client/build/FLSHDRMS.EXE"
    )
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    settings = arguments[separator + 1 :]
    model = args.model or ("stub-gradient" if not args.list_models else None)
    real_check = model in {
        entry.name for entry in entry_points(group="flashdreams.applications_v2")
    }
    listing = args.list_models or model is None or settings == ["--help"]
    if args.mode == "mp4" and not listing and args.output_path is None:
        parser.error("--output-path is required for --mode mp4")
    if args.output_path is not None and (
        args.output_path.name != str(args.output_path)
        or not str(args.output_path).isascii()
        or any(char in str(args.output_path) for char in '\\:*?"<>|')
    ):
        parser.error("--output-path must be a DOS filename in the check drive")
    if args.mode != "mp4" and args.output_path is not None:
        parser.error("--output-path is only used with --mode mp4")
    if args.mode == "mp4" and not listing:
        if not all(shutil.which(tool) for tool in ("ffprobe", "ffmpeg")):
            parser.error("The MP4 check needs host ffprobe and ffmpeg")
    if args.list_models and args.model:
        parser.error("--list-models does not accept a model")
    if not 1 <= args.check_frames <= 65535:
        parser.error("--check-frames must be 1..65535")
    if args.seed is not None and not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must be uint32")
    if args.timeout is not None and not 1 <= args.timeout <= 3600:
        parser.error("--timeout must be 1..3600 seconds")
    for value in (args.ip, args.netmask, args.gateway):
        if value is not None:
            try:
                ipaddress.IPv4Address(value)
            except ipaddress.AddressValueError:
                parser.error(f"Invalid IPv4 address: {value}")
    server = None
    thread = None
    log_handler = None
    try:
        from model_service.service import ModelServer
        from model_service.stub_service import StubBackend, StubSession

        if real_check:
            from model_service.flashdreams_backend import T2VBackend
            from flashdreams.runtime_v2.application_registry import create_application

            logging.basicConfig(level=logging.INFO)
            backend = T2VBackend(model, lambda: create_application(model))
        else:
            backend = StubBackend()
        directory = (args.output_dir or app / "client/build/network-check").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        log_handler = logging.FileHandler(directory / "SERVER.TXT", mode="w")
        log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger = logging.getLogger("model_service.service")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(log_handler)
        server = ModelServer(("127.0.0.1", 0), backend, capture_frames=real_check)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        guest = ["FLSHDRMS", "remote"]
        if model:
            guest.append(model)
        application_args = []
        guest.extend(["--host", "10.0.2.2", "--port", str(server.server_address[1])])
        guest.extend(["--error-path", "ERROR.TXT", "--mode", args.mode])
        (directory / "ERROR.TXT").unlink(missing_ok=True)
        if args.output_path is not None:
            guest.extend(["--output-path", str(args.output_path)])
        if args.list_models:
            guest.append("--list-models")
        if args.prompt is not None or args.prompt_file:
            if args.prompt_file:
                prompt = args.prompt_file.read_bytes()
            else:
                prompt = args.prompt.encode("utf-8")
            (directory / "PROMPT.TXT").write_bytes(prompt)
            application_args.extend(["--prompt-file", "PROMPT.TXT"])
        if args.seed is not None:
            application_args.extend(["--seed", str(args.seed)])
        if args.initial_rgb:
            (directory / "IMAGE.RGB").write_bytes(args.initial_rgb.read_bytes())
            application_args.extend(["--initial-rgb", "IMAGE.RGB"])
        if args.ip:
            guest.extend(
                ["--ip", args.ip, "--netmask", args.netmask, "--gateway", args.gateway]
            )
        if args.bpp:
            guest.extend(["--bpp", str(args.bpp)])
        if args.timeout is not None or real_check:
            guest.extend(["--network-timeout", str(args.timeout or 600)])
        guest.extend(["--check-frames", str(args.check_frames)])
        # The diagnostic's catalog arguments become the native remote CLI.
        for index in range(0, len(settings), 2):
            key = settings[index]
            value = settings[index + 1 : index + 2]
            if key in ("--width", "--height", "--fps"):
                guest.extend([key if key == "--fps" else "--pixel-" + key[2:], *value])
            elif key == "--compile" and value in (["true"], ["false"]):
                application_args.append(
                    "--compile" if value == ["true"] else "--no-compile"
                )
            else:
                application_args.extend([key, *value])
        if application_args:
            guest.extend(["--", *application_args])
        success = launch(
            directory,
            guest,
            dosbox_path=args.dosbox_path,
            client_path=args.client_path,
            dosbox_config=args.dosbox_config,
            headless=True,
            timeout=max(120, (args.timeout or 600) * 2) if real_check else 120,
        )
        client_log = (directory / "CLIENT.TXT").read_text()
        print(client_log.strip())
        errors = directory / "ERROR.TXT"
        if errors.is_file() and errors.stat().st_size:
            raise ValueError(errors.read_text().strip())
        if not success:
            raise ValueError(f"DOS command failed; see {directory}")
        if listing:
            marker = (
                f"Model: {model}\nRuntime options before --:"
                if settings == ["--help"]
                else "Select a model with FLSHDRMS remote APPLICATION"
            )
            if marker not in client_log or "Session:" in client_log:
                raise ValueError("Native catalog/help check failed")
            if (
                "Models: " + " ".join(item["id"] for item in backend.catalog())
                not in client_log
            ):
                raise ValueError("Native catalog check failed")
            print(f"Native catalog check passed; logs: {directory}")
            return
        if (
            f"PASS: model-service session; frames={args.check_frames} "
            not in client_log
        ):
            raise ValueError(f"Native network check failed; see {directory}")
        parameters, inputs = _check_settings(args, model, settings, backend.catalog())
        negotiated = json.loads(
            next(
                line.removeprefix("Session: ")
                for line in client_log.splitlines()
                if line.startswith("Session: ")
            )
        )
        if negotiated["model"] != model or negotiated["parameters"] != parameters:
            raise ValueError("Native client did not apply the requested CLI settings")
        width, height = parameters["width"], parameters["height"]
        if real_check:
            if server.last_frames is None:
                raise ValueError("The model service did not return RGB frames")
            metadata, expected = server.last_frames
            if (
                metadata["first_frame"] != args.check_frames - 1
                or metadata["frame_count"] != 1
            ):
                raise ValueError("Unexpected model output captured for verification")
            (directory / "source.ppm").write_bytes(
                f"P6\n{width} {height}\n255\n".encode() + expected
            )
        else:
            image = args.initial_rgb.read_bytes() if args.initial_rgb else b""
            session = StubSession(model, parameters, image)
            for index in range(0, args.check_frames, 4):
                _, expected = session.step(min(4, args.check_frames - index), inputs)
        expected = expected[-width * height * 3 :]
        if args.mode == "mp4":
            _verify_mp4(
                directory / args.output_path,
                expected,
                width,
                height,
                parameters["fps"],
                args.check_frames,
            )
        else:
            _verify(
                directory, _display_rgb(expected, width, height), args.bpp, directory
            )
        print(f"Native TCP check passed; logs: {directory}")
    except (OSError, ValueError, LookupError, subprocess.SubprocessError) as error:
        parser.exit(1, f"DOS check failed: {error}\n")
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join()
        if log_handler:
            logging.getLogger("model_service.service").removeHandler(log_handler)
            log_handler.close()


if __name__ == "__main__":
    main()
