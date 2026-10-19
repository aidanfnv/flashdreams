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

"""Native CLI workflows through a locally built DOSBox-X and DJGPP client."""

import subprocess
import sys
import tempfile
import json
import threading
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual


def test_native_cli_workflows():
    """Check discovery, quoted settings, image upload, display, and CLI errors."""
    with tempfile.TemporaryDirectory(prefix="native-cli-check-") as temporary:
        directory = Path(temporary)
        image = directory / "image.rgb"
        image.write_bytes(
            (bytes(range(256)) * (800 * 450 * 3 // 256 + 1))[: 800 * 450 * 3]
        )
        cases = [
            (["--list-models"], "Native catalog check passed", True),
            (["stub-image", "--", "--help"], "Image input: required_rgb24", True),
            (
                [
                    "stub-gradient",
                    "--",
                    "--width",
                    "320",
                    "--height",
                    "200",
                    "--fps",
                    "24",
                    "--pan",
                    "-3",
                    "--seed",
                    "4294967295",
                    "--prompt",
                    'Neon café, "taxi", driver\'s [*]\npath\\',
                ],
                "Native TCP check passed",
                True,
            ),
            (
                [
                    "stub-image",
                    "--initial-rgb",
                    str(image),
                    "--ip",
                    "10.0.2.15",
                    "--bpp",
                    "24",
                    "--",
                    "--width",
                    "800",
                    "--height",
                    "450",
                    "--pan",
                    "-2",
                ],
                "Native TCP check passed",
                True,
            ),
            (["missing-model"], "Service does not offer model", False),
            (["stub-gradient", "--", "--fps", "0"], "invalid setting: --fps 0", False),
            (
                ["stub-gradient", "--", "--fps", "1e1"],
                "invalid setting: --fps 1e1",
                False,
            ),
            (
                ["stub-gradient", "--", "--fps", "24", "--fps", "30"],
                "duplicate runtime option: --fps",
                False,
            ),
            (["stub-gradient", "--", "--width"], "Missing value for option", False),
            (["stub-image"], "image requirement does not match", False),
            (
                [
                    "stub-gradient",
                    "--mode",
                    "mp4",
                    "--output-path",
                    "gradient.mp4",
                    "--",
                    "--width",
                    "34",
                    "--height",
                    "18",
                    "--fps",
                    "119",
                    "--pan",
                    "3",
                ],
                "DOS MP4 check passed: 34x18, 119 fps, 4 frames",
                True,
            ),
            (["stub-gradient", "--mode", "mp4"], "--output-path is required", False),
            (
                [
                    "stub-gradient",
                    "--check-frames",
                    "9",
                    "--mode",
                    "mp4",
                    "--output-path",
                    "nine.mp4",
                    "--",
                    "--width",
                    "34",
                    "--height",
                    "18",
                ],
                "9 frames; mean RGB error",
                True,
            ),
            (["--check-frames", "0"], "--check-frames must be 1..65535", False),
            (
                [
                    "stub-gradient",
                    "--mode",
                    "mp4",
                    "--output-path",
                    "odd.mp4",
                    "--",
                    "--width",
                    "33",
                    "--height",
                    "18",
                ],
                "MP4 requires even frame dimensions",
                False,
            ),
        ]
        for index, (arguments, expected, success) in enumerate(cases):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dos_client.network_demo",
                    "--check",
                    "--output-dir",
                    str(directory / str(index)),
                    *arguments,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=150,
            )
            assert (result.returncode == 0) == success, result.stdout
            assert expected in result.stdout, result.stdout


def test_native_device_discovery_and_routing(tmp_path):
    """Keep a busy device isolated while DOS displays frames from another device."""
    import socket

    from dos_client.emulator import launch
    from dos_client.network_demo import _display_rgb
    from dos_client.vbe_demo import _verify
    from model_service.protocol import PROTOCOL, VERSION, receive_message, send_message
    from model_service.stub_service import StubServer

    app = Path(__file__).resolve().parents[1]
    emulator = app.parents[2] / ".local-build/dosbox-x-prefix/bin/dosbox-x"
    with StubServer(("127.0.0.1", 0), devices=["stub:0", "stub:1"]) as server:
        server.capture_frames = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        client = socket.create_connection(server.server_address, timeout=5)
        request_id = 0

        def request(op, **fields):
            nonlocal request_id
            request_id += 1
            send_message(client, {"id": request_id, "op": op, **fields})
            return receive_message(client)

        def dos(name, options, success=True):
            directory = tmp_path / name
            result = launch(
                directory,
                [
                    "FLSHDRMS",
                    "remote",
                    *options,
                    "--port",
                    str(server.server_address[1]),
                    "--error-path",
                    "ERROR.TXT",
                ],
                dosbox_path=emulator,
                client_path=app / "client/build/FLSHDRMS.EXE",
                headless=True,
                timeout=120,
            )
            log = (directory / "CLIENT.TXT").read_text()
            error = directory / "ERROR.TXT"
            assert result == success, log + (
                error.read_text() if error.exists() else ""
            )
            return directory, log, error

        try:
            request("hello", protocol=PROTOCOL, version=VERSION)
            assert (
                request("open", model="stub-gradient", device="stub:0")[0]["device"]
                == "stub:0"
            )
            _, log, _ = dos("listing", ["--list-devices"])
            assert "default stub:0" in log
            assert (
                "stub:0" in log
                and "busy" in log
                and "stub:1" in log
                and "available" in log
            )
            directory, log, _ = dos(
                "selected", ["stub-gradient", "--device=stub:1", "--check-frames", "3"]
            )
            session = json.loads(
                next(
                    line[9:]
                    for line in log.splitlines()
                    if line.startswith("Session: ")
                )
            )
            assert session["device"] == "stub:1"
            assert "frames=3 " in log
            metadata, source = server.last_frames
            _verify(
                directory,
                _display_rgb(source, metadata["width"], metadata["height"]),
                None,
                directory,
            )
            _, _, error = dos("busy", ["stub-gradient", "--device", "stub:0"], False)
            assert "busy" in error.read_text()
            _, _, error = dos(
                "missing", ["stub-gradient", "--device", "missing"], False
            )
            assert "does not offer the requested device" in error.read_text()
            # The held job survives both DOS errors, then yields the default GPU.
            assert request("step", count=1)[0]["first_frame"] == 0
            request("close")
            _, log, _ = dos("default", ["stub-gradient", "--check-frames", "1"])
            session = json.loads(
                next(
                    line[9:]
                    for line in log.splitlines()
                    if line.startswith("Session: ")
                )
            )
            assert session["device"] == "stub:0"
        finally:
            client.close()
            server.shutdown()
            worker.join(timeout=5)


def test_native_named_servers_use_selected_catalog_and_device(tmp_path):
    """Discover two hosts, isolate busy ordinals, and display the selected host's RGB."""
    from contextlib import ExitStack

    from dos_client.emulator import launch
    from dos_client.network_demo import _display_rgb
    from dos_client.vbe_demo import _verify
    from model_service.devices import DevicePool
    from model_service.service import ModelServer, _parameters
    from model_service.stub_service import StubBackend

    class SettingsBackend(StubBackend):
        def __init__(self, extra):
            self.extra = extra
            self.jobs = []

        def catalog(self):
            entries = super().catalog()
            if self.extra:
                entries[0]["parameters"] = {
                    **entries[0]["parameters"],
                    "gain": {
                        "type": "number",
                        "default": 1.0,
                        "minimum": 0,
                        "maximum": 2,
                    },
                }
            return entries

        def open(self, model, values, payload, *, status=None):
            entry = next(entry for entry in self.catalog() if entry["id"] == model)
            effective = _parameters(values, entry["parameters"])
            self.jobs.append(effective)
            session = super().open(
                model,
                {k: v for k, v in values.items() if k != "gain"},
                payload,
                status=status,
            )
            session.parameters = effective
            return session

    app = Path(__file__).resolve().parents[1]
    emulator = app.parents[2] / ".local-build/dosbox-x-prefix/bin/dosbox-x"
    backends = [SettingsBackend(False), SettingsBackend(True)]
    pools = [
        DevicePool(
            {"cuda:0": backend},
            [{"id": "cuda:0", "name": name, "type": "stub", "total_memory": None}],
        )
        for name, backend in zip(("left", "right"), backends, strict=True)
    ]
    with ExitStack() as stack:
        servers = [
            stack.enter_context(ModelServer(("127.0.0.1", 0), pool)) for pool in pools
        ]
        workers = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in servers
        ]
        for server, worker in zip(servers, workers, strict=True):
            server.capture_frames = True
            worker.start()
        endpoints = [
            "--server",
            f"left=10.0.2.2:{servers[0].server_address[1]}",
            "--server",
            f"right=10.0.2.2:{servers[1].server_address[1]}",
        ]

        def dos(name, options, success=True, addresses=None):
            drive = tmp_path / name
            separator = options.index("--") if "--" in options else len(options)
            result = launch(
                drive,
                [
                    "FLSHDRMS",
                    "remote",
                    *options[:separator],
                    *(endpoints if addresses is None else addresses),
                    "--error-path",
                    "ERROR.TXT",
                    "--network-timeout",
                    "30",
                    *options[separator:],
                ],
                dosbox_path=emulator,
                client_path=app / "client/build/FLSHDRMS.EXE",
                headless=True,
                timeout=60,
            )
            log = (drive / "CLIENT.TXT").read_text()
            if (drive / "ERROR.TXT").exists():
                log += (drive / "ERROR.TXT").read_text()
            assert result == success, log
            return drive, log

        held = pools[0].open("stub-gradient", {}, b"", device="cuda:0")
        try:
            _, log = dos("devices", ["--list-devices"])
            assert "left/cuda:0" in log and "right/cuda:0" in log
            assert "busy" in log.split("Server right:")[0]
            assert "available" in log.split("Server right:")[1]
            _, log = dos("models", ["--list-models"])
            assert log.count("Models: stub-gradient stub-image") == 2
            for name, extra in (("left", False), ("right", True)):
                _, log = dos(
                    f"help-{name}",
                    ["stub-gradient", "--device", f"{name}/cuda:0", "--", "--help"],
                )
                assert ("--gain" in log) == extra
            assert len(backends[0].jobs) == 1 and not backends[1].jobs
            drive, log = dos(
                "selected",
                [
                    "stub-gradient",
                    "--device=right/cuda",
                    "--check-frames",
                    "2",
                    "--pixel-width",
                    "32",
                    "--pixel-height",
                    "16",
                    "--",
                    "--gain",
                    "1.25",
                ],
            )
            assert '"device":"cuda:0"' in log and "frames=2 " in log
            assert backends[1].jobs[0]["gain"] == 1.25 and len(backends[0].jobs) == 1
            metadata, source = servers[1].last_frames
            _verify(
                drive,
                _display_rgb(source, metadata["width"], metadata["height"]),
                None,
                drive,
            )
            _, log = dos("busy", ["stub-gradient", "--device", "left/cuda:0"], False)
            assert "busy" in log and held.step(1, {})[0]["first_frame"] == 0
            _, log = dos(
                "wrong-catalog",
                ["stub-gradient", "--device", "left/cuda:0", "--", "--gain", "1.25"],
                False,
            )
            assert "Unknown, duplicate, or invalid setting: --gain" in log, log
            _, log = dos(
                "missing-device", ["stub-gradient", "--device", "right/missing"], False
            )
            assert "does not offer the requested device" in log
            # An unselected offline host must not prevent model help on right.
            addresses = ["--server", "left=10.0.2.2:1", *endpoints[2:]]
            _, log = dos(
                "offline-unselected",
                ["stub-gradient", "--device", "right/cuda:0", "--", "--help"],
                addresses=addresses,
            )
            assert "--gain" in log and "Server left:" not in log
            for name, options, addresses, marker in (
                (
                    "ambiguous",
                    ["--list-models", "--device", "cuda:0"],
                    endpoints,
                    "must use NAME/ID",
                ),
                (
                    "unknown-server",
                    ["--list-models", "--device", "other/cuda:0"],
                    endpoints,
                    "configured server",
                ),
                (
                    "duplicate",
                    ["--list-models"],
                    [*endpoints[:2], *endpoints[:2]],
                    "Duplicate --server",
                ),
                (
                    "mixed",
                    ["--list-models", "--host", "10.0.2.2"],
                    endpoints,
                    "cannot be combined",
                ),
                (
                    "invalid",
                    ["--list-models"],
                    ["--server", "left=10.0.2.2:65536"],
                    "Expected --server",
                ),
            ):
                _, log = dos(name, options, False, addresses)
                assert marker in log, log
            held.close()
            _, log = dos("default", ["stub-gradient", "--check-frames", "1"])
            assert "Server left:" in log and "Server right:" not in log
            assert len(backends[0].jobs) == 2
        finally:
            held.close()
            for server, worker in zip(servers, workers, strict=True):
                server.shutdown()
                worker.join(timeout=5)


def test_mp4_finishes_with_model_and_stays_on_client():
    """Encode a complete finite model rollout on DOS and exit at end of generation."""
    from model_service.flashdreams_backend import T2VBackend
    from model_service.service import ModelServer, _Handler
    from t2v import T2VApplication, T2VApplicationDefaults
    from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

    pipelines = []
    overrides = []

    class CLIApplication(T2VApplication):
        def _apply_compile_override(self, config, enabled):
            overrides.append(("compile", enabled))
            return config

        def _apply_seed_override(self, config, seed):
            overrides.append(("seed", seed))
            return config

    def factory():
        pipeline = FakeT2VPipeline(width=32, height=16, first_block_frames=3)
        pipelines.append(pipeline)
        return CLIApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakeT2VPipelineConfig(pipeline),
                pixel_width=32,
                pixel_height=16,
                total_blocks=2,
                device="cpu",
                fps=24,
            )
        )

    backend = T2VBackend("cpu-t2v", factory, device="cpu")
    with tempfile.TemporaryDirectory(prefix="native-mp4-check-") as temporary:
        directory = Path(temporary) / "DOS drive"
        destination = directory / "A local clip.mp4"
        visible_during_session = []

        class StorageHandler(_Handler):
            def _capture(self, metadata, pixels):
                if metadata["first_frame"] == 1:
                    visible_during_session.append(
                        destination.is_file() and destination.stat().st_size > 0
                    )
                super()._capture(metadata, pixels)

        with ModelServer(
            ("127.0.0.1", 0), backend, handler=StorageHandler, capture_frames=True
        ) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dos_client.emulator",
                        "--storage-dir",
                        str(directory),
                        "--headless",
                        "--",
                        "FLSHDRMS",
                        "remote",
                        "cpu-t2v",
                        "--port",
                        str(server.server_address[1]),
                        "--mode",
                        "mp4",
                        "--output-path",
                        destination.name,
                        "--error-path",
                        "ERROR.TXT",
                        "--pixel-width=48",
                        "--pixel-height",
                        "32",
                        "--fps",
                        "17",
                        "--timeout",
                        "unbound",
                        "--",
                        "--prompt=A neon city",
                        "--no-compile",
                        "--seed",
                        "4294967295",
                        "--total-blocks",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=150,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert (
                    "Generation finished." in result.stdout
                    and "frames=3 " in result.stdout
                )
                assert "Local MP4:" in result.stdout and "VBE " not in result.stdout
                assert visible_during_session == [True]
                assert not pipelines[-1].closed
                assert all(
                    phase in result.stdout
                    for phase in (
                        "Loading model...",
                        "Preparing prompt and generation state...",
                        "Generating block 1/1...",
                        "Transferring RGB frames...",
                        "Encoding frame 1 on DOS...",
                    )
                ), result.stdout
                assert overrides == [("compile", False), ("seed", 4294967295)]
                assert pipelines[-1].generated == [0]
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
                            str(destination),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=30,
                    ).stdout
                )["streams"][0]
                assert (
                    metadata["codec_name"],
                    metadata["width"],
                    metadata["height"],
                ) == ("h264", 48, 32)
                assert metadata["nb_read_frames"] == "3"
                # MP4 rounds each frame duration to the nearest 90 kHz clock tick.
                assert float(Fraction(metadata["avg_frame_rate"])) == pytest.approx(
                    17, abs=17 * 17 / 90000
                )
                second_command = [
                    "--prompt=Another city"
                    if argument == "--prompt=A neon city"
                    else "SECOND.MP4"
                    if argument == destination.name
                    else argument
                    for argument in result.args
                ]
                second = subprocess.run(
                    second_command, capture_output=True, text=True, timeout=150
                )
                assert second.returncode == 0, second.stdout + second.stderr
                assert (
                    "Reusing resident model." in second.stdout
                    and "Loading model..." not in second.stdout
                )
                assert (
                    "frames=3 " in second.stdout
                    and (directory / "SECOND.MP4").is_file()
                )
                assert len(pipelines) == 2  # One catalog config, one actual model load.
                assert pipelines[-1].caches[-1]["text"] == ["Another city"]
                assert pipelines[-1].generated == [0, 0]
                assert overrides == [("compile", False), ("seed", 4294967295)]
                failed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dos_client.emulator",
                        "--storage-dir",
                        str(directory),
                        "--headless",
                        "--",
                        "FLSHDRMS",
                        "remote",
                        "cpu-t2v",
                        "--port",
                        str(server.server_address[1]),
                        "--mode",
                        "mp4",
                        "--output-path",
                        "invalid.mp4",
                        "--error-path",
                        "ERROR.TXT",
                        "--",
                        "--prompt",
                        " ",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=150,
                )
                assert (
                    failed.returncode != 0
                    and "prompt must be non-empty"
                    in (directory / "ERROR.TXT").read_text()
                )
                assert not (directory / "invalid.mp4").exists()
            finally:
                server.shutdown()
                worker.join(timeout=5)
                assert not worker.is_alive()


def test_remote_namespace_placement_and_timeout():
    """Reject misplaced flags and finalize DOS output at an explicit runtime limit."""
    from dos_client.emulator import launch
    from model_service.service import ModelServer
    from model_service.stub_service import StubBackend

    workspace = Path(__file__).resolve().parents[4]
    app = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native-runner-check-") as temporary:
        with ModelServer(("127.0.0.1", 0), StubBackend()) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for index, (arguments, marker, success) in enumerate(
                    [
                        (["remote", "--help"], "Usage: FLSHDRMS remote", True),
                        (["run-v2"], "Usage: FLSHDRMS remote", False),
                        (["stub-gradient"], "Usage: FLSHDRMS remote", False),
                        (["run"], "Usage: FLSHDRMS remote", False),
                        (
                            ["remote", "stub-gradient", "--prompt", "A city"],
                            "misplaced, or duplicate runtime option: --prompt",
                            False,
                        ),
                        (
                            ["remote", "stub-gradient", "--", "--width", "32"],
                            "Use --pixel-width, --pixel-height, or --fps before --.",
                            False,
                        ),
                        (
                            [
                                "remote",
                                "stub-gradient",
                                "--mode",
                                "mp4",
                                "--output-path",
                                "TIMED.MP4",
                                "--pixel-width",
                                "32",
                                "--pixel-height",
                                "16",
                                "--timeout",
                                "2.5",
                                "--",
                                "--prompt",
                                "An endless city",
                            ],
                            "Stopped at --timeout.",
                            True,
                        ),
                    ]
                ):
                    drive = Path(temporary) / str(index)
                    separator = (
                        arguments.index("--") if "--" in arguments else len(arguments)
                    )
                    command = [
                        "FLSHDRMS",
                        *arguments[:separator],
                        "--port",
                        str(server.server_address[1]),
                        "--error-path",
                        "ERROR.TXT",
                        *arguments[separator:],
                    ]
                    # Help/invalid command invocations do not use the service options.
                    actual = launch(
                        drive,
                        command,
                        dosbox_path=workspace
                        / ".local-build/dosbox-x-prefix/bin/dosbox-x",
                        client_path=app / "client/build/FLSHDRMS.EXE",
                        headless=True,
                        timeout=60,
                    )
                    assert actual == success
                    log = (drive / "CLIENT.TXT").read_text()
                    if (drive / "ERROR.TXT").is_file():
                        log += (drive / "ERROR.TXT").read_text()
                    assert marker in log, log
                    if "--timeout" in arguments:
                        assert "MP4 saved: TIMED.MP4" in log, log
                        decoded = subprocess.run(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-i",
                                str(drive / "TIMED.MP4"),
                                "-f",
                                "null",
                                "-",
                            ],
                            capture_output=True,
                            timeout=30,
                        )
                        assert decoded.returncode == 0, decoded.stderr
            finally:
                server.shutdown()
                worker.join(timeout=5)
                assert not worker.is_alive()
