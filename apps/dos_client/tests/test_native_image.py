# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode real files in DOS and check the pixels received by the model pipeline."""

import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from model_service.flashdreams_backend import T2VBackend
from model_service.service import ModelServer
from t2v import T2VApplicationDefaults
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

from apps.model_service.tests.image_application import ImageApplication

pytestmark = pytest.mark.manual


def test_dos_png_jpeg_resize_and_file_errors():
    """Keep decoding/resizing on DOS, preserving RGB order and reporting bad files."""
    pipelines = []

    def factory():
        pipeline = FakeT2VPipeline(width=32, height=16, first_block_frames=1)
        pipelines.append(pipeline)
        return ImageApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakeT2VPipelineConfig(pipeline),
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
            ),
            input_image="required",
        )

    backend = T2VBackend("cpu-image", factory, device="cpu")
    with tempfile.TemporaryDirectory(prefix="native-image-check-") as temporary:
        drive = Path(temporary)
        y, x = np.indices((16, 32))
        pixels = np.stack((x * 7, y * 15, (x + y) * 5), axis=-1).astype(np.uint8)
        rgb = Image.fromarray(pixels)
        rgb.save(drive / "RGB.PNG")
        rgb.convert("RGBA").save(drive / "RGBA.PNG")
        rgb.quantize(colors=32).save(drive / "INDEX.PNG")
        rgb.save(drive / "BASE.JPG", quality=95, subsampling=0)
        rgb.save(drive / "PROG.JPG", quality=95, subsampling=0, progressive=True)
        rgb.resize((16, 8)).save(drive / "SMALL.PNG")
        (drive / "BROKEN.PNG").write_bytes(b"not a PNG")
        (drive / "SHORT.RGB").write_bytes(pixels.tobytes()[:-1])
        Image.new("RGB", (2048, 2049)).save(drive / "HUGE.PNG")

        with ModelServer(("127.0.0.1", 0), backend) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()

            def run(arguments, runtime=()):
                return subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "dos_client.emulator",
                        "--storage-dir",
                        str(drive),
                        "--headless",
                        "--",
                        "FLSHDRMS",
                        "remote",
                        "cpu-image",
                        "--port",
                        str(server.server_address[1]),
                        "--mode",
                        "mp4",
                        "--output-path",
                        "CLIP.MP4",
                        "--error-path",
                        "ERROR.TXT",
                        *runtime,
                        "--",
                        "--prompt",
                        "Colors move across a canvas",
                        *arguments,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )

            try:
                for name in (
                    "RGB.PNG",
                    "RGBA.PNG",
                    "INDEX.PNG",
                    "BASE.JPG",
                    "PROG.JPG",
                    "SMALL.PNG",
                ):
                    resize = name == "SMALL.PNG"
                    width, height = (64, 32) if resize else (32, 16)
                    runtime = (
                        ["--pixel-width", str(width), "--pixel-height", str(height)]
                        if resize
                        else []
                    )
                    result = run(["--image-path", name], runtime)
                    assert result.returncode == 0, (
                        result.stdout
                        + result.stderr
                        + (drive / "ERROR.TXT").read_text()
                    )
                    assert (
                        "decoded on DOS" in result.stdout
                        and "Generation finished" in result.stdout
                    )
                    with Image.open(drive / name) as source:
                        expected = np.asarray(
                            source.convert("RGB").resize(
                                (width, height), Image.Resampling.BILINEAR
                            )
                        )
                    conditioned = pipelines[-1].caches[-1]["image"]
                    assert tuple(conditioned.shape) == (1, 3, height, width)
                    received = (
                        conditioned[0].float().permute(1, 2, 0).numpy() + 1
                    ) * 127.5
                    # JPEG decoder rounding and bfloat16 normalization can differ by a few levels.
                    assert np.max(np.abs(received - expected.astype(float))) <= 4
                    assert not pipelines[-1].closed

                for arguments, error in (
                    (["--image-path", "MISSING.PNG"], "Cannot read PNG/JPEG"),
                    (["--image-path", "BROKEN.PNG"], "Cannot read PNG/JPEG"),
                    (["--image-path", "HUGE.PNG"], "Image exceeds"),
                    (["--initial-rgb", "SHORT.RGB"], "must match the requested"),
                    (
                        ["--image-path", "RGB.PNG", "--initial-rgb", "SHORT.RGB"],
                        "not both",
                    ),
                ):
                    before = len(pipelines)
                    (drive / "CLIP.MP4").unlink(missing_ok=True)
                    result = run(arguments)
                    assert result.returncode != 0, result.stdout
                    assert error in (drive / "ERROR.TXT").read_text()
                    assert (
                        len(pipelines) == before and not (drive / "CLIP.MP4").exists()
                    )
            finally:
                server.shutdown()
                worker.join(timeout=5)
                assert not worker.is_alive()
