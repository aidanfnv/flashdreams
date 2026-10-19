# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the real application/session API through the service with a CPU pipeline."""

import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from model_service.flashdreams_backend import ApplicationBackend, T2VBackend
from model_service.protocol import PROTOCOL, VERSION, receive_message, send_message
from model_service.service import serve_connection
from t2v import T2VApplication, T2VApplicationDefaults
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

from apps.model_service.tests.image_application import ImageApplication

pytestmark = pytest.mark.ci_cpu


def test_resident_jobs_reuse_reload_and_shutdown():
    """Reuse weights with fresh prompts, reload incompatible jobs, and release on stop."""
    configs = []

    class Application(ImageApplication):
        def _apply_compile_override(self, config, enabled):
            return config

        def _apply_seed_override(self, config, seed):
            return config

    def factory():
        config = FakeT2VPipelineConfig(FakeT2VPipeline(width=32, height=16))
        config.pipeline.diffusion_model = SimpleNamespace(rng=torch.Generator())
        configs.append(config)
        return Application(
            defaults=T2VApplicationDefaults(
                pipeline_config=config,
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
                device="cpu",
            ),
            input_image="optional",
        )

    backend = T2VBackend("cpu-resident", factory, device="cpu")
    phases = []
    parameters = {"prompt": "First scene"}
    image = bytes([255, 128, 0]) * (32 * 16)

    def status(phase, **fields):
        phases.append(phase)

    first = backend.open("cpu-resident", parameters, image, status=status)
    assert phases == ["loading", "preparing"]
    config = configs[-1]
    image_path = backend._resident.application.image_path
    assert image_path.is_file()
    original_noise = torch.rand(4, generator=config.pipeline.diffusion_model.rng)
    loop = first._session.model_loop
    metadata, original = first.step(1, {}, status=status)
    assert metadata["first_frame"] == 0 and phases[-1] == "generating"
    first.close()
    assert loop.state.cache is None and not config.pipeline.closed
    assert image_path.is_file()
    phases.clear()
    second = backend.open(
        "cpu-resident",
        {**parameters, "prompt": "Second scene", "fps": 24},
        image,
        status=status,
    )
    assert phases == ["reusing", "preparing"]
    assert len(configs) == 2 and config.setup_count == 1
    assert backend._resident.application.image_path == image_path
    assert torch.equal(
        torch.rand(4, generator=config.pipeline.diffusion_model.rng), original_noise
    )
    assert config.pipeline.caches[-1]["text"] == ["Second scene"]
    metadata, fresh = second.step(1, {})
    assert metadata["first_frame"] == 0 and fresh == original
    second.close()

    # The existing application API prepares the image and rollout length in init.
    # These changes therefore reload, along with model construction settings.
    parameters["prompt"] = "Third scene"
    for changed in (
        {},
        {"width": 64},
        {"seed": 7},
        {"compile": True},
        {"total_blocks": 2},
    ):
        parameters.update(changed)
        image = b""  # Dropping image conditioning must not reuse the previous image.
        previous = configs[-1]
        phases.clear()
        job = backend.open("cpu-resident", parameters, image, status=status)
        assert phases == ["loading", "preparing"]
        assert previous.pipeline.closed and configs[-1].setup_count == 1
        assert not image_path.exists()
        assert configs[-1].pipeline.caches[-1]["image"] is None
        job.close()

    # Invalid requests and another busy client leave resident weights alone.
    config = configs[-1]
    with pytest.raises(ValueError):
        backend.open("cpu-resident", {**parameters, "total_blocks": 0}, b"")
    active = backend.open("cpu-resident", parameters, b"")
    with pytest.raises(ValueError, match="busy"):
        backend.open("cpu-resident", parameters, b"")
    backend.close()
    assert not config.pipeline.closed
    active.close()
    assert config.pipeline.closed
    with pytest.raises(ValueError, match="shutting down"):
        backend.open("cpu-resident", parameters, b"")
    backend.close()


def test_catalog_selection_across_applications_and_shared_device_lease():
    """Select text/image applications over one connection without loading on discovery."""
    configs = {"cpu-text": [], "cpu-image": []}

    def factory(model, width, height, input_image):
        def create():
            config = FakeT2VPipelineConfig(FakeT2VPipeline(width=width, height=height))
            configs[model].append(config)
            return ImageApplication(
                defaults=T2VApplicationDefaults(
                    pipeline_config=config,
                    pixel_width=width,
                    pixel_height=height,
                    total_blocks=1,
                ),
                input_image=input_image,
            )

        return create

    def broken():
        raise ImportError("An optional dependency is missing")

    backend = ApplicationBackend(
        {
            "cpu-text": factory("cpu-text", 32, 16, "none"),
            "cpu-image": factory("cpu-image", 64, 32, "required"),
            "unsupported": object,
            "broken": broken,
        },
        device="cpu",
    )
    assert all(
        items[0].setup_count == 0 and len(items) == 1 for items in configs.values()
    )
    sender, receiver = socket.socketpair()
    worker = threading.Thread(target=serve_connection, args=(receiver, backend))
    worker.start()
    request_id = 0

    def request(operation, payload=b"", **fields):
        nonlocal request_id
        request_id += 1
        send_message(sender, {"id": request_id, "op": operation, **fields}, payload)
        response, pixels = receive_message(sender)
        assert response["id"] == request_id
        return response, pixels

    parameters = {"prompt": "A moving scene"}
    image = bytes([255, 128, 0]) * (64 * 32)
    try:
        sender.settimeout(10)
        request("hello", protocol=PROTOCOL, version=VERSION)
        response, _ = request("models")
        assert [
            (m["id"], m["input_image"], m["parameters"]["width"]["default"])
            for m in response["models"]
        ] == [
            ("cpu-image", "required_rgb24", 64),
            ("cpu-text", "none", 32),
        ]
        response, _ = request("open", model="unsupported", parameters=parameters)
        assert response["type"] == "error" and response["code"] == "request"
        response, _ = request("open", model="cpu-text", parameters=parameters)
        assert response["model"] == "cpu-text" and response["parameters"]["width"] == 32
        with pytest.raises(ValueError, match="busy"):
            backend.open("cpu-image", parameters, image)
        assert len(configs["cpu-image"]) == 1
        response, pixels = request("step", count=1)
        assert response["width"] == 32 and len(pixels) == 32 * 16 * 3
        request("close")
        assert not configs["cpu-text"][-1].pipeline.closed
        response, _ = request("open", model="cpu-image", parameters=parameters)
        assert (
            response["type"] == "error" and "requires an initial" in response["message"]
        )
        response, _ = request(
            "open", model="cpu-image", parameters=parameters, payload=image
        )
        assert configs["cpu-text"][-1].pipeline.closed
        assert (
            response["model"] == "cpu-image" and response["parameters"]["width"] == 64
        )
        response, pixels = request("step", count=1)
        assert response["width"] == 64 and len(pixels) == len(image)
        request("close")
        assert not configs["cpu-image"][-1].pipeline.closed
    finally:
        sender.close()
        worker.join(timeout=10)
        receiver.close()
        assert not worker.is_alive()
        backend.close()
    assert configs["cpu-image"][-1].pipeline.closed


@pytest.mark.parametrize("capability", ["none", "optional", "required"])
def test_image_capabilities_validation_and_reset(capability):
    """Validate uploads before model loading and retain RGB conditioning on reset."""
    configs = []

    def factory():
        config = FakeT2VPipelineConfig(FakeT2VPipeline(width=32, height=16))
        configs.append(config)
        return ImageApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=config,
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
            ),
            input_image=capability,
        )

    backend = T2VBackend("cpu-video", factory, device="cpu")
    assert backend.catalog()[0]["input_image"] == (
        "none" if capability == "none" else f"{capability}_rgb24"
    )
    image = np.arange(16 * 32 * 3, dtype=np.uint8).reshape(16, 32, 3)
    parameters = {"prompt": "A colorful city"}
    if capability == "none":
        with pytest.raises(ValueError, match="does not accept"):
            backend.open("cpu-video", parameters, image.tobytes())
    else:
        with pytest.raises(ValueError, match="match the requested"):
            backend.open("cpu-video", parameters, image.tobytes()[:-1])
    if capability == "required":
        with pytest.raises(ValueError, match="requires an initial"):
            backend.open("cpu-video", parameters, b"")
    assert len(configs) == 1 and configs[0].setup_count == 0
    if capability != "required":
        session = backend.open("cpu-video", parameters, b"")
        assert configs[-1].pipeline.caches[0]["image"] is None
        session.close()
    if capability != "none":
        session = backend.open("cpu-video", parameters, image.tobytes())
        try:
            expected = (
                torch.from_numpy(image).to(torch.bfloat16).permute(2, 0, 1)[None]
                / 127.5
                - 1
            )
            pipeline = configs[-1].pipeline
            assert torch.equal(pipeline.caches[-1]["image"], expected)
            session.reset()
            assert torch.equal(pipeline.caches[-1]["image"], expected)
            assert pipeline.caches[-1]["text"] == [parameters["prompt"]]
        finally:
            session.close()
        assert not pipeline.closed
    backend.close()
    assert backend._resident.image_directory is None


def test_image_upload_cleanup_after_initialization_failure():
    """Remove a temporary upload even when the application's normal init fails."""
    paths = []

    class FailingApplication(ImageApplication):
        def _apply_parsed_arguments(self, args):
            super()._apply_parsed_arguments(args)
            paths.append(self.image_path)
            assert self.image_path.is_file()
            raise RuntimeError("preparation failed")

    def factory():
        return FailingApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakeT2VPipelineConfig(FakeT2VPipeline()),
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
            ),
            input_image="required",
        )

    backend = T2VBackend("cpu-image", factory, device="cpu")
    with pytest.raises(RuntimeError, match="preparation failed"):
        backend.open("cpu-image", {"prompt": "A scene"}, bytes(32 * 16 * 3))
    assert not paths[0].parent.exists()
    assert backend._resident.application is backend._resident.image_directory is None
    backend.close()


def test_disconnect_during_generation_status_keeps_healthy_weights():
    """Release a departed peer's cache and reuse the model after a failed progress send."""
    configs = []

    def factory():
        config = FakeT2VPipelineConfig(FakeT2VPipeline(width=32, height=16))
        configs.append(config)
        return T2VApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=config,
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
            )
        )

    def disconnected(phase, **fields):
        raise BrokenPipeError("peer disconnected")

    backend = T2VBackend("cpu-stream", factory, device="cpu")
    parameters = {"prompt": "A streaming scene"}
    first = backend.open("cpu-stream", parameters, b"")
    with pytest.raises(BrokenPipeError, match="peer disconnected"):
        first.step(1, {}, status=disconnected)
    first.close()
    assert not configs[-1].pipeline.closed
    second = backend.open("cpu-stream", parameters, b"")
    assert len(configs) == 2 and configs[-1].setup_count == 1
    metadata, _ = second.step(1, {})
    assert metadata["first_frame"] == 0
    second.close()
    backend.close()


def test_model_blocks_reset_failures_and_disconnect_cleanup():
    """Drain blocks across requests, rewind, and release the device after failure/EOF."""
    configs = []
    fail_at = None

    def factory():
        config = FakeT2VPipelineConfig(
            FakeT2VPipeline(width=32, height=16, fail_at=fail_at)
        )
        configs.append(config)
        return T2VApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=config,
                pixel_width=32,
                pixel_height=16,
                total_blocks=2,
                device="cpu",
            )
        )

    backend = T2VBackend("cpu-t2v", factory, device="cpu")
    assert backend.catalog()[0]["stub"] is False
    assert configs[0].setup_count == 0  # Discovery must not load a model.
    parameters = {"prompt": "A neon café", "fps": 24}
    sender, receiver = socket.socketpair()
    worker = threading.Thread(target=serve_connection, args=(receiver, backend))
    worker.start()
    request_id = 0

    def request(operation, **fields):
        nonlocal request_id
        request_id += 1
        send_message(sender, {"id": request_id, "op": operation, **fields})
        metadata, payload = receive_message(sender)
        assert metadata["id"] == request_id
        return metadata, payload

    try:
        sender.settimeout(10)
        response, _ = request("hello", protocol=PROTOCOL, version=VERSION)
        assert response["end_of_stream"] == "finished"
        response, _ = request("open", model="cpu-t2v", parameters={"prompt": " "})
        assert response["type"] == "error" and len(configs) == 1
        response, _ = request("open", model="cpu-t2v", parameters=parameters)
        assert response["type"] == "session" and response["parameters"]["fps"] == 24
        pipeline = configs[-1].pipeline
        assert pipeline.device == "cpu" and configs[-1].setup_count == 1
        assert pipeline.caches == [
            {"text": ["A neon café"], "image": None, "height": 2, "width": 4}
        ]
        with pytest.raises(ValueError, match="busy"):
            backend.open("cpu-t2v", parameters, b"")
        response, _ = request("step", count=1, inputs={"pan": 1})
        assert response["code"] == "request" and pipeline.generated == []

        received = bytearray()
        for first, count in ((0, 4), (4, 4), (8, 4), (12, 4), (16, 4), (20, 1)):
            response, pixels = request("step", count=4)
            assert response["type"] == "frames"
            assert (response["first_frame"], response["frame_count"]) == (first, count)
            assert (response["width"], response["height"], response["fps"]) == (
                32,
                16,
                24,
            )
            assert len(pixels) == count * 32 * 16 * 3
            received.extend(pixels)
        assert pipeline.generated == pipeline.finalized == [0, 1]
        # Independently normalize the pipeline's grey sequence to byte pixels.
        shades = [round((0.5 + (index % 8) / 8) * 127.5) for index in range(21)]
        assert received == b"".join(bytes([shade]) * (32 * 16 * 3) for shade in shades)
        response, pixels = request("step", count=4)
        assert response["type"] == "finished" and response["next_frame"] == 21
        assert pixels == b"" and pipeline.generated == [0, 1]
        request("reset")
        response, pixels = request("step", count=4)
        assert response["first_frame"] == 0 and pixels == received[: len(pixels)]
        assert pipeline.generated == [0, 1, 0]
        request("close")
        assert not pipeline.closed

        fail_at = 0
        pipeline._fail_at = 0
        request("open", model="cpu-t2v", parameters=parameters)
        broken = configs[-1].pipeline
        response, pixels = request("step")
        assert (
            response["code"] == "execution" and "generate failed" in response["message"]
        )
        assert pixels == b"" and broken.closed
        response, _ = request("step")
        assert response["code"] == "request" and "Open" in response["message"]
        fail_at = None
        request("open", model="cpu-t2v", parameters=parameters)
        disconnected = configs[-1].pipeline
    finally:
        sender.close()
        worker.join(timeout=10)
        receiver.close()
        assert not worker.is_alive()
    assert not disconnected.closed
    session = backend.open("cpu-t2v", parameters, b"")
    session.close()
    session.close()  # Idempotent close must not release the lease twice.
    backend.close()
    assert disconnected.closed
