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

"""Device reservations, worker isolation, and protocol selection on CPU."""

import os
import socket
import threading
from functools import partial
from pathlib import Path

import pytest
import torch
from model_service.devices import DevicePool, ProcessBackend
from model_service.flashdreams_backend import T2VBackend, _device_descriptions
from model_service.protocol import PROTOCOL, VERSION, receive_message, send_message
from model_service.stub_service import StubServer
from t2v import T2VApplication, T2VApplicationDefaults
from t2v.testing import FakeT2VPipeline, FakeT2VPipelineConfig

pytestmark = pytest.mark.ci_cpu


def fake_backend(dtype=None):
    """Construct a fake pipeline in the worker's own PyTorch environment."""
    if dtype is not None:
        torch.set_default_dtype(getattr(torch, dtype))

    def application():
        return T2VApplication(
            defaults=T2VApplicationDefaults(
                pipeline_config=FakeT2VPipelineConfig(
                    FakeT2VPipeline(width=32, height=16)
                ),
                pixel_width=32,
                pixel_height=16,
                total_blocks=1,
                device="cpu",
            )
        )

    class Backend(T2VBackend):
        def open(self, model, values, payload, *, status=None):
            session = super().open(model, values, payload, status=status)
            session._session.model_loop.state.pipeline._fail_at = (
                0 if values.get("prompt") == "Fail" else None
            )
            if status:
                status("worker", pid=os.getpid(), dtype=str(torch.get_default_dtype()))
            return session

    return Backend("cpu-video", application, device="cpu")


def test_two_workers_residency_failure_and_process_isolation(monkeypatch):
    """Keep jobs, default dtype, failures, and resident weights separate per device."""
    # Spawned workers import the test factory without pytest's import hook.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    catalog = fake_backend().catalog()
    backends = {
        name: ProcessBackend(partial(fake_backend, dtype), catalog)
        for name, dtype in (("sim:0", "float64"), ("sim:1", "float32"))
    }
    pool = DevicePool(
        backends,
        [
            {"id": name, "name": name, "type": "stub", "total_memory": None}
            for name in backends
        ],
    )
    phases, jobs = [], []
    original_dtype = torch.get_default_dtype()

    def status(phase, **fields):
        phases.append((phase, fields))

    try:
        assert all(backend.process is None for backend in backends.values())
        assert pool.devices()["default_device"] == "sim:0"
        assert pool.catalog()[0]["devices"] == ["sim:0", "sim:1"]
        first = pool.open("cpu-video", {"prompt": "First"}, b"", status=status)
        jobs.append(first)
        second = pool.open(
            "cpu-video",
            {"prompt": "Second", "width": 64},
            b"",
            device="sim:1",
            status=status,
        )
        jobs.append(second)
        assert first.device == "sim:0" and second.device == "sim:1"
        assert backends["sim:0"].process.pid != backends["sim:1"].process.pid
        assert {fields["dtype"] for phase, fields in phases if phase == "worker"} == {
            "torch.float32",
            "torch.float64",
        }
        assert torch.get_default_dtype() == original_dtype
        assert all(d["busy"] for d in pool.devices()["devices"])
        with pytest.raises(ValueError, match="busy"):
            pool.open("cpu-video", {"prompt": "Another"}, b"", device="sim:0")
        with pytest.raises(ValueError, match="Unknown"):
            pool.open("cpu-video", {}, b"", device="missing")
        metadata, original = first.step(1, {}, status=status)
        assert metadata["width"] == 32 and len(original) == 32 * 16 * 3
        metadata, _ = second.step(1, {})
        assert metadata["width"] == 64 and metadata["first_frame"] == 0
        first.reset()
        assert first.step(1, {})[1] == original
        first.close()
        phases.clear()
        reused = pool.open("cpu-video", {"prompt": "Fresh"}, b"", status=status)
        jobs.append(reused)
        assert phases[0][0] == "reusing"
        reused.close()
        failed = pool.open("cpu-video", {"prompt": "Fail"}, b"", device="sim:0")
        jobs.append(failed)
        with pytest.raises(RuntimeError, match="generate failed"):
            failed.step(1, {})
        failed.close()
        assert second.step(1, {})[0]["first_frame"] == 1
        recovered = pool.open("cpu-video", {"prompt": "Recovered"}, b"", device="sim:0")
        jobs.append(recovered)
        assert recovered.step(1, {})[0]["first_frame"] == 0
        recovered.close()
        interrupted = pool.open(
            "cpu-video", {"prompt": "Interrupted"}, b"", device="sim:0"
        )
        jobs.append(interrupted)

        def disconnected(phase, **fields):
            raise BrokenPipeError("client left")

        with pytest.raises(BrokenPipeError, match="client left"):
            interrupted.step(1, {}, status=disconnected)
        interrupted.close()
        phases.clear()
        healthy = pool.open("cpu-video", {"prompt": "Healthy"}, b"", status=status)
        jobs.append(healthy)
        assert phases[0][0] == "reusing"
        process = backends["sim:0"].process
        process.terminate()
        process.join(timeout=5)
        with pytest.raises(RuntimeError, match="worker disconnected"):
            healthy.step(1, {})
        healthy.close()
        assert second.step(1, {})[0]["first_frame"] == 2
        restarted = pool.open("cpu-video", {"prompt": "Restarted"}, b"")
        jobs.append(restarted)
        assert backends["sim:0"].process.pid != process.pid
        assert restarted.step(1, {})[0]["first_frame"] == 0
    finally:
        for job in jobs:
            job.close()
        pool.close()
    assert all(b.process is None for b in backends.values())


def test_protocol_discovery_selection_busy_and_disconnect():
    """Select independent jobs over TCP and recover a reservation after disconnect."""
    with StubServer(("127.0.0.1", 0), devices=["stub:0", "stub:1"]) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        clients = []

        def connect():
            client = socket.create_connection(server.server_address, timeout=5)
            clients.append(client)
            send_message(
                client,
                {"id": 1, "op": "hello", "protocol": PROTOCOL, "version": VERSION},
            )
            assert receive_message(client)[0]["device_selection"]
            return client

        def request(client, op, **fields):
            send_message(client, {"id": 2, "op": op, **fields})
            return receive_message(client)

        try:
            a, b = connect(), connect()
            devices, _ = request(a, "devices")
            assert devices["default_device"] == "stub:0"
            assert [d["id"] for d in devices["devices"]] == ["stub:0", "stub:1"]
            for invalid in ("missing", 1, "", None):
                response, _ = request(a, "open", model="stub-gradient", device=invalid)
                assert response["code"] == "request"
            response, _ = request(a, "open", model="stub-gradient")
            assert response["device"] == "stub:0"
            response, _ = request(b, "open", model="stub-gradient", device="stub:0")
            assert response["code"] == "request" and "busy" in response["message"]
            response, _ = request(b, "open", model="stub-image", device="stub:1")
            assert response["code"] == "request"
            response, _ = request(
                b,
                "open",
                model="stub-gradient",
                device="stub:1",
                parameters={"width": 4, "height": 2},
            )
            assert response["device"] == "stub:1"
            assert request(b, "step", count=1)[0]["first_frame"] == 0
            request(a, "close")
            devices, _ = request(a, "devices")
            assert [d["busy"] for d in devices["devices"]] == [False, True]
            b.close()
            # A new open is ordered after the handler releases the disconnected job.
            assert server.backend._leases["stub:1"].acquire(timeout=5)
            server.backend._leases["stub:1"].release()
            response, _ = request(a, "open", model="stub-gradient", device="stub:1")
            assert response["device"] == "stub:1"
        finally:
            for client in clients:
                client.close()
            server.shutdown()
            thread.join(timeout=5)


def test_visible_gpu_ordinals_and_validation(monkeypatch):
    """Honor CUDA-visible ordinals and reject duplicate, absent, or invalid devices."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def properties(index):
        if index not in (0, 1):
            raise AssertionError("Invalid device id")
        return SimpleNamespace(name=f"GPU {index}", total_memory=1024**3 * (index + 1))

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert [d["id"] for d in _device_descriptions(None)] == ["cuda:0", "cuda:1"]
    assert _device_descriptions(["cuda:1"])[0]["name"] == "GPU 1"
    assert _device_descriptions(["cpu"])[0]["total_memory"] is None
    with pytest.raises(ValueError, match="unique"):
        _device_descriptions(["cuda", "cuda:0"])
    with pytest.raises(AssertionError):
        _device_descriptions(["cuda:2"])
    with pytest.raises(ValueError, match="Only"):
        _device_descriptions(["meta"])
