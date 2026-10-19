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

"""Client-selected FlashDreams applications behind the model-service API."""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import threading
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from model_service.protocol import MAX_PAYLOAD
from model_service.service import MAX_FRAMES, ModelServer, _parameters
from model_service.devices import DevicePool, ProcessBackend


class UnsupportedApplication(ValueError):
    """An installed application needs an execution backend we do not provide yet."""


class ResidentModel:
    """Keep one prepared application on a device between isolated jobs.

    The lease covers a whole job because pipelines can own mutable decoder and
    shape state. Shutdown releases weights after an active job returns its lease.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.application = self.key = None
        self.image_directory = None
        self._lease = threading.Lock()
        self._closing = False

    def acquire(self) -> None:
        """Reserve the device for one job without waiting behind another client."""
        if not self._lease.acquire(blocking=False):
            raise ValueError("Model device is busy with another session")
        if self._closing:
            self._lease.release()
            raise ValueError("Model service is shutting down")

    def discard(self) -> None:
        """Unload the cached application while holding the device lease."""
        application = self.application
        image_directory = self.image_directory
        self.application = self.key = None
        self.image_directory = None
        try:
            if application is not None:
                application.close()
        finally:
            if image_directory is not None:
                image_directory.cleanup()
            if application is not None:
                gc.collect()
                if self.device.startswith("cuda"):
                    import torch

                    if torch.cuda.is_initialized():
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()

    def release(self, *, invalidate: bool = False) -> None:
        """Return the device lease, retaining only a healthy prepared model."""
        try:
            if invalidate or self._closing:
                self.discard()
        finally:
            self._lease.release()

    def close(self) -> None:
        """Unload idle weights, or defer unloading until the active job closes."""
        self._closing = True
        if self._lease.acquire(blocking=False):
            self.release()


class ApplicationBackend:
    """Discover supported applications and dispatch the client's selection."""

    def __init__(self, factories: dict, *, device: str = "cuda") -> None:
        self._backends = {}
        self._resident = ResidentModel(device)
        for model, factory in sorted(factories.items()):
            try:
                try:
                    backend = T2VBackend(
                        model, factory, device=device, resident=self._resident
                    )
                except UnsupportedApplication:
                    from model_service.cam2v_backend import Cam2VBackend

                    backend = Cam2VBackend(
                        model, factory, device=device, resident=self._resident
                    )
                self._backends[model] = backend
            except UnsupportedApplication:
                logging.getLogger(__name__).info(
                    "Skipping %s: its application type is not supported yet", model
                )
            except (ImportError, LookupError, TypeError, ValueError) as error:
                logging.getLogger(__name__).warning(
                    "Cannot advertise %s: %s", model, error
                )

    def catalog(self) -> list[dict]:
        return [
            entry for backend in self._backends.values() for entry in backend.catalog()
        ]

    def open(self, model, values, payload, *, status=None) -> T2VSession:
        if not isinstance(model, str) or model not in self._backends:
            raise ValueError("Unknown or unsupported model")
        return self._backends[model].open(model, values, payload, status=status)

    def close(self) -> None:
        """Release the service's resident model when the listener closes."""
        self._resident.close()


class T2VBackend:
    """Advertise one application and load its model when a client opens a session."""

    def __init__(
        self, model: str, factory, *, device: str = "cuda", resident=None
    ) -> None:
        application = factory()
        self._validate_application(application)
        desc = application.session_desc()
        diffusion = getattr(application.pipeline_config, "diffusion_model", None)
        transformer = getattr(diffusion, "transformer", None)
        self.model, self.factory, self.device = model, factory, device
        # Read the adapter's existing CLI hook without initializing its model.
        parser = argparse.ArgumentParser(add_help=False)
        application._configure_argument_parser(parser)
        image_argument = next(
            (a for a in parser._actions if "--image-path" in a.option_strings), None
        )
        self._input_image = (
            "none"
            if image_argument is None
            else "required"
            if image_argument.required
            else "optional"
        )
        self._resident = resident if resident is not None else ResidentModel(device)
        self._schema = {
            "width": {
                "type": "integer",
                "default": desc.video_width,
                "minimum": 16,
                "maximum": 2048,
            },
            "height": {
                "type": "integer",
                "default": desc.video_height,
                "minimum": 16,
                "maximum": 2048,
            },
            "fps": {
                "type": "integer",
                "default": desc.frames_per_second_for_step,
                "minimum": 1,
                "maximum": 120,
            },
            "prompt": {"type": "string", "default": "", "max_length": 2048},
            "seed": {
                "type": "integer",
                "default": getattr(diffusion, "seed", 42),
                "minimum": 0,
                "maximum": 0xFFFFFFFF,
            },
            "compile": {
                "type": "boolean",
                "default": getattr(transformer, "compile_network", False),
            },
            "total_blocks": {
                "type": "integer",
                "default": application.defaults.total_blocks,
                "minimum": 1,
                "maximum": 0x7FFFFFFF,
            },
        }

    def _validate_application(self, application):
        """Require the application contract supported by this adapter."""
        from t2v import T2VApplication

        if not isinstance(application, T2VApplication):
            raise UnsupportedApplication("Expected a T2V application")

    def _prepare_application(self, application, parameters, arguments):
        """Initialize one application with its existing command-line interface."""
        application.init(arguments)

    def _extra_cache_key(self, parameters):
        """Return additional settings that affect prepared application state."""
        return ()

    def _wrap_session(self, model, parameters, session):
        """Expose the session through bounded RGB frame requests."""
        return T2VSession(model, parameters, session, self._resident)

    def catalog(self) -> list[dict]:
        return [
            {
                "id": self.model,
                "name": self.model,
                "stub": False,
                "backend": "flashdreams",
                "device": self.device,
                "input_image": (
                    "none"
                    if self._input_image == "none"
                    else f"{self._input_image}_rgb24"
                ),
                "parameters": self._schema,
                "inputs": {},
            }
        ]

    def open(self, model, values, payload, *, status=None) -> T2VSession:
        if model != self.model:
            raise ValueError("Unknown model")
        parameters = _parameters(values, self._schema)
        if payload and self._input_image == "none":
            raise ValueError(
                "This text-to-video backend does not accept an image payload"
            )
        if not parameters["prompt"].strip():
            raise ValueError("prompt must be non-empty text")
        if parameters["width"] * parameters["height"] * 3 > MAX_PAYLOAD:
            raise ValueError("Frame exceeds the RGB payload limit")
        if not payload and self._input_image == "required":
            raise ValueError("This model requires an initial RGB image")
        if payload and len(payload) != parameters["width"] * parameters["height"] * 3:
            raise ValueError("Initial image must match the requested width and height")
        self._resident.acquire()
        session = None
        try:
            key = (
                model,
                self.device,
                *(
                    parameters[name]
                    for name in ("width", "height", "seed", "compile", "total_blocks")
                ),
                hashlib.sha256(payload).digest(),
                self._extra_cache_key(parameters),
            )
            arguments = [
                "--prompt",
                parameters["prompt"],
                "--device",
                self.device,
                "--total-blocks",
                str(parameters["total_blocks"]),
                "--no-ui",
            ]
            if "compile" in values:
                arguments.append(
                    "--compile" if parameters["compile"] else "--no-compile"
                )
            if "seed" in values:
                arguments.extend(["--seed", str(parameters["seed"])])
            reused = self._resident.key == key
            if not reused:
                self._resident.discard()
            if status is not None:
                status("reusing" if reused else "loading", model=model)
            logging.getLogger(__name__).info(
                "%s %s on %s", "Reusing" if reused else "Loading", model, self.device
            )
            if not reused:
                if payload:
                    from PIL import Image

                    # Keep the lossless upload for the resident application's
                    # normal --image-path API, including later fresh sessions.
                    self._resident.image_directory = TemporaryDirectory(
                        prefix="flashdreams-dos-image-"
                    )
                    image_path = (
                        Path(self._resident.image_directory.name) / "initial.png"
                    )
                    Image.frombytes(
                        "RGB", (parameters["width"], parameters["height"]), payload
                    ).save(image_path)
                    arguments.extend(["--image-path", str(image_path)])
                self._resident.application = self.factory()
                self._prepare_application(
                    self._resident.application, parameters, arguments
                )
                self._resident.key = key
            application = self._resident.application
            desc = replace(
                application.session_desc(),
                video_width=parameters["width"],
                video_height=parameters["height"],
                frames_per_second_for_step=parameters["fps"],
                metadata={
                    **application.session_desc().metadata,
                    "prompt": parameters["prompt"],
                },
            )
            session = application.create_session(desc)
            if status is not None:
                status("preparing")
            session.init()
            # The model's generator survives session caches; a new CLI job
            # must restart its seed rather than continue the preceding job's noise.
            rng = getattr(
                getattr(session.model_loop.state.pipeline, "diffusion_model", None),
                "rng",
                None,
            )
            if rng is not None:
                rng.manual_seed(parameters["seed"])
            return self._wrap_session(model, parameters, session)
        except BaseException:
            try:
                if session is not None:
                    session.close()
            finally:
                self._resident.release(invalidate=True)
            raise

    def close(self) -> None:
        """Release resident weights when the service stops."""
        self._resident.close()


class T2VSession:
    """Buffer one generated block and return its RGB frames on demand."""

    def __init__(self, model, parameters, session, resident):
        self.model, self.parameters = model, parameters
        self._session, self._resident = session, resident
        self._pending = None
        self._offset = self._step = self.frame_index = 0
        self._closed = False
        self._broken = False
        self._inputs = {}
        self._input_schema = {}
        self._block_metadata = {}
        self._one_block_per_reply = False

    def _events(self):
        """Return application input for the next generated block."""
        from flashdreams.runtime_v2.user_input_events import UserInputEvents

        return UserInputEvents([])

    def step(self, count: int, inputs: dict, *, status=None) -> tuple[dict, bytes]:
        if self._closed:
            raise ValueError("Session is closed")
        if type(count) is not int or not 1 <= count <= MAX_FRAMES:
            raise ValueError(f"count must be an integer in 1..{MAX_FRAMES}")
        self._inputs = _parameters(inputs, self._input_schema)
        width, height = self.parameters["width"], self.parameters["height"]
        if count * width * height * 3 > MAX_PAYLOAD:
            raise ValueError("Requested batch exceeds the RGB payload limit")
        first = self.frame_index
        try:
            return self._generate(count, first, width, height, status)
        except ConnectionError:
            # A status reply can fail without damaging the resident model.
            raise
        except Exception as error:
            self._broken = True
            raise RuntimeError(str(error)) from error

    def _generate(self, count, first, width, height, status):
        import torch
        from flashdreams.runtime_v2.video_encoder import result_to_rgb24_tensor

        frames = []
        with torch.inference_mode():
            while len(frames) < count:
                if self._pending is None:
                    loop = self._session.model_loop
                    if loop.is_finished():
                        break
                    logging.getLogger(__name__).info(
                        "Generating block %s for %s", self._step, self.model
                    )
                    if status is not None:
                        status(
                            "generating",
                            block=self._step + 1,
                            total_blocks=self.parameters["total_blocks"],
                        )
                    results = loop.step(self._step, self._events())
                    if len(results) != 1 or results[0].frame_count < 1:
                        raise RuntimeError(
                            "Expected one non-empty video output channel"
                        )
                    self._step += 1
                    # Retain only this decoded block while the client fetches it.
                    self._pending = (
                        result_to_rgb24_tensor(
                            results[0], self._session.session_desc, (width, height)
                        )
                        .cpu()
                        .numpy()
                    )
                    self._offset = 0
                frames.append(self._pending[self._offset].tobytes())
                self._offset += 1
                if self._offset == len(self._pending):
                    self._pending = None
                    if self._one_block_per_reply:
                        break
        self.frame_index += len(frames)
        if not frames:
            return {"type": "finished", "next_frame": self.frame_index}, b""
        return {
            "type": "frames",
            "format": "rgb24",
            "width": width,
            "height": height,
            "first_frame": first,
            "frame_count": len(frames),
            "fps": self.parameters["fps"],
            **self._block_metadata,
        }, b"".join(frames)

    def reset(self) -> None:
        import torch

        with torch.inference_mode():
            self._session.model_loop.reset()
        self._pending = None
        self._offset = self._step = self.frame_index = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending = None
        try:
            self._session.model_loop.close()
            self._session.close()
        except BaseException:
            self._broken = True
            raise
        finally:
            self._session = None
            self._resident.release(invalidate=self._broken)


def _device_backend(device, applications):
    """Construct one device's supported adapters inside its inference worker."""
    from flashdreams.runtime_v2.application_registry import create_application

    if device.startswith("cuda:"):
        import torch

        torch.cuda.set_device(device)
    return ApplicationBackend(
        {slug: partial(create_application, slug) for slug in applications},
        device=device,
    )


def _device_descriptions(requested):
    """Resolve exposed devices against CUDA's visible ordinal numbering."""
    import torch

    names = requested or [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    descriptions = []
    for name in names:
        device = torch.device(name)
        if device.type == "cuda":
            index = device.index if device.index is not None else 0
            properties = torch.cuda.get_device_properties(index)
            descriptions.append(
                {
                    "id": f"cuda:{index}",
                    "name": properties.name,
                    "type": "cuda",
                    "total_memory": properties.total_memory,
                }
            )
        elif device.type == "cpu" and device.index is None:
            descriptions.append(
                {"id": "cpu", "name": "Server CPU", "type": "cpu", "total_memory": None}
            )
        else:
            raise ValueError("Only CPU and visible CUDA devices are supported")
    if not descriptions:
        raise ValueError(
            "No visible CUDA devices; specify --device cpu for a CPU backend"
        )
    if len({d["id"] for d in descriptions}) != len(descriptions):
        raise ValueError("Exposed devices must be unique")
    return descriptions


def main() -> None:
    """Serve installed supported applications and let the client select one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "application",
        nargs="?",
        help="Optional application restriction; omit to advertise installed supported apps.",
    )
    parser.add_argument(
        "--device",
        action="append",
        help="Expose a device; repeat for several. Default: all visible CUDA GPUs.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9390)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0..65535")
    from flashdreams.runtime_v2.application_registry import (
        create_application,
        registered_application_slugs,
    )

    logging.basicConfig(level=logging.INFO)
    try:
        descriptions = _device_descriptions(args.device)
    except (ValueError, RuntimeError, AssertionError) as error:
        parser.error(str(error))
    applications = (
        [args.application] if args.application else registered_application_slugs()
    )
    # Catalog construction reads adapter configuration without preparing models.
    catalog_backend = ApplicationBackend(
        {slug: partial(create_application, slug) for slug in applications},
        device=descriptions[0]["id"],
    )
    catalog = catalog_backend.catalog()
    catalog_backend.close()
    if not catalog:
        parser.error(
            "No supported applications installed; install a T2V or Cam2V integration."
        )
    backend = DevicePool(
        {
            d["id"]: ProcessBackend(
                partial(_device_backend, d["id"], [entry["id"] for entry in catalog]),
                catalog,
            )
            for d in descriptions
        },
        descriptions,
    )
    with ModelServer((args.host, args.port), backend) as server:
        print(
            f"FlashDreams model service at {server.server_address[0]}:{server.server_address[1]} "
            f"({len(backend.catalog())} applications)",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
