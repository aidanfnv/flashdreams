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

"""Device routing and isolated inference workers using the standard library."""

import multiprocessing
import threading


class DevicePool:
    """Reserve one job per device while retaining independent backend weights."""

    def __init__(self, backends: dict, descriptions: list[dict]):
        if not backends or [d["id"] for d in descriptions] != list(backends):
            raise ValueError("Device descriptions must match the configured backends")
        self.backends = backends
        self.descriptions = descriptions
        self.default_device = next(iter(backends))
        self._leases = {device: threading.Lock() for device in backends}
        self._lock = threading.Lock()
        self._closed = False

    def devices(self) -> dict:
        """Describe configured devices and this service's current job reservations."""
        return {
            "default_device": self.default_device,
            "devices": [
                {**d, "busy": self._leases[d["id"]].locked()} for d in self.descriptions
            ],
        }

    def catalog(self) -> list[dict]:
        return [
            {**entry, "device": self.default_device, "devices": list(self.backends)}
            for entry in self.backends[self.default_device].catalog()
        ]

    def open(self, model, values, payload, *, device=None, status=None):
        if device == "cuda":
            device = "cuda:0"
        if device is None:
            device = self.default_device
        if not isinstance(device, str) or device not in self.backends:
            raise ValueError("Unknown or unavailable device")
        lease = self._leases[device]
        with self._lock:
            if self._closed:
                raise ValueError("Model service is shutting down")
            if not lease.acquire(blocking=False):
                raise ValueError(f"Model device {device} is busy with another session")
        try:
            session = self.backends[device].open(model, values, payload, status=status)
            return _DeviceSession(session, device, lease)
        except BaseException:
            lease.release()
            raise

    def close(self) -> None:
        """Stop every owned backend, including its current job and resident weights."""
        with self._lock:
            self._closed = True
        failure = None
        for backend in self.backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as error:
                    failure = failure or error
        if failure is not None:
            raise failure


class _DeviceSession:
    """Return a device reservation when its job closes or disconnects."""

    def __init__(self, session, device, lease):
        self.session = session
        self.model, self.parameters = session.model, session.parameters
        self.device, self.lease = device, lease
        self._closed = False

    def step(self, count, inputs, *, status=None):
        return self.session.step(count, inputs, status=status)

    def reset(self):
        return self.session.reset()

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self.session.close()
            finally:
                self.lease.release()


def _worker(connection, factory):
    """Run a device backend in a fresh process with its own framework globals."""
    backend = session = None

    def status(phase, **fields):
        connection.send(("status", phase, fields))
        if not connection.recv():
            raise ConnectionError("The client disconnected during preparation")

    try:
        backend = factory()
        connection.send(("ready",))
        while True:
            operation, fields, payload = connection.recv()
            try:
                result = b""
                if operation == "shutdown":
                    break
                if operation == "open":
                    if session is not None:
                        raise ValueError("Close the current job before opening another")
                    session = backend.open(
                        fields["model"], fields["parameters"], payload, status=status
                    )
                    response = {
                        "model": session.model,
                        "parameters": session.parameters,
                    }
                elif operation == "step":
                    response, result = session.step(
                        fields["count"], fields["inputs"], status=status
                    )
                elif operation == "reset":
                    session.reset()
                    response = {}
                elif operation == "close":
                    if session is not None:
                        session.close()
                        session = None
                    response = {}
                else:
                    raise ValueError("Unknown worker operation")
                connection.send(("result", response, result))
            except ValueError as error:
                connection.send(("error", "request", str(error)))
            except ConnectionError as error:
                connection.send(("error", "connection", str(error)))
            except Exception as error:
                if session is not None:
                    session.close()
                    session = None
                connection.send(("error", "execution", str(error)))
    except (EOFError, ConnectionError):
        pass
    except Exception as error:
        connection.send(("error", "execution", str(error)))
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            try:
                if backend is not None:
                    close = getattr(backend, "close", None)
                    if close is not None:
                        close()
            finally:
                connection.close()


class ProcessBackend:
    """Start one persistent worker lazily, serializing its device's requests."""

    def __init__(self, factory, catalog):
        self.factory, self._catalog = factory, catalog
        self.process = self.connection = None
        self._lock = threading.RLock()
        self._closed = False

    def catalog(self):
        return self._catalog

    def _discard_worker(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.process is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()
            self.process = None

    def _io(self, operation, *args):
        try:
            return getattr(self.connection, operation)(*args)
        except (OSError, EOFError) as error:
            self._discard_worker()
            raise RuntimeError("Model worker disconnected") from error

    def _request(self, operation, payload=b"", *, status=None, **fields):
        with self._lock:
            if self._closed:
                raise RuntimeError("Model worker is closed")
            if self.process is None:
                context = multiprocessing.get_context("spawn")
                self.connection, child = context.Pipe()
                self.process = context.Process(
                    target=_worker, args=(child, self.factory)
                )
                # Compiler helpers may need to start subprocesses of their own.
                try:
                    self.process.start()
                except BaseException:
                    self.connection.close()
                    self.process = self.connection = None
                    raise
                finally:
                    child.close()
                ready = self._io("recv")
                if ready != ("ready",):
                    self._discard_worker()
                    raise RuntimeError(f"Model worker failed to start: {ready[-1]}")
            self._io("send", (operation, fields, payload))
            failure = None
            while True:
                reply = self._io("recv")
                if reply[0] == "status":
                    try:
                        if status is not None:
                            status(reply[1], **reply[2])
                    except Exception as error:
                        failure = error
                    self._io("send", failure is None)
                    continue
                if failure is not None:
                    raise failure
                if reply[0] == "error":
                    kind = {"request": ValueError, "connection": ConnectionError}.get(
                        reply[1], RuntimeError
                    )
                    raise kind(reply[2])
                return reply[1], reply[2]

    def open(self, model, values, payload, *, status=None):
        with self._lock:
            if self.process is not None and not self.process.is_alive():
                self._discard_worker()
        response, _ = self._request(
            "open", payload, status=status, model=model, parameters=values
        )
        return _ProcessSession(self, response["model"], response["parameters"])

    def close(self):
        with self._lock:
            self._closed = True
            if self.connection is not None:
                try:
                    self.connection.send(("shutdown", {}, b""))
                except (OSError, EOFError):
                    pass
                self.connection.close()
                self.connection = None
            if self.process is not None:
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join()
                self.process = None


class _ProcessSession:
    """Keep rollout operations on the device worker that owns its model cache."""

    def __init__(self, backend, model, parameters):
        self.backend, self.model, self.parameters = backend, model, parameters
        self.process = backend.process
        self._closed = False

    def step(self, count, inputs, *, status=None):
        if self.backend.process is not self.process:
            raise RuntimeError("Model worker lost its session")
        return self.backend._request("step", count=count, inputs=inputs, status=status)

    def reset(self):
        if self.backend.process is not self.process:
            raise RuntimeError("Model worker lost its session")
        self.backend._request("reset")

    def close(self):
        if not self._closed:
            self._closed = True
            if not self.backend._closed and self.backend.process is self.process:
                self.backend._request("close")
