# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the installed stream plugin through the normal runtime parser."""

from importlib.metadata import entry_points

import pytest
from dos_stunt import stream_client_window
from dos_stunt.stream_mode import StreamMode

from flashdreams.runtime_v2.cli import _parser
from flashdreams.runtime_v2.client_window_factory import (
    client_window_mode,
    create_client_window,
)

pytestmark = pytest.mark.ci_cpu


def test_installed_stream_mode_owns_its_endpoint(monkeypatch) -> None:
    """Discover the package entry point and defer connections until creation."""
    registered = [
        point
        for point in entry_points(group="flashdreams.client_windows_v2")
        if point.name == "stream"
    ]
    assert len(registered) == 1
    assert registered[0].value == "dos_stunt.stream_mode:StreamMode"
    endpoints = []
    monkeypatch.setattr(
        stream_client_window,
        "StreamClientWindow",
        lambda **kwargs: endpoints.append(kwargs),
    )
    parsed = _parser().parse_args(
        [
            "test-application",
            "--mode",
            "stream",
            "--stream-host",
            "localhost",
            "--stream-port",
            "9321",
        ]
    )
    mode = client_window_mode(parsed.mode, modes=parsed._client_window_modes)
    assert isinstance(mode, StreamMode)
    assert not endpoints
    create_client_window(parsed)
    assert endpoints == [{"host": "localhost", "port": 9321}]


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--stream-port", "0", "between"),
        ("--stream-port", "-1", "between"),
        ("--stream-port", "65536", "between"),
        ("--stream-host", " ", "nonempty"),
    ],
)
def test_stream_mode_rejects_invalid_endpoints(option, value, message) -> None:
    """Reject invalid endpoints before opening a socket."""
    parsed = _parser().parse_args(
        ["test-application", "--mode", "stream", option, value]
    )
    with pytest.raises(ValueError, match=message):
        create_client_window(parsed)
