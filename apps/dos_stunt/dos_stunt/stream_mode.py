# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register RGB streaming through FlashDreams' client-window plugin interface."""

import argparse

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode


class StreamMode(ClientWindowMode):
    """Exchange frames and input with an external output service."""

    name = "stream"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the output service's TCP endpoint options."""
        parser.add_argument(
            "--stream-host",
            default="127.0.0.1",
            help="External output service address.",
        )
        parser.add_argument(
            "--stream-port",
            type=int,
            default=9386,
            help="External output service TCP port.",
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        """Reject empty hosts and invalid TCP ports."""
        if not parsed_args.stream_host.strip():
            raise ValueError("--stream-host must be nonempty")
        if not 1 <= parsed_args.stream_port <= 65535:
            raise ValueError("--stream-port must be between 1 and 65535")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        """Connect to the selected service when this mode is selected."""
        self.check_arguments(parsed_args)
        from dos_stunt.stream_client_window import StreamClientWindow

        return StreamClientWindow(
            host=parsed_args.stream_host, port=parsed_args.stream_port
        )

    def starting(self, client_window: IClientWindow) -> str:
        """Describe where frames and input are exchanged."""
        return "Streaming RGB frames and input through the external output service."
