#!/usr/bin/env python3
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

"""Open one Ethernet socket with sudo, then run the bridge as the original user."""

from __future__ import annotations

import argparse
import fcntl
import os
import pwd
from pathlib import Path

from dos_stunt.ethernet_transport import PACKET_FD_ENV, open_packet_socket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a bridge command after --")
    if os.geteuid() != 0 or "SUDO_UID" not in os.environ:
        parser.error("run this launcher with sudo from your normal user account")
    user = pwd.getpwuid(int(os.environ["SUDO_UID"]))
    if user.pw_uid == 0:
        parser.error("the bridge must run as a non-root user")
    if (Path("/sys/class/net") / args.interface / "master").exists():
        parser.error("use a dedicated Ethernet interface outside a bridge or bond")
    sock = open_packet_socket(args.interface)
    fd = fcntl.fcntl(sock.fileno(), fcntl.F_DUPFD, 512)
    sock.close()
    os.set_inheritable(fd, True)
    os.initgroups(user.pw_name, user.pw_gid)
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)
    os.environ[PACKET_FD_ENV] = str(fd)
    os.environ.update(HOME=user.pw_dir, USER=user.pw_name, LOGNAME=user.pw_name)
    print(
        f"Ethernet socket bound to {args.interface}; running bridge as {user.pw_name}",
        flush=True,
    )
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
