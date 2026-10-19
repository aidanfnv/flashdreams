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

"""Mount a persistent DOS drive in DOSBox-X and run a supplied DOS command."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _response_argument(value: str) -> str:
    """Quote one DJGPP response-file argument, preserving UTF-8 and backslashes."""
    trailing = len(value) - len(value.rstrip("\\"))
    prefix = value[: len(value) - trailing]
    # Keep trailing backslashes outside the quotes to preserve the closing quote.
    return '"' + prefix.replace('"', '\\"').replace("'", "\\'") + '"' + "\\" * trailing


def launch(
    storage: Path,
    command: list[str],
    *,
    dosbox_path: Path,
    client_path: Path,
    dosbox_config: Path | None = None,
    headless: bool = False,
    timeout: int | None = None,
) -> bool:
    """Mount ``storage`` as C: and run ``command`` without interpreting its settings.

    Headless runs save command stdout in ``CLIENT.TXT``, report DOS ERRORLEVEL,
    and exit the emulator. Visible runs return to the DOS prompt. The optional
    timeout is for automated checks; normal launches have no timeout.
    """
    app = Path(__file__).resolve().parents[1]
    storage = storage.resolve()
    if not dosbox_path.is_file():
        raise ValueError(f"DOSBox-X does not exist: {dosbox_path}")
    if not client_path.is_file():
        raise ValueError("Build first: bash apps/dos_client/client/build.sh")
    if any("\0" in value for value in command):
        raise ValueError("DOS command arguments must contain no NUL")
    storage.mkdir(parents=True, exist_ok=True)
    target = storage / "FLSHDRMS.EXE"
    if client_path.resolve() != target.resolve():
        shutil.copy2(client_path, target)
    notices = client_path.with_name("THIRDP.TXT")
    if notices.is_file() and notices.resolve() != (storage / notices.name).resolve():
        shutil.copy2(notices, storage / notices.name)
    # FLSHDRMS uses DJGPP response files to avoid DOS's 126-byte command tail.
    if command and command[0].upper() in ("FLSHDRMS", "FLSHDRMS.EXE"):
        (storage / "COMMAND.RSP").write_text(
            " ".join(_response_argument(value) for value in command[1:]),
            encoding="utf-8",
        )
        guest = command[0] + " @COMMAND.RSP"
    else:
        guest = " ".join(
            '"' + value + '"' if " " in value else value for value in command
        )
    invocation = [
        str(dosbox_path.resolve()),
        "-fastlaunch",
        "-nopromptfolder",
        "-conf",
        str(app / "client/dosbox-vbe.conf"),
        "-conf",
        str(app / "client/dosbox-network.conf"),
    ]
    if dosbox_config:
        invocation.extend(["-conf", str(dosbox_config.resolve())])
    if headless:
        invocation.append("-silent")
        (storage / "FAILED.TXT").unlink(missing_ok=True)
    invocation.extend(
        [
            "-c",
            "mount c .",
            "-c",
            "c:",
            "-c",
            r"z:\system\ne2000.com 0x60 10 0x300 > DRIVER.TXT",
        ]
    )
    if guest:
        invocation.extend(["-c", guest + (" > CLIENT.TXT" if headless else "")])
    if headless:
        invocation.extend(
            [
                "-c",
                "if errorlevel 1 echo DOS command failed. > FAILED.TXT",
                "-c",
                "exit",
            ]
        )
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = (
        "dummy" if headless else ("x11" if env.get("DISPLAY") else "wayland")
    )
    env["SDL_AUDIODRIVER"] = "dummy"
    with tempfile.TemporaryDirectory(prefix="dosbox-config-") as config:
        env["XDG_CONFIG_HOME"] = config
        with (storage / "DOSBOX.LOG").open("w") as log:
            subprocess.run(
                invocation,
                cwd=storage,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
    return not headless or (storage / "FAILED.TXT").stat().st_size == 0


def main() -> None:
    """Open DOSBox-X with a persistent drive and optional DOS command after ``--``."""
    app = Path(__file__).resolve().parents[1]
    workspace = app.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=app / "client/build/dos-drive",
        help="Persistent host directory mounted as C:; input and output files live here.",
    )
    parser.add_argument(
        "--dosbox-config", type=Path, help="Additional emulator settings."
    )
    parser.add_argument(
        "--dosbox-path",
        type=Path,
        default=workspace / ".local-build/dosbox-x-prefix/bin/dosbox-x",
    )
    parser.add_argument(
        "--client-path",
        type=Path,
        default=app / "client/build/FLSHDRMS.EXE",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window, print command stdout, and exit after the DOS command.",
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="DOS command after --."
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.headless and not command:
        parser.error("--headless requires a DOS command after --")
    try:
        print(f"DOS C: = {args.storage_dir.resolve()}", flush=True)
        success = launch(
            args.storage_dir,
            command,
            dosbox_path=args.dosbox_path,
            client_path=args.client_path,
            dosbox_config=args.dosbox_config,
            headless=args.headless,
        )
        if args.headless:
            print((args.storage_dir / "CLIENT.TXT").read_text().strip())
        if not success:
            raise ValueError(
                f"DOS command failed; logs are in {args.storage_dir.resolve()}"
            )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Emulator failed: {error}\n")


if __name__ == "__main__":
    main()
