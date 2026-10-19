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

"""DOS display service for the general FlashDreams frame stream.

The default paths use this app's source checkout and the adjacent local
IA16/DOSBox-X builds. Server command-line options can override device paths.
"""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
"""Directory containing this app's README and native client sources."""
ROOT = APP_DIR.parents[2]
"""Workspace containing the FlashDreams checkout and local tool builds."""
CLIENT_DIR = APP_DIR / "client/build"
"""Writable native client build and emulator artifact directory."""
DOSBOX_PATH = ROOT / ".local-build/dosbox-x-prefix/bin/dosbox-x"
"""Local DOSBox-X executable with the Unix Ethernet backend."""
DOSBOX_CONFIG = Path(__file__).resolve().parent / "dosbox-ne2000.conf"
"""Packaged emulator configuration, without a forced CPU type or speed."""
