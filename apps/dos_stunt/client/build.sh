#!/usr/bin/env bash
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

set -euo pipefail
cd "$(dirname "$0")"
compiler="${IA16_GCC:-../../../../build-ia16/prefix/bin/ia16-elf-gcc}"
if [[ ! -x "$compiler" ]]; then
    compiler=ia16-elf-gcc
fi
mkdir -p build
"$compiler" -mcmodel=small -Os -Wall -Wextra -c client.c -o build/client.o
"$compiler" -mcmodel=small -c packet.asm.S -o build/packet.o
"$compiler" -mcmodel=small -Os build/client.o build/packet.o -li86 -o build/DOSRX.EXE
file build/DOSRX.EXE
