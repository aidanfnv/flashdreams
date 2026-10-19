// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DOS_CLIENT_PACKET_H
#define DOS_CLIENT_PACKET_H

#define PACKET_SLOTS 32
#define PACKET_BYTES 1600

#ifndef __ASSEMBLER__
#include "lwip/netif.h"

int packet_open(struct netif *interface, unsigned interrupt);
void packet_poll(struct netif *interface);
void packet_close(void);
unsigned packet_dropped(void);
unsigned long packet_received(void);
void packet_trace_enable(void);
void packet_trace_save(void);
#endif
#endif
