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

#ifndef DOS_LWIP_OPTIONS_H
#define DOS_LWIP_OPTIONS_H
#define NO_SYS 1
#define SYS_LIGHTWEIGHT_PROT 0
#define LWIP_SOCKET 0
#define LWIP_NETCONN 0
#define LWIP_IPV4 1
#define LWIP_IPV6 0
#define LWIP_RAW 0
#define LWIP_DNS 0
#define LWIP_DHCP 1
#define LWIP_ARP 1
#define LWIP_ICMP 1
#define LWIP_IGMP 0
#define LWIP_AUTOIP 0
#define LWIP_NETIF_LOOPBACK 0
#define MEM_ALIGNMENT 4
#define MEM_SIZE (128 * 1024)
#define TCP_MSS 1460
#define TCP_WND 32768
#define TCP_SND_BUF 8192
#define TCP_SND_QUEUELEN 32
#define MEMP_NUM_TCP_SEG 32
#define PBUF_POOL_SIZE 64
#define PBUF_POOL_BUFSIZE 1600
#endif
