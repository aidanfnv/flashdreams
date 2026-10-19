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

#include <dpmi.h>
#include <go32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/farptr.h>
#include "lwip/etharp.h"
#include "lwip/sys.h"
#include "netif/ethernet.h"
#include "packet.h"

extern const unsigned char packet_blob[], packet_blob_end[];
extern const unsigned char packet_head[], packet_tail[], packet_rejected[];
extern const unsigned char packet_lengths[], packet_rx[], packet_tx[];

static int selector = -1, handle = -1;
static unsigned segment, vector, base;
static unsigned long received;
typedef struct {
    unsigned time, size;
    unsigned char bytes[PACKET_BYTES];
} PacketTrace;
static PacketTrace *trace;
static unsigned trace_count;

void packet_trace_enable(void)
{
    trace = calloc(256, sizeof(*trace));
}

static void trace_packet(const unsigned char *data, unsigned size)
{
    if (!trace) return;
    PacketTrace *entry = trace + trace_count++ % 256;
    entry->time = sys_now();
    entry->size = size;
    memcpy(entry->bytes, data, size);
}

void packet_trace_save(void)
{
    if (!trace) return;
    FILE *file = fopen("NET.PCP", "wb");
    if (!file) return;
    const unsigned header[] = {0xa1b2c3d4, 0x00040002, 0, 0, PACKET_BYTES, 1};
    fwrite(header, sizeof(header), 1, file);
    unsigned first = trace_count > 256 ? trace_count - 256 : 0;
    for (unsigned i = first; i < trace_count; ++i) {
        PacketTrace *entry = trace + i % 256;
        unsigned record[] = {entry->time / 1000, (entry->time % 1000) * 1000,
                             entry->size, entry->size};
        fwrite(record, sizeof(record), 1, file);
        fwrite(entry->bytes, entry->size, 1, file);
    }
    fclose(file);
}

#define OFFSET(symbol) ((unsigned)((symbol) - packet_blob))

u32_t sys_now(void)
{
    static unsigned previous, days;
    unsigned ticks = _farpeekl(_dos_ds, 0x46c);
    /* BIOS IRQ0 ticks are monotonic within a day. DJGPP's interpolated uclock()
     * can step backward under DOSBox-X. TCP timers need no PIT interpolation. */
    if (ticks < previous)
        ++days;
    previous = ticks;
    return (u32_t)(((unsigned long long)days * 0x1800b0 + ticks) *
                   65536000ull / 1193182);
}

static int driver_call(__dpmi_regs *regs)
{
    if (__dpmi_int(vector, regs) || (regs->x.flags & 1)) {
        fprintf(stderr, "Packet driver INT %02x error %u.\n", vector, regs->h.dh);
        return -1;
    }
    return 0;
}

static err_t transmit(struct netif *interface, struct pbuf *packet)
{
    unsigned char data[PACKET_BYTES] = {0};
    __dpmi_regs regs = {0};
    unsigned size = packet->tot_len;
    (void)interface;
    if (size > sizeof(data))
        return ERR_BUF;
    pbuf_copy_partial(packet, data, size, 0);
    if (size < 60)
        size = 60;
    trace_packet(data, size);
    dosmemput(data, size, base + OFFSET(packet_tx));
    regs.x.ax = 0x0400;
    regs.x.cx = size;
    regs.x.ds = segment;
    regs.x.si = OFFSET(packet_tx);
    return driver_call(&regs) ? ERR_IF : ERR_OK;
}

int packet_open(struct netif *interface, unsigned interrupt)
{
    __dpmi_raddr address;
    __dpmi_regs regs = {0};
    char signature[8];
    vector = interrupt;
    if (__dpmi_get_real_mode_interrupt_vector(vector, &address))
        return -1;
    dosmemget(address.segment * 16u + address.offset16 + 3, 8, signature);
    if (memcmp(signature, "PKT DRVR", 8)) {
        fprintf(stderr, "No packet driver at INT %02x. Load your NIC driver first.\n", vector);
        return -1;
    }
    unsigned size = OFFSET(packet_blob_end);
    segment = __dpmi_allocate_dos_memory((size + 15) / 16, &selector);
    if ((int)segment < 0) {
        selector = -1;
        fprintf(stderr, "Cannot allocate DOS packet buffers.\n");
        return -1;
    }
    base = segment * 16u;
    dosmemput(packet_blob, size, base);
    regs.x.ax = 0x0201; /* Ethernet class; zero-length filter receives all types. */
    regs.x.bx = 0xffff;
    regs.x.es = segment;
    if (driver_call(&regs))
        goto failed;
    handle = regs.x.ax;
    memset(&regs, 0, sizeof(regs));
    regs.x.ax = 0x0600;
    regs.x.bx = handle;
    regs.x.cx = 6;
    regs.x.es = segment;
    regs.x.di = OFFSET(packet_tx);
    if (driver_call(&regs) || regs.x.cx != 6)
        goto failed;
    dosmemget(base + OFFSET(packet_tx), 6, interface->hwaddr);
    interface->hwaddr_len = 6;
    interface->mtu = 1500;
    interface->name[0] = 'p';
    interface->name[1] = 'd';
    interface->flags = NETIF_FLAG_BROADCAST | NETIF_FLAG_ETHARP;
    interface->output = etharp_output;
    interface->linkoutput = transmit;
    printf("Packet driver INT %02x: %02x:%02x:%02x:%02x:%02x:%02x\n", vector,
           interface->hwaddr[0], interface->hwaddr[1], interface->hwaddr[2],
           interface->hwaddr[3], interface->hwaddr[4], interface->hwaddr[5]);
    return 0;
failed:
    packet_close();
    return -1;
}

void packet_poll(struct netif *interface)
{
    unsigned head = _farpeekw(_dos_ds, base + OFFSET(packet_head));
    /* Bound each poll so timers and keyboard processing still get a turn. */
    for (unsigned i = 0; i < PACKET_SLOTS; ++i) {
        if (head == _farpeekw(_dos_ds, base + OFFSET(packet_tail)))
            break;
        unsigned size = _farpeekw(_dos_ds, base + OFFSET(packet_lengths) + head * 2);
        struct pbuf *packet = pbuf_alloc(PBUF_RAW, size, PBUF_POOL);
        if (!packet)
            break; /* Leave the slot queued while lwIP has no receive memory. */
        unsigned offset = base + OFFSET(packet_rx) + head * PACKET_BYTES;
        if (trace) {
            unsigned char data[PACKET_BYTES];
            dosmemget(offset, size, data);
            trace_packet(data, size);
        }
        for (struct pbuf *part = packet; part; part = part->next) {
            dosmemget(offset, part->len, part->payload);
            offset += part->len;
        }
        head = (head + 1) & (PACKET_SLOTS - 1);
        _farpokew(_dos_ds, base + OFFSET(packet_head), head);
        ++received;
        if (interface->input(packet, interface) != ERR_OK)
            pbuf_free(packet);
    }
}

unsigned packet_dropped(void)
{
    return selector < 0 ? 0 : _farpeekw(_dos_ds, base + OFFSET(packet_rejected));
}

unsigned long packet_received(void)
{
    return received;
}

void packet_close(void)
{
    free(trace);
    trace = NULL;
    if (handle >= 0) {
        __dpmi_regs regs = {0};
        regs.x.ax = 0x0300;
        regs.x.bx = handle;
        if (driver_call(&regs))
            return; /* Keep callback memory valid if the driver refuses release. */
        handle = -1;
    }
    if (selector >= 0) {
        __dpmi_free_dos_memory(selector);
        selector = -1;
    }
}
