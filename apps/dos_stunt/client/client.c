// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <conio.h>
#include <dos.h>
#include <i86.h>
#include <libi86/malloc.h>
#include <libi86/string.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "keyboard.h"
#include "packet.h"

#define FRAME_SIZE 64000u
#define PALETTE_SIZE 768u
#define BUFFER_SIZE (FRAME_SIZE + PALETTE_SIZE)
#define CHUNK_SIZE 1400u
#define CHUNK_COUNT 46u
#define PALETTE_CHUNK_COUNT 47u
#define ETHERTYPE 0x88b5u
/* The IRQ producer publishes filled slots; main owns the head until consumption.
 * RX_SLOTS is a power of two, shared with packet.asm.S through packet.h. */
const unsigned char packet_type[2] = {0x88, 0xb5};
volatile unsigned char rx_head, rx_tail, rx_count, rx_peak;
volatile unsigned short rx_length[RX_SLOTS];
volatile unsigned short rx_rejected;
unsigned char rx_packet[RX_SLOTS][RX_PACKET_SIZE];

extern int packet_open(void);
extern int packet_get_address(unsigned handle, void *address);
extern void packet_release(unsigned handle);
extern int packet_send(const void *frame, unsigned length);

static unsigned char tx_packet[60];
static unsigned char file_buffer[1024];
static unsigned char guest_mac[6];
static unsigned char host_mac[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static volatile struct keyboard_state keyboard;

/* IRQ 1 gives both make and break scan codes. BIOS getch() cannot report
 * releases, which the driving controls need. Keep the handler independent of
 * DOS and the packet driver; only the main loop sends Ethernet packets. */
static void __far __interrupt keyboard_interrupt(void)
{
    unsigned char code = inp(0x60);
    unsigned char control;

    keyboard_record(&keyboard, code);
    control = inp(0x61);
    outp(0x61, control | 0x80);
    outp(0x61, control);
    outp(0x20, 0x20);
}

static unsigned word_at(const unsigned char *p)
{
    return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static void consume_packet(void)
{
    _disable();
    rx_head = (rx_head + 1) & (RX_SLOTS - 1);
    --rx_count;
    _enable();
}

static void set_video_mode(unsigned mode)
{
    union REGS in = {0}, out;
    in.x.ax = mode;
    int86(0x10, &in, &out);
}

static void set_rgb332_palette(void)
{
    unsigned index;
    outp(0x3c8, 0);
    for (index = 0; index < 256; ++index) {
        outp(0x3c9, ((index >> 5) * 63) / 7);
        outp(0x3c9, (((index >> 2) & 7) * 63) / 7);
        outp(0x3c9, ((index & 3) * 63) / 3);
    }
}

static void set_frame_palette(const unsigned char __far *palette)
{
    unsigned index;
    outp(0x3c8, 0);
    for (index = 0; index < PALETTE_SIZE; ++index)
        outp(0x3c9, palette[index]);
}

static int save_palette(const char *name)
{
    FILE *out;
    unsigned index;
    outp(0x3c7, 0);
    for (index = 0; index < PALETTE_SIZE; ++index)
        file_buffer[index] = inp(0x3c9) & 63;
    out = fopen(name, "wb");
    if (out == NULL)
        return -1;
    if (fwrite(file_buffer, 1, PALETTE_SIZE, out) != PALETTE_SIZE) {
        fclose(out);
        return -1;
    }
    return fclose(out);
}

static void send_ack(unsigned sequence, unsigned chunk)
{
    memset(tx_packet, 0, sizeof(tx_packet));
    memcpy(tx_packet, host_mac, 6);
    memcpy(tx_packet + 6, guest_mac, 6);
    tx_packet[12] = ETHERTYPE >> 8;
    tx_packet[13] = ETHERTYPE & 255;
    tx_packet[14] = 'F';
    tx_packet[15] = 'D';
    tx_packet[16] = 1;
    tx_packet[17] = 2; /* acknowledgement */
    tx_packet[18] = sequence & 255;
    tx_packet[19] = sequence >> 8;
    tx_packet[20] = chunk;
    packet_send(tx_packet, sizeof(tx_packet));
}

static void send_keys(unsigned sequence, const unsigned char *state)
{
    memset(tx_packet, 0, sizeof(tx_packet));
    memcpy(tx_packet, host_mac, 6);
    memcpy(tx_packet + 6, guest_mac, 6);
    tx_packet[12] = ETHERTYPE >> 8;
    tx_packet[13] = ETHERTYPE & 255;
    tx_packet[14] = 'F';
    tx_packet[15] = 'D';
    tx_packet[16] = 1;
    tx_packet[17] = 6; /* full scan-set-1 keyboard snapshot */
    tx_packet[18] = sequence & 255;
    tx_packet[19] = sequence >> 8;
    memcpy(tx_packet + 20, state, KEY_BYTES);
    tx_packet[52] = RX_SLOTS;
    tx_packet[53] = 1; /* Supports progressive RGB332 image kind 7. */
    packet_send(tx_packet, sizeof(tx_packet));
}

static int save_screen(const char *name, const unsigned char __far *vga)
{
    FILE *out = fopen(name, "wb");
    unsigned offset;
    if (out == NULL)
        return -1;
    for (offset = 0; offset < FRAME_SIZE;) {
        unsigned count = FRAME_SIZE - offset;
        const unsigned char __far *source = vga + offset;
        if (count > sizeof(file_buffer))
            count = sizeof(file_buffer);
        _fmemcpy(_CV_FP(file_buffer), source, count);
        if (fwrite(file_buffer, 1, count, out) != count) {
            fclose(out);
            return -1;
        }
        offset += count;
    }
    return fclose(out);
}

int main(int argc, char **argv)
{
    int hold = argc > 1 && strcmp(argv[1], "/hold") == 0;
    int continuous = argc > 1 && strcmp(argv[1], "/continuous") == 0;
    unsigned char __far *frame = _fmalloc(BUFFER_SIZE);
    unsigned char __far *vga = MK_FP(0xa000, 0);
    volatile const uint32_t __far *bios_ticks = MK_FP(0x40, 0x6c);
    uint32_t started;
    unsigned sequence = 0xffff;
    unsigned received_chunks = 0;
    unsigned frame_chunk_count = 0;
    unsigned frame_kind = 0;
    unsigned char chunk_mask[6] = {0};
    unsigned adaptive_palette = 0;
    uint32_t frames = 0;
    unsigned key_sequence = 0;
    uint32_t key_packets = 0;
    uint32_t key_changes = 0;
    uint32_t key_snapshots = 0;
    unsigned char last_keys[KEY_BYTES] = {0};
    uint32_t last_key_tick = 0;
    uint32_t packets = 0;
    uint32_t dropped = 0;
    uint32_t accepted = 0;
    unsigned done = 0;
    unsigned host_known = 0;
    int handle;
    FILE *stats;
    __libi86_isr_t old_keyboard_interrupt;

    if (frame == NULL) {
        puts("Cannot allocate frame buffer");
        _ffree(frame);
        return 1;
    }
    handle = packet_open();
    if (handle < 0) {
        puts("Packet driver missing or access_type failed");
        _ffree(frame);
        return 2;
    }
    if (packet_get_address(handle, guest_mac) < 0) {
        puts("Packet driver could not return a six-byte Ethernet address");
        packet_release(handle);
        _ffree(frame);
        return 2;
    }
    set_video_mode(0x13);
    set_rgb332_palette();
    keyboard.dirty = 1; /* Send the initial ready snapshot immediately. */
    old_keyboard_interrupt = _dos_getvect(9);
    _dos_setvect(9, keyboard_interrupt);
    started = *bios_ticks;
    last_key_tick = started;

    while (!done && (continuous ||
           (uint32_t)(*bios_ticks - started) < 18u * 60u)) {
        unsigned len, seq, chunk, count, offset, payload_len, expected_len;
        unsigned char current_keys[KEY_BYTES];
        uint32_t now = *bios_ticks;
        const unsigned char *p;
        if (keyboard.dirty || now != last_key_tick) {
            int changed;
            _disable();
            keyboard_snapshot(&keyboard, current_keys);
            _enable();
            ++key_snapshots;
            changed = memcmp(current_keys, last_keys, KEY_BYTES);
            if (changed || now != last_key_tick || key_packets == 0) {
                send_keys(key_sequence++, current_keys);
                ++key_packets;
                if (changed)
                    ++key_changes;
                memcpy(last_keys, current_keys, KEY_BYTES);
                last_key_tick = now;
            }
        }
        if (!rx_count)
            continue;
        len = rx_length[rx_head];
        p = rx_packet[rx_head];
        ++packets;
        if (len < 21 || p[12] != 0x88 || p[13] != 0xb5 ||
            p[14] != 'F' || p[15] != 'D' || p[16] != 1)
            goto consumed;
        if (memcmp(p, guest_mac, 6) != 0 || (p[6] & 1) ||
            (host_known && memcmp(p + 6, host_mac, 6) != 0))
            goto consumed;
        if (p[17] == 3) {
            seq = word_at(p + 18);
            chunk = 255;
            memcpy(host_mac, p + 6, 6);
            done = 1;
            goto acknowledged;
        }
        if ((p[17] != 1 && p[17] != 7) || len < 26)
            goto consumed;
        seq = word_at(p + 18);
        chunk = p[20];
        count = p[21];
        offset = word_at(p + 22);
        payload_len = word_at(p + 24);
        if ((count != CHUNK_COUNT && count != PALETTE_CHUNK_COUNT) ||
            (p[17] == 7 && count != CHUNK_COUNT) ||
            chunk >= count || offset != chunk * CHUNK_SIZE) {
            ++dropped;
            goto consumed;
        }
        expected_len = (count == PALETTE_CHUNK_COUNT ? BUFFER_SIZE : FRAME_SIZE) - offset;
        if (expected_len > CHUNK_SIZE)
            expected_len = CHUNK_SIZE;
        if (payload_len != expected_len || len < 26 + payload_len) {
            ++dropped;
            goto consumed;
        }
        if (!host_known) {
            memcpy(host_mac, p + 6, 6);
            host_known = 1;
        }
        if (seq != sequence || frame_chunk_count == 0) {
            if (frame_chunk_count && (unsigned)(seq - sequence) >= 0x8000u) {
                ++dropped;
                goto consumed;
            }
            sequence = seq;
            received_chunks = 0;
            frame_chunk_count = count;
            frame_kind = p[17];
            memset(chunk_mask, 0, sizeof(chunk_mask));
            if (frame_kind == 7 && adaptive_palette) {
                set_rgb332_palette();
                adaptive_palette = 0;
            }
        }
        if (count != frame_chunk_count || p[17] != frame_kind) {
            ++dropped;
            goto consumed;
        }
        if (chunk_mask[chunk >> 3] & (1u << (chunk & 7)))
            goto acknowledged;
        _fmemcpy((frame_kind == 7 ? vga : frame) + offset,
                 _CV_FP(p + 26), payload_len);
        chunk_mask[chunk >> 3] |= 1u << (chunk & 7);
        started = *bios_ticks;
        ++received_chunks;
        ++accepted;
        if (received_chunks == frame_chunk_count) {
            if (frame_kind != 7)
                _fmemcpy(vga, frame, FRAME_SIZE);
            if (frame_chunk_count == PALETTE_CHUNK_COUNT) {
                set_frame_palette(frame + FRAME_SIZE);
                adaptive_palette = 1;
            } else if (adaptive_palette) {
                set_rgb332_palette();
                adaptive_palette = 0;
            }
            ++frames;
        }
acknowledged:
        consume_packet();
        send_ack(seq, chunk);
        continue;
consumed:
        consume_packet();
    }

    _dos_setvect(9, old_keyboard_interrupt);
    if (frames || (frame_kind == 7 && received_chunks)) {
        save_screen("FRAME.RAW", vga);
        save_palette("PALETTE.RAW");
    }
    if (hold && frames)
        getch();
    set_video_mode(3);
    packet_release(handle);
    _ffree(frame);
    stats = fopen("STATS.TXT", "w");
    if (stats != NULL) {
        unsigned i;
        fprintf(stats, "frames=%lu packets=%lu accepted=%lu dropped=%lu last_sequence=%u key_packets=%lu key_changes=%lu key_snapshots=%lu rx_peak=%u rx_rejected=%u chunk_mask=",
                (unsigned long)frames, (unsigned long)packets,
                (unsigned long)accepted, (unsigned long)dropped, sequence,
                (unsigned long)key_packets, (unsigned long)key_changes,
                (unsigned long)key_snapshots, (unsigned)rx_peak,
                (unsigned)rx_rejected);
        for (i = 0; i < sizeof(chunk_mask); ++i)
            fprintf(stats, "%02x", chunk_mask[i]);
        fputc('\n', stats);
        fclose(stats);
    }
    printf("frames=%lu packets=%lu dropped=%lu\n",
           (unsigned long)frames, (unsigned long)packets, (unsigned long)dropped);
    return frames || done ? 0 : 3;
}
