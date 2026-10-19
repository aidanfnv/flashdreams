/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Shared VBE display for the standalone probe and network client. */
#include "vbe.h"
#include <dpmi.h>
#include <go32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/farptr.h>

enum { WIDTH = 640, HEIGHT = 480, MAX_MODES = 1024 };

typedef struct {
    unsigned number, bpp, pitch;
    unsigned red, green, blue;
    uint32_t physical;
} VideoMode;

static int video_changed;
static unsigned previous_mode, previous_legacy;

static unsigned word(const unsigned char *data)
{
    return data[0] | ((unsigned)data[1] << 8);
}

static uint32_t dword(const unsigned char *data)
{
    return word(data) | ((uint32_t)word(data + 2) << 16);
}

static void restore_video(void)
{
    __dpmi_regs regs = {0};
    if (!video_changed)
        return;
    regs.x.ax = 0x4f02;
    regs.x.bx = previous_mode;
    __dpmi_int(0x10, &regs);
    if (regs.x.ax != 0x004f) {
        memset(&regs, 0, sizeof(regs));
        regs.h.al = previous_legacy;
        __dpmi_int(0x10, &regs);
    }
    video_changed = 0;
}

static int bios_info(unsigned function, unsigned mode, unsigned char *data,
                     unsigned size)
{
    __dpmi_regs regs = {0};
    memset(data, 0, size);
    if (function == 0x4f00)
        memcpy(data, "VBE2", 4);
    dosmemput(data, size, __tb);
    regs.x.ax = function;
    regs.x.cx = mode;
    regs.x.es = __tb >> 4;
    regs.x.di = __tb & 15;
    __dpmi_int(0x10, &regs);
    if (regs.x.ax != 0x004f)
        return -1;
    dosmemget(__tb, size, data);
    return 0;
}

static int find_mode(VideoMode *selected, unsigned wanted_bpp, unsigned *version)
{
    unsigned char controller[512], info[256];
    unsigned short modes[MAX_MODES];
    unsigned count, address, memory;
    if (bios_info(0x4f00, 0, controller, sizeof(controller)) ||
        memcmp(controller, "VESA", 4) || word(controller + 4) < 0x0200) {
        fprintf(stderr, "VBE 2.0 or newer is required.\n");
        return -1;
    }
    *version = word(controller + 4);
    memory = word(controller + 18) * 65536u;
    address = word(controller + 16) * 16u + word(controller + 14);
    /* The list can be inside the transfer buffer: copy it before querying modes. */
    for (count = 0; count < MAX_MODES; ++count) {
        if (address + count * 2 + 1 >= 0x100000u)
            break;
        modes[count] = _farpeekw(_dos_ds, address + count * 2);
        if (modes[count] == 0xffff)
            break;
    }
    if (count == MAX_MODES || address + count * 2 + 1 >= 0x100000u) {
        fprintf(stderr, "Invalid VBE mode list.\n");
        return -1;
    }
    for (unsigned i = 0; i < count; ++i) {
        VideoMode candidate;
        unsigned masks, pitch;
        uint32_t red, green, blue;
        if (bios_info(0x4f01, modes[i], info, sizeof(info)))
            continue;
        if ((word(info) & 0x99) != 0x99 || word(info + 18) != WIDTH ||
            word(info + 20) != HEIGHT || info[24] != 1 || info[27] != 6 ||
            (info[25] != 24 && info[25] != 32) ||
            (wanted_bpp && info[25] != wanted_bpp))
            continue;
        /* Some BIOSes report VBE 3 but omit its linear fields (including DOSBox-X). */
        int linear_info = *version >= 0x0300 && word(info + 50);
        masks = linear_info ? 54 : 31;
        pitch = word(info + (linear_info ? 50 : 16));
        candidate.number = modes[i];
        candidate.bpp = info[25];
        candidate.pitch = pitch;
        candidate.red = info[masks + 1];
        candidate.green = info[masks + 3];
        candidate.blue = info[masks + 5];
        candidate.physical = dword(info + 40);
        if (info[masks] != 8 || info[masks + 2] != 8 || info[masks + 4] != 8 ||
            candidate.red + 8 > candidate.bpp ||
            candidate.green + 8 > candidate.bpp ||
            candidate.blue + 8 > candidate.bpp ||
            pitch < WIDTH * (candidate.bpp / 8) || pitch * HEIGHT > memory ||
            candidate.physical < 0x100000u ||
            candidate.physical > UINT32_MAX - pitch * HEIGHT)
            continue;
        red = 255u << candidate.red;
        green = 255u << candidate.green;
        blue = 255u << candidate.blue;
        if ((red & green) || (red & blue) || (green & blue))
            continue;
        if (candidate.bpp > selected->bpp)
            *selected = candidate;
    }
    if (!selected->bpp) {
        fprintf(stderr, "No supported 640x480 RGB888 linear framebuffer mode.\n");
        return -1;
    }
    return 0;
}

static int save_check(const VideoMode *mode, unsigned version,
                      const unsigned char *readback)
{
    FILE *file = fopen("VBE.RAW", "wb");
    unsigned size = mode->pitch * HEIGHT;
    int valid;
    if (!file)
        return -1;
    valid = fwrite(readback, 1, size, file) == size;
    if (fclose(file))
        valid = 0;
    if (!valid || !(file = fopen("MODE.TXT", "w")))
        return -1;
    valid = fprintf(file,
        "version=%u mode=%u width=%u height=%u bpp=%u pitch=%u "
        "red=%u green=%u blue=%u lfb=%lu\n",
        version, mode->number, WIDTH, HEIGHT, mode->bpp, mode->pitch,
        mode->red, mode->green, mode->blue, (unsigned long)mode->physical) > 0;
    if (fclose(file))
        valid = 0;
    return valid ? 0 : -1;
}

static VideoMode mode;
static unsigned version, size;
static unsigned char *pixels;
static __dpmi_meminfo mapping;
static int mapped, selector = -1;

void vbe_shutdown(void)
{
    restore_video();
    if (selector >= 0) {
        __dpmi_free_ldt_descriptor(selector);
        selector = -1;
    }
    if (mapped) {
        __dpmi_free_physical_address_mapping(&mapping);
        mapped = 0;
    }
    free(pixels);
    pixels = NULL;
}

int vbe_init(unsigned wanted_bpp)
{
    __dpmi_regs regs = {0};
    if (find_mode(&mode, wanted_bpp, &version))
        return -1;
    size = mode.pitch * HEIGHT;
    pixels = calloc(1, size);
    if (!pixels) {
        fprintf(stderr, "Cannot allocate framebuffer conversion buffer.\n");
        return -1;
    }
    mapping.address = mode.physical;
    mapping.size = size;
    if (__dpmi_physical_address_mapping(&mapping)) {
        fprintf(stderr, "DPMI cannot map the linear framebuffer.\n");
        goto failed;
    }
    mapped = 1;
    selector = __dpmi_allocate_ldt_descriptors(1);
    if (selector < 0 || __dpmi_set_segment_base_address(selector, mapping.address) ||
        __dpmi_set_segment_limit(selector, (size - 1) | 4095u)) {
        fprintf(stderr, "Cannot create framebuffer selector.\n");
        goto failed;
    }
    regs.h.ah = 0x0f;
    __dpmi_int(0x10, &regs);
    previous_legacy = regs.h.al & 0x7f;
    memset(&regs, 0, sizeof(regs));
    regs.x.ax = 0x4f03;
    __dpmi_int(0x10, &regs);
    previous_mode = regs.x.ax == 0x004f ? regs.x.bx : previous_legacy;
    atexit(vbe_shutdown);
    memset(&regs, 0, sizeof(regs));
    regs.x.ax = 0x4f02;
    regs.x.bx = mode.number | 0x4000;
    video_changed = 1;
    __dpmi_int(0x10, &regs);
    if (regs.x.ax != 0x004f) {
        fprintf(stderr, "VBE mode set failed.\n");
        goto failed;
    }
    printf("VBE %u.%u mode 0x%03x: %ux%u, %u bpp, pitch %u, LFB 0x%08lx\n",
           version >> 8, version & 255, mode.number, WIDTH, HEIGHT, mode.bpp,
           mode.pitch, (unsigned long)mode.physical);
    return 0;
failed:
    vbe_shutdown();
    return -1;
}

void vbe_draw(const unsigned char *rgb, unsigned width, unsigned height)
{
    unsigned draw_width = WIDTH, draw_height = HEIGHT, columns[WIDTH];
    if ((uint64_t)width * HEIGHT > (uint64_t)height * WIDTH)
        draw_height = (uint64_t)height * WIDTH / width;
    else
        draw_width = (uint64_t)width * HEIGHT / height;
    if (!draw_width) draw_width = 1;
    if (!draw_height) draw_height = 1;
    unsigned left = (WIDTH - draw_width) / 2, top = (HEIGHT - draw_height) / 2;
    memset(pixels, 0, size);
    for (unsigned x = 0; x < draw_width; ++x)
        columns[x] = (uint64_t)x * width / draw_width;
    for (unsigned y = 0; y < draw_height; ++y) {
        unsigned row = (uint64_t)y * height / draw_height;
        for (unsigned x = 0; x < draw_width; ++x) {
            const unsigned char *source = rgb + (row * width + columns[x]) * 3;
            uint32_t color = ((uint32_t)source[0] << mode.red) |
                             ((uint32_t)source[1] << mode.green) |
                             ((uint32_t)source[2] << mode.blue);
            memcpy(pixels + (y + top) * mode.pitch + (x + left) * (mode.bpp / 8),
                   &color, mode.bpp / 8);
        }
    }
    movedata(_my_ds(), (unsigned)pixels, selector, 0, size);
}

int vbe_save(void)
{
    movedata(selector, 0, _my_ds(), (unsigned)pixels, size);
    return save_check(&mode, version, pixels);
}
