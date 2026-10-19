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

#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vbe.h"
enum { WIDTH = VBE_WIDTH, HEIGHT = VBE_HEIGHT };

static int load_rgb(unsigned char *rgb, const char *path)
{
    static const unsigned char colors[8][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {0, 255, 255},
        {255, 0, 255}, {255, 255, 0}, {255, 255, 255}, {0, 0, 0},
    };
    if (path) {
        FILE *file = fopen(path, "rb");
        int valid;
        if (!file) {
            perror(path);
            return -1;
        }
        valid = fread(rgb, 1, WIDTH * HEIGHT * 3, file) == WIDTH * HEIGHT * 3 &&
                fgetc(file) == EOF && !ferror(file);
        fclose(file);
        if (!valid)
            fprintf(stderr, "Input must contain exactly 640x480 packed RGB24 pixels.\n");
        return valid ? 0 : -1;
    }
    for (unsigned y = 0; y < HEIGHT; ++y) {
        for (unsigned x = 0; x < WIDTH; ++x) {
            unsigned char *pixel = rgb + (y * WIDTH + x) * 3;
            if (y >= 400) {
                memcpy(pixel, colors[x / 80], 3);
            } else {
                pixel[0] = x * 255 / (WIDTH - 1);
                pixel[1] = y * 255 / 399;
                pixel[2] = (x + y) & 255;
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *rgb_path = NULL;
    unsigned wanted_bpp = 0;
    int check = 0, status = 1;
    unsigned char *rgb;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--check")) {
            check = 1;
        } else if (!strcmp(argv[i], "--bpp") && i + 1 < argc &&
                   (!strcmp(argv[i + 1], "24") || !strcmp(argv[i + 1], "32"))) {
            wanted_bpp = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--rgb") && i + 1 < argc) {
            rgb_path = argv[++i];
        } else {
            fprintf(stderr, "Usage: VBEDEMO [--bpp 24|32] [--rgb FILE] [--check]\n");
            return 2;
        }
    }
    if (check && !freopen("ERROR.TXT", "w", stderr))
        return 1;
    rgb = malloc(VBE_RGB_BYTES);
    if (!rgb || load_rgb(rgb, rgb_path) || vbe_init(wanted_bpp))
        goto cleanup;
    vbe_draw(rgb, WIDTH, HEIGHT);
    if (check) {
        if (vbe_save()) {
            fprintf(stderr, "Cannot save framebuffer check.\n");
            goto cleanup;
        }
    } else {
        while (getch() != 27)
            ;
    }
    status = 0;
cleanup:
    vbe_shutdown();
    free(rgb);
    return status;
}
