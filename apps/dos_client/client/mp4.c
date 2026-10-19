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

/* CPU encoding and file writing on the DOS client, independent of VBE. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define H264E_SVC_API 0
#define H264E_MAX_THREADS 0
#include "minih264e.h"
#define MINIMP4_IMPLEMENTATION
#include "minimp4.h"
#include "mp4.h"

static FILE *file;
static MP4E_mux_t *mux;
static mp4_h26x_writer_t writer;
static H264E_persist_t *encoder;
static H264E_scratch_t *scratch;
static unsigned char *yuv;
static unsigned width, height, fps;
static uint32_t frame_index;

static int write_bytes(int64_t offset, const void *bytes, size_t count, void *token)
{
    FILE *destination = token;
    // ponytail: DJGPP's FILE seeks are signed 32-bit; stop at 2 GiB instead
    // of wrapping. A large-file DOS API is needed for longer recordings.
    if (offset < 0 || offset > LONG_MAX || count > (uint64_t)(LONG_MAX - offset) ||
        fseek(destination, (long)offset, SEEK_SET))
        return -1;
    return fwrite(bytes, 1, count, destination) == count ? 0 : -1;
}

int mp4_close(void)
{
    int result = 0;
    if (mux) {
        result = MP4E_close(mux) != MP4E_STATUS_OK;
        mux = NULL;
        mp4_h26x_write_close(&writer);
    }
    if (file) {
        if (fclose(file)) result = 1;
        file = NULL;
    }
    free(encoder);
    free(scratch);
    free(yuv);
    encoder = NULL;
    scratch = NULL;
    yuv = NULL;
    frame_index = 0;
    if (result) fprintf(stderr, "Cannot finalize the MP4 file.\n");
    return result ? -1 : 0;
}

static int open_video(const char *path, unsigned w, unsigned h, unsigned rate)
{
    H264E_create_param_t options = {0};
    int persist_size, scratch_size;
    if (!w || !h || w > 2048 || h > 2048 || ((w | h) & 1) || !rate || rate > 120) {
        fprintf(stderr, "MP4 requires even frame dimensions up to 2048x2048 and fps in 1..120.\n");
        return -1;
    }
    options.width = w;
    options.height = h;
    options.gop = rate * 2;
    options.const_input_flag = 1;  /* Encoder pads non-macroblock dimensions. */
    if (H264E_sizeof(&options, &persist_size, &scratch_size)) goto failed;
    encoder = malloc(persist_size);
    scratch = malloc(scratch_size);
    yuv = malloc(w * h * 3 / 2);
    if (!encoder || !scratch || !yuv || H264E_init(encoder, &options)) goto failed;
    file = fopen(path, "w+b");
    if (!file) {
        fprintf(stderr, "Cannot write MP4: %s\n", path);
        goto failed;
    }
    /* Fragmented MP4 retains completed frames if the run is interrupted. */
    mux = MP4E_open(0, 1, file, write_bytes);
    if (!mux || mp4_h26x_write_init(&writer, mux, w, h, 0) != MP4E_STATUS_OK ||
        writer.mux_track_id < 0) goto failed;
    width = w;
    height = h;
    fps = rate;
    printf("Local MP4: %s, %ux%u, %u fps, CPU H.264 encoder\n", path, w, h, rate);
    return 0;
failed:
    fprintf(stderr, "Cannot initialize the local MP4 encoder.\n");
    mp4_close();
    return -1;
}

static void rgb_to_yuv420(const unsigned char *rgb)
{
    unsigned size = width * height;
    unsigned char *u = yuv + size, *v = u + size / 4;
    for (unsigned i = 0; i < size; ++i) {
        const unsigned char *pixel = rgb + i * 3;
        yuv[i] = ((66 * pixel[0] + 129 * pixel[1] + 25 * pixel[2] + 128) >> 8) + 16;
    }
    for (unsigned row = 0; row < height; row += 2)
        for (unsigned column = 0; column < width; column += 2) {
            int r = 0, g = 0, b = 0;
            for (unsigned dy = 0; dy < 2; ++dy)
                for (unsigned dx = 0; dx < 2; ++dx) {
                    const unsigned char *pixel = rgb + ((row + dy) * width + column + dx) * 3;
                    r += pixel[0]; g += pixel[1]; b += pixel[2];
                }
            r = (r + 2) / 4; g = (g + 2) / 4; b = (b + 2) / 4;
            unsigned index = row / 2 * (width / 2) + column / 2;
            u[index] = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            v[index] = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        }
}

int mp4_write_frame(const char *path, const unsigned char *rgb,
                    unsigned w, unsigned h, unsigned rate)
{
    if (!file && open_video(path, w, h, rate)) return -1;
    if (w != width || h != height || rate != fps) {
        fprintf(stderr, "MP4 frame dimensions or fps changed during the session.\n");
        return -1;
    }
    rgb_to_yuv420(rgb);
    H264E_io_yuv_t input = {
        .yuv = {yuv, yuv + width * height, yuv + width * height * 5 / 4},
        .stride = {width, width / 2, width / 2},
    };
    H264E_run_param_t options = {
        .encode_speed = H264E_SPEED_FASTEST,
        .qp_min = 24, .qp_max = 24,
    };
    unsigned char *coded;
    int length;
    unsigned duration = ((uint64_t)frame_index + 1) * 90000 / fps -
                         (uint64_t)frame_index * 90000 / fps;
    if (H264E_encode(encoder, scratch, &options, &input, &coded, &length) ||
        length <= 0 || mp4_h26x_write_nal(&writer, coded, length, duration) != MP4E_STATUS_OK ||
        fflush(file)) {
        fprintf(stderr, "Cannot encode or write MP4 frame %lu.\n", (unsigned long)frame_index);
        return -1;
    }
    ++frame_index;
    return 0;
}
