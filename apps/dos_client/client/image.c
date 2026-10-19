// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>

#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_SIMD
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 4096
#define STBI_FAILURE_USERMSG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "image.h"

enum { MAX_IMAGE_BYTES = 12 * 1024 * 1024 };

unsigned char *image_load(const char *path, unsigned width, unsigned height)
{
    int source_width, source_height, channels;
    if (!width || !height || width > 2048 || height > 2048 ||
        width * height * 3 > MAX_IMAGE_BYTES) {
        fprintf(stderr, "Invalid target image dimensions.\n");
        return NULL;
    }
    if (!stbi_info(path, &source_width, &source_height, &channels)) {
        fprintf(stderr, "Cannot read PNG/JPEG %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }
    if (source_width < 1 || source_height < 1 ||
        source_width > STBI_MAX_DIMENSIONS || source_height > STBI_MAX_DIMENSIONS ||
        (unsigned long long)source_width * source_height * 3 > MAX_IMAGE_BYTES) {
        fprintf(stderr, "Image exceeds 4096 pixels per side or 12 MiB of RGB pixels.\n");
        return NULL;
    }
    unsigned char *source = stbi_load(path, &source_width, &source_height, &channels, 3);
    if (!source) {
        fprintf(stderr, "Cannot decode PNG/JPEG %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }
    printf("Image: %s %dx%d -> %ux%u RGB24 (decoded on DOS)\n",
           path, source_width, source_height, width, height);
    if ((unsigned)source_width == width && (unsigned)source_height == height)
        return source;
    unsigned char *output = malloc(width * height * 3);
    if (!output) {
        stbi_image_free(source);
        fprintf(stderr, "Not enough memory to resize the image.\n");
        return NULL;
    }
    /* Pixel centers and clamped edges; no implicit crop or letterboxing. */
    for (unsigned y = 0; y < height; ++y) {
        double sy = (y + 0.5) * source_height / height - 0.5;
        if (sy < 0) sy = 0;
        if (sy > source_height - 1) sy = source_height - 1;
        unsigned y0 = sy, y1 = y0 + (y0 + 1 < (unsigned)source_height);
        double fy = sy - y0;
        for (unsigned x = 0; x < width; ++x) {
            double sx = (x + 0.5) * source_width / width - 0.5;
            if (sx < 0) sx = 0;
            if (sx > source_width - 1) sx = source_width - 1;
            unsigned x0 = sx, x1 = x0 + (x0 + 1 < (unsigned)source_width);
            double fx = sx - x0;
            for (unsigned c = 0; c < 3; ++c) {
                double top = source[(y0 * source_width + x0) * 3 + c] * (1 - fx) +
                             source[(y0 * source_width + x1) * 3 + c] * fx;
                double bottom = source[(y1 * source_width + x0) * 3 + c] * (1 - fx) +
                                source[(y1 * source_width + x1) * 3 + c] * fx;
                output[(y * width + x) * 3 + c] = top * (1 - fy) + bottom * fy + 0.5;
            }
        }
    }
    stbi_image_free(source);
    return output;
}
