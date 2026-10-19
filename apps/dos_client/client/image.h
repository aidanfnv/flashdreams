// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef DOS_CLIENT_IMAGE_H
#define DOS_CLIENT_IMAGE_H

/* Decode PNG/JPEG and bilinearly resize to tightly packed RGB24. Caller frees. */
unsigned char *image_load(const char *path, unsigned width, unsigned height);

#endif
