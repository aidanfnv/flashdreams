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

#ifndef DOS_CLIENT_VBE_H
#define DOS_CLIENT_VBE_H
#define VBE_WIDTH 640
#define VBE_HEIGHT 480
#define VBE_RGB_BYTES (VBE_WIDTH * VBE_HEIGHT * 3)
int vbe_init(unsigned wanted_bpp);
void vbe_draw(const unsigned char *rgb, unsigned width, unsigned height);
int vbe_save(void);
void vbe_shutdown(void);
#endif
