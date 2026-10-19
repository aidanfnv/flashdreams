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


#ifndef DOS_STUNT_KEYBOARD_H
#define DOS_STUNT_KEYBOARD_H

#define KEY_BYTES 32u

struct keyboard_state {
    unsigned char down[KEY_BYTES];
    unsigned char tapped[KEY_BYTES];
    unsigned char extended;
    unsigned char pause;
    unsigned char dirty;
};

/* Scan set 1: ordinary keys 0..127, E0 keys 128..255, Pause at 255.
 * No application actions or bindings belong in the DOS keyboard decoder. */
static void keyboard_record(volatile struct keyboard_state *state,
                            unsigned char code)
{
    static const unsigned char pause_tail[5] = {0x1d, 0x45, 0xe1, 0x9d, 0xc5};
    unsigned index, bit;
    if (state->pause) {
        if (code != pause_tail[state->pause - 1])
            state->pause = 0;
        else if (++state->pause == 6) {
            state->down[31] |= 0x80;
            state->tapped[31] |= 0x80;
            state->pause = 0;
            state->dirty = 1;
        }
        return;
    }
    if (code == 0xe1) {
        state->extended = 0;
        state->pause = 1;
        return;
    }
    if (code == 0xe0) {
        state->extended = 1;
        return;
    }
    if (code >= 0xe0) /* Keyboard acknowledgments and error bytes. */
        return;
    index = code & 0x7f;
    if (state->extended) {
        state->extended = 0;
        if (index == 0x2a || index == 0x36) /* Print Screen's fake Shift. */
            return;
        index += 128;
    }
    bit = 1u << (index & 7);
    state->dirty = 1;
    if (code & 0x80)
        state->down[index >> 3] &= (unsigned char)~bit;
    else {
        state->down[index >> 3] |= bit;
        state->tapped[index >> 3] |= bit;
    }
}

/* Call with IRQs disabled. Latch quick taps until one snapshot is sent,
 * then publish their release on the next snapshot. Pause has no break code. */
static void keyboard_snapshot(volatile struct keyboard_state *state,
                              unsigned char *snapshot)
{
    unsigned i;
    unsigned char pending_release = 0;
    for (i = 0; i < KEY_BYTES; ++i) {
        snapshot[i] = state->down[i] | state->tapped[i];
        pending_release |= snapshot[i] ^ state->down[i];
        state->tapped[i] = 0;
    }
    state->down[31] &= 0x7f;
    /* A latched tap or Pause needs a release snapshot without another IRQ. */
    state->dirty = pending_release || (snapshot[31] & 0x80);
}

#endif
