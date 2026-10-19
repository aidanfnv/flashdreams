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


#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../client/keyboard.h"

static int down(unsigned char *keys, unsigned scan)
{
    return !!(keys[scan >> 3] & (1u << (scan & 7)));
}

int main(void)
{
    struct keyboard_state state = {0};
    unsigned char keys[KEY_BYTES];
    unsigned i;
    const unsigned char pause[] = {0xe1, 0x1d, 0x45, 0xe1, 0x9d, 0xc5};
    const unsigned char print[] = {0xe0, 0x2a, 0xe0, 0x37};
    const unsigned char print_break[] = {0xe0, 0xb7, 0xe0, 0xaa};
    assert(!state.dirty);
    keyboard_record(&state, 0x32); /* M */
    keyboard_record(&state, 0xb2); /* A quick release before the main loop. */
    assert(state.dirty);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0x32) && state.dirty);
    keyboard_snapshot(&state, keys);
    assert(!down(keys, 0x32) && !state.dirty);

    keyboard_record(&state, 0x11); /* Hold W while tapping M. */
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0x11) && !state.dirty);
    keyboard_record(&state, 0x11); /* Typematic must not schedule a release. */
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0x11) && !state.dirty);
    keyboard_record(&state, 0x32);
    keyboard_record(&state, 0xb2);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0x11) && down(keys, 0x32) && state.dirty);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0x11) && !down(keys, 0x32) && !state.dirty);
    keyboard_record(&state, 0x91);
    assert(state.dirty);
    keyboard_snapshot(&state, keys);
    assert(!down(keys, 0x11) && !state.dirty);

    keyboard_record(&state, 0xe0);
    assert(!state.dirty); /* A prefix alone does not change logical key state. */
    keyboard_record(&state, 0x48);
    assert(state.dirty);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0xc8) && !down(keys, 0x48) && !state.dirty);
    keyboard_record(&state, 0xe0);
    keyboard_record(&state, 0xc8);
    keyboard_snapshot(&state, keys);
    assert(!down(keys, 0xc8) && !state.dirty);
    for (i = 0; i < sizeof(pause); ++i)
        keyboard_record(&state, pause[i]);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 255) && !down(keys, 0x1d) && !down(keys, 0x45) && state.dirty);
    keyboard_snapshot(&state, keys);
    assert(!down(keys, 255) && !state.dirty);
    for (i = 0; i < sizeof(print); ++i)
        keyboard_record(&state, print[i]);
    keyboard_snapshot(&state, keys);
    assert(down(keys, 0xb7) && !down(keys, 0x2a) && !down(keys, 0xaa));
    for (i = 0; i < sizeof(print_break); ++i)
        keyboard_record(&state, print_break[i]);
    keyboard_snapshot(&state, keys);
    assert(!down(keys, 0xb7));
    for (i = 0; i < KEY_BYTES; ++i)
        assert(keys[i] == 0);
    assert(!state.dirty);
    puts("Native keyboard decoder: event-driven snapshots, quick taps, held keys, E0, Pause, Print Screen checked");
    return 0;
}
