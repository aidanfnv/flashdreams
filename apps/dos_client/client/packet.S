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

#include "packet.h"
    .text
    .code16
    .balign 16
    .global _packet_blob, _packet_blob_end
    .global _packet_head, _packet_tail, _packet_rejected, _packet_lengths
    .global _packet_rx, _packet_tx

/* Copied unchanged into one DOS memory block. All addresses are relative to
 * the block's segment. No protected-mode callback, allocation, or lwIP call
 * occurs inside the packet driver's interrupt context. */
_packet_blob:
    pushf
    cli
    pushw %ax
    pushw %bx
    pushw %cx
    pushw %dx
    pushw %si
    pushw %ds
    pushw %cs
    popw %ds
    testw %ax, %ax
    jnz completed
    cmpw $0, pending-_packet_blob
    jne rejected
    cmpw $14, %cx
    jb rejected
    cmpw $PACKET_BYTES, %cx
    ja rejected
    movw _packet_tail-_packet_blob, %bx
    movw %bx, %ax
    incw %ax
    andw $(PACKET_SLOTS-1), %ax
    cmpw _packet_head-_packet_blob, %ax
    je rejected
    movw %cx, pending_length-_packet_blob
    movw $1, pending-_packet_blob
    movw %bx, %ax
    movw $PACKET_BYTES, %dx
    mulw %dx
    movw %ax, %di
    addw $(_packet_rx-_packet_blob), %di
    pushw %cs
    popw %es
    jmp done
completed:
    cmpw $1, %ax
    jne done
    cmpw $1, pending-_packet_blob
    jne done
    movw _packet_tail-_packet_blob, %bx
    shlw $1, %bx
    movw pending_length-_packet_blob, %cx
    movw %cx, _packet_lengths-_packet_blob(%bx)
    movw _packet_tail-_packet_blob, %ax
    incw %ax
    andw $(PACKET_SLOTS-1), %ax
    movw %ax, _packet_tail-_packet_blob
    movw $0, pending-_packet_blob
    jmp done
rejected:
    incw _packet_rejected-_packet_blob
    xorw %ax, %ax
    movw %ax, %es
    movw %ax, %di
done:
    popw %ds
    popw %si
    popw %dx
    popw %cx
    popw %bx
    popw %ax
    popf
    lret
    .balign 2
_packet_head: .word 0
_packet_tail: .word 0
_packet_rejected: .word 0
pending: .word 0
pending_length: .word 0
_packet_lengths: .space PACKET_SLOTS*2
    .balign 16
_packet_rx: .space PACKET_SLOTS*PACKET_BYTES
_packet_tx: .space PACKET_BYTES
_packet_blob_end:
    .code32
