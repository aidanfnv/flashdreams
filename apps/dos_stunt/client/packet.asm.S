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


#include "packet.h"
	.arch i8086,jumps
	.code16
	.att_syntax prefix
	.text

	.global packet_open
packet_open:
	pushw %bx
	pushw %si
	pushw %di
	pushw %ds
	pushw %es
	movw %ds, %cs:packet_data_segment
	movw $0x0201, %ax	/* access_type, Ethernet class */
	movw $0xffff, %bx	/* any Ethernet hardware type */
	movw $2, %cx		/* two-byte EtherType filter */
	xorw %dx, %dx		/* first interface */
	movw $packet_type, %si
	pushw %cs
	popw %es
	movw $packet_callback, %di
	int $0x60
	jnc 1f
	movw $0xffff, %ax
1:	popw %es
	popw %ds
	popw %di
	popw %si
	popw %bx
	ret

	.global packet_get_address
packet_get_address:
	pushw %bp
	movw %sp, %bp
	pushw %bx
	pushw %di
	pushw %es
	movw 4(%bp), %bx
	movw 6(%bp), %di
	pushw %ds
	popw %es
	movw $6, %cx
	movw $0x0600, %ax
	int $0x60
	jc 1f
	cmpw $6, %cx
	jne 1f
	xorw %ax, %ax
	jmp 2f
1:	movw $0xffff, %ax
2:	popw %es
	popw %di
	popw %bx
	popw %bp
	ret

	.global packet_release
packet_release:
	pushw %bp
	movw %sp, %bp
	pushw %bx
	movw 4(%bp), %bx
	movw $0x0300, %ax
	int $0x60
	popw %bx
	popw %bp
	ret

	.global packet_send
packet_send:
	pushw %bp
	movw %sp, %bp
	pushw %si
	pushw %ds
	movw 4(%bp), %si
	movw 6(%bp), %cx
	movw $0x0400, %ax
	int $0x60
	jnc 1f
	movw $0xffff, %ax
	jmp 2f
1:	xorw %ax, %ax
2:	popw %ds
	popw %si
	popw %bp
	ret

/* The packet driver calls this far routine twice.  Phase zero asks for a
 * receive buffer in ES:DI; phase one says that buffer is filled.  No C code
 * runs inside the driver's callback. */
packet_callback:
	pushf
	pushw %ax
	pushw %bx
	pushw %cx
	pushw %dx
	pushw %si
	pushw %bp
	pushw %ds
	movw %cs:packet_data_segment, %dx
	movw %dx, %ds
	cmpw $0, %ax
	jne 2f
	cmpb $0, %cs:packet_receiving
	jne 3f
	cmpb $RX_SLOTS, rx_count
	jae 3f
	cmpw $RX_PACKET_SIZE, %cx
	ja 3f
	movw %dx, %es
	xorw %ax, %ax
	movb rx_tail, %al
	movw $RX_PACKET_SIZE, %bx
	mulw %bx
	movw %ax, %di
	addw $rx_packet, %di
	movb $1, %cs:packet_receiving
	jmp 4f
2:	cmpw $1, %ax
	jne 4f
	cmpb $0, %cs:packet_receiving
	je 4f
	xorw %bx, %bx
	movb rx_tail, %bl
	shlw $1, %bx
	movw %cx, rx_length(%bx)
	movb rx_tail, %al
	incb %al
	andb $(RX_SLOTS-1), %al
	movb %al, rx_tail
	incb rx_count
	movb rx_count, %al
	cmpb rx_peak, %al
	jbe 5f
	movb %al, rx_peak
5:	movb $0, %cs:packet_receiving
	jmp 4f
3:	incw rx_rejected
	xorw %ax, %ax
	movw %ax, %es
	movw %ax, %di
4:	popw %ds
	popw %bp
	popw %si
	popw %dx
	popw %cx
	popw %bx
	popw %ax
	popf
	lret

packet_data_segment:
	.word 0
packet_receiving:
	.byte 0
