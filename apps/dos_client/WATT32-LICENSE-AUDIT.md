<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Watt-32 license investigation

Checked on 2026-10-05 against
[gvanem/Watt-32](https://github.com/gvanem/Watt-32) commit
`c727114d209138e3057c122134cf7c3f3e67c374`.

Watt-32 has not been integrated into the DOS client. This investigation built
the library locally and linked small diagnostic executables to identify the
code a client would contain. It does not establish that every component is
cleared for distribution.

## What actually links

The first probe calls `sock_init`, `socket`, `connect`, `send`, `recv`, and
`close_s`, using a numeric IPv4 address. The default upstream DJGPP build pulls
88 Watt-32 object files into that executable.

| Component | License in source | Default probe | Practical consequence |
| --- | --- | --- | --- |
| TFTP, `src/tftp.c` and `.h` | GPL-2.0-or-later | Included | Disable `USE_TFTP` for a client intended to avoid ordinary GPL code. Initialization references this module even if the application never requests TFTP. |
| `_w32_ffs`, inside `src/misc.c` | LGPL-2.0-or-later | Included | This unused routine enters the executable with the rest of `misc.o`. Disabling `USE_BSD_API` excludes it. Removing TFTP alone does not remove this LGPL code. |
| Internationalized DNS, `src/idna.c` and `src/iconv/*.h` | GNU libiconv portions are LGPL-2.0-or-later | Included | Disable `USE_IDNA` to exclude the character conversion code and tables. A numeric destination address alone does not exclude them. |
| BSD socket and other modules authored by Gisle Vanem | BSD style terms with an advertising acknowledgment | Included | Preserve copyright, conditions, and disclaimer with binaries. Advertising materials mentioning features or use must include the specified acknowledgment. This is not standard BSD-3-Clause. |
| Original WatTCP TCP/IP and packet-driver code | Custom terms in `src/copyrigh.h` | Included | Applications may be distributed or sold. Selling the library, or distributing modified library source/object/archive files, requires the author's written permission. |
| CPU detection, `src/cpumodel.S` and related headers | Reference to the historical RTEMS license | Included | RTEMS has a GPL linking exception permitting applications under other licenses. Retain its applicable notices; do not describe this component as ordinary GPL without the exception. |
| zlib modules | Zlib | Included | Permissive terms. Preserve source notices and identify changes if redistributing modified source. |
| Dynamic DNS, part of `src/dynip.c` | Artistic License reference for borrowed HTGET code | Included | Excluded in both reduced probes. The referenced license text is not supplied with that source file. |
| `str_tok` inside `src/misc_str.c` | Attribution to libcurl; permission text omitted | Included | libcurl uses permissive curl terms. Its copyright and permission notice must be accounted for in a distribution. |

The ordinary GPL and LGPL obligations are different. LGPL can permit an
application to retain its own license, but a static executable needs the
applicable library source, notices, and sufficient application source/object
code and build materials for recipients to modify the library and relink.
See [LGPL v2, section 6](https://spdx.org/licenses/LGPL-2.0-or-later.html).
Ordinary GPL has broader combined-work requirements; see
[GPL v2, sections 2 and 3](https://spdx.org/licenses/GPL-2.0-or-later.html).

The WatTCP and advertising restrictions also need consideration before
claiming compliance for a combination containing GPL/LGPL material. Merely
adding a license file does not settle that compatibility question.

The advertising acknowledgment is:

> This product includes software developed by Gisle Vanem Bergen, Norway.

Berkeley's removal of its own advertising clause does not remove Gisle
Vanem's separately stated requirement.

## Reduced builds checked

1. **BSD socket API retained:** disabling `USE_TFTP`, `USE_IDNA`, and
   `USE_DYNIP_CLI` removes those modules from the link map. The executable still
   contains `_w32_ffs` from `misc.o`; 83 Watt-32 objects link.
2. **Native WatTCP API:** a separate probe uses `tcp_open`, `tcp_tick`,
   `sock_fastwrite`, `sock_fastread`, and `sock_close`. Disabling the BSD API and
   related optional features excludes the identified ordinary GPL/LGPL code.
   This executable links 41 Watt-32 objects, including RTEMS CPU detection and
   the custom WatTCP/BSD style code.

The native probe additionally disables `USE_BIND`, `USE_IPV6`, `USE_FSEXT`,
`USE_TCP_MD5`, `USE_GZIP`, `USE_DEBUG`, `USE_LANGUAGE`, `USE_PROFILER`,
`USE_LOOPBACK`, and `USE_FRAGMENTS`. These are the configurations actually
checked, not a claim that every disabled feature is necessary for avoiding
copyleft code. BOOTP, DHCP, and ordinary DNS remain enabled.

All three libraries and probe executables built and linked successfully with
the workspace DJGPP 12.2.0 toolchain. These are link investigations, not network
functionality tests. No probe was used to exchange model-service traffic.

## Remaining limits and distribution scope

Some linked sources provide attribution or provenance without a complete
permission notice; some have no explicit grant. The legacy `copyrigh.h` notice
expressly limits itself to files including that notice. Consequently, a
complete blanket license for all linked code cannot be established from this
snapshot alone. Examples needing clarification include `src/chksum.c`,
`src/timer.c`, and borrowed portions of `src/netaddr.c` and `src/misc_str.c`.

Using the native API is a demonstrated way to exclude the identified ordinary
GPL/LGPL portions. It does not turn Watt-32 into a uniformly permissive library
or resolve those remaining notice/provenance gaps. A distribution using it
still needs the applicable notices and an agreed treatment of its custom
restrictions. lwIP remains the simpler candidate if predictable permissive
licensing is the objective.

CWSDPMI and NIC packet drivers remain separately installed prerequisites.
Neither enters these link maps. Local compilers, build utilities, and DOSBox-X
are outside the planned client distribution. The separate 16-bit `dos_stunt`
client does not use Watt-32.

For this workspace, probe sources, link maps, exact configuration diffs, object
lists, and build logs are saved under
`../.local-build/watt32-license-audit/` relative to the FlashDreams repository.
Private modified source copies used for the reduced builds are under
`/tmp/watt32-link-audit/`. The original Watt-32 checkout has no tracked changes.

Upstream references:

- [WatTCP custom terms](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/copyrigh.h)
- [Default build options](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/config.h)
- [Initialization that pulls optional features into the executable](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/sock_ini.c)
- [TFTP notice](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/tftp.c)
- [Bit-scan routine and notice](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/misc.c)
- [Character conversion notice](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/iconv/cp437.h)
- [Socket implementation notice](https://github.com/gvanem/Watt-32/blob/c727114d209138e3057c122134cf7c3f3e67c374/src/connect.c)
- [Historical RTEMS license and linking exception](https://www.rtems.org/license/license/)
- [curl copyright and permission notice](https://curl.se/docs/copyright.html)
