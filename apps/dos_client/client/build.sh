#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "$0")"
compiler="${DJGPP_GCC:-../../../../build-djgpp/prefix/bin/i586-pc-msdosdjgpp-gcc}"
if [[ ! -x "$compiler" ]]; then
    compiler=i586-pc-msdosdjgpp-gcc
fi
mkdir -p build
flags=(-std=gnu11 -O2 -Wall -Wextra -Werror)
"$compiler" "${flags[@]}" vbe_demo.c vbe.c -o build/VBEDEMO.EXE

lwip_source="${LWIP_SOURCE:-../../../../.local-build/lwip-2.2.1-source}"
cjson_source="${CJSON_SOURCE:-../../../../cjson}"
minih264_source="${MINIH264_SOURCE:-../../../../.local-build/minih264-source}"
minimp4_source="${MINIMP4_SOURCE:-../../../../.local-build/minimp4-source}"
stb_source="${STB_SOURCE:-../../../../.local-build/stb-source}"
if [[ ! -f "$lwip_source/src/core/tcp.c" || ! -f "$cjson_source/cJSON.c" ]]; then
    echo "Native networking needs lwIP 2.2.1 and cJSON 1.7.19 source trees." >&2
    echo "See apps/dos_client/README.md; override LWIP_SOURCE and CJSON_SOURCE if needed." >&2
    exit 1
fi
if [[ ! -f "$stb_source/stb_image.h" ]]; then
    echo "PNG/JPEG loading needs stb_image source; see apps/dos_client/README.md (STB_SOURCE)." >&2
    exit 1
fi
if [[ ! -f "$minih264_source/minih264e.h" || ! -f "$minimp4_source/minimp4.h" ]]; then
    echo "Local MP4 encoding needs minih264 and minimp4 source trees." >&2
    echo "See apps/dos_client/README.md; override MINIH264_SOURCE and MINIMP4_SOURCE if needed." >&2
    exit 1
fi
for dependency in "$lwip_source:77dcd25a72509eb83f72b033d219b1d40cd8eb95" \
                  "$cjson_source:6d9f2443ab071f86e5d9b43025a40929ec41c46c" \
                  "$minih264_source:b0baea7a80ef9d12da97301dd1099b8791b5ba43" \
                  "$minimp4_source:2a3014a7753cfcba4442adeb469431967238fe6a" \
                  "$stb_source:2c980bb59875b0d32144a71867fbdebb2f77cd20"; do
    path="${dependency%:*}"
    revision="${dependency##*:}"
    if [[ "$(git -C "$path" rev-parse HEAD)" != "$revision" ]]; then
        echo "Unexpected dependency revision: $path (expected $revision)" >&2
        exit 1
    fi
done
# The encoder's platform detection predates DJGPP support. Extend only the
# little-endian declaration in a generated header; leave its source untouched.
python3 - "$minih264_source" <<'PY'
from pathlib import Path
import sys
source = (Path(sys.argv[1]) / 'minih264e.h').read_text()
marker = '#elif defined(__linux__) || defined(__CYGWIN__)'
assert source.count(marker) == 1
source = source.replace(marker, '#elif defined(__DJGPP__)\n#define __BYTE_ORDER 0\n#define __BIG_ENDIAN 1\n' + marker)
Path('build/minih264e.h').write_text(source)
PY
includes=(-I. -I"$lwip_source/src/include" -I"$cjson_source" -isystem build -isystem "$minimp4_source" -isystem "$stb_source")
objects=()
for source in "$lwip_source"/src/core/*.c "$lwip_source"/src/core/ipv4/*.c \
              "$lwip_source/src/netif/ethernet.c"; do
    object="build/lwip-$(basename "${source%.c}").o"
    "$compiler" "${flags[@]}" "${includes[@]}" -MMD -c "$source" -o "$object"
    objects+=("$object")
done
"${compiler%-gcc}-ar" rcs build/liblwip.a "${objects[@]}"
"$compiler" "${flags[@]}" "${includes[@]}" -DCJSON_NESTING_LIMIT=64 \
    -c "$cjson_source/cJSON.c" -o build/cJSON.o
"$compiler" "${flags[@]}" "${includes[@]}" model_client.c packet.c packet.S vbe.c mp4.c h264.c image.c \
    build/liblwip.a build/cJSON.o -lm -Wl,-Map,build/FLSHDRMS.MAP -o build/FLSHDRMS.EXE

# Reproduce file-specific upstream notices alongside the executable. Source
# stays in separate checkouts; packet drivers and DPMI hosts are not bundled.
python3 - "$lwip_source" "$cjson_source" "$minih264_source" "$minimp4_source" "$stb_source" <<'PY'
from pathlib import Path
import re, sys
lwip, cjson, minih264, minimp4, stb = map(Path, sys.argv[1:])
notices = ['FlashDreams native DOS client — third-party notices\n',
           'lwIP 2.2.1\nhttps://github.com/lwip-tcpip/lwip\n' + (lwip/'COPYING').read_text()]
seen = set()
lwip = lwip.resolve()
files = set()
members = set(re.findall(r'liblwip\.a\((lwip-[^)]+)\)', Path('build/FLSHDRMS.MAP').read_text()))
for member in members:
    dependency = Path('build', member).with_suffix('.d').read_text().replace('\\\n', ' ')
    files.update(Path(name).resolve() for name in dependency.split()[1:])
for path in sorted(path for path in files if path.is_relative_to(lwip)):
    for block in re.findall(r'/\*.*?\*/', path.read_text(), flags=re.S):
        if 'Copyright' in block and ('Redistribution' in block or 'permission to use' in block):
            if block not in seen:
                notices.append(str(path.relative_to(lwip)) + '\n' + block)
                seen.add(block)
            break
notices.append('cJSON 1.7.19\nhttps://github.com/DaveGamble/cJSON\n' + (cjson/'LICENSE').read_text())
notices.append('minih264 b0baea7a80ef9d12da97301dd1099b8791b5ba43\n'
               'https://github.com/lieff/minih264\n'
               'Built with a DJGPP little-endian platform declaration; encoder logic unchanged.\n' +
               (minih264/'LICENSE').read_text())
notices.append('minimp4 2a3014a7753cfcba4442adeb469431967238fe6a\n'
               'https://github.com/lieff/minimp4\n' + (minimp4/'LICENSE').read_text())
notices.append('stb_image 2.30 (2c980bb59875b0d32144a71867fbdebb2f77cd20)\n'
               'https://github.com/nothings/stb\nPNG/JPEG only, SIMD disabled; MIT option selected.\n' +
               (stb/'LICENSE').read_text().split('ALTERNATIVE A - MIT License\n', 1)[1]
               .split('ALTERNATIVE B', 1)[0].rstrip('-\n '))
notices.append('Unmodified DJGPP 2.05 startup and C runtime; source and licenses:\n'
               'https://www.delorie.com/djgpp/\n'
               'https://www.mirrorservice.org/sites/ftp.delorie.com/pub/djgpp/current/v2/djcrx205.zip\n'
               'https://www.mirrorservice.org/sites/ftp.delorie.com/pub/djgpp/current/v2/djlsr205.zip\n')
notices.append('FlashDreams client: Apache-2.0\n' + Path('../../../LICENSES/Apache-2.0.txt').read_text())
notices.append(Path('../../../NOTICE').read_text())
Path('build/THIRDP.TXT').write_text('\n\n'.join(notices), encoding='utf-8')
PY
echo "Built VBEDEMO.EXE and FLSHDRMS.EXE; binary notices: build/THIRDP.TXT"
