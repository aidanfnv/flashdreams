# DOS VGA display app

A separate output service and a FlashDreams client-window plugin providing
`--mode stream`.
It selects a 256-color VGA palette for each RGB frame and sends all 320×200 pixels to
`DOSRX.EXE`, through either DOSBox-X's emulated NE2000 or a dedicated Ethernet
port. The DOS client forwards keyboard state and normally draws complete frames;
optional direct RGB332 drawing displays chunks as they arrive.
The application renders its own HUD into the source image; DOS has no local HUD.

## DOS prerequisites

- A DOS PC with VGA and an Ethernet network card.
- An Ethernet packet driver for that card, installed separately and loaded at
  software interrupt `0x60` before starting `DOSRX.EXE`. Obtain it from the card's
  vendor or a packet-driver project; the NE2000 options are linked in
  [Physical DOS PC](#physical-dos-pc).

`DOSRX.EXE` is a 16-bit real-mode program and does not require CWSDPMI or another
DPMI host. We distribute our client executable with its required runtime license
notices; packet drivers are user-installed prerequisites and are not bundled.
For emulator runs, the launcher uses DOSBox-X's built-in NE2000 packet driver.

## Layout and local build

```text
dos_stunt/       stream plugin/protocol, server, Ethernet transport, keyboard decoder
client/          16-bit DOS sources and build.sh
client/build/    ignored executable, object files, emulator output and logs
tests/           CPU checks and explicit headless emulator checks
```

From the FlashDreams checkout, install into the existing local venv and build:

```sh
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e apps/dos_stunt
apps/dos_stunt/client/build.sh
```

Installing the app registers `stream` through `flashdreams.client_windows_v2`.
It uses the plugin discovery API merged in [PR #682](https://github.com/NVIDIA/flashdreams/pull/682).
The stream mode, TCP protocol, and DOS conversion live in this package;
FlashDreams core discovers the plugin without importing a DOS-specific backend.
Discovery does not connect to a server or launch DOSBox-X. The existing
`--stream-host` and `--stream-port` options remain available.

This app uses the venv's existing FlashDreams, NumPy, and Pillow. A fresh
workspace environment can be configured with `--package flashdreams-dos-stunt`;
the `dev` extra includes Crazy Robotaxi for input checks.

The build defaults to the adjacent local
`build-ia16/prefix/bin/ia16-elf-gcc`, falling back to `ia16-elf-gcc` on PATH.
Set `IA16_GCC` to override it. The emulator defaults to the adjacent local
`.local-build/dosbox-x-prefix/bin/dosbox-x`, with our Unix Ethernet backend.
Compiler and emulator builds remain external to this app.

The optional test-emulator backend is preserved in
[`emulator/dosbox-x-unix-ethernet.patch`](emulator/dosbox-x-unix-ethernet.patch),
against DOSBox-X commit `b13702c0af413b46061129aea0924122e5b59a8b`.
For a fresh DOSBox-X source checkout at that revision, apply it from the
FlashDreams repository root:

```sh
git -C ../dosbox-x apply --unidiff-zero "$PWD/apps/dos_stunt/emulator/dosbox-x-unix-ethernet.patch"
```

Build DOSBox-X using its upstream Linux build instructions, with a local install
prefix such as `../.local-build/dosbox-x-prefix`. This patch is GPL-2.0-or-later
and belongs to the separate emulator; it is not linked into `DOSRX.EXE`.
Physical Ethernet runs do not need the emulator or this patch.

**Rebuild/copy DOSRX.EXE when updating the client:** full-keyboard packets replace
the earlier eight-action keyboard mask. This server requires the new client.

## Crazy Robotaxi

Use `uv run --no-sync` with the prepared local venv. It adds the venv's
executables to PATH; `--no-sync` keeps the installed model and client packages.

Terminal one, from the FlashDreams checkout:

```sh
uv run --no-sync flashdreams-dos-server --visible
```

Terminal two, from the same checkout:

```sh
uv run --no-sync flashdreams-run-v2 crazy-robotaxi-omnidreams-fast-perf \
  --mode stream -- \
  --display-width 320 --display-height 200 \
  --game-mode race \
  --map apps/crazy_robotaxi/crazy_robotaxi/maps/flashdreams_raceway.robotaxi.yaml \
  --race-course grand-prix
```

The server listens on localhost:9386 and opens DOSBox-X on the first image.
The source UI renders at 320×200; no server-side resize is needed for these
frames. Color quantization happens before Ethernet delivery, using VGA's
six-bit-per-channel DAC. The default `--palette adaptive` selects up to 256 colors
for each image without dithering. DOS receives the indexed image and its palette
together, then displays the complete frame. This adds 768 bytes per frame.

The first adaptive palette is ordered by brightness. Before each later delivery,
the server matches its colors to the last delivered palette's indices: exact
colors keep their slots first, then nearby colors use remaining slots through
greedy nearest-color matching. Pixel indices are remapped along with the
palette, preserving the quantized image exactly. Matching happens after pending
frame replacement, so dropped frames do not change the reference palette.
This helps prevent unchanged HUD colors such as red and magenta from flashing
as unrelated colors when palette entries move between frames.

DOS receives the complete indices and palette before drawing. The VGA update
still copies pixels first and writes the palette afterwards; those operations
are not atomic. Stable slots reduce the resulting color mismatch, but cannot
remove every transition artifact when palette colors themselves change.

To boost contrast before quantization, add `--contrast 1.4` to the server command.
Values above 1 increase contrast; the default 1 leaves the image unchanged.
The adjustment runs on Linux after resizing and before quantization.
For comparison with the previous fixed palette, use:

```sh
uv run --no-sync flashdreams-dos-server --visible --palette rgb332 --contrast 1.4
```

Rebuild `DOSRX.EXE` for adaptive palettes; older clients only accept RGB332.

### Faster RGB332 delivery

The server now keeps up to four image packets in flight, matched to the DOS
client's four-slot receive queue. This is the default for both palettes.
Older clients without the capacity announcement retain one-packet pacing.
Use `--packet-window 1` to wait for every ACK individually.

For optional direct VGA writes:

```sh
uv run --no-sync flashdreams-dos-server --visible --palette rgb332 --direct-vga
```

Keep your usual FlashDreams command. `--contrast 1.4` can still be added.
`--direct-vga` skips the full-frame RAM-to-VGA copy by writing received chunks
straight into video memory. It requires RGB332 and the rebuilt `DOSRX.EXE`;
adaptive palettes continue to use complete buffered frames. Direct drawing
may show a mixture of old and new pixels during delivery, especially if a
packet needs retransmission. Omit the flag to retain buffered drawing.
The same option works with `--interface` on physical hardware.

The queue adds about 4.5 KiB of conventional RAM compared with the old single
packet buffer. DOS accepts chunks out of order, copies each chunk once, and
ACKs duplicates without displaying the same frame twice. Linux retries only
missing ACKs, bounds outstanding chunks to the negotiated capacity, and keeps
processing keyboard packets during retries. This queues packets within one
frame; pending source images still use the newest-frame policy.

Focus DOSBox-X for keyboard input. W/A/S/D and arrows drive, space brakes,
R restarts, H hides the control hints, and M toggles the HD-map view.
Configured Crazy Robotaxi key bindings also work. Escape reaches the app's
normal return-to-menu action. **Ctrl+C in the server terminal stops the run.**

Menus and HUD use a fixed 5×8 bitmap font when either display dimension is
below 640×400. Menus lay out directly in the native viewport, with compact
padding and buttons sized to the text. Small text stays at the font's native
size; headings and speed digits use Droid Sans. Compass geometry and position
scale with the viewport.
The score/timer bar uses a scaled top margin, and the
compass follows its measured bottom edge with a scaled gap. The control hints
can still be hidden with H. Speedometer edge margins and the BEV's edge
margins and window padding use the same resolution scale.

Menus and HUD are drawn directly into the source framebuffer at its native
resolution, using SlangPy's default texture filtering and font rasterization.

The bitmap face covers small menu and HUD text: settings, buttons, score/time,
speed units, control hints, leaderboard rows, and world-marker labels.
At 640×400 and above, all UI text uses Droid Sans.

Race mode and normal warmup remain in use. There is no default frame cap.
FlashDreams, the stream server, and DOS display keep independent frame rates:
they replace pending frames rather than accumulating old ones. DOS draws each
complete received frame. Input stays active during pauses or slow display.

### Emulator configuration

The live default config uses `core=normal` without forcing CPU type or cycles.
Provide another configuration on the server command:

```sh
uv run --no-sync flashdreams-dos-server --visible --dosbox-config /path/to/dosbox.conf
```

For the 286 type and DOSBox-X's approximate 8 MHz speed preset:

```ini
[cpu]
core=normal
cputype=286
cycles=fixed 750
```

A CPU-only configuration suffices. The server adds its required Ethernet
configuration after loading the supplied file. `--dosbox-path` overrides the
emulator executable; `--client-dir` chooses a writable directory containing
DOSRX.EXE. `--host`/`--port` select the TCP listener; FlashDreams' `--stream-host`
and `--stream-port` select that endpoint.

The server accepts one producer per invocation. Optional `--max-frames N`
stops after N completed frames; a capped visible check holds its last image
until a key is pressed. Continuous runs close directly.

## Synthetic and headless checks

```sh
uv run --no-sync flashdreams-dos-demo --visible
uv run --no-sync flashdreams-dos-demo --fps 0
```

The synthetic sender is a bounded 30-frame diagnostic. Visible mode holds its
last frame for a keypress; `--fps 0` sends as quickly as ACKs allow. It verifies
all 64,000 saved VGA bytes. FRAME.RAW, PALETTE.RAW, STATS.TXT, CLIENT.TXT, DRIVER.TXT and logs
are written under `apps/dos_stunt/client/build/`.

For a stream without model weights, start the server and run:

```sh
uv run --no-sync flashdreams-run-v2 color_fade.app --mode stream --layout bcthw -- \
  --seconds 3 --frames-per-step 8
```

Application completion closes the stream and DOS client. The automated checks:

```sh
uv run --no-sync python -m pytest apps/dos_stunt/tests -m ci_cpu
uv run --no-sync python apps/dos_stunt/tests/check_controls.py
uv run --no-sync python apps/dos_stunt/tests/check_transport_retry.py
uv run --no-sync python apps/dos_stunt/tests/check_server.py
uv run --no-sync python apps/dos_stunt/tests/check_client_delivery.py
```

The control check compiles the same native scan decoder used by the DOS IRQ
handler, then checks keyboard packets through FlashDreams and Crazy Robotaxi.
It covers M, rebound I, extended arrows, both Shift keys, stale releases,
sequence wrap, quick taps, Pause, and Print Screen. It sends no desktop keys.
The native check also covers pending-sample scheduling, including a quick tap
while another key stays held and the automatic Pause release.
The server script launches headless DOSBox-X for capped and uncapped runs and
verifies VGA bytes. Tests need local sockets; the emulator scripts also need
the local emulator and built client. No GPU model generation is needed.

A local before/after keyboard-polling check used the normal core, `cputype=286`,
and `cycles=fixed 750` (DOSBox-X's 8 MHz 286 preset), with synthetic RGB332
frames over its Unix Ethernet backend. Two runs per binary, each excluding
five warmup frames and measuring 30 frames, averaged 3.35 fps before and
3.70 fps after. All 64,000 saved framebuffer bytes matched, with no retries.
Idle `key_snapshots` matched the heartbeat packet count. These are emulator
results; physical network-card and VGA performance still need measurement.

The delivery check uses private DOS drives and the same normal-core 8 MHz 286
preset. It verifies direct/buffered loss recovery, lost final ACKs, sequence
wrap, out-of-order partial direct frames at shutdown, and adaptive palette
delivery. Its benchmark excludes startup and five warmup frames and writes
settings, binary hashes, timings, and client statistics to
`/tmp/dos-client-delivery.json`.
To compare a saved previous executable using two runs of 30 measured frames:

```sh
uv run --no-sync python apps/dos_stunt/tests/check_client_delivery.py \
  --baseline /path/to/previous/DOSRX.EXE --runs 2 --frames 30
```

Measured locally over DOSBox-X's Unix Ethernet backend, at 320×200 RGB332:

| Client/delivery | Outstanding packets | Mean displayed fps |
| --- | ---: | ---: |
| Previous keyboard-optimized client | 1 | 3.76 |
| New buffered client | 1 | 3.56 |
| New buffered client (default) | 4 | 3.87 |
| Direct VGA | 1 | 4.46 |
| Direct VGA plus queue | 4 | 4.97 |

The queue reached four packets with no rejected packets, retransmissions, or
framebuffer mismatches in these benchmark runs. The new receive bookkeeping
adds overhead when using one-packet buffered delivery. Default queuing recovers
that cost; direct VGA plus queuing improved throughput about 32% over the
previous client. These measurements cover synthetic delivery in the emulator,
not model generation or physical ISA/VGA bus performance.

## Frame stream protocol

Each TCP message is two unsigned 32-bit network-order lengths, then UTF-8 JSON,
then binary payload bytes. The JSON limit is 64 KiB and the payload limit is
64 MiB. Both endpoints first exchange:

```json
{"type":"hello","protocol":"flashdreams-frame-stream","version":1}
```

An image message contains packed RGB bytes in row order and an envelope such as:

```json
{"type":"frame","format":"rgb24","width":640,"height":360}
```

Images may change dimensions between messages. The receiver verifies
`width * height * 3` payload bytes. The application draws menus and the HUD into
the image before transmission.

Reverse messages have no binary payload. Supported envelopes are keyboard
(`type`, `key`, `pressed`), mouse (`type`, `action`, `x`, `y`, `button`, `pressed`,
`wheel_x`, `wheel_y`), focus (`type`, `focused`), and `reset` or `close` with only
`type`. Mouse actions are `move`, `button`, or `wheel`; omitted numeric fields
default to zero. Input timestamps are assigned by the stream plugin on receipt.
An `error` envelope with a `message` makes the run report a service failure.
FlashDreams sends `close` on normal shutdown; disconnection also ends the stream.

## Ethernet protocol

Frames use EtherType `0x88B5`, ASCII `FD`, protocol version 1. DOS initially
broadcasts keyboard state; Linux learns its MAC, and DOS learns Linux's MAC
from the first image packet. Subsequent traffic uses unicast addresses.

| Offset | Bytes | Image packet field |
| --- | ---: | --- |
| 14 | 2 | `FD` |
| 16 | 1 | Version 1 |
| 17 | 1 | Kind: buffered image 1, ACK 2, stop 3, full keyboard 6, direct RGB332 7 |
| 18 | 2 | Frame sequence, little endian |
| 20 | 1 | Chunk index |
| 21 | 1 | Chunk count: 47 for adaptive, 46 for RGB332 |
| 22 | 2 | Frame payload offset, little endian |
| 24 | 2 | Frame payload byte count, little endian |

An adaptive frame contains 64,000 pixel indices followed by 768 palette bytes
(RGB triplets, each channel 0–63). A fixed RGB332 frame contains only the
64,000 indices. For kind 1, the client stages the full payload before updating the VGA
framebuffer and palette; incomplete frames leave the displayed palette intact.
Kind 7 carries only RGB332 indices and writes each unique chunk directly to
VGA. A six-byte received-chunk bitmap permits reordering and prevents duplicate
copies or repeated frame completion. `STATS.TXT` includes that bitmap as
`chunk_mask`, plus `rx_peak` and `rx_rejected` for queue diagnostics. On emulator
shutdown, the server verifies the actual partial direct framebuffer using that
bitmap if a transfer was cancelled.
Each chunk carries up to 1,400 payload bytes and requires an ACK. Live transfers retry
while the receiver remains available and cancel when the stream stops.
The stop packet uses ACK chunk 255. Kinds 4 (old action mask) and 5 (local HUD)
are retired.

Keyboard kind 6 uses bytes 18–19 for its sequence and bytes 20–51 for a 256-bit
scan-set-1 bitmap: ordinary codes 0–127, E0-prefixed codes 128–255, Pause at 255.
Byte 52 advertises the receive capacity (four); byte 53 bit 0 announces support
for direct RGB332 kind 7. Zero or absent capacity means legacy one-packet pacing.
DOS sends state changes and a heartbeat about 18 times per second. Quick taps
are latched for one snapshot; Pause and Print Screen's special sequences are
handled without fake modifier presses. Linux converts standard US PC key
positions to normal key names; no application actions live in the DOS client.
The client samples the bitmap after decoded keyboard events or a BIOS-tick
heartbeat. A latched quick tap or Pause schedules a follow-up release sample,
so these releases do not wait for another keyboard interrupt. Idle iterations
skip the bitmap scan and its interrupt-disabled section. `STATS.TXT` includes
`key_snapshots` to count actual samples.
Arrow and keypad keys remain distinct. Both physical Shift/Control/Alt keys
share their logical modifier name. Other keyboard layouts currently use this
US key-name mapping. The bitmap represents key state, not OS text/IME input.

Linux rejects old/out-of-order snapshots and releases retained keys if no
keyboard packet arrives for half a second. Later snapshots recover lost held
state. Multiple taps of the same key during one blocked DOS loop iteration
collapse to one tap; a queued event protocol could preserve those separately.

## Physical DOS PC

The physical mode uses [Linux `AF_PACKET`](https://man7.org/linux/man-pages/man7/packet.7.html)
on the explicitly selected interface.
The cable carries our Ethernet frames directly; neither end needs an IP
address, a TCP/IP stack on DOS, or a network bridge. On this workstation,
`eno2` is the spare port (no address or carrier at inspection); `eno1` carries
the normal network connection. Connect the DOS PC only to the spare port.
`ethtool eno2` reports support for 10 Mb/s half duplex and automatic MDI-X.

For a 286 PC with 8-bit ISA slots, both the VGA card and
NE2000-compatible card must work in those slots. The network card and driver
must support byte transfers. The
[selected VOGONS archive](https://vogonsdrivers.com/getfile.php?fileid=854&menustate=)
contains Crynwr NE2000 version 11.4.3; its listed options have no 8-bit transfer
switch (`-n` means NetWare conversion). This exact driver passed our emulator
test, but that does not verify an actual card's behavior on an 8-bit ISA bus.
An [8-bit NE2000 driver variant](https://github.com/skiselev/isa8_eth/tree/main/software/driver)
exists if the card needs it. The card's chipset and configuration determine
which driver to use.

Use the network card's actual IRQ and I/O base, with an IRQ available on the
PC's ISA bus. The emulator's IRQ 10 setting is not suitable for an 8-bit ISA
slot. Do not select an IRQ already used by another device. The command below
uses IRQ 3 and I/O base `0x300` **as an example**; set the card and command to
matching values. If the card has RJ45, the link must support its 10 Mb/s
Ethernet mode. A crossover cable or a small, isolated 10/100 switch may be
needed if a direct cable does not establish a link.

Copy the newly built `DOSRX.EXE` to a writable directory on the DOS PC. Install
a suitable packet driver separately; the example below assumes your driver is
`NE2000.COM` and is available in the current directory or on the DOS path.
Ensure the VGA card is the active display. From DOS:

```dos
NE2000.COM 0x60 3 0x300
DOSRX.EXE /continuous
```

The driver should report a valid Ethernet address; all zeroes or all `FF`
indicate a card/driver initialization problem. The client uses packet-driver
interrupt `0x60`. No DOS networking TSR besides that driver is required.

For a dedicated link, leave NetworkManager off the spare port and bring its
link up. These are commands to run when preparing the physical connection;
the app does not change interface configuration:

```sh
sudo nmcli device set eno2 managed no
sudo ip link set dev eno2 up
ethtool eno2
```

Check `Link detected: yes` after connecting the cable. Do not add the spare
port to a bridge or use a shared connection profile.

First send the synthetic pattern. From the FlashDreams checkout:

```sh
sudo .venv/bin/flashdreams-dos-ethernet eno2 -- \
  .venv/bin/flashdreams-dos-demo --interface eno2 --fps 0
```

The launcher opens a socket bound only to that port and EtherType, then drops
root privileges before executing the command as the original user. It makes
no permanent capability changes to Python. The synthetic check sends 30
frames as quickly as DOS accepts them, asks DOS to stop, and prints the
acknowledged display rate. `FRAME.RAW`, `PALETTE.RAW`, and `STATS.TXT` are saved on the DOS PC;
Linux cannot verify those files until they are copied back. Start
`DOSRX.EXE /continuous` again for the next run.

Start the DOS server on the spare port, from the FlashDreams checkout:

```sh
sudo .venv/bin/flashdreams-dos-ethernet eno2 -- \
  .venv/bin/flashdreams-dos-server --interface eno2
```

Then launch Crazy Robotaxi in the second terminal using the same `--mode stream`
command as the emulator run. FlashDreams connects only to the local output
service; the DOS server handles the dedicated Ethernet link. Start
`DOSRX.EXE /continuous` on DOS. Keyboard packets remain active while no world frames are available.
Ctrl+C in the DOS server terminal stops the stream and the DOS client.

The reproducible hardware transport check creates two virtual Ethernet ports
inside a temporary Linux user/network namespace, relays their packets to the
DOSBox-X NE2000 card, and verifies the final synthetic VGA bytes. It does not
use either physical port. To test with a downloaded driver and the standalone server
physical mode:

```sh
uv run --no-sync python apps/dos_stunt/tests/check_hardware_transport.py \
  --packet-driver /path/to/NE2000.COM --flashdreams
```

With the selected VOGONS driver, this passed using an emulated 286, IRQ 3,
different MAC addresses from the original demo, five byte-verified synthetic
frames, and twelve FlashDreams frames, all with zero retransmissions. Actual
card compatibility and performance on a physical 8 MHz 286 remain to be measured.
