<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native DOS FlashDreams client

This is a general DOS client prototype. The DJGPP frontend follows the existing
Linux client's CLI convention for model selection, settings, prompts, and input
files. It connects over TCP through a DOS packet driver, handles the CLI and
output locally: `--mode native-window` displays RGB frames using VBE;
`--mode mp4` encodes and saves them on DOS. A Linux service runs FlashDreams
model execution. Application logic for interactive models is future work.

The real model service uses existing FlashDreams v2 text/image-to-video application
adapters. Model selection, help, settings, and per-step inputs come from the
remote catalog. The [shared model service](../model_service/README.md) owns the synthetic
server, Python diagnostic client, and RGB protocol, all using only the Python
standard library. This client remains independent of `dos_stunt`.

The shared service API will evolve through these clients. What we learn from
the prototype will inform a broader service for remote GPU devices across
DOS, Linux, and Windows clients.

The [Watt-32 license investigation](WATT32-LICENSE-AUDIT.md) records the linked
components and reduced configurations checked while evaluating native networking.

## Networking stack choice

The selected stack is [lwIP 2.2.1](https://github.com/lwip-tcpip/lwip/tree/STABLE-2_2_1_RELEASE),
at commit `77dcd25a72509eb83f72b033d219b1d40cd8eb95`. Its
[BSD 3-Clause license](https://github.com/lwip-tcpip/lwip/blob/STABLE-2_2_1_RELEASE/COPYING)
allows commercial use and redistribution, including modified versions. Source
distributions retain the notices; binary distributions reproduce them in their
accompanying materials. There is no advertising acknowledgment or requirement to
publish application source. Preserve the individual files' copyright notices as
well as the license text when preparing a distribution.

Use its callback TCP API with `NO_SYS=1`, which supports a polling application
without threads or an operating-system scheduler. The DOS adapter will pass
Ethernet frames between lwIP and the installed packet driver, and service lwIP's
timers. This interface keeps the client independent of the NIC model.

A workspace-local compatibility probe compiled 30 upstream source files with
DJGPP and linked 23 lwIP objects, including TCP, IPv4, Ethernet, DNS, and DHCP.
All 23 linked objects carry BSD 3-Clause notices. The probe ran successfully in
headless DOSBox-X and generated a TCP SYN; the captured frame's Ethernet
addresses, IP addresses, destination port, and IPv4/TCP checksums were verified.
This initial probe validated the core build and packet layout. The native
network client now adds the packet-driver adapter and the model-service protocol.

## DOS prerequisites

- A DOS PC compatible with the executable's CPU target (Pentium by default in
  this build). Native-window mode also needs VBE 2.0+ video support with a
  640×480, 24-bit or 32-bit mode and a linear framebuffer. MP4 mode needs no VBE.
- A compatible 32-bit DPMI host. Install `CWSDPMI.EXE` separately in the client's
  directory or on the DOS path, or use a DOS environment that already provides
  compatible DPMI services. CWSDPMI is available through the
  [DJGPP download site](https://www.delorie.com/djgpp/getting.html).
- For the network client, an Ethernet packet driver for the
  installed NIC, obtained separately from its vendor or a packet-driver project
  and loaded before starting the client. The current standalone display probe
  does not use networking. The network client uses DHCP by default, or accepts
  static IPv4 settings. Its service address must currently be numeric IPv4.

We distribute our client executable with its required runtime license notices.
CWSDPMI and NIC packet drivers are user-installed prerequisites and are not
bundled. DOSBox-X supplies its own CWSDPMI and NE2000 packet driver for emulator
runs.

## Native networking prototype

`client/build/FLSHDRMS.EXE` negotiates the model-service protocol and discovers
the catalog. Use `FLSHDRMS remote APPLICATION [runtime options] --
[application options]`, corresponding to `uv run --no-sync flashdreams-run-v2 remote`
on Linux. `FLSHDRMS` is the DOS frontend; `remote` selects remote inference.
Setting names, types, defaults, and bounds come from the
catalog. Per-step inputs such as `--pan` use the same syntax. Unknown, duplicate,
and invalid settings are rejected. Omitted settings use the server's defaults.

The supported Linux runtime options are `--mode native-window|mp4`,
`--output-path`, `--pixel-width`, `--pixel-height`, `--fps`, and `--timeout`.
They go before `--`. The selected application's options go after it: `--prompt`,
`--seed`, `--compile` / `--no-compile`, `--total-blocks`, and `--image-path`
where supported. Both `--option VALUE` and `--option=VALUE` work for options
with values. `--help` shows runtime help; `-- --help` shows catalog options
without loading model weights. Unsupported commands/options are rejected.
This prototype does not implement other Linux runtime options such as
`--total-model-steps`, tensor layout, preload, or presentation policies.
`--device ID` before `--` selects an inference device on the server.
Omit it to use the server's default. Use `--list-devices` to discover the
exposed IDs, GPU names, memory sizes, and service job reservations:

```dos
FLSHDRMS remote --host 192.168.1.2 --list-devices
FLSHDRMS remote t2v-self-forcing-wan2.1-t2v-1.3b --host 192.168.1.2 --device cuda:1 -- --prompt "A neon city"
```

The server exposes all visible CUDA GPUs by default; repeat its `--device`
option to restrict the available set. Each device owns an independent worker
and model cache. A busy selected device rejects another job. `busy` refers to
this service's reservations, not GPU use by unrelated programs.
Explicit discovery/selection requires a service advertising that capability;
omitting selection remains compatible with earlier services.

### Multiple servers

Replace `--host`/`--port` with repeatable `--server NAME=IPv4:PORT` options.
`--list-devices` queries each endpoint and prints qualified IDs such as
`workstation/cuda:0` and `colossus/cuda:0`. Use `--device NAME/ID` to choose
the server and device for a job. Only that server is contacted, and its catalog
supplies the available applications and their options. Different servers can
advertise different options for the same application ID.

For longer DOS commands, place the arguments in a text response file, for
example `RUN.RSP`:

```text
remote t2v-self-forcing-wan2.1-t2v-1.3b
--server workstation=192.168.50.1:9390
--server colossus=192.168.50.3:9390
--device colossus/cuda:0
--mode native-window --pixel-width 320 --pixel-height 192 --fps 16
-- --prompt "A small robot walking along a sunny beach"
```

Run it with `FLSHDRMS @RUN.RSP`. For discovery, use the same endpoint options
with `remote --list-devices` or `remote --list-models`, omitting the application
and its options. Model listings are grouped by server; adding `--device NAME/ID`
restricts the model listing to that server. `APPLICATION -- --help` also uses
the selected server's catalog. Discovery opens no model job.

Without `--device`, a job uses the first server's default device. With multiple
servers, explicit selections must include `NAME/`. An offline unselected server
does not prevent a job; discovery fails if any queried server cannot be read.
The client supports up to 16 named endpoints, numeric IPv4 addresses, and unique
names of up to 63 letters, digits, `_`, or `-`, starting with a letter.
In DOSBox-X, use `10.0.2.2` for services or SSH forwards on the emulator host,
with different ports for each endpoint. The emulator launcher only mounts
storage and starts the supplied DOS command.

### Other client options

DOS extensions before `--` configure networking, VBE, and diagnostics:
`--host`, `--port`, `--server`, `--ip`, `--netmask`, `--gateway`, `--packet-int`, `--bpp`,
`--network-timeout`, `--error-path`, `--list-models`, `--list-devices`, `--device`,
`--check`, and `--check-frames`.
After `--`, `--prompt-file` and `--initial-rgb` load DOS files.

The client requests one RGB frame at a time. TCP transfer, image upload, RGB
conversion, and output handling run on DOS. The service supplies model output;
no application UI is streamed. The output mode is a client setting and is
independent of the server.

### Client output modes

`--mode native-window` (the current default) presents frames locally using the
640×480 VBE display. It fits source frames with nearest-pixel scaling and black
borders to preserve their aspect ratio. Native-window mode remains open until
Escape, including after a finite model finishes.

`--mode mp4 --output-path VIDEO.MP4` converts RGB to YUV420 and encodes H.264
entirely on the DOS CPU, then writes an MP4 to the DOS filesystem. It preserves
the source dimensions and uses the returned model fps for file timing. Encoding
speed does not change playback timing. It ends automatically when the model
reports generation finished; on physical DOS, Escape also stops and finalizes
the recording. MP4 mode requires even dimensions up to 2048×2048. This build's
DOS file API limits recordings to less than 2 GiB. The encoder uses fixed QP 24
and its fastest preset. Quality/speed controls are future work.

Completed frames are written as MP4 fragments, preserving them if a recording
is interrupted. The service's synthetic models generate indefinitely; the
client's explicit `--check` limits a diagnostic run to four frames.
`--check-frames N` runs the same diagnostic for N frames, including video-memory
readback in native-window mode. These limits apply only when requested.

### Emulator launcher

The launcher configures DOSBox-X, mounts a persistent host folder as `C:`, places
`FLSHDRMS.EXE` and its notices there, and loads the emulator's NE2000 packet driver.
Everything after the launcher's first `--` is the command to run **inside DOS**.
`FLSHDRMS` selects models, resolves DOS file paths, and handles model output.

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-dos -- \
  FLSHDRMS remote stub-gradient --mode native-window -- --prompt "A neon city"
```

Here the first `--` ends launcher options; the second starts the selected application's options. The DOS command is the same command you would type on a physical DOS
machine. `FLSHDRMS` has its own `--help`; launcher's help is available before the
separator. Long `FLSHDRMS` commands use a DJGPP response file to bypass DOS's
126-byte command tail, preserving the supplied arguments.

Omit the DOS command to open a DOS prompt. Visible runs return to that prompt
when the command finishes; type `EXIT` to close DOSBox-X. `--headless` runs
without a window, prints command stdout, and closes the emulator after the
command completes. It does not impose a frame cap or timeout. Use
`--dosbox-config /path/to/cpu.conf` before the separator for CPU/emulator settings.

`--storage-dir` defaults to `apps/dos_client/client/build/dos-drive`. Input and
output paths passed to `FLSHDRMS` refer to this DOS filesystem. For example,
`--output-path VIDEO.MP4` writes `C:\VIDEO.MP4`, which is immediately the host file
`/tmp/flshdrms-dos/VIDEO.MP4` when using the folder above. Outputs remain on that
mounted drive; there is no copy after the run. DOSBox-X long filename support
is enabled, including names with spaces. Choose filenames representable in the
DOS code page. Put prompt/image files into the mounted folder before starting
their DOS command, and refer to them by their DOS paths.

Emulator and packet-driver logs stay in the mounted folder as `DOSBOX.LOG` and
`DRIVER.TXT`. Headless command stdout is `CLIENT.TXT`; a nonzero DOS `ERRORLEVEL`
is reported as a failed command. For unattended native errors, pass
`--error-path ERROR.TXT` to `FLSHDRMS` and read that file on the mounted drive.

### Run a FlashDreams model

The current development venv already contains this package and several T2V
integrations. In another environment, install `flashdreams-dos-client` with its
`model-service` extra and the integration for the desired application. The
service resolves installed applications through FlashDreams' v2 registry.

From the FlashDreams repository, start the model service in one terminal:

```bash
uv run --no-sync flashdreams-model-service
```

In another terminal, inspect its settings and start the DOS client:

```bash
uv run --no-sync flashdreams-dos-client --headless -- FLSHDRMS remote --list-models
uv run --no-sync flashdreams-dos-client --headless -- FLSHDRMS remote t2v-wan21-t2v-1.3b-480p -- --help
uv run --no-sync flashdreams-dos-client -- FLSHDRMS remote t2v-wan21-t2v-1.3b-480p --mode native-window -- \
  --prompt "A white cat wearing sunglasses sits on a surfboard."
```

To encode the complete clip on the DOS client instead:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-dos --headless -- \
  FLSHDRMS remote t2v-wan21-t2v-1.3b-480p --mode mp4 --output-path CAT.MP4 --error-path ERROR.TXT -- \
  --prompt "A white cat wearing sunglasses sits on a surfboard." --no-compile
```

The server command is the same for either client mode.

The server discovers installed v2 applications supported by its execution
backends and advertises their settings without loading weights. DOS selects
the application ID; the server loads that application when the session opens.
Different client runs can select different applications through the same
server. The current backend supports shared `T2VApplication` adapters for text
and image conditioning. Other application types are skipped. Installed apps
whose dependencies are missing are skipped with a server log explaining why.
An optional positional application slug restricts the catalog to that one
application, preserving the earlier server commands.

Settings include width, height, fps, prompt, seed, compile, and
total-blocks; their defaults come from the selected adapter. The existing
adapter validates model-specific restrictions such as latent dimensions and
rollout length. A nonempty prompt is required. `--no-compile` disables
network compilation for a short trial; omit it to use the adapter's default.
The Linux service accepts `--device`, `--host`, and `--port`.

Wan 2.1 produces its entire 81-frame clip in one block before the first frame is
available. Other text-to-video adapters can produce several blocks. The server
buffers one decoded block and returns its frames individually as DOS requests
them. Native-window holds the final image until Escape; MP4 finalizes the file
and exits at the end of generation. Generation uses the adapter's rollout
limit; the client adds no frame cap.
Each job gets a fresh generation cache. Closing or disconnecting releases that
job's cache after any in-flight generation returns, while healthy model weights
stay resident in the server. A second job with the same initialization settings
reuses those weights and can supply a new prompt or output fps. Reuse requires
the same application, device, dimensions, seed, compilation setting, rollout
length, and input RGB pixels. Image and rollout changes reload because the
application retains their startup configuration between sessions. This behavior is
implemented in the DOS service without extending the shared T2V application API.
The service restarts the model's random generator at each job's seed. Normal
prompt preparation can still reload temporary text/image encoders when the
pipeline releases those encoders between rollouts.

The service retains one model and admits one active job at a time. Switching
applications/settings evicts the previous model before loading the next one.
Execution failures invalidate resident weights; stopping the server releases
them. There is no idle eviction timer.

The DOS CLI reports loading or reusing weights, preparing prompt/cache,
generating each model block, transferring RGB frames, and encoding frames on
DOS. Transfer and encoding reports are throttled to avoid logging every frame.
Native-window keeps text mode for the initial status messages and switches to
VBE when the first frame arrives. MP4 keeps reporting progress in text mode.

This backend supports `T2VApplication` adapters with text and optional or
required image input. Interactive world-model inputs and concurrent jobs sharing
one GPU need further adapters/service work. The synthetic
image and input paths remain available below.

### Streaming generation

Self-Forcing produces successive blocks while preserving its autoregressive
cache. Start the same model service, then run a short streaming clip locally:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-streaming -- \
  FLSHDRMS remote t2v-self-forcing-wan2.1-t2v-1.3b \
  --mode native-window --pixel-width 320 --pixel-height 192 -- \
  --prompt "A white cat wearing sunglasses rides a surfboard through gentle ocean waves." \
  --total-blocks 8 --no-compile
```

Eight blocks produce 93 frames: nine in the initial block, then twelve per
block. This crosses the model's 21-latent-frame attention window. Each block is
generated when DOS needs its frames, using the same rollout cache. Native-window
keeps the last image visible until Escape. Omit `--total-blocks 8` to use the
application's default rollout length.

For local recording, add `--headless` to the launcher and replace
`--mode native-window` with `--mode mp4 --output-path STREAM.MP4`.
The file is written directly to the mounted DOS drive and finishes when the
rollout ends. Escape or an explicit runtime timeout finalizes frames already
encoded; disconnect releases the server's cache after any active inference
returns.

### Image-conditioned generation

The Wan 2.2 TI2V-5B adapter requires a first-frame image. Use the same generic
service above, with the Wan 2.2 integration installed:

```bash
uv run --no-sync flashdreams-model-service
```

Put a PNG or JPEG named `INPUT.PNG` in the persistent mounted drive, then run:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-dos --headless -- \
  FLSHDRMS remote t2v-wan22-ti2v-5b \
  --mode mp4 --output-path VIDEO.MP4 --error-path ERROR.TXT \
  --pixel-width 512 --pixel-height 288 -- \
  --image-path INPUT.PNG --prompt "Describe how the scene in the input image should move." --no-compile
```

`INPUT.PNG` is a DOS path, resolved by `FLSHDRMS` on the mounted `C:` drive.
DOS decodes the file, converts it to RGB24, and bilinearly resizes it to the
requested model width and height. Resizing stretches to fit without cropping;
choose dimensions with the input's aspect ratio when possible. The service
receives those pixels and normalizes them for the model. It never reads the
client's image file. The completed MP4 is `/tmp/flshdrms-dos/VIDEO.MP4`.
For local VBE display, use `--mode native-window` and omit `--headless`.

Image capability is discovered from the adapter's existing `--image-path`
argument. The DOS service saves the uploaded RGB pixels as a lossless temporary
PNG and passes its own path through that argument. The file survives compatible
jobs and is removed when the resident model is evicted or the server stops,
including initialization failures. The shared T2V app and model adapters require
no DOS-specific changes.

PNG supports RGB, RGBA (alpha discarded), grayscale, and indexed images;
baseline and progressive JPEG are supported. EXIF rotation and color profiles
are not applied. Source dimensions are limited to 4096 pixels per side and
12 MiB of decoded RGB pixels; memory needed for decoding can be higher.
The target is limited to 2048 pixels per side and the same 12 MiB payload limit.
The source image need not match the model dimensions. The raw `--initial-rgb`
alternative still requires exactly `width * height * 3` bytes. Supply one
image option; the catalog declares whether image input is absent, optional,
or required. Reset retains the uploaded image.

### Separate GPU machine

On the GPU machine, install the model service and desired integration, check
that its GPU is idle, then run `uv run --no-sync flashdreams-model-service`.
From the client workstation, run `ssh -N -L 9391:127.0.0.1:9390 colossus`,
replacing `colossus` with the GPU machine's SSH hostname. The tunnel forwards
the workstation's `127.0.0.1:9391` to the remote service.
Inside DOSBox-X, `10.0.2.2` reaches this workstation; `FLSHDRMS` therefore uses
`--host 10.0.2.2 --port 9391`. The emulator launcher and mounted storage remain
local.

Display a short real-model rollout in DOS:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-colossus -- \
  FLSHDRMS remote t2v-self-forcing-wan2.1-t2v-1.3b \
  --host 10.0.2.2 --port 9391 --mode native-window \
  --pixel-width 320 --pixel-height 192 --fps 16 -- \
  --prompt "A white cat wearing sunglasses rides a surfboard through gentle ocean waves. The camera follows smoothly." \
  --seed 42 --no-compile --total-blocks 2
```

Escape returns to the DOS prompt; type `EXIT` there to close the emulator.
The two-block limit is explicit for this short example. Omit `--total-blocks`
to use the model's normal rollout length.

Encode and save the same rollout on DOS:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-colossus --headless -- \
  FLSHDRMS remote t2v-self-forcing-wan2.1-t2v-1.3b \
  --host 10.0.2.2 --port 9391 --mode mp4 --output-path CLIP.MP4 \
  --pixel-width 320 --pixel-height 192 --fps 16 -- \
  --prompt "A white cat wearing sunglasses rides a surfboard through gentle ocean waves. The camera follows smoothly." \
  --seed 42 --no-compile --total-blocks 2
```

The file is written directly to the mounted drive as
`/tmp/flshdrms-colossus/CLIP.MP4`. For the image-conditioned command above,
use the same storage directory and add `--host 10.0.2.2 --port 9391` before
the application's `--` separator. Put its `INPUT.PNG` on that mounted drive.
Stop the service and SSH tunnel when finished.

#### Verified cross-machine results

On 2026-10-07, the native executable passed five runs in local DOSBox-X with
inference on Colossus's GB300. TCP crossed SLiRP and an SSH port forward;
the local launcher had CUDA hidden. Each run used the CLI above, seed 42,
`--no-compile`, and 16 FPS.

| Run | Model output | Check |
| --- | --- | --- |
| Complete DOS MP4 | Self-Forcing, 320×192, 21 frames | Natural end of generation; file encoded and saved on the mounted DOS drive |
| DOS VBE | Self-Forcing, 320×192, 21 frames | 640×480, 32-bit framebuffer matched the scaled server RGB byte for byte |
| Forced TCP disconnect | Self-Forcing, during block 2 | DOS reported the closed connection; its nine-frame partial MP4 remained decodable; server released the cache and device lease |
| Reconnect | Self-Forcing, 320×192, 21 frames | A fresh DOS run reused the resident model and completed normally |
| Image-conditioned DOS MP4 | Wan 2.2 TI2V-5B, 512×288, 81 frames | PNG decoded on DOS; uploaded RGB matched the input exactly; complete clip encoded and saved on DOS |

The MP4 checks decoded every frame and verified H.264, dimensions, frame counts,
and playback timing. Final-frame mean RGB error against independent YUV420
conversion of the captured source was 2.97–3.41 levels out of 255. The three
complete Self-Forcing rollouts produced identical source RGB. VBE used the
explicit `--check-frames 21` diagnostic to save video-memory readback and exit.

Fifteen GPU process checks, before startup, jobs, and generation blocks, found
no external GPU process. The test server and SSH tunnel were stopped afterward,
and Colossus reported no remaining GPU processes. Server software was PyTorch
2.12.1+cu130, CUDA 13.0, cuDNN 9.20.0, driver 580.173.02, and bf16 model weights.

The exact commands, scripts, client logs, input image, server RGB captures,
framebuffer readback, MP4s, and result JSON are preserved in
`client/build/colossus-dos-20261007-8jgcu2co/` (ignored build artifacts).
No production client, model-service, shared T2V, integration, or core code change
was needed for these checks.

### Run the synthetic stub

From the FlashDreams repository, run the service in one terminal:

```bash
uv run --no-sync flashdreams-model-stub
```

In another terminal, list models and inspect their settings:

```bash
uv run --no-sync flashdreams-dos-client --headless -- FLSHDRMS remote --list-models
uv run --no-sync flashdreams-dos-client --headless -- FLSHDRMS remote stub-gradient -- --help
```

Then launch a model in DOSBox-X:

```bash
uv run --no-sync flashdreams-dos-client -- FLSHDRMS remote stub-gradient --pixel-width 640 --pixel-height 480 --fps 30 -- \
  --prompt "A neon city" --seed 42 --pan 1
```

The DOS client obtains an address through DHCP and reaches the workstation's
loopback service as `10.0.2.2:9390` through SLIRP. Native-window remains open
until Escape; there is no frame cap or default inactivity timeout. An optional
`--network-timeout 30` limits a network operation to 30 seconds without progress.
For catalog listing and model help, use the launcher's explicit `--headless`
option as above. `FLSHDRMS remote APPLICATION --bpp 24` selects 24-bit VBE explicitly.

For a UTF-8 prompt file, put `PROMPT.TXT` in the mounted drive and use
`FLSHDRMS remote APPLICATION -- --prompt-file PROMPT.TXT`. Prompt files preserve line breaks
and are limited to 8192 bytes; the selected model's character limit also applies.
`--prompt` and `--seed` belong after `--`. Use each setting once.

`--timeout SECONDS` limits wall-clock application runtime, including model
initialization, and finalizes any frames already encoded on DOS.
`--timeout unbound` (or omission) adds no runtime limit. It is separate from
`--network-timeout`, which measures inactivity during a network operation.
MP4 can run to the adapter's natural end without specifying a runtime limit.

To exercise local image loading with the stub, put `IMAGE.PNG` in the mounted
drive, select `stub-image`, and supply its DOS filename:

```bash
uv run --no-sync flashdreams-dos-client --storage-dir /tmp/flshdrms-dos -- \
  FLSHDRMS remote stub-image --pixel-width 800 --pixel-height 450 -- --image-path IMAGE.PNG --pan -2
```

`--initial-rgb IMAGE.RGB` accepts already decoded packed RGB24 as an alternative
(three bytes per pixel, row-major, no header or padding).
Binary messages are limited to 12 MiB, and the client processes one frame
per response. Selecting another catalog model requires no client-side ID list.

The launcher is an emulator helper; the service runs independently. On a physical
DOS PC, load your NIC's packet driver, ensure a DPMI host is available, and run
the executable directly. For example, with the workstation on `192.168.50.1` and
the DOS PC on `192.168.50.2`:

```text
FLSHDRMS remote stub-gradient --host 192.168.50.1 --ip 192.168.50.2 --pixel-width 640 --pixel-height 480
```

On physical DOS, the selected output and output path belong directly to that
machine, for example:

```text
FLSHDRMS remote t2v-wan21-t2v-1.3b-480p --host 192.168.50.1 --mode mp4 --output-path VIDEO.MP4 -- --prompt "A cat surfing"
```

Bind the Linux model service to that workstation interface with
`uv run --no-sync flashdreams-model-service --host 192.168.50.1`
(use `flashdreams-model-stub` for synthetic models). Omit `--ip` when the DOS PC's network
has DHCP. `--netmask`, `--gateway`, `--port`, and `--packet-int 0x60` are also
available. The native executable uses the packet-driver interface and does not
require an NE2000 card. The emulator helper's NE2000 configuration is for testing.

### Network and framebuffer check

```bash
uv run --no-sync python -m dos_client.network_demo --check
```

This separate diagnostic harness starts a private local stub on an automatically
chosen TCP port, runs the native client headlessly through DOSBox-X's SLIRP
backend, and transfers four frames. With no model argument, this check selects `stub-gradient` with its
320×240 defaults. It verifies discovery/session setup, requested CLI settings,
frame metadata, and the last frame's complete video-memory readback against
the expected stub output, including display scaling and borders.
The harness uses `client/build/network-check` as its persistent mounted drive;
override with `--output-dir`. All artifacts are written there directly.
The finite frame count and timeout apply to this check only. To exercise static
addressing and 24-bit display, add `--ip 10.0.2.15 --bpp 24`. Model settings after
`--` and `--initial-rgb` also work with the check. `--check --list-models` or
`--check MODEL -- --help` verify catalog operations without generating frames.

To check MP4 output, add the client mode and a destination:

```bash
uv run --no-sync python -m dos_client.network_demo --check --mode mp4 \
  --output-path CHECK.MP4 --output-dir /tmp/dos-check -- --width 320 --height 200 --fps 24
```

This runs the encoder and muxer inside DOS. Host `ffprobe` and `ffmpeg` are used
only for this check: verify H.264, dimensions, frame count, and playback rate;
decode all four frames; compare the last frame with an independent 4:2:0
conversion of the source. Normal MP4 generation does not need host video tools.

To check real inference through the same native networking and display path:

```bash
uv run --no-sync python -m dos_client.network_demo --check t2v-wan21-t2v-1.3b-480p \
  --output-dir apps/dos_client/client/build/real-wan21 -- \
  --prompt "A white cat wearing sunglasses sits on a surfboard." --compile false --seed 42
```

For an installed application ID, the private service runs real inference and
compares the last received frame with every byte of DOS video memory. It saves
the source frame as `source.ppm` and the decoded framebuffer as `frame.ppm`.
This check transfers four frames; the model can generate a larger block to
supply them. Its network inactivity limit is 600 seconds unless `--timeout` is
supplied, with a host subprocess limit of twice that value (at least 120
seconds). These limits apply to the finite check; normal launches have no
default timeout.

To check a complete eight-block Self-Forcing rollout and its final framebuffer:

```bash
uv run --no-sync python -m dos_client.network_demo --check --check-frames 93 \
  t2v-self-forcing-wan2.1-t2v-1.3b \
  --output-dir apps/dos_client/client/build/streaming-check -- \
  --prompt "A cat surfing through gentle waves." \
  --width 320 --height 192 --total-blocks 8 --compile false --seed 42
```

`--check-frames` is a diagnostic limit; normal client commands have no frame cap.

The same real-model check accepts `--mode mp4 --output-path CLIP.MP4`
to verify DOS encoding of actual model output instead of VBE readback. The MP4
filename is relative to its mounted `--output-dir` drive.

With the native client and local DOSBox-X built, run the CLI regression check:

```bash
uv run --no-sync pytest apps/dos_client/tests/test_native_cli.py -m manual -q
uv run --no-sync pytest apps/dos_client/tests/test_native_image.py -m manual -q
```

It checks listing/help, UTF-8 and quoted settings, image upload, both display
scaling directions, and errors for unavailable models, invalid/duplicate
settings, missing values, and missing required images. It also checks local MP4
encoding with non-macroblock dimensions and a non-divisor fps, rejected odd
dimensions, automatic completion of a finite rollout, and error reporting in
normal headless MP4 mode, including a file visible on mounted storage before
generation finishes and a filename/folder containing spaces.
It runs two successive jobs against one service, checking that the second uses
resident weights, starts a fresh rollout, and receives the new prompt. Status
messages are checked through the native DOS TCP receiver.
The named-server check lists two independent servers with the same raw device
ID, checks each server's own help and settings, displays exact source pixels
from the selected server, rejects a busy device without affecting its held
job, and checks default routing, invalid endpoint selections, and an offline
unselected server. On 2026-10-08, discovery and exact VBE output also passed
across this workstation and Colossus using CPU stubs while their GPUs were
occupied. Logs and the runnable check are in
`apps/model_service/artifacts/multi-server-20261008-cpu/`; the test servers and
SSH forward stopped afterward.

On 2026-10-14, real named-server selection passed through DOSBox-X against
both the workstation's RTX PRO 6000 Blackwell and Colossus's GB300. Each DOS
command supplied both endpoints and selected `workstation/cuda:0` or
`colossus/cuda:0`. Self-Forcing produced 320×192 RGB at 16 FPS with seed 42,
`--no-compile`, and two blocks; an explicit three-frame diagnostic limit saved
the 640×480, 32-bit VBE framebuffer. Readback matched the scaled source
exactly on both hosts. Each job released its device lease.

Before each DOS run, a Python-client pending-request cancellation released
that device; DOS then reused the same resident worker and its seeded source
matched the beginning of that host's complete Linux rollout. Eighteen GPU
ownership snapshots across the two hosts found no unrelated compute process.
The owned servers, workers, and SSH forward stopped; final checks found no
compute process on either GPU. Both servers used PyTorch 2.12.1+cu130 and
CUDA 13.0, with drivers 595.84 (workstation) and 580.173.02 (Colossus).
Commands, source RGB, framebuffer dumps, scripts, and results are in
`apps/model_service/artifacts/multi-server-gpu-20261014/` (ignored generated
artifacts).

The image check runs PNG/JPEG decoding and resizing in DOS, checks the uploaded
pixels at the model cache, and rejects missing, corrupt, oversized, conflicting,
and incorrectly sized raw image inputs before loading the model.

DOSBox-X must be built with libslirp support. The workspace-local emulator now
uses locally unpacked libslirp and GLib headers/libraries; no system installation
is needed. Neither emulator libraries nor DOS packet drivers are linked into
`FLSHDRMS.EXE`.

## DOS graphics prototype

`client/build/VBEDEMO.EXE` uses 640×480 with a VBE 2.0+ linear
framebuffer. It enumerates the modes the card provides, preferring 32-bit direct
color and otherwise using 24-bit direct color. Packed RGB24 is converted locally
to the selected mode's channel layout and scanline pitch. The service's frame
dimensions and format are independent of the display mode; 640×480 is the
initial target, not a protocol requirement. See the
[VBE specification](https://www.phatcode.net/res/221/files/vbe20.pdf).

The workspace-local [build-djgpp](https://github.com/andrewwutw/build-djgpp)
toolchain is built at `../build-djgpp/prefix`, using GCC 12.2.0, binutils 2.30,
and DJGPP 2.05. The host build tools are under `../.local-build/sysroot`; no
system installation is required. The native build script also accepts
`DJGPP_GCC=/path/to/i586-pc-msdosdjgpp-gcc` for another toolchain.
The final DJLSR host-utility stage was completed serially because its makefile
races under parallel make; both upstream C and C++ compiler checks then passed.

Build the executable from the FlashDreams repository:

```bash
bash apps/dos_client/client/build.sh
```

This builds both `VBEDEMO.EXE` and `FLSHDRMS.EXE`. Native networking also needs
the pinned lwIP and cJSON source checkouts. The existing workspace has them at
`../.local-build/lwip-2.2.1-source` and `../cjson`. For another workspace, clone
[lwIP](https://github.com/lwip-tcpip/lwip) at
`77dcd25a72509eb83f72b033d219b1d40cd8eb95` and
[cJSON](https://github.com/DaveGamble/cJSON) at
`6d9f2443ab071f86e5d9b43025a40929ec41c46c`. Set `LWIP_SOURCE` and
`CJSON_SOURCE` to their paths when invoking the build script. Dependencies remain
outside the FlashDreams source tree; the build checks their revisions.

Local MP4 encoding also uses [minih264](https://github.com/lieff/minih264) at
`b0baea7a80ef9d12da97301dd1099b8791b5ba43` and
[minimp4](https://github.com/lieff/minimp4) at
`2a3014a7753cfcba4442adeb469431967238fe6a`. Both upstream licenses are CC0-1.0.
The workspace has their sources at `../.local-build/minih264-source` and
`../.local-build/minimp4-source`; override `MINIH264_SOURCE` and `MINIMP4_SOURCE`
for other locations. The encoder is built without SSE, SVC, or threads. The
build adds only a DJGPP little-endian platform declaration to a generated copy
of its header. Upstream source checkouts remain unchanged. The binary notices
include both projects' full CC0 licenses.

PNG/JPEG decoding uses [stb_image 2.30](https://github.com/nothings/stb) at
`2c980bb59875b0d32144a71867fbdebb2f77cd20`, with only those codecs enabled and
SIMD disabled. Its MIT license option is selected and reproduced in the binary
notices. The workspace source is `../.local-build/stb-source`; clone that
revision there or set `STB_SOURCE` to its checkout when building.

Copy `FLSHDRMS.EXE` and its generated `THIRDP.TXT` notices when preparing a client
distribution. Obtain the DPMI host and the NIC packet driver separately as
described above.

After the editable package installation below, open the test pattern in DOSBox-X:

```bash
uv run --no-sync flashdreams-dos-vbe
```

It remains open until you press **Escape**, then restores the previous video
mode and exits. `--bpp 24` or `--bpp 32` requests that depth explicitly. To display
a 640×480 packed RGB24 file, add `--rgb /path/to/image.rgb`. This milestone
displays one static image. Use `flashdreams-dos-client` above for the separate
network executable that receives frames from the model service.

The launcher accepts `--dosbox-config /path/to/cpu.conf`. Its settings are read
after the supplied SVGA/memory defaults, so it can override them. No CPU type or
cycle count is fixed in the default config. `--dosbox-path` selects another
DOSBox-X executable.

The DOS executable also runs directly at a DOS prompt:

```text
VBEDEMO.EXE
VBEDEMO.EXE --rgb IMAGE.RGB
```

Ensure the DPMI prerequisite above is available before running it. In DOSBox-X,
the executable uses the emulator's CWSDPMI from `Z:\BIN`.

### Framebuffer check

```bash
uv run --no-sync flashdreams-dos-vbe --check --bpp 32
uv run --no-sync flashdreams-dos-vbe --check --bpp 24
```

Each check generates a 640×480 frame with the stub, loads that RGB file in the
native executable, and reads back the mapped emulated video memory. The launcher
compares every framebuffer byte, including row padding, with the expected native
pixel layout. It saves `MODE.TXT`, `VBE.RAW`, logs, and a decoded `frame.ppm` in
`client/build/check-32` or `client/build/check-24`. Override the directory with
`--output-dir`. These checks are headless and exit automatically; the normal
visible launch has no timeout. `--check --rgb ...` checks an existing RGB file.

### Toolchain attribution

The executable links unmodified DJGPP 2.05 startup code and C library. DJGPP's
licenses and runtime distribution exceptions are described in `copying.dj`,
`copying`, and `copying.lib` in the
[DJGPP 2.05 cross-development archive](https://www.mirrorservice.org/sites/ftp.delorie.com/pub/djgpp/current/v2/djcrx205.zip).
DJGPP itself and its source are available from the
[DJGPP download site](https://www.delorie.com/djgpp/getting.html). The compiler
sources and downloaded libraries remain in the separate build-djgpp checkout.

## Shared service and protocol

The inference service, synthetic service, Python diagnostic client, and
version 1 RGB protocol now live in
[`apps/model_service`](../model_service/README.md). The console commands and
wire protocol are unchanged. This package owns the native DOS executable,
its build and emulator launch tools, and DOS-specific checks.

## Check

Shared protocol and backend CPU checks:

```bash
uv run --no-sync pytest apps/model_service/tests -m ci_cpu -q
```

DOS CLI, VBE, image decoding, and MP4 checks remain under
`apps/dos_client/tests` and are marked `manual` because they run DOSBox-X.
The real-model and native framebuffer checks above use the shared service.
