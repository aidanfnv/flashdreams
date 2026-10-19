<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared FlashDreams model service

This package owns the model catalog, inference server, synthetic server,
RGB protocol, and diagnostic Python client. The
[DOS client](../dos_client/README.md) uses it for remote inference.
It contains no DOS executable, compiler, emulator, or platform display code.

The base package uses only the Python standard library. The optional
`flashdreams` extra adds the framework and shared T2V application; an installed
model integration supplies its application adapter. The server currently
supports T2VApplication adapters, including required/optional image input
advertised by their existing CLI hooks, and installed Cam2VApplication adapters
through a camera session backend. Cam2V requires `flashdreams-cam2v`, supplied by
its model integration. Arbitrary interactive game applications remain unsupported.

## Install in this development workspace

```bash
uv pip install --no-deps -e apps/model_service
```

For the real service, use an environment with FlashDreams, `flashdreams-t2v`,
and the desired integration already installed. The standalone service's
`flashdreams` extra declares its framework dependencies. The base package,
synthetic server, protocol, and diagnostic client need none of them.

The three existing commands retain their names:

- `flashdreams-model-service`: real inference and installed model catalog.
- `flashdreams-model-stub`: deterministic synthetic models.
- `flashdreams-model-probe`: Python client saving RGB frames as PPM files.

## Run inference

```bash
uv run --no-sync flashdreams-model-service
```

With no positional application, the server advertises all installed supported
v2 applications. An optional application ID restricts the catalog for isolated
tests or operator policy; clients can discover the catalog without it. Catalog
pagination keeps each response within the 16 KiB header limit. The client
selects the application and supplies its prompt, image pixels, and settings.
The CLI also accepts `--host`, `--port`, and repeatable `--device` options.
By default, it exposes all visible CUDA GPUs; the first is the default device.
Use `--device cuda:1` to expose only that GPU, or repeat the option to expose
several. IDs follow the server's `CUDA_VISIBLE_DEVICES` ordinal numbering.
`--device cpu` is available for CPU-compatible adapters.
It listens on 127.0.0.1:9390 by default. For a separate client machine, bind
an appropriate host interface or forward the port over SSH.

```bash
uv run --no-sync flashdreams-model-service --host 0.0.0.0
```

See each client's README for its output commands. The normal Linux runner
owns MP4 encoding and native-window presentation on the client; DOS owns
its VBE display and MP4 encoder locally. The server returns packed RGB24
frames, not files or rendered application menus.

## Model residency and sessions

Each GPU gets a separate, lazily started worker process and its own prepared
application. This isolates PyTorch defaults, model caches, and inference failures
between devices. Catalog and device discovery do not load model weights or
start workers. A connection owns one active job; the device lease covers that
job and rejects competing opens on that device. Jobs on different devices can
run concurrently. An omitted client device uses the first exposed device;
the service does not silently move a job to another GPU when it is busy.
Compatible subsequent jobs reuse weights with a fresh rollout/cache and seed.
Changing model, geometry, seed, compilation settings, rollout length,
or initial image pixels reloads the prepared application. Prompt and output
FPS can change without reloading for T2V. Camera sessions also reload when the
prompt or calibration changes; output FPS alone does not require a reload. A model execution failure invalidates the
prepared application; request validation errors leave it intact.

`close` and disconnect release job state while retaining healthy weights.
Service shutdown closes the owned workers and releases their weights.
A disconnected client cannot cancel a GPU kernel
already executing; its job is released at the next request boundary.

## Camera-controlled generation

Use the same service with an installed Cam2V integration:

```bash
uv run --no-sync flashdreams-model-service cam2v-lingbot --device cuda:0
```

For the same application loop with a CPU dummy pipeline:

```bash
uv run --no-sync flashdreams-model-service cam2v-dummy --device cpu
```

Camera entries advertise `interaction: "camera"` and require an uploaded RGB
image and nonempty prompt. Their `horizontal_fov` parameter (degrees, default 90)
and `world_scale` (default 1) define explicit synthetic camera calibration.
Intrinsics use square pixels and a centered principal point at the requested
output dimensions. Tune these for the source image; the defaults are estimates.
The backend constructs the shared Cam2V conditioning object and reuses the
adapter's pipeline, generation function, and camera coordinate convention.
It does not load example images, server-local pose files, or a remote UI.

Each `step.inputs` is a complete boolean snapshot of `forward`, `backward`,
`turn_left`, `turn_right`, `strafe_left`, `strafe_right`, `look_up`, and `look_down`.
Omitted actions are released. Inputs apply when the next model block starts;
requests draining an existing block cannot change its already-generated frames.
Movement advances in model time (`frames / fps`), without accumulating motion
for time spent waiting on inference. Short taps between block boundaries may be
coalesced away. Clients must supply their own release/disconnect behavior.

Camera replies never mix blocks: a batch may contain fewer frames than requested.
They add `block`, `control_time_s`, and `applied_inputs` to identify the controls
that actually produced those pixels. This is an additive protocol-v1 extension.
The existing `total_blocks` setting controls finite completion; there is no added
client frame cap. Reset/reopen starts a new rollout from initial conditions,
without promising exact continuation of a previous model cache.

## Run the stub

From the FlashDreams repository, install into the existing development venv:

```bash
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e apps/model_service
```

Start the service in one terminal:

```bash
uv run --no-sync flashdreams-model-stub
```

It exposes `stub:0` and `stub:1` as two simulated devices. Repeat `--device ID`
to choose another set of simulated IDs. Each has its own job reservation.
It listens on `127.0.0.1:9390`. Use `--host` and `--port` to choose another
address. `--port 0` chooses a free port and prints it. Stop with Ctrl+C.

In another terminal, discover the models and save four test frames:

```bash
uv run --no-sync flashdreams-model-probe \
  --prompt "A neon city" --seed 42 --width 640 --height 480 \
  --frames 4 --output-dir /tmp/flashdreams-model-preview
```

The probe writes `frame-00000.ppm` and subsequent files. These contain the raw RGB
received from the service, wrapped in a PPM header by the probe. The gradient
model hashes the prompt and seed to choose colors; it performs no AI inference.

For image input, provide packed RGB bytes with exactly the requested dimensions:

```bash
uv run --no-sync flashdreams-model-probe \
  --initial-rgb /path/to/image.rgb --width 640 --height 480 \
  --pan 2 --frames 4 --output-dir /tmp/flashdreams-image-preview
```

The first returned frame is the uploaded image. Later frames scroll its rows
horizontally. Image file decoding belongs to the native client; this
diagnostic accepts already decoded RGB. `--pan 0` holds the image still, and
negative values reverse the motion. The probe also accepts `--host` and `--port`.

## Prototype protocol, version 1

This is a model request/response protocol. A connection has one active session
and one outstanding request at a time. Different connections have independent
sessions, subject to the backend's device capacity. Each response echoes the
request's positive uint32 `id` and contains a `type`. Each request contains an
`op`.

### Wire format

Every message begins with this 12-byte prefix:

| Offset | Size | Value |
| --- | --- | --- |
| 0 | 4 | ASCII `FDM1` |
| 4 | 4 | UTF-8 JSON header length, unsigned big-endian integer |
| 8 | 4 | Binary payload length, unsigned big-endian integer |

The JSON object follows the prefix, followed by the binary payload. JSON headers
are limited to 16 KiB, and payloads to 12 MiB. Read the declared lengths in full:
TCP may split a message across multiple reads. JSON must use standard numbers,
with no NaN or Infinity literals.

The first request must negotiate the protocol:

```json
{"id":1,"op":"hello","protocol":"flashdreams-model-service","version":1}
```

Its response has `type: "hello"`, the same protocol and version,
`frame_format: "rgb24"`, `max_frames_per_step: 4`, and
`end_of_stream: "finished"`. An incompatible handshake receives an error, then
the connection closes.

### Optional status messages

Set `"status_messages": true` in the hello request to opt in; the server's hello
response advertises `"status_messages": true`. An opted-in client may receive
zero or more messages with `type: "status"`, the outstanding request's `id`, and
no binary payload before the request's final response. The `phase` is `loading`,
`reusing`, `preparing`, `generating`, or `transferring`. Generation includes
`block` and, for finite T2V jobs, `total_blocks`; transfer includes `first_frame`,
`frame_count`, and `bytes`. For example:

```json
{"id":3,"type":"status","phase":"generating","block":1,"total_blocks":1}
```

Clients that omit the hello flag receive only final responses. Status and final
responses may share a TCP read or arrive fragmented; both use the same bounded
framing above. DOS encoding progress is reported locally.

### Operations

| Operation | Request fields / payload | Response |
| --- | --- | --- |
| `models` | No payload; optional zero-based `offset` | `models`: catalog with IDs, setting schemas, and input schemas; paged replies include `next_offset` (integer or null) |
| `devices` | No payload; requires `device_selection` capability | `devices`: default device and device descriptions |
| `open` | `model`, optional `device` and `parameters`; RGB payload for an image model | `session`: model ID, selected device, and effective parameters |
| `step` | Optional `count` (default 1) and `inputs` | `frames`: metadata and packed RGB payload; or `finished` with `next_frame` and no payload |
| `reset` | No payload | `reset`: rewind the rollout and frame index |
| `close` | No payload | `closed`: release the session, keeping the connection open |

Close the current session before opening another. Settings and inputs are
validated against the discovered schemas; unknown fields in these maps are
rejected. Request errors return `type: "error"`, `code: "request"`, and a
`message`; the connection remains usable. Invalid framing or JSON returns an
error with `id: 0` and `code: "protocol"`, then closes the connection. Execution
failures return `code: "execution"` and discard the failed session; the client
can open another session on the same connection. Disconnecting releases the
active session.

### Device discovery and selection

The hello response advertises `"device_selection": true`. Request `devices`
before opening a job to discover IDs, names, device types, total memory in bytes
(`null` when unavailable), and job reservations:

```json
{"id":2,"type":"devices","default_device":"cuda:0","devices":[{"id":"cuda:0","name":"GPU name","type":"cuda","total_memory":85899345920,"busy":false}]}
```

`busy` reports a job owned by this service. It does not measure utilization or
detect unrelated GPU workloads. Check those separately before starting inference.

The hello reply advertises `catalog_pagination: true`. Clients can request
`models` with `offset: 0`, then follow `next_offset` until
null. Catalog order is fixed for the server lifetime. Legacy requests without
`offset` still return the full catalog if it fits; otherwise they receive an
error requesting a client with pagination support. DOS/probe clients currently
use that legacy request. An individual model entry must fit within 16 KiB.

The catalog retains one entry per application and adds its available `devices`
and default `device`. Select a device explicitly in `open`:

```json
{"id":3,"op":"open","model":"t2v-self-forcing-wan2.1-t2v-1.3b","device":"cuda:1","parameters":{"prompt":"A neon city"}}
```

The session response confirms the selected ID. Unknown or occupied devices
return a request error without affecting jobs on other devices. `cuda` aliases
`cuda:0`. These are additive protocol-v1 fields: clients omitting selection
can still use older services; explicit discovery/selection requires the hello
capability. Backend implementations without this capability retain their
existing session responses.

Both the Linux remote application and `FLSHDRMS remote` accept repeatable
`--server NAME=HOST:PORT` endpoints and qualify discovered IDs as `NAME/ID`.
The clients use the selected endpoint's catalog and send only its raw device ID
(for example `cuda:0`) in `open`. Server names are client configuration; the
service protocol and server configuration remain unchanged. See the
[DOS client](../dos_client/README.md#multiple-servers) for examples.

### Image input and synthetic model settings

Each catalog entry's `input_image` is `none`, `optional_rgb24`, or
`required_rgb24`. Image payloads contain exactly `width * height * 3` bytes at
the effective model dimensions. Decoding and resizing happen on the client.
For T2V, the real service derives this capability from the adapter's existing
`--image-path` argument; it does not maintain a list of image-capable model IDs.

The stub service advertises two models:

- `stub-gradient`: width, height, fps, prompt, and seed settings; no image input.
- `stub-image`: width, height, and fps settings; requires a packed RGB image with
  exactly `width * height * 3` bytes in the `open` payload.

Their common defaults are 320×240 at 30 simulation fps. Width is 1–1024, height
1–768, and fps 1–120. The prompt defaults to empty and allows 2048 characters.
Seed defaults to 0 and is an unsigned 32-bit integer. Both models expose a `pan`
input from −16 to 16 pixels per frame, defaulting to 1. Integer fields reject
booleans. The discovery response includes these defaults and bounds so the DOS
client can describe and validate model settings through its CLI.

Example requests after discovery:

```json
{"id":3,"op":"open","model":"stub-gradient","parameters":{"width":640,"height":480,"prompt":"A neon city","seed":42}}
{"id":4,"op":"step","count":2,"inputs":{"pan":1}}
```

The `step` response header is:

```json
{"id":4,"type":"frames","format":"rgb24","width":640,"height":480,"first_frame":0,"frame_count":2,"fps":30}
```

Its payload contains consecutive frames, with top-to-bottom rows and
left-to-right RGB byte triplets. There is no row padding, palette, compression,
or rendered application UI. Frame `i` in the batch has simulation time
`(first_frame + i) / fps` seconds. `reset` reproduces the initial sequence with
the same settings and image.

Generation advances only when the client requests more frames than remain in
the buffered block. The `fps` setting describes the frames' intended timing;
it does not pace responses. The DOS native-window client displays frames as
quickly as transfer and conversion allow; the Linux client uses the normal
runner's presentation timing. MP4 records each frame at its declared time.
Each request returns up to the requested
1–4 frames, allowing a smaller final batch. After all generated frames are
drained, `step` returns `finished` with `next_frame` equal to the total emitted
frame count. The stub continues indefinitely; a real model respects its
adapter's rollout limit. A client can reset a finished session.

## Check

```bash
uv run --no-sync pytest apps/model_service/tests -m ci_cpu -q
```

The CPU checks cover discovery, settings, RGB uploads, frame ordering,
fragmented/truncated messages, malformed framing, status replies, session
isolation, end of stream, reset, model reuse/reload, device leasing, execution
failure, disconnect cleanup, and shutdown through a fake T2V pipeline.
They import no model integration.

Client-specific emulator and output checks remain under `apps/dos_client/tests`.

After extraction, 26 CPU checks and three client checks passed, including
DOSBox-X VBE/MP4 workflows and the normal Linux CLI's MP4 encoding. A built
service wheel imported and printed CLI help in a clean venv containing no
FlashDreams or DOS package. Real inference on an idle Colossus GB300 also
passed with local Linux MP4/native-window output over SSH forwarding. See the
[DOS cross-machine commands and results](../dos_client/README.md#separate-gpu-machine)
for the corresponding native DOS checks.

## Device-routing verification

```bash
uv run --no-sync pytest apps/model_service/tests -m ci_cpu -q
uv run --no-sync pytest apps/dos_client/tests/test_native_cli.py::test_native_device_discovery_and_routing -m manual -q
uv run --no-sync pytest apps/dos_client/tests/test_native_cli.py::test_native_named_servers_use_selected_catalog_and_device -m manual -q
```

The CPU checks run two concurrent isolated workers with different PyTorch
default dtypes. They verify separate jobs and cached models, rejection of an
occupied device, failure and disconnect cleanup, reset, and worker restart.
The Linux CLI check selects `stub:1` with CUDA hidden locally. The DOS check
keeps `stub:0` occupied while displaying exact VBE pixels from `stub:1`, rejects
occupied/unknown devices, and then uses the released default device.

Named-server client checks use two independent CPU servers offering the same
raw device ID and different catalogs. Both clients discover qualified IDs and
use the selected server's settings; busy jobs remain isolated across hosts.
Linux MP4 output matches a direct source encode, and DOS VBE readback matches
the scaled RGB exactly. On 2026-10-08 the same routing passed across this
workstation and Colossus using CPU stubs, with no PyTorch import on the remote
service. The owned servers and SSH forward stopped afterward. Both physical
GPUs had unrelated workloads during that check. Its scripts and results are in
`artifacts/multi-server-20261008-cpu/` (ignored generated artifacts).

On 2026-10-14, real named-server selection passed against both physical GPUs:
the workstation's RTX PRO 6000 Blackwell and Colossus's GB300. Both endpoints
advertised `cuda:0`; qualified selections reached the requested host. On each
server, the Linux CLI completed 21 Self-Forcing frames at 320×192 and 16 FPS
with seed 42, `--no-compile`, and two blocks. The client MP4 matched an
independent source encode byte for byte. Pending-request cancellation through
the Python client's stop event released its device lease, and a following
three-frame DOS VBE diagnostic reused the same worker and matched source
pixels exactly. All six jobs released their reservations.

Nine ownership snapshots per host found no unrelated compute process. The
owned servers, workers, and SSH forward stopped; final GPU checks found no
compute process. Both servers used PyTorch 2.12.1+cu130 and CUDA 13.0, with
drivers 595.84 on the workstation and 580.173.02 on Colossus. Commands,
source RGB, MP4s, VBE dumps, check/capture scripts, ownership snapshots, and
results are in `artifacts/multi-server-gpu-20261014/` (ignored generated
artifacts). Client processes had CUDA hidden; the DOS client ran in DOSBox-X.

On 2026-10-08, real Self-Forcing inference passed through an isolated `cuda:0`
worker on the RTX PRO 6000 Blackwell workstation. Linux received 21 frames at
320×192 and 16 FPS; its MP4 was byte-identical to encoding the captured server
RGB directly. DOS then reused the worker's model, received three frames, and
matched the scaled source exactly in VBE readback. Discovery did not start the
worker; an occupied real GPU rejected another job. Seven GPU process checks
found only the owned worker, and shutdown removed it. The commands, runnable
check, source frames, and results are in
`artifacts/devices-20261008-1ivv_goc/` (ignored generated artifacts).

Two-device concurrency has been verified with CPU workers. The physical GPU
checks selected each host in turn; simultaneous inference on two physical GPUs
remains unverified.
