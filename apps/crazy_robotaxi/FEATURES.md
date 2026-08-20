# Crazy Robotaxi feature ledger

This ledger is the scope authority for the ground-up application rewrite. A
feature may not be removed, deferred, or declared complete without updating
this file and the tests named by the relevant entry.

The 2026-08-20 rebase ports the game onto the official V2 `IApplication`,
`ISession`, input-event, session-description, result, reset, and two-thread
session-runner contracts. The current CLI/client-window host and lower-level
OmniDreams model-session boundaries remain behind explicitly named V1
compatibility surfaces because V2 does not define replacements yet. The
user-approved exception for features the WIP API cannot express is recorded
below.

## Included in the application-API milestone

| ID | Capability | Owner | Acceptance criteria | Tests |
|---|---|---|---|---|
| CR-CORE-001 | Arcade vehicle handling | `omnidreams_game_engine` | Throttle, braking/reverse, steering return, and handbrake produce bounded deterministic motion. | `test_simulation.py` |
| CR-CORE-002 | V2 application input | application/engine | The V2 session consumes timestamped `UserInputEvents`; populated keyboard, analog, and normalized driver-command event data produce the same engine `DriverCommand`. | `test_application.py`, `test_input.py` |
| CR-CORE-003 | V2 application lifecycle | application | `CrazyRobotaxiV2Application` and `CrazyRobotaxiV2Session` implement the official V2 contracts; `run_session` drives reset and close, while explicitly named V1 adapters preserve current CLI discovery and stock local-window/WebRTC hosting. | `test_application.py`; upstream V2 runner tests |
| CR-CORE-004 | Scene conditioning | engine | Simulated poses rasterize directly through Ludus into OmniDreams HD-map chunks without importing Interactive Drive. | provider tests; manual GPU smoke |
| CR-CORE-005 | Causal presentation alignment | engine | Generated frame zero uses rollout-boundary state; later frames use preceding conditioning state. | `test_alignment.py`, `test_provider.py` |
| CR-CORE-006 | Synchronized V2 game output | application/engine | `CrazyRobotaxiStepResult` extends V2 `StepResult` with collision-checked `application_frames` containing score, timers, fare, passenger, target, and session state aligned to video. | `test_application.py`, `test_provider.py` |
| CR-GAME-001 | Route-valid fares | `crazy_robotaxi` | Pickups lie on the scene route and dropoffs are reachable and sufficiently separated when the route permits. | `test_game.py` |
| CR-GAME-002 | Timer and scoring | game | Pickup starts a fare, arrival awards remaining-time score and bonus time, expiry fails the fare, and global expiry ends the hosted session. | `test_game.py`, `test_application.py` |
| CR-GAME-003 | Pickup passengers | game | Visible pickups create pedestrian trajectories in conditioning and collected pickups disappear on the completion frame. | `test_game.py`, `test_provider.py` |
| CR-LAUNCH-001 | Transitional application launch | game | `flashdreams-run crazy-robotaxi` is discovered from `flashdreams.applications`; the named V1 host adapter drives the same V2 session for local-window and WebRTC output. | `test_application.py`, CLI help smoke |
| CR-LAUNCH-002 | Application-owned model runtime | application | One OmniDreams runtime is created lazily by the application, reused across isolated sessions, and closed by the V2 application lifecycle. | `test_application.py` |
| CR-LAUNCH-003 | Warmup and performance presets | application | WebRTC can warm leading AR specializations, and standard/perf/native-perf select integration-owned OmniDreams configs without importing Interactive Drive. | `test_application.py`; manual GPU validation |
| CR-LAUNCH-004 | Sequential completed games | application/host | A completed WebRTC generation can start a fresh game on the retained peer and shared model runtime. | upstream application WebRTC tests; `test_application.py` session-isolation test |

## Implemented FlashDreams integration

`CrazyRobotaxiV2Application` and `CrazyRobotaxiV2Session` implement the official
V2 application/session contracts added by #490. The session consumes V2
`UserInputEvents`, publishes V2 `SessionDesc`, and returns
`CrazyRobotaxiStepResult`, a V2 `StepResult` subtype carrying synchronized game
output. It accepts the standard populated keyboard edges directly, retains held
key state, gives a connected analog source priority, and accepts normalized
driver-command events for the temporary V1 host boundary. The official
`run_session` reset path atomically rebuilds the model cache, simulation, game,
renderer timeline, and causal alignment before the first post-reset step.

`crazy-robotaxi` remains launchable through `flashdreams-run` without a private
runner or launch loop. Because #490 did not add V2 `ApplicationRunner`, CLI
discovery, or concrete client windows, `CrazyRobotaxiV1ApplicationAdapter` and
`CrazyRobotaxiV1SessionAdapter` are the temporary playable-host boundary.
Their exact conversions and deletion criteria are recorded in
[API_FINDINGS.md](API_FINDINGS.md#compatibility-surfaces-to-remove).

The V2 application lazily constructs one public `OmnidreamsRuntime` and closes
it through its lifecycle. This is the remaining lower-level V1 model boundary
because V2 has no model runtime/session/input contract.
Sessions retain only their own OmniDreams cache, provider, simulation, and game
state. Normal game expiry completes through the temporary model requirements,
and subsequent WebRTC games reuse the peer and model runtime while creating
isolated session state.

The standalone engine advances simulation, fares, timers, scoring, passengers,
and HD-map conditioning without importing Interactive Drive. Each generated V2
result carries collision-checked, causally aligned `application_frames`;
generic discovery and rendering of that extension remain API-blocked.

WebRTC warmup uses neutral driving windows to exercise the leading seven
autoregressive specializations. `--model-preset standard|perf|native-perf`
selects only integration-owned OmniDreams configurations, with `perf` as the
default. Session reset rebuilds model and game state on the generation thread,
while delivery from a concrete V2 client window remains deferred below.

## API-blocked deferrals

### CR-DEFER-004 — In-session restart

Status: V2 reset input and sequential completed games are included; delivery
from a concrete V2 client window remains API-blocked, not cancelled.

`ResetUserInputEventData` now makes V2 `run_session` reset the session, restart
at step zero, and discard queued or in-flight results from the abandoned
generation. The game reset rebuilds its model cache, simulation, game, renderer
timeline, causal alignment, and held input state. No concrete V2 client window
exists, and the V1 host command has no reset field, so Escape cannot deliver the
standard event in the playable host yet.

Restoration target:

- Make concrete local-window and WebRTC V2 clients translate the reset control
  into `ResetUserInputEventData`.
- Preserve the runner's generation tagging and stale-result discard behavior.
- Acceptance requires deterministic reset parity, held-key clearing, no stale
  video delivery, and at least two reset cycles in a CPU fake-session test.

API dependency: concrete V2 client windows and their reset-control mapping. See
[API_FINDINGS.md](API_FINDINGS.md#in-session-restart).

### CR-DEFER-005 — High-score persistence and name entry

Status: game/store code retained and tested; application-host integration is
blocked.

The host cannot express an optional edge-triggered name action beside a
required level-triggered driving snapshot, and its stock input handlers expose
no app-defined text/action path. The milestone therefore ends the hosted
session when the game leaves `playing`; it does not silently record a default
name.

Restoration target:

- Deliver validated name submit/skip actions through public canonical input.
- Use the existing locked, atomic, top-ten `HighScoreStore` and its 1–12
  character validation.
- Present validation errors and the saved leaderboard in both supported live
  frontends.
- Acceptance requires application-session tests plus local-window and WebRTC
  action-delivery tests; existing store tests must remain unchanged.

API dependency: optional/action canonical modalities and text-capable input
handlers. See
[API_FINDINGS.md](API_FINDINGS.md#high-score-persistence-and-name-entry).

### CR-DEFER-006 — Full wheel support

Status: engine converters and evdev calibration now use V2 event types and are
tested; no V2 client window can connect them end-to-end.

The stock local window supplies generic SDL gamepad axes, but there is no
application input-handler composition hook for calibrated evdev devices.
Application WebRTC input is keyboard-only and cannot accept continuous browser
Gamepad snapshots.

Restoration target:

- Compose the existing `EvdevWheelReader` with host-owned local input.
- Carry browser Gamepad snapshots through populated V2 events, including
  disconnect fallback.
- Preserve steering/pedal calibration, handbrake, reverse, and reset mappings.
- Acceptance requires native profile tests, browser payload tests, disconnect
  fallback, and command parity between both transports.

API dependency: populated V2 gamepad/wheel event contracts and concrete native
and WebRTC producers. See
[API_FINDINGS.md](API_FINDINGS.md#full-wheel-support).

### CR-DEFER-007 — Minimal game HUD

Status: presentation payload is carried by a V2 `StepResult` subtype; generic
renderer integration is blocked.

The stock application sinks do not accept application-owned browser resources
or a local presenter/view plug-in. Video and synchronized
`application_frames` metadata still flow through `StepResult`, but no public
hook can render score, timers, fare state, target direction, name entry, or the
leaderboard.

Restoration target:

- Render one typed payload in both local-window and WebRTC views without
  querying the game controller.
- Keep HUD work outside model/session execution and honor sink backpressure.
- Acceptance requires payload/view tests in both frontends and a manual frame
  pacing check.

API dependency: a typed output/presentation schema plus
application-contributed paired `IOFactory` selection or presentation plug-ins.
See
[API_FINDINGS.md](API_FINDINGS.md#minimal-game-hud).

### CR-DEFER-008 — Deterministic replay, MP4, and null input

Status: output sinks exist, but the CLI pairs MP4/null output with
`NullInputHandler`, which cannot satisfy the application's required
`driver_command` modality.

Restoration target:

- Select a replay/CI input router paired with MP4 or null presentation.
- Consume scripted V2 driver events with deterministic time windows; do not
  special-case replay inside the game session.
- Acceptance requires a two-step CPU replay through the V2 application runner,
  an MP4 fake-writer test, and deterministic command/result metadata parity.

API dependency: V2 `ApplicationRunner`, CLI/client-window selection, replay
input, and MP4/null client windows. See
[API_FINDINGS.md](API_FINDINGS.md#deterministic-replay-and-headless-output).

### CR-DEFER-009 — WebRTC geometry from application sessions

Status: blocked by output-target setup, not cancelled.

The V2 application declares output geometry in `SessionDesc`, but the currently
playable V1 WebRTC serving independently defaults to 1280x720 at 30 FPS and the
CLI has no generic geometry override. This blocks a truthful 1280x704 default
and the 1168x640 performance-manifest shape through `flashdreams-run`.

Restoration target:

- Resolve encoder and browser metadata geometry from validated `SessionDesc`,
  or validate explicit generic host overrides against it.
- Acceptance requires non-720p application WebRTC tests for software and NVENC
  setup plus browser chunk metadata parity.

API dependency: session-aware WebRTC output-target setup. See
[API_FINDINGS.md](API_FINDINGS.md#webrtc-output-geometry).

### CR-DEFER-010 — Game-driven session completion and model chunk requests

Status: the game and lower-level OmniDreams session know when to stop and how
many conditioning frames to request, but V2 cannot express either decision.

The V1 host adapter remains playable because it exposes the lower-level
`next_step_requirements()` result and returns `None` when the game leaves
`playing`. Direct V2 `run_session(steps=None)` can only stop on a client close;
after game expiry it would attempt another step and the game would have to
raise. Fixed-step runs also cannot request OmniDreams' variable next chunk.

Restoration target:

- Add a clean V2 application-completion result or control signal.
- Add a V2 model-step requirement (or equivalent scheduling contract) that can
  request the next step index and conditioning-frame count.
- Acceptance requires natural timer expiry through `run_session`, no exception
  for normal completion, and variable-chunk fake-model coverage.

API dependency: V2 session completion and model-driven step requirements. See
[API_FINDINGS.md](API_FINDINGS.md#application-completion-and-model-step-requirements).

## Non-API follow-up

### CR-FOLLOWUP-001 — Exact performance-manifest extraction and validation

Status: not blocked by the application API; not yet implemented or validated.

The selectable neutral OmniDreams presets do not consume the Interactive Drive
performance manifest. Its native DiT/FP8 KV-cache backend, `[1000, 100]`
denoising schedule, `skip_finalize_kv_cache`, and 1168x640 runtime shape still
need an integration-neutral OmniDreams configuration before the game can adopt
them without depending on Interactive Drive.

Acceptance requires a fixed-input GPU comparison of startup/prewarm and
steady-state timings, reset and sequential-session coverage, recorded active
backend/configuration, and quality artifacts. Cold compile/cache-fill chunks
must remain separate from steady-state results. Until that validation exists,
the current preset selector is configuration reuse rather than a Crazy
Robotaxi performance claim.

## Previously approved deferrals

### CR-DEFER-001 — Alignment capture tooling

Add an engine-level diagnostic output sink over `StepResult` and
`application_frames`, with a manifest for model preset, scene, layouts,
timestamps, and missing optional artifacts. It must never affect simulation or
presentation timing when disabled. Acceptance requires a CPU artifact-schema
test and a manual GPU capture with bijective frame/timestamp counts. Behavioral
reference remains in branch history at
`interactive_drive/crazy_robotaxi/alignment_diagnostics.py`.

### CR-DEFER-002 — Legacy MJPEG serving

If an active consumer is confirmed, implement MJPEG as an `OutputSink` over the
same results and metadata, with `/state` compatibility explicitly decided
before coding. Acceptance requires disconnect cleanup, slow-client
backpressure, and payload parity with WebRTC. Behavioral reference remains in
branch history at `interactive_drive/crazy_robotaxi/streaming_presenter.py`.

### CR-DEFER-003 — Polished HUD parity

After CR-DEFER-007 is unblocked and restored, produce a reviewed visual spec
and asset inventory before porting the large bespoke typography, animation,
BEV styling, effects, responsive layout, and presentation optimizations.
Acceptance requires approved reference screenshots, frontend parity tests, and
frame-time measurements. Behavioral references remain in branch history at
`hud_presenter.py` and `streaming_presenter.py`.

## Scope-change rule

Any pull request touching game scope must update this ledger first. Moving an
item into or out of the milestone requires explicit acceptance criteria and a
named test location. Deferred items remain visible here, in the package README,
and in API findings until completed or explicitly cancelled by the team.
