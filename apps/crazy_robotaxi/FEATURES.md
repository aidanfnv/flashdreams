# Crazy Robotaxi feature ledger

This ledger is the scope authority for the ground-up application rewrite. A
feature may not be removed, deferred, or declared complete without updating
this file and the tests named by the relevant entry.

The 2026-08-14 rebase moved Crazy Robotaxi onto
`IFlashDreamsApplication` / `IFlashDreamsApplicationSession`. The user-approved
exception for features the WIP API cannot express is recorded below. Those
features are deferred only until their explicit API unblock conditions exist;
they must not be recreated with private compatibility shims.

## Included in the application-API milestone

| ID | Capability | Owner | Acceptance criteria | Tests |
|---|---|---|---|---|
| CR-CORE-001 | Arcade vehicle handling | `omnidreams_game_engine` | Throttle, braking/reverse, steering return, and handbrake produce bounded deterministic motion. | `test_simulation.py` |
| CR-CORE-002 | Canonical application input | application/engine | The app declares the stock `driver_command`; host-provided keyboard/SDL gamepad values become the same engine `DriverCommand`. | `test_application.py`, `test_input.py` |
| CR-CORE-003 | Hosted application lifecycle | application | The installed app uses the public application/session contracts for discovery, stepping, completion, and cleanup with stock local-window and WebRTC hosting. | `test_application.py`; upstream application bridge tests |
| CR-CORE-004 | Scene conditioning | engine | Simulated poses rasterize directly through Ludus into OmniDreams HD-map chunks without importing Interactive Drive. | provider tests; manual GPU smoke |
| CR-CORE-005 | Causal presentation alignment | engine | Generated frame zero uses rollout-boundary state; later frames use preceding conditioning state. | `test_alignment.py`, `test_provider.py` |
| CR-CORE-006 | Synchronized game output | application/engine | Generated results carry collision-checked `application_frames` containing score, timers, fare, passenger, target, and session state aligned to the video frames. | `test_application.py`, `test_provider.py` |
| CR-GAME-001 | Route-valid fares | `crazy_robotaxi` | Pickups lie on the scene route and dropoffs are reachable and sufficiently separated when the route permits. | `test_game.py` |
| CR-GAME-002 | Timer and scoring | game | Pickup starts a fare, arrival awards remaining-time score and bonus time, expiry fails the fare, and global expiry ends the hosted session. | `test_game.py`, `test_application.py` |
| CR-GAME-003 | Pickup passengers | game | Visible pickups create pedestrian trajectories in conditioning and collected pickups disappear on the completion frame. | `test_game.py`, `test_provider.py` |
| CR-LAUNCH-001 | New application launch | game | `flashdreams-run crazy-robotaxi` is discovered from `flashdreams.applications`; stock local-window and WebRTC outputs use the same application session. | `test_application.py`, CLI no-GPU smoke |
| CR-LAUNCH-002 | Application-owned model runtime | application | One OmniDreams runtime is created lazily on the model worker, reused across isolated sessions, and closed by the application lifecycle. | `test_application.py` |
| CR-LAUNCH-003 | Warmup and performance presets | application | WebRTC can warm leading AR specializations, and standard/perf/native-perf select integration-owned OmniDreams configs without importing Interactive Drive. | `test_application.py`; manual GPU validation |
| CR-LAUNCH-004 | Sequential completed games | application/host | A completed WebRTC generation can start a fresh game on the retained peer and shared model runtime. | upstream application WebRTC tests; `test_application.py` session-isolation test |

## Implemented FlashDreams integration

`crazy-robotaxi` is registered through `flashdreams.applications` and runs from
the shared `flashdreams-run` entry point without a private runner or launch
loop. Stock local-window and WebRTC hosting consume the same application
session. Keyboard and generic SDL gamepad input arrive through the required
canonical `driver_command` modality and are translated once into the engine's
transport-neutral `DriverCommand`.

The application lazily constructs one public `OmnidreamsRuntime` on the model
worker and closes it through the application lifecycle. Sessions retain only
their own OmniDreams cache, provider, simulation, and game state. Normal game
expiry completes through `next_step_requirements() -> None`, and subsequent
WebRTC games reuse the peer and model runtime while creating isolated session
state.

The standalone engine advances simulation, fares, timers, scoring, passengers,
and HD-map conditioning without importing Interactive Drive. Each generated
result carries collision-checked, causally aligned `application_frames`
metadata for the output sink; rendering that payload remains separately
API-blocked.

WebRTC warmup uses neutral driving windows to exercise the leading seven
autoregressive specializations. `--model-preset standard|perf|native-perf`
selects only integration-owned OmniDreams configurations, with `perf` as the
default. Session reset rebuilds model and game state on the model worker, while
the missing active-game reset input/control path remains deferred below.

## API-blocked deferrals

### CR-DEFER-004 — In-session restart

Status: reset implementation and sequential completed games are included;
active-game user initiation remains API-blocked, not cancelled.

The public session now supports reset, and Crazy Robotaxi resets its model
cache, simulation, game, renderer timeline, and causal alignment on the model
worker. The application bridge still cannot turn a stock canonical input into
a host reset decision, so the previous Escape/restart action cannot reach that
implementation during an active game.

Restoration target:

- Add a public edge-triggered reset action and bridge it to the host's reset
  decision.
- Clear held canonical input and call `OutputSink.begin_generation` exactly
  once after the implemented state reset.
- Acceptance requires deterministic reset parity, held-key clearing, no stale
  video delivery, and at least two reset cycles in a CPU fake-session test.

API dependency: public canonical reset action and application-to-host control
mapping. See
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

Status: engine converters, evdev calibration, and tests retained; the new host
cannot connect them end-to-end.

The stock local window supplies generic SDL gamepad axes, but there is no
application input-handler composition hook for calibrated evdev devices.
Application WebRTC input is keyboard-only and cannot accept continuous browser
Gamepad snapshots.

Restoration target:

- Compose the existing `EvdevWheelReader` with host-owned local input.
- Carry browser Gamepad snapshots through the same declared canonical driver
  modality, including disconnect fallback.
- Preserve steering/pedal calibration, handbrake, reverse, and reset mappings.
- Acceptance requires native profile tests, browser payload tests, disconnect
  fallback, and command parity between both transports.

API dependency: application-contributed paired `IOFactory` selection and
schema-validated continuous WebRTC snapshots. See
[API_FINDINGS.md](API_FINDINGS.md#full-wheel-support).

### CR-DEFER-007 — Minimal game HUD

Status: presentation payload retained; renderer integration is blocked.

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
- Consume scripted canonical commands with deterministic time windows; do not
  special-case replay inside the game session.
- Acceptance requires a two-step CPU replay through `run_application`, an MP4
  fake-writer test, and deterministic command/result metadata parity.

API dependency: CLI/output-target selection of an application-compatible
paired `IOFactory`. See
[API_FINDINGS.md](API_FINDINGS.md#deterministic-replay-and-headless-output).

### CR-DEFER-009 — WebRTC geometry from application sessions

Status: blocked by output-target setup, not cancelled.

The application declares output geometry in `SessionInfo`, but stock WebRTC
serving independently defaults to 1280x720 at 30 FPS and the CLI has no generic
geometry override. This blocks a truthful 1280x704 default and the 1168x640
performance-manifest shape through `flashdreams-run`.

Restoration target:

- Resolve encoder and browser metadata geometry from validated session info,
  or validate explicit generic host overrides against it.
- Acceptance requires non-720p application WebRTC tests for software and NVENC
  setup plus browser chunk metadata parity.

API dependency: session-aware WebRTC output-target setup. See
[API_FINDINGS.md](API_FINDINGS.md#webrtc-output-geometry).

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
