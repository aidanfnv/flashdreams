# Crazy Robotaxi application API status

This document answers two questions:

1. Which parts of Crazy Robotaxi are running through the merged FlashDreams
   application API?
2. Which parts cannot yet be expressed by that API, why are they blocked, and
   how should they be connected when the API supports them?

It is not a comparison with an earlier branch revision.

## Ported to the application API

### Application discovery and launch

Crazy Robotaxi registers `crazy-robotaxi` in the
`flashdreams.applications` entry-point group and exposes `create_app()`. It is
launched by the shared CLI as `flashdreams-run crazy-robotaxi`; it has no custom
runner or launch loop.

API path: `create_application()` -> `IFlashDreamsApplication.init()` ->
`IFlashDreamsApplication.create_session()`.

### Native-window and WebRTC hosting

The stock local-window and WebRTC `IOFactory` implementations host the same
application session. Input acquisition and video delivery remain transport
concerns; the game and engine do not import either frontend.

API path: `IOFactory.create_input_handler()` and
`IOFactory.create_output_sink()` -> host-owned session driver.

### Keyboard and generic SDL driving input

The application declares the stock required `DRIVER_COMMAND` canonical
modality. Host keyboard input, and the generic SDL gamepad state supported by
the local-window handler, arrive as a `CanonicalInputWindow`. The application
translates that validated payload once into the engine's transport-neutral
`DriverCommand`.

API path: `IFlashDreamsApplication.input_schema` -> `InputHandler` ->
`CanonicalInputWindow` -> `CrazyRobotaxiApplicationSession.step()`.

### OmniDreams model initialization and autoregressive stepping

The application session constructs the public lower-level
`OmnidreamsRuntime`, creates its inference session from the scene's initial
frame and prompt, delegates chunk sizing through `StepRequirements`, and calls
the model once per application step. This uses the merged application session
lifecycle and the public OmniDreams integration; it does not use
`flashdreams.runtime.demo` from game or engine code.

API path: `IFlashDreamsApplicationSession.init()` ->
`next_step_requirements()` -> `step()` -> `close()`.

### Standalone game engine and game rules

Arcade vehicle simulation, route-valid fare selection, pickup/drop-off rules,
score, fare timers, global game time, and passenger trajectories run inside
`omnidreams_game_engine` and `crazy_robotaxi`. They do not call Interactive
Drive. Each model chunk advances this state from the canonical command before
HD-map conditioning is rendered.

API path: application `step()` -> `OmnidreamsGameInputProvider.prepare_step()`
-> game/simulation -> model `InferenceInput`.

Passengers are included now: waiting passengers contribute dynamic actor
trajectories to model conditioning, and collection removes them on the
completion frame.

### Causally aligned game-state output

The provider freezes the state corresponding to each conditioning frame. The
application attaches those frames under `application_frames` on the generated
`StepResult`, rejecting metadata-key collisions. Both supported presenters
receive the same video result and synchronized game payload.

API path: model `StepResult` -> application-owned metadata attachment ->
`OutputSink.write()`.

The payload transport is ported. Rendering that payload as a game HUD is not;
that separate blocker is documented under Minimal game HUD.

### Normal game completion

When game time expires, the application session stops advertising another
`StepRequirements`. The host completes the session and closes model, provider,
input, and output resources through their public lifecycle methods.

API path: `next_step_requirements() -> None` and
`IFlashDreamsApplicationSession.close()`.

## Currently blocked by the application API

These features must not be implemented with game-private transport or host
shims. Each returns to scope when its stated public API dependency exists.

### In-session restart

Current state: the engine can reset its simulator, game, renderer, and causal
aligner, but the hosted application does not expose restart.

Why blocked: `IFlashDreamsApplicationSession` has no reset/recreate operation,
and the upstream application adapter's `_ApplicationSession.reset()` rejects
every reset. A correct restart must atomically reset the OmniDreams cache,
simulation, game, canonical held-input state, renderer, output generation, and
causal aligner. Resetting only the game objects would desynchronize generated
video and game state.

How to port: consume a public host-coordinated session reset/recreate decision;
recreate the model cache and all per-session game state on the model thread,
clear the input handler, and invoke `OutputSink.begin_generation()` once for
the new generation.

### High-score persistence and name entry

Current state: `HighScoreStore` is implemented and tested with validated names,
file locking, atomic replacement, and top-ten ordering. It is not called by the
hosted session, and the game ends without silently inventing a player name.

Why blocked: `CanonicalInputSchema` has no optional/event modality, so the app
cannot declare intermittent name-submit/skip actions beside the required
level-triggered driving snapshot. Stock native and WebRTC application handlers
also expose no application-defined text input. Finally, stock presenters have
no application-owned view for name entry, validation errors, or the resulting
leaderboard.

How to port: declare optional text/action canonical input, deliver it through
both paired input handlers, submit it to the existing `HighScoreStore`, and
render validation and leaderboard state through the same typed presentation
contract required by the minimal HUD.

### Full wheel support

Current state: engine-owned wheel profiles, axis calibration, evdev reading,
keyboard conversion, analog conversion, disconnect fallback, and their unit
tests are retained. The stock local-window path can provide only its generic
SDL gamepad mapping.

Why blocked: output-target setup cannot compose an application-provided evdev
input handler with the stock local presenter. The WebRTC application input
handler converts keyboard actions only and has no schema-validated continuous
browser Gamepad snapshot path.

How to port: allow an application to contribute or compose a paired
`IOFactory`; feed calibrated evdev and browser Gamepad snapshots through the
same canonical driver modality; preserve disconnect fallback and input-schema
validation in both transports.

### Minimal game HUD

Current state: score, timer, fare, target, passenger, and session state are
present in synchronized `application_frames` metadata. No stock presenter
renders them.

Why blocked: the API has no `OutputSchema`, application-owned presenter/view
plug-in, or application-owned WebRTC resource bundle. `SessionInfo` describes
video geometry after session initialization, while `StepResult.metadata` is an
untyped delivery convention. The stock native and WebRTC sinks therefore
cannot discover or render the game-state payload as a supported contract.

How to port: add a typed output/presentation schema that output-target setup can
inspect, plus application-contributed views or paired `IOFactory`
implementations for local-window and WebRTC. Both views must render the same
payload without querying the game controller.

### Deterministic replay and headless output

Current state: FlashDreams ships MP4 and null output sinks, but those CLI modes
cannot run this input-driven application.

Why blocked: the CLI pairs MP4 and null output with `NullInputHandler`. Crazy
Robotaxi correctly requires `driver_command`, so canonical input validation
rejects the empty windows before a game step. The diagram's Replay/CI input
router is not selectable for an application.

How to port: let output-target setup select a scripted/replay `InputHandler`
paired with MP4 or null output. It must emit deterministic canonical commands
and time windows; replay logic must remain outside the game session.

### Application-level model Runtime reuse

Current state: `CrazyRobotaxiApplicationSession.init()` creates one
`OmnidreamsRuntime`, inference cache, provider, simulation, and game. This is
correct for the current single-session host and keeps CUDA initialization on
the model thread.

Why blocked: the merged application API exposes no long-lived application
Runtime lifecycle. `IFlashDreamsApplication.init()` is for command-line parsing
and runs before model-thread session initialization; the application contract
has no `destroy()`. Moving weights there would create unsafe thread ownership
and no reliable cleanup path.

How to port: when FlashDreams adds the discussed
`initialize()`/`create_session()`/`destroy()` model-runtime ABI, move model
weights and one-time CUDA allocations into it. Keep the inference cache,
provider, simulation, and game in each per-user session. Then delete the
session's `runtime_factory` field. Until that API exists, do not create a
branch-owned imitation.

## Upstream compatibility shims still in the execution path

Crazy Robotaxi owns no compatibility shims.

Every application launched by the current upstream `run_application()` passes
through `flashdreams/flashdreams/runtime/demo/application_runtime.py`. That
upstream file explicitly calls itself a temporary/hacky shim. It adapts the new
`IFlashDreamsApplicationSession` contract to the older `RuntimeHost`, session
drivers, `StepPipeline`, and `InferenceInput` contracts through:

- `_ApplicationRuntime`
- `_ApplicationSession`
- `_ApplicationInputSource`
- `_ApplicationInputProvider`
- `_ApplicationOutputEdges`

This branch neither changes nor calls those private classes directly. Removing
that shim is FlashDreams application-framework work. Crazy Robotaxi should need
no change if the public application, input-handler, and output-sink contracts
remain stable.

`IFlashDreamsApplication.createSession()` is also an upstream compatibility
alias for `create_session()`. Crazy Robotaxi implements and calls only
`create_session()` and does not depend on that alias.

Direct use of the public lower-level `OmnidreamsRuntime` is the current model
integration boundary, not a compatibility shim.

## Deliberately deferred work that is not an API finding

Alignment capture tooling, legacy MJPEG serving, and polished-HUD parity were
explicitly deferred for scope reasons. They are tracked in
[FEATURES.md](FEATURES.md), but they are not presented here as defects in the
application API. Polished-HUD work remains downstream of the API-blocked
minimal HUD.
