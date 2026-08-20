# Crazy Robotaxi FlashDreams API findings

This document records only public FlashDreams API gaps exposed by Crazy
Robotaxi and compatibility surfaces that may need removal. Successfully ported
features and non-API follow-up work belong in [FEATURES.md](FEATURES.md).

Each feature below must remain deferred until its required public API exists;
it must not be implemented with a game-private transport or host shim.

## Remaining V2 platform gaps after #490

This branch was rebased onto upstream commit `abb468e`, which includes the V2
I/O types from #488 and the V2 application/session runner from #490.
The game-facing session has been moved to the installable
`flashdreams.api_v2`/`flashdreams.runtime_v2` types. Successfully ported V2
behavior is listed in [FEATURES.md](FEATURES.md). This section records the API
surface that is still missing.

V2 now contains `IApplication`, `ISession`, `InputSource`, `OutputSink`,
`IClientWindow`, timestamped input events, lifecycle reset/close events,
`SessionDesc`, a video-only `StepResult`, and the two-thread `run_session` loop.
It still does not contain application discovery or `flashdreams-run`
orchestration, model-session construction, or local-window/WebRTC/replay
implementations of `IClientWindow`.

The remaining V2 host gate is:

- an `ApplicationRunner` connected to `flashdreams-run` discovery and session
  creation;
- concrete local-window and WebRTC client windows that deliver populated
  keyboard/gamepad events and consume generated video;
- a typed application-output extension for synchronized HUD/game state and
  output time windows, including an output path from `ISession.step_ui`;
- a session-completion signal and model-driven step requirements; and
- a public model-session boundary able to initialize and step OmniDreams.

Until that gate exists, the V2 core is exposed to the existing playable host by
the explicitly named V1 adapters inventoried at the end of this document. The
findings below record the feature impact of each missing V2 contract.

## Required API work

### In-session restart

Current state: the engine and application session can atomically reset the
OmniDreams cache, simulator, game, renderer timeline, and causal aligner. A
completed WebRTC game can start another generation on the same peer.

Why still blocked: the public application bridge does not map a canonical
application input to the runtime's `ControlDecision(reset=True)`, and the stock
`driver_command` modality has no reset action. Therefore no stock local-window
or WebRTC input can request the implemented reset while a game is active. A
private provider/model-adapter hook would cross the application boundary and is
not used.

V2 status: #490 adds `ResetUserInputEventData`; `run_session` calls the game
session's reset, restarts at step zero, and drops queued or in-flight results
from the abandoned generation. The remaining blocker is concrete input
delivery: no V2 local/WebRTC window emits the reset event, and the V1
`driver_command` adapter has no reset field.

How to finish: make concrete V2 local/WebRTC windows emit the standard reset
event. Verify their held-key policy against `run_session`'s documented behavior
that the entire reset batch reaches the first new-generation step.

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

V2 status: `ISession.step_ui` supplies an asynchronous UI-rate input hook, but
it cannot produce output. V2 still ships no text event, input
schema/capability declaration, application event dispatch, or client-window UI
implementation. Its base `StepResult` cannot carry a typed leaderboard or
validation state back to a presenter.

How to port: add public V2 text/action event types and application dispatch,
deliver them through both client windows, submit them to the existing
`HighScoreStore`, and render validation and leaderboard state through the same
typed presentation contract required by the minimal HUD.

### Full wheel support

Current state: engine-owned wheel profiles, axis calibration, evdev reading,
keyboard conversion, analog conversion, disconnect fallback, and their unit
tests are retained. The stock local-window path can provide only its generic
SDL gamepad mapping.

Why blocked: output-target setup cannot compose an application-provided evdev
input handler with the stock local presenter. The WebRTC application input
handler converts keyboard actions only and has no schema-validated continuous
browser Gamepad snapshot path.

V2 status: #490 populates standard keyboard edges. The engine consumes them and
defines game-owned analog-state and normalized driver-command event data.
`GamepadUserInputEventData` and `GameWheelUserInputEventData` remain empty stubs
with no axes, buttons, connection state, capability schema, or native/browser
producer. There is also no V2 converter registry or priority contract for
selecting analog input over keyboard input.

How to port: implement V2 native and browser input sources plus a public
converter-selection contract; feed calibrated evdev and browser Gamepad
snapshots into the same normalized driver event; preserve disconnect fallback
and capability validation in both transports.

### Minimal game HUD

Current state: score, timer, fare, target, passenger, and session state are
present in synchronized `application_frames` metadata. No stock presenter
renders them.

Why blocked: the API has no `OutputSchema`, application-owned presenter/view
plug-in, or application-owned WebRTC resource bundle. `SessionInfo` describes
video geometry after session initialization, while `StepResult.metadata` is an
untyped delivery convention. The stock native and WebRTC sinks therefore
cannot discover or render the game-state payload as a supported contract.

V2 status: `CrazyRobotaxiStepResult` temporarily subclasses V2 `StepResult` to
carry synchronized `application_frames`, model metadata, and an output time
window. The base type still has only a video tensor and numeric `metrics`.
`OutputSink` has no schema negotiation or application-view extension, so a
generic sink cannot discover or render the added fields.

How to port: add a typed V2 output/presentation schema that client-window setup
can inspect, plus application-contributed local-window and WebRTC views. Both
views must render the same payload without querying the game controller.

### Deterministic replay and headless output

Current state: FlashDreams ships MP4 and null output sinks, but those CLI modes
cannot run this input-driven application.

Why blocked: the CLI pairs MP4 and null output with `NullInputHandler`. Crazy
Robotaxi correctly requires `driver_command`, so canonical input validation
rejects the empty windows before a game step. The diagram's Replay/CI input
router is not selectable for an application.

V2 status: the game session can be driven directly with deterministic
`UserInputEvents`, but V2 has no replay `InputSource`, deterministic step-window
contract, application runner, or output-target selector that can pair a source
with MP4/null output.

How to port: let V2 client-window setup select a scripted/replay `InputSource`
paired with MP4 or null output. It must emit deterministic timestamped driver
events and step windows; replay logic must remain outside the game session.

### Application completion and model step requirements

Current state: Crazy Robotaxi knows when its game timer ends, and the
lower-level OmniDreams session declares each chunk's step index and input frame
count. The V1 host adapter can stop when `next_step_requirements()` returns
`None`.

Why blocked: V2 `ISession.step()` can return only `StepResult`. It has no
completion result, next-step requirement, or variable frame-count request.
`run_session(steps=None)` runs until the client sends `CloseUserInputEventData`,
while a fixed `steps` count is runtime-owned. A V2 game therefore cannot end the
run when its timer expires without raising, and the runner cannot ask the
OmniDreams session how many conditioning frames the next model step needs.

How to port: add an application-completion signal and a public next-step
requirements contract, or define an equivalent result/control protocol that
lets `run_session` stop cleanly and schedule variable model chunks.

### WebRTC output geometry

Current state: application sessions publish their actual FPS, width, and height
through `SessionInfo`. The stock WebRTC application launcher independently
constructs `WebRTCApplicationServing` with 1280x720 at 30 FPS. Crazy Robotaxi
defaults to 1280x704, and the Interactive Drive performance manifest uses
1168x640.

Why blocked: the WebRTC application path does not derive encoder geometry from
the initialized session, and `flashdreams-run` exposes no WebRTC width/height
override. NVENC setup and browser chunk metadata therefore use host defaults
instead of the application's declared dimensions.

V2 status: the game session publishes its resolved `SessionDesc`, and #490
passes that description to `OutputSink.open`. The abstract geometry contract is
there; no V2 WebRTC client window exists to apply it to encoder and browser
metadata.

How to port: V2 client-window setup must resolve WebRTC geometry from the
application session's validated `SessionDesc`, or validate generic host
overrides against it, before constructing encoder and browser metadata. Until
then, non-default geometry requires the V1 programmatic serving configuration
and is not claimed as a supported CLI path.

## Compatibility surfaces to remove

Crazy Robotaxi currently owns two deliberately named V1 host adapters in
`crazy_robotaxi.app`:

- `CrazyRobotaxiV1ApplicationAdapter` implements
  `IFlashDreamsApplication` solely so `flashdreams-run` and application entry
  point discovery can create the V2 application. It also translates V2 warmup
  events into `ApplicationWarmupSessionInputs`.
- `CrazyRobotaxiV1SessionAdapter` translates V1 `CanonicalInputWindow` driver
  snapshots into V2 `UserInputEvents`, maps V2 `SessionDesc` back to
  `SessionInfo`, exposes the current `StepRequirements`, and converts
  `CrazyRobotaxiStepResult` back to the V1 result plus metadata expected by the
  stock local-window/WebRTC sinks.

Delete both adapters and point the entry point at `CrazyRobotaxiV2Application`
once V2 supplies `ApplicationRunner`, `flashdreams-run` discovery, and concrete
client windows. V2 application/session lifecycle itself is already implemented
and used by the core. The adapter classes are tested by name so they cannot
silently become an untracked compatibility layer.

The V2 core also retains one lower-level V1 model boundary. It constructs
`OmnidreamsRuntime` with `InferenceConfig`; the provider produces
`InferenceInput`; and the model session returns V1 `StepRequirements` and
`StepResult`. V2 has application/session reset and close, but no lower-level
model runtime/session equivalents for runtime creation, initial versus per-step
model input, or model-driven next-step requirements. Replace this boundary when
those model contracts land; do not add a second model adapter around it.

`CrazyRobotaxiStepResult` is a temporary V2 extension rather than a V1 adapter,
but its `application_frames`, `model_metadata`, and `output_window_us` fields
must be removed or reduced once the V2 base result gains typed application
output and output timing.

## Related UI work

ArielG-NV's `api-refactor/ui-async-thread` branch is based on the older V1
application stack, so its framework types are not directly reusable in this V2
port. It does demonstrate useful behavior concretely: a presentation cadence
independent of model generation, a thread-safe mailbox separating level state
from drained edge events, raw keyboard/text/pointer transport, compositing on a
presentation thread, and generation-aware queue flushing.

PR #490 already adopts the two-thread cadence, bounded presentation queue, and
generation-aware result discarding in V2 `run_session`. Crazy Robotaxi now uses
that runner in its lifecycle-reset test and consumes the standard populated V2
keyboard edges. The remaining reusable ideas need V2-native contracts:
populated text/pointer events and client-window transport, a concurrency-safe
level/edge UI state handoff, and an output path from `step_ui` for asynchronously
rendered HUD frames. Until those land, copying the V1
`ServerUI`/`AsyncPresentationCoordinator` types would create another temporary
framework rather than finish the V2 port.

## Upstream compatibility surfaces

Upstream removed the temporary
`flashdreams/runtime/demo/application_runtime.py` shim in commit `5d30407`.
V1 applications now run through the shared `flashdreams.demo.bridge` and
`run_demo_session` implementation. Crazy Robotaxi's V1 host adapters consume
only the public V1 application, canonical-input, session, and result contracts;
the V2 core consumes only public V2 contracts. Neither imports the bridge or
`flashdreams.runtime.demo`.

`flashdreams.demo.bridge.ApplicationRuntime` and `ApplicationSession` are the
current upstream adapters from the public application contract to
`RuntimeHost`; upstream does not label them compatibility shims. Crazy Robotaxi
executes through them indirectly via `run_application()` but neither imports
nor replaces them.

The WebRTC manager still contains `_LegacyWebRTCRuntimeAdapter` for older
demos. The application path supplies its shared `RuntimeHost`, adapter, spec,
and scenario, so Crazy Robotaxi does not execute through that legacy adapter.

`IFlashDreamsApplication.createSession()` is also an upstream compatibility
alias for `create_session()`. Crazy Robotaxi implements and calls only
`create_session()` and does not depend on that alias.

The merged `flashdreams.api_v2` and `flashdreams.runtime_v2` packages are a
parallel WIP API surface, not compatibility shims. They do not replace or adapt
the current application bridge.
