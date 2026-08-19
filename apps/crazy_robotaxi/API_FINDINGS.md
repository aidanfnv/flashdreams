# Crazy Robotaxi FlashDreams API findings

This document records only public FlashDreams API gaps exposed by Crazy
Robotaxi and compatibility surfaces that may need removal. Successfully ported
features and non-API follow-up work belong in [FEATURES.md](FEATURES.md).

Each feature below must remain deferred until its required public API exists;
it must not be implemented with a game-private transport or host shim.

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

How to finish: add a public edge-triggered canonical action and an application
bridge mapping from that action to a host-coordinated reset. The host must clear
held input and invoke `OutputSink.begin_generation()` exactly once after the
session and provider resets complete.

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

How to port: output-target setup must resolve WebRTC geometry from validated
`SessionInfo`, or expose generic host overrides that are checked against it,
before constructing the encoder and browser metadata. Until then, non-default
geometry requires programmatic `WebRTCApplicationServing` configuration and is
not claimed as a supported CLI path.

## Upstream compatibility surfaces

Crazy Robotaxi owns no compatibility shims.

Upstream removed the temporary
`flashdreams/runtime/demo/application_runtime.py` shim in commit `5d30407`.
Applications now run through the shared `flashdreams.demo.bridge` and
`run_demo_session` implementation. Crazy Robotaxi consumes only the public
application, canonical-input, session, and output contracts; it does not import
the bridge or `flashdreams.runtime.demo`.

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

Direct use of the public lower-level `OmnidreamsRuntime` is the current model
integration boundary, not a compatibility shim.
