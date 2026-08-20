# Crazy Robotaxi

Crazy Robotaxi is a consumer application built on the standalone
`omnidreams_game_engine` and the latest `flashdreams.api_v2` /
`flashdreams.runtime_v2` I/O types. It does not import or modify the enterprise
`omnidreams.interactive_drive` demo.

The complete implementation/deferred-work ledger is
[FEATURES.md](FEATURES.md). Every remaining API-blocked feature and upstream
compatibility surface is inventoried in [API_FINDINGS.md](API_FINDINGS.md).
Review both files before changing scope.

The game implements the official V2 `IApplication` and `ISession` contracts,
consumes timestamped events, publishes `SessionDesc`, and returns a V2 result
subtype with synchronized game state. PR #490 still does not include V2
`ApplicationRunner`/CLI discovery, an OmniDreams model-session contract, or
concrete local/WebRTC client windows. Two explicitly named V1 adapters retain
the playable `flashdreams-run` path until those V2 pieces land. Every conversion
and deletion condition is documented in
[API_FINDINGS.md](API_FINDINGS.md#compatibility-surfaces-to-remove).

Native local window (default):

```bash
flashdreams-run crazy-robotaxi
```

WebRTC browser output:

```bash
flashdreams-run crazy-robotaxi --output webrtc --host 0.0.0.0 --port 8080
```

Application options include `--scene-path`, `--scene-dir`, `--scene-uuid`,
`--scene-variant`, `--camera-name`, `--prompt`, `--pixel-height`,
`--pixel-width`, `--fps`, `--total-blocks`, `--game-time-s`, `--game-seed`,
`--device`, `--model-preset standard|perf|native-perf`, and `--compile` /
`--no-compile`. The default model preset is `perf`. The performance presets
reuse integration-owned OmniDreams configuration; they do not import the
Interactive Drive manifest.

The V2 session directly understands the standard populated keyboard events;
the temporary host adapter converts its V1 `driver_command` keyboard/SDL
gamepad snapshot into a V2 normalized driving event. Keyboard controls are
`W`/`S` for throttle and brake, `A`/`D` to steer, and Space for stop/handbrake.
Concrete V2 input clients are still needed for the implemented standard reset
event, calibrated evdev/browser wheel parity, and name entry. Typed
presentation, natural game completion, replay/headless input, and WebRTC
geometry setup are also API-blocked. These items and the temporary
compatibility code are explicitly inventoried in the documents above.
