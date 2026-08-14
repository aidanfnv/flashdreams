# Crazy Robotaxi

Crazy Robotaxi is a consumer application built on the standalone
`omnidreams_game_engine` and the new
`flashdreams.demo.IFlashDreamsApplication` API. It does not import or modify the
enterprise `omnidreams.interactive_drive` demo.

The complete implementation/deferred-work ledger is
[FEATURES.md](FEATURES.md). The remaining upstream compatibility shim and every
API-blocked feature are inventoried in [API_FINDINGS.md](API_FINDINGS.md).
Review both files before changing scope.

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
`--device`, and `--compile` / `--no-compile`.

The current public application host supplies its stock `driver_command`
keyboard/SDL-gamepad modality. Keyboard controls are `W`/`S` for throttle and
brake, `A`/`D` to steer, and Space for the stop/handbrake command. The API does
not yet expose the hooks needed for atomic restart, calibrated evdev/browser
wheel parity, name submission/high-score persistence, or application-owned HUD
views. Its MP4/null factories also provide empty input despite the required
driving modality, so deterministic replay/headless capture is blocked pending a
scripted input-router selection. Those items are explicitly deferred with exact
blockers and restoration tests in the documents above; none are hidden behind
compatibility code in this package.
