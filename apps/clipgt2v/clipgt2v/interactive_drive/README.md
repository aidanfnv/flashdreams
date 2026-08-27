# ClipGT2V driving support

This package contains model-neutral scene loading, vehicle simulation,
conditioning rasterization, presentation, and input handling used by the
`clipgt2v` and `interactive-drive` applications.

World-model construction is injected through application hooks. Pipeline
configs, checkpoints, scene download policy, and session implementations belong
in `integrations_v2/<model>/`; this package does not select a model.

Use the registered application entry points rather than importing an
integration directly:

```bash
uv run flashdreams-run-v2 clipgt2v --mode webrtc -- --scene scene.usdz
uv run flashdreams-run-v2 interactive-drive --mode webrtc
```
