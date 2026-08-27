# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import clipgt2v.app as clipgt_app_module
import interactive_drive.app as app_module
import numpy as np
import pytest
import torch
from clipgt2v.app import (
    ClipGT2VApplication,
    ClipGT2VApplicationHooks,
    ClipGT2VModelLoop,
    ClipGT2VModelState,
    DriveTelemetry,
)
from interactive_drive import (
    DEFAULT_SCENE_FILENAME,
    DEFAULT_SCENE_REPO_ID,
    InteractiveDriveApplication,
    InteractiveDriveConfig,
    InteractiveDriveSession,
    InteractiveDriveUILoop,
    download_default_scene,
)

from flashdreams.runtime_v2.user_input_event import (
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _FakeUI:
    Cond_ = SimpleNamespace(once="once")

    def __init__(self) -> None:
        self.text_lines: list[str] = []
        self.images: list[str] = []

    @staticmethod
    def ImVec2(x: float, y: float) -> tuple[float, float]:
        return (x, y)

    def set_next_window_pos(self, position: Any, condition: Any) -> None:
        del position, condition

    def set_next_window_size(self, size: Any, condition: Any) -> None:
        del size, condition

    def begin(self, title: str) -> None:
        del title

    def end(self) -> None:
        pass

    def text(self, value: str) -> None:
        self.text_lines.append(value)

    def combo(self, label: str, index: int, options: list[str]) -> tuple[bool, int]:
        del label, options
        return False, index

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        del label
        return False, value

    def button(self, label: str) -> bool:
        del label
        return False

    def same_line(self) -> None:
        pass

    def progress_bar(self, fraction: float, size: Any) -> None:
        del fraction, size

    def image(
        self,
        key: str,
        pixels: Any,
        *,
        size: tuple[float, float],
    ) -> None:
        del pixels, size
        self.images.append(key)

    def separator(self) -> None:
        pass


def test_model_step_publishes_bev_channel_and_complete_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = SimpleNamespace(speed_mps=0.0, steer_rad=0.0)
    trajectory = SimpleNamespace(
        boundary_state_after_chunk=vehicle,
        timestamps_us=np.array([100], dtype=np.int64),
    )
    bev_frames = [
        np.full((2, 3, 3), fill_value=value, dtype=np.uint8) for value in (17, 29)
    ]
    chunk = SimpleNamespace(
        frames=[SimpleNamespace(bev_host_uint8=bev_frame) for bev_frame in bev_frames]
    )
    backend = SimpleNamespace(
        initial_chunk_frames=2,
        chunk_frames=2,
        render_first_chunk=lambda _: chunk,
    )
    config = SimpleNamespace(
        total_blocks=2,
        app=SimpleNamespace(
            chunk=SimpleNamespace(frame_interval_us=33_333),
            vehicle=object(),
        ),
    )
    physics_world: Any = object()
    state = ClipGT2VModelState(
        backend_factory=lambda _: backend,
        config=config,
        desc=SimpleNamespace(output_layout="tchw"),
        scene_loader=lambda *args: object(),
        scene=object(),
        vehicle=vehicle,
        backend=backend,
        physics_world=physics_world,
        view_mode="physx",
    )
    elapsed: list[float] = []
    trajectory_calls: list[dict[str, Any]] = []
    clock = iter((10.0, 10.123))
    monkeypatch.setattr(clipgt_app_module.time, "perf_counter", lambda: next(clock))

    def sample_trajectory(**kwargs: Any) -> Any:
        trajectory_calls.append(kwargs)
        return trajectory

    monkeypatch.setattr(clipgt_app_module, "sample_chunk_trajectory", sample_trajectory)
    monkeypatch.setattr(
        clipgt_app_module,
        "_frame_chunk_tensor",
        lambda frame_chunk, view_mode: torch.zeros((2, 3, 1, 1)),
    )
    monkeypatch.setattr(clipgt_app_module, "_telemetry_status", lambda *args: "ready")
    monkeypatch.setattr(
        ClipGT2VModelState,
        "_publish_drive_telemetry",
        lambda self, chunk, model_loop_ms: elapsed.append(model_loop_ms),
    )
    loop = ClipGT2VModelLoop()
    loop.state = state

    results = loop.step(0, UserInputEvents([]))

    assert len(results) == 2
    assert results[1].frame_count == 2
    assert tuple(results[1].output.shape) == (2, 3, 2, 3)
    assert results[1].output[:, 0, 0, 0].tolist() == [17, 29]
    assert elapsed == [pytest.approx(123.0)]
    assert trajectory_calls[0]["physics_world"] is physics_world
    assert trajectory_calls[0]["capture_physics_debug"] is True


def test_drive_telemetry_publishes_frame_chunk_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[DriveTelemetry] = []
    ui_state = SimpleNamespace(set_drive_telemetry=published.append)
    monkeypatch.setattr(
        clipgt_app_module,
        "invoke_async",
        lambda _loop, callback: callback(ui_state),
    )
    state = ClipGT2VModelState(
        backend_factory=lambda _: None,
        config=SimpleNamespace(
            app=SimpleNamespace(scene_path=tmp_path / "scene.usdz", variant="default")
        ),
        desc=SimpleNamespace(),
        scene_loader=lambda *args: object(),
        vehicle=SimpleNamespace(speed_mps=0.0, steer_rad=0.0),
        ui_loop=object(),
    )
    chunk = SimpleNamespace(
        frames=[
            SimpleNamespace(bev_host_uint8=None),
            SimpleNamespace(bev_host_uint8=None),
            SimpleNamespace(bev_host_uint8=None),
        ]
    )

    state._publish_drive_telemetry(chunk, model_loop_ms=12.5)

    assert len(published) == 1
    assert published[0].frames_in_chunk == 3


def test_interactive_drive_uses_regular_application_contract() -> None:
    app = InteractiveDriveApplication()
    assert app.session_desc().video_width == 1280
    assert app.session_desc().video_height == 704


def test_clipgt2v_no_ui_registers_only_the_model_loop(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = ClipGT2VApplication()
    app.init(["--scene", str(scene), "--backend", "raster", "--no-ui"])

    session = app.create_session(app.session_desc())
    session.init()

    assert isinstance(session.model_loop, ClipGT2VModelLoop)
    assert session._registered_ui_loop is None


def test_interactive_drive_resolves_default_scene_when_omitted(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[None] = []

    def resolve_default_scene() -> Path:
        calls.append(None)
        return scene

    app = InteractiveDriveApplication(default_scene_resolver=resolve_default_scene)
    app.init(["--backend", "raster", "--total-blocks", "0"])

    assert calls == [None]
    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_exposes_game_mode(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(scene), "--backend", "raster", "--game-mode"])

    assert app._config is not None
    assert app._config.app.game_mode is True
    assert app._config.app.vehicle.speed_limit_enabled is True
    assert app._config.app.vehicle.actor_collision_enabled is True
    assert app._config.app.vehicle.static_collision_enabled is True


def test_world_model_accepts_postprocess_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    monkeypatch.setattr(
        clipgt_app_module,
        "discover_postprocess_presets",
        lambda: {"example-preset": object()},
    )
    hooks = ClipGT2VApplicationHooks(
        backend_factory=lambda config: config,
    )
    app = InteractiveDriveApplication(
        hooks=hooks,
        default_scene_resolver=lambda: scene,
    )

    app.init(["--postprocess-preset", "example-preset"])

    assert app._config is not None
    assert isinstance(app._config, InteractiveDriveConfig)
    assert app._config.app.postprocess.preset == "example-preset"


def test_default_scene_uses_original_hugging_face_location(tmp_path: Path) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[dict[str, str]] = []

    def fake_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(scene)

    assert download_default_scene(fake_download) == scene
    assert calls == [
        {
            "repo_id": DEFAULT_SCENE_REPO_ID,
            "repo_type": "dataset",
            "filename": DEFAULT_SCENE_FILENAME,
        }
    ]


def test_standalone_application_downloads_default_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "default.usdz"
    scene.touch()
    calls: list[None] = []

    def fake_default_scene() -> Path:
        calls.append(None)
        return scene

    monkeypatch.setattr(app_module, "download_default_scene", fake_default_scene)
    app = InteractiveDriveApplication()

    app.init(["--backend", "raster", "--total-blocks", "0"])

    assert calls == [None]
    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_prefers_explicit_scene(tmp_path: Path) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()

    def unexpected_default_scene() -> Path:
        raise AssertionError("default scene should not be resolved")

    app = InteractiveDriveApplication(default_scene_resolver=unexpected_default_scene)
    app.init(["--scene", str(scene), "--backend", "raster"])

    assert app._config is not None
    assert app._config.app.scene_path == scene


def test_interactive_drive_owns_a_separate_session_and_ui_loop(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene), "--backend", "raster"])

    session = app.create_session(app.session_desc())
    assert isinstance(session, InteractiveDriveSession)
    assert app._config is not None
    assert app._config.app.bev.enabled is True
    assert app._config.app.bev.show_ego_car is True

    session.init()
    assert isinstance(session.ui_loop, InteractiveDriveUILoop)


def test_interactive_drive_hud_draws_imgui_controls_and_images(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene), "--backend", "raster"])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.ui_loop
    assert isinstance(loop, InteractiveDriveUILoop)
    pixel = np.zeros((4, 4, 4), dtype=np.uint8)
    loop.state.sprites = {
        "steering_wheel": pixel,
        "throttle_pressed": pixel,
        "throttle_unpressed": pixel,
        "brake_pressed": pixel,
        "brake_unpressed": pixel,
    }
    loop.state.set_drive_telemetry(
        DriveTelemetry(
            speed_mps=12.0,
            steering_rad=0.2,
            throttle=0.8,
            brake=0.0,
            reverse=False,
            blocks_generated=7,
            frames_in_chunk=13,
            scene_path=scene,
            variant="default",
            postprocess_enabled=True,
            input_source="wheel/gamepad",
            model_loop_ms=123.45,
            bev_frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )
    )
    ui = _FakeUI()

    loop.step_ui(ui, 0, UserInputEvents([]))

    assert "Block 7" in ui.text_lines
    assert "Input  wheel/gamepad" in ui.text_lines
    assert "Speed   26.8 mph" in ui.text_lines
    assert "Gear   D" in ui.text_lines
    assert "Steer  +0.20 rad" in ui.text_lines
    assert "frames_in_chunk: 13" in ui.text_lines
    assert "model_loop_ms: 123.5" in ui.text_lines
    assert ui.images == [
        "steering-wheel",
        "brake-pedal",
        "throttle-pedal",
        "bev-minimap",
    ]
    # Positive steering means left, so the HUD wheel rotates counterclockwise.
    assert loop.state.wheel_cache_angle == 36


def test_interactive_drive_view_button_cycles_all_three_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = tmp_path / "local.usdz"
    scene.touch()
    app = InteractiveDriveApplication()
    app.init(["--scene", str(scene), "--backend", "raster"])
    session = app.create_session(app.session_desc())
    session.init()
    loop = session.ui_loop
    assert isinstance(loop, InteractiveDriveUILoop)
    selected: list[str] = []
    model_state = SimpleNamespace(set_view_mode=selected.append)
    monkeypatch.setattr(
        app_module,
        "invoke_async",
        lambda _loop, callback: callback(model_state),
    )

    for expected in ("hdmap", "physx", "rgb"):
        loop._toggle_view()
        assert loop.state.view_mode == expected

    assert selected == ["hdmap", "physx", "rgb"]


def test_number_keys_select_rgb_hdmap_and_physx_views() -> None:
    state = ClipGT2VModelState(
        backend_factory=cast(Any, lambda _: None),
        config=cast(Any, SimpleNamespace()),
        desc=cast(Any, SimpleNamespace()),
        scene_loader=cast(Any, lambda *args: object()),
    )
    loop = ClipGT2VModelLoop()
    loop.state = state

    for key, expected in (("1", "rgb"), ("2", "hdmap"), ("3", "physx")):
        loop._apply_events(
            UserInputEvents(
                [
                    KeyboardUserInputEvent(
                        timestamp=np.uint64(0),
                        key=key,
                        state=KeyboardInputState.PRESSED,
                    )
                ]
            )
        )
        assert state.view_mode == expected
        assert key not in state.pressed_keys


def test_frame_view_selects_rgb_hdmap_and_physx_streams() -> None:
    hdmap = np.full((2, 3, 3), 11, dtype=np.uint8)
    rgb = np.full((2, 3, 3), 22, dtype=np.uint8)
    physx = np.full((2, 3, 3), 33, dtype=np.uint8)
    chunk: Any = SimpleNamespace(
        frames=[
            SimpleNamespace(
                rgb_host_uint8=hdmap,
                model_rgb_host_uint8=rgb,
                physx_rgb_host_uint8=physx,
            )
        ]
    )

    assert clipgt_app_module._frame_chunk_tensor(chunk, "rgb")[0, 0, 0, 0] == 22
    assert clipgt_app_module._frame_chunk_tensor(chunk, "hdmap")[0, 0, 0, 0] == 11
    assert clipgt_app_module._frame_chunk_tensor(chunk, "physx")[0, 0, 0, 0] == 33


def test_interactive_drive_discovers_scenes_and_weather_variants(
    tmp_path: Path,
) -> None:
    first_uuid = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
    second_uuid = "11111111-2222-3333-4444-555555555555"
    base = tmp_path / f"clipgt-{first_uuid}.usdz"
    rain = tmp_path / f"clipgt-{first_uuid}-rain.usdz"
    other = tmp_path / f"clipgt-{second_uuid}.usdz"
    for scene in (base, rain, other):
        scene.touch()
    app = InteractiveDriveApplication()

    app.init(["--scene", str(rain), "--backend", "raster"])

    assert len(app._interactive_scene_options) == 2
    selected = next(
        option for option in app._interactive_scene_options if option.path == base
    )
    assert selected.variants == ("default", "rain")
    assert app._config is not None
    assert app._config.app.scene_path == base
    assert app._config.app.variant == "rain"
