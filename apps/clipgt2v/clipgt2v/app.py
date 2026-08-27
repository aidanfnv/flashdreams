# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HD-map-conditioned video driving with native v2 and SlangPy UI loops."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.plugins.registry import discover_postprocess_presets
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    GamepadUserInputEvent,
    GameWheelUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from torch import Tensor

from clipgt2v.interactive_drive.backends.base import RenderBackend
from clipgt2v.interactive_drive.backends.raster import RasterRenderBackend
from clipgt2v.interactive_drive.config import (
    AppConfig,
    BevConfig,
    ChunkConfig,
    RasterConfig,
    VehicleConfig,
)
from clipgt2v.interactive_drive.input.keyboard import command_from_snapshot
from clipgt2v.interactive_drive.scene_loader import load_scene_bundle
from clipgt2v.interactive_drive.simulation.ego_vehicle_kinematics import (
    build_ground_snapper,
    sample_chunk_trajectory,
    state_from_initial_pose,
)
from clipgt2v.interactive_drive.simulation.game_physics import GamePhysicsWorld
from clipgt2v.interactive_drive.types import (
    ControlSnapshot,
    DriverCommand,
    FrameChunk,
    SceneBundle,
    VehicleState,
)

BackendFactory = Callable[[AppConfig], RenderBackend]
SceneLoader = Callable[..., SceneBundle]
SceneResolver = Callable[[], Path]
ManifestLoader = Callable[[Path], Any]
ManifestResolver = Callable[[str | Path], Path]
ViewMode = Literal["rgb", "hdmap", "physx"]

_VIEW_MODES: tuple[ViewMode, ...] = ("rgb", "hdmap", "physx")
_VIEW_MODE_KEYS: dict[str, ViewMode] = dict(zip(("1", "2", "3"), _VIEW_MODES))


@dataclass(frozen=True, slots=True)
class ClipGT2VApplicationDefaults:
    """Defaults supplied by a scene-driving model integration."""

    title: str = "Scene Drive"
    """Window title."""

    slug: str = "scene-drive"
    """Application slug shown in parser diagnostics."""

    backend: Literal["raster", "world_model"] = "raster"
    """Backend selected when ``--backend`` is omitted."""

    total_blocks: int = 60
    """Default number of generated blocks."""


@dataclass(frozen=True, slots=True)
class ClipGT2VApplicationHooks:
    """Model-owned callables consumed by the scene-driving application."""

    backend_factory: BackendFactory
    """Create the selected render backend on the model thread."""

    manifest_loader: ManifestLoader | None = None
    """Load model-backend metadata; ``None`` disables model manifests."""

    manifest_resolver: ManifestResolver | None = None
    """Resolve model manifest names to local paths."""


@dataclass(frozen=True, slots=True)
class ClipGT2VConfig:
    app: AppConfig
    total_blocks: int
    view_mode: ViewMode
    no_ui: bool = False


@dataclass(frozen=True, slots=True)
class DriveTelemetry:
    """Thread-safe driving values published to an application UI loop."""

    speed_mps: float
    steering_rad: float
    throttle: float
    brake: float
    reverse: bool
    blocks_generated: int
    frames_in_chunk: int
    scene_path: Path
    variant: str
    postprocess_enabled: bool
    input_source: str
    model_loop_ms: float
    bev_frame: Any | None = None


@dataclass(slots=True)
class ClipGT2VModelState:
    backend_factory: BackendFactory
    config: ClipGT2VConfig
    desc: SessionDesc
    scene_loader: SceneLoader
    scene: SceneBundle | None = None
    vehicle: VehicleState | None = None
    ground_snapper: Any | None = None
    next_timestamp_us: int = 0
    blocks_generated: int = 0
    first_chunk: bool = True
    pressed_keys: set[str] = field(default_factory=set)
    controller_command: DriverCommand | None = None
    last_command: DriverCommand = field(default_factory=DriverCommand)
    view_mode: ViewMode = "rgb"
    postprocess_enabled: bool = True
    pending_prompt: str | None = None
    reset_pending: bool = False
    ui_loop: IUILoop[Any] | None = None
    backend: RenderBackend | None = None
    physics_world: GamePhysicsWorld | None = None

    def restart(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty.")
        self.pending_prompt = prompt
        self.reset_pending = True
        self._notify("Restart queued on the model loop.")

    def set_view_mode(self, view_mode: ViewMode) -> None:
        self.view_mode = view_mode
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, view_mode=view_mode: state.set_view_mode(view_mode),
            )

    def select_scene(self, scene_path: Path, variant: str) -> None:
        """Queue a fresh rollout for a scene and weather variant."""
        self.config = replace(
            self.config,
            app=replace(
                self.config.app,
                scene_path=Path(scene_path),
                variant=variant,
            ),
        )
        self.reset_pending = True
        self._notify(f"Scene change queued: {Path(scene_path).stem} ({variant}).")

    def select_variant(self, variant: str) -> None:
        """Queue a fresh rollout using another variant of the current scene."""
        self.select_scene(self.config.app.scene_path, variant)

    def set_postprocess_enabled(self, enabled: bool) -> None:
        """Toggle generated-video post-processing without rebuilding the model."""
        self.postprocess_enabled = bool(enabled)
        if self.backend is not None:
            self.backend.set_postprocess_enabled(self.postprocess_enabled)
        self._notify(
            "Post-processing enabled."
            if self.postprocess_enabled
            else "Post-processing disabled."
        )

    def _notify(self, status: str) -> None:
        if self.ui_loop is not None:
            invoke_async(
                self.ui_loop,
                lambda state, status=status: state.set_status(status),
            )

    def _publish_drive_telemetry(self, chunk: FrameChunk, model_loop_ms: float) -> None:
        """Send controls, vehicle state, chunk metrics, and BEV to the UI."""
        if self.ui_loop is None or self.vehicle is None:
            return
        bev_frame = chunk.frames[-1].bev_host_uint8 if chunk.frames else None
        command = self.last_command
        telemetry = DriveTelemetry(
            speed_mps=self.vehicle.speed_mps,
            steering_rad=self.vehicle.steer_rad,
            throttle=command.throttle,
            brake=command.brake,
            reverse=command.reverse or self.vehicle.speed_mps < -0.01,
            blocks_generated=self.blocks_generated,
            frames_in_chunk=len(chunk.frames),
            scene_path=self.config.app.scene_path,
            variant=self.config.app.variant,
            postprocess_enabled=self.postprocess_enabled,
            input_source=(
                "wheel/gamepad"
                if self.controller_command is not None
                else "keyboard"
                if self.pressed_keys
                else "idle"
            ),
            model_loop_ms=model_loop_ms,
            bev_frame=bev_frame,
        )
        invoke_async(
            self.ui_loop,
            lambda state, telemetry=telemetry: state.set_drive_telemetry(telemetry),
        )


class ClipGT2VModelLoop(IModelLoop[ClipGT2VModelState]):
    """Own scene state, simulation, model cache, and backend execution."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        step_started_at = time.perf_counter()
        state = self.state
        self._apply_events(events)
        if state.scene is None or state.reset_pending:
            self._initialize_rollout()
        assert state.scene is not None
        assert state.vehicle is not None
        assert state.backend is not None
        chunk_size = (
            state.backend.initial_chunk_frames
            if state.first_chunk
            else state.backend.chunk_frames
        )
        command = self._command()
        state.last_command = command
        trajectory = sample_chunk_trajectory(
            start_state=state.vehicle,
            start_timestamp_us=state.next_timestamp_us,
            command=command,
            chunk_size=chunk_size,
            chunk_config=state.config.app.chunk,
            vehicle_config=state.config.app.vehicle,
            ground_snapper=state.ground_snapper,
            physics_world=state.physics_world,
            capture_physics_debug=state.view_mode == "physx",
        )
        chunk = (
            state.backend.render_first_chunk(trajectory)
            if state.first_chunk
            else state.backend.render_next_chunk(trajectory)
        )
        state.vehicle = trajectory.boundary_state_after_chunk
        state.next_timestamp_us = int(
            trajectory.timestamps_us[-1] + state.config.app.chunk.frame_interval_us
        )
        state.first_chunk = False
        state.blocks_generated += 1
        output = _frame_chunk_tensor(chunk, state.view_mode)
        state._notify(
            "Rollout complete."
            if self.is_finished()
            else _telemetry_status(state.vehicle, state.blocks_generated)
        )
        results = [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=int(output.shape[0]),
                output_layout=state.desc.output_layout,
                metrics={},
            )
        ]
        bev_output = self._bev_chunk_tensor(chunk)
        if bev_output is not None:
            results.append(
                StepResult(
                    step_index=step_index,
                    output=bev_output,
                    frame_count=int(bev_output.shape[0]),
                    output_layout=state.desc.output_layout,
                    metrics={},
                )
            )
            if state.ui_loop is not None:
                telemetry_frame = self._PresentedBevFrame(
                    ui_loop=state.ui_loop,
                    fallback=chunk.frames[-1].bev_host_uint8,
                )
                chunk = replace(
                    chunk,
                    frames=(
                        *chunk.frames[:-1],
                        replace(chunk.frames[-1], bev_host_uint8=telemetry_frame),
                    ),
                )
        model_loop_ms = (time.perf_counter() - step_started_at) * 1000.0
        state._publish_drive_telemetry(chunk, model_loop_ms)
        return results

    @dataclass(frozen=True, slots=True)
    class _PresentedBevFrame:
        """Resolve the BEV image at the UI's current presentation frame."""

        ui_loop: Any
        fallback: Any

        def to_numpy(self) -> Any:
            try:
                frame = self.ui_loop.presented_model_frame(1)
            except IndexError:
                frame = None
            if frame is None:
                return self.fallback
            return frame.permute(1, 2, 0).detach().cpu().numpy()

    @staticmethod
    def _bev_chunk_tensor(chunk: FrameChunk) -> Tensor | None:
        """Build a BEV channel aligned one-to-one with the emitted video frames."""
        bev_values = [frame.bev_host_uint8 for frame in chunk.frames]
        if not any(value is not None for value in bev_values):
            return None
        if any(value is None for value in bev_values):
            raise ValueError(
                "The render backend must provide one BEV frame per emitted frame."
            )

        frames: list[Tensor] = []
        for value in bev_values:
            array = np.asarray(value)
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected HWC BEV frame, received shape {tuple(tensor.shape)}"
                )
            frames.append(tensor.permute(2, 0, 1))
        return torch.stack(frames)

    def is_finished(self) -> bool:
        total = self.state.config.total_blocks
        return total > 0 and self.state.blocks_generated >= total

    def reset(self) -> None:
        self.state.reset_pending = True

    def close(self) -> None:
        if self.state.backend is not None:
            self.state.backend.close()
            self.state.backend = None
        if self.state.physics_world is not None:
            self.state.physics_world.close()
            self.state.physics_world = None
        self.state.scene = None

    def _initialize_rollout(self) -> None:
        state = self.state
        app_config = state.config.app
        prompt = state.pending_prompt
        state._notify("Loading scene conditioning on the model loop...")
        if state.backend is None:
            state.backend = state.backend_factory(app_config)
        backend = state.backend
        scene = state.scene_loader(
            app_config.scene_path,
            app_config.camera_name,
            app_config.variant,
            prompt if prompt is not None else app_config.prompt_override,
            app_config.raster,
        )
        if state.scene is None:
            backend.warmup_model()
        else:
            backend.reset_scene_conditioning()
        backend.load_scene(scene)
        backend.set_postprocess_enabled(state.postprocess_enabled)
        state.scene = scene
        state.vehicle = state_from_initial_pose(
            scene.initial_rig_to_world,
            scene.initial_yaw_rad,
            scene.initial_speed_mps,
        )
        state.ground_snapper = build_ground_snapper(scene)
        if state.physics_world is not None:
            state.physics_world.close()
        state.physics_world = GamePhysicsWorld(scene, app_config.vehicle)
        state.next_timestamp_us = scene.initial_timestamp_us
        state.blocks_generated = 0
        state.first_chunk = True
        state.reset_pending = False
        state.pending_prompt = None

    def _apply_events(self, events: UserInputEvents) -> None:
        state = self.state
        for event in events.get_events():
            if isinstance(event, KeyboardUserInputEvent):
                view_mode = _VIEW_MODE_KEYS.get(event.key.strip().lower())
                if view_mode is not None:
                    if event.state is KeyboardInputState.PRESSED:
                        state.set_view_mode(view_mode)
                    continue
                key = _normalize_drive_key(event.key)
                if key is None:
                    continue
                if event.state is KeyboardInputState.PRESSED:
                    state.pressed_keys.add(key)
                else:
                    state.pressed_keys.discard(key)
            elif isinstance(event, GameWheelUserInputEvent):
                state.controller_command = (
                    None
                    if event.action == "disconnected"
                    else DriverCommand(
                        throttle=event.throttle,
                        brake=event.brake,
                        steer=-event.steering,
                        steer_is_direct=True,
                        manual_control=True,
                    )
                )
            elif isinstance(event, GamepadUserInputEvent):
                state.controller_command = _gamepad_command(event)

    def _command(self) -> DriverCommand:
        if self.state.controller_command is not None:
            return self.state.controller_command
        return command_from_snapshot(ControlSnapshot(pressed=self.state.pressed_keys))


@dataclass(slots=True)
class ClipGT2VUIState:
    model_loop: IModelLoop[ClipGT2VModelState]
    title: str
    prompt: str
    status: str = "W/S accelerate, A/D steer; gamepads and wheels are supported."
    view_mode: ViewMode = "rgb"
    prompt_widget: Any | None = field(default=None, init=False, repr=False)
    view_widget: Any | None = field(default=None, init=False, repr=False)
    status_widget: Any | None = field(default=None, init=False, repr=False)

    def set_status(self, status: str) -> None:
        self.status = status
        if self.status_widget is not None:
            self.status_widget.text = status

    def set_view_mode(self, view_mode: ViewMode) -> None:
        self.view_mode = view_mode

    def set_drive_telemetry(self, telemetry: DriveTelemetry) -> None:
        """Accept model telemetry; the compact ClipGT2V UI only shows status."""
        del telemetry


class ClipGT2VUILoop(SlangPyUILoop[ClipGT2VUIState]):
    """Own retained SlangPy controls and composite model output."""

    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        del step_index, events
        if self.state.prompt_widget is None:
            window = ui.Window(
                ui.screen, self.state.title, position=(16, 16), size=(600, 265)
            )
            self.state.prompt_widget = ui.InputText(
                window, "Prompt", self.state.prompt, self._set_prompt, multi_line=True
            )
            self.state.view_widget = ui.InputInt(
                window,
                "View (0=RGB, 1=HD map, 2=PhysX)",
                _VIEW_MODES.index(self.state.view_mode),
                self._set_view,
            )
            ui.Button(window, "Restart rollout", self._restart)
            self.state.status_widget = ui.Text(window, self.state.status)
        return self.presented_model_frame()

    def _set_prompt(self, value: str) -> None:
        self.state.prompt = value

    def _set_view(self, value: int) -> None:
        view = _VIEW_MODES[max(0, min(int(value), len(_VIEW_MODES) - 1))]
        self.state.set_view_mode(view)
        invoke_async(
            self.state.model_loop,
            lambda state, view=view: state.set_view_mode(view),
        )

    def _restart(self) -> None:
        prompt = self.state.prompt.strip()
        if not prompt:
            self.state.set_status("Enter a prompt before restarting.")
            return
        self.state.set_status("Restart queued.")
        invoke_async(
            self.state.model_loop,
            lambda state, prompt=prompt: state.restart(prompt),
        )


class ClipGT2VSession(ISession):
    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        config: ClipGT2VConfig,
        desc: SessionDesc,
        scene_loader: SceneLoader,
        title: str,
    ) -> None:
        self._backend_factory = backend_factory
        self._config = config
        self._desc = desc
        self._scene_loader = scene_loader
        self._title = title

    @property
    def session_desc(self) -> SessionDesc:
        return self._desc

    def init(self) -> None:
        model_state = ClipGT2VModelState(
            backend_factory=self._backend_factory,
            config=self._config,
            desc=self._desc,
            scene_loader=self._scene_loader,
            view_mode=self._config.view_mode,
            postprocess_enabled=self._config.app.postprocess.is_enabled(),
        )
        model_loop = self.register_model_loop(ClipGT2VModelLoop, state=model_state)
        if self._config.no_ui:
            return
        ui_loop = self.register_ui_loop(
            ClipGT2VUILoop,
            state=ClipGT2VUIState(
                model_loop=model_loop,
                title=self._title,
                prompt=self._config.app.prompt_override or "",
                view_mode=self._config.view_mode,
            ),
            width=self._desc.video_width,
            height=self._desc.video_height,
        )
        model_state.ui_loop = ui_loop


class ClipGT2VApplication(IApplication):
    """Create native v2 ClipGT2V sessions without owning a client window."""

    def __init__(
        self,
        *,
        defaults: ClipGT2VApplicationDefaults | None = None,
        hooks: ClipGT2VApplicationHooks | None = None,
        backend_factory: BackendFactory | None = None,
        scene_loader: SceneLoader = load_scene_bundle,
        default_scene_resolver: SceneResolver | None = None,
    ) -> None:
        defaults = defaults or ClipGT2VApplicationDefaults()
        hooks = hooks or ClipGT2VApplicationHooks(backend_factory=_build_backend)
        self._title = defaults.title
        self._slug = defaults.slug
        self._default_backend = defaults.backend
        self._default_blocks = defaults.total_blocks
        self._hooks = hooks
        self._backend_factory = backend_factory or hooks.backend_factory
        self._scene_loader = scene_loader
        self._default_scene_resolver = default_scene_resolver
        self._config: ClipGT2VConfig | None = None
        self._desc = SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            frames_per_second_for_ui=60,
            frames_per_second_for_step=30,
            video_width=1280,
            video_height=704,
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(prog=f"flashdreams-run-v2 {self._slug} --")
        parser.add_argument(
            "--scene",
            type=Path,
            required=self._default_scene_resolver is None,
            help=(
                "Local USDZ scene. If omitted, download the default scene from "
                "Hugging Face."
                if self._default_scene_resolver is not None
                else "Local USDZ scene."
            ),
        )
        parser.add_argument(
            "--backend",
            choices=("raster", "world_model"),
            default=self._default_backend,
        )
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--prompt")
        parser.add_argument("--camera", default="camera_front_wide_120fov")
        parser.add_argument("--variant", default="default")
        parser.add_argument("--total-blocks", type=int, default=self._default_blocks)
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--width", type=int, default=1280)
        parser.add_argument("--height", type=int, default=704)
        parser.add_argument("--view", choices=_VIEW_MODES, default="rgb")
        parser.add_argument(
            "--no-ui",
            action="store_true",
            help="Disable the application UI and present model output directly.",
        )
        parser.add_argument(
            "--game-mode",
            action="store_true",
            help=("Enable the vehicle speed limit and actor/static collisions."),
        )
        parser.add_argument(
            "--postprocess-preset",
            default="",
            choices=tuple(sorted(discover_postprocess_presets())),
            help=(
                "Video post-processing preset for generated world-model frames. "
                "A configured preset starts enabled and can be toggled in the HUD."
            ),
        )
        args = parser.parse_args(list(commandline_args))
        scene = args.scene
        if scene is None:
            assert self._default_scene_resolver is not None
            scene = self._default_scene_resolver()
        if not scene.is_file():
            raise FileNotFoundError(scene)
        if args.total_blocks < 0:
            raise ValueError("--total-blocks must be >= 0 (0 means unbounded).")
        if args.fps <= 0 or args.width <= 0 or args.height <= 0:
            raise ValueError("--fps, --width, and --height must be > 0.")
        manifest = args.manifest
        if args.backend == "world_model" and manifest is None:
            if self._hooks.manifest_resolver is None:
                raise ValueError(
                    "The selected model backend does not provide a manifest resolver."
                )
            manifest = self._hooks.manifest_resolver("example_world_model.yaml")
        if manifest is not None and not manifest.is_file():
            raise FileNotFoundError(manifest)
        chunk = ChunkConfig(fps=args.fps)
        raster = RasterConfig(width=args.width, height=args.height)
        app_config = AppConfig(
            scene_path=scene,
            backend=args.backend,
            game_mode=args.game_mode,
            camera_name=args.camera,
            variant=args.variant,
            prompt_override=args.prompt,
            manifest_path=manifest,
            chunk=chunk,
            raster=raster,
            postprocess=VideoPostprocessChainConfig(
                preset=args.postprocess_preset,
            ),
            bev=BevConfig(enabled=False),
            vehicle=VehicleConfig(),
        )
        if args.backend == "world_model":
            if self._hooks.manifest_loader is None:
                raise ValueError(
                    "The selected model backend does not provide a manifest loader."
                )
            assert manifest is not None
            manifest_config = self._hooks.manifest_loader(manifest)
            app_config = replace(
                app_config,
                chunk=replace(
                    chunk,
                    fps=manifest_config.fps,
                    chunk_frames=manifest_config.num_frames_per_block,
                ),
                raster=replace(
                    raster,
                    width=manifest_config.resolution_wh[0],
                    height=manifest_config.resolution_wh[1],
                ),
            )
        self._config = ClipGT2VConfig(
            app=app_config,
            total_blocks=args.total_blocks,
            view_mode=args.view,
            no_ui=args.no_ui,
        )
        self._desc = replace(
            self._desc,
            frames_per_second_for_step=app_config.chunk.fps,
            video_width=app_config.raster.width,
            video_height=app_config.raster.height,
        )

    def session_desc(self) -> SessionDesc:
        return self._desc

    def create_session(self, session_desc: SessionDesc) -> ISession:
        if self._config is None:
            raise RuntimeError("init() must run before create_session().")
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("ClipGT2V requires tchw output.")
        return ClipGT2VSession(
            backend_factory=self._backend_factory,
            config=self._config,
            desc=session_desc,
            scene_loader=self._scene_loader,
            title=self._title,
        )

    def close(self) -> None:
        return


def _build_backend(config: AppConfig) -> RenderBackend:
    if config.backend == "raster":
        return RasterRenderBackend(
            config.chunk,
            config.raster,
            bev=config.bev,
            vehicle=config.vehicle,
        )
    raise ValueError(
        "The model integration must provide a backend factory for model rendering."
    )


def _normalize_drive_key(key: str) -> str | None:
    key = key.strip().lower()
    aliases = {
        "arrowup": "w",
        "arrowdown": "s",
        "arrowleft": "a",
        "arrowright": "d",
    }
    key = aliases.get(key, key)
    return key if key in {"w", "a", "s", "d", "space"} else None


def _gamepad_command(event: GamepadUserInputEvent) -> DriverCommand | None:
    if event.action == "disconnected":
        return None
    if event.action != "state":
        return None
    steer = -(event.axes[0] if event.axes else 0.0)
    throttle = event.buttons[7] if len(event.buttons) > 7 else 0.0
    brake = event.buttons[6] if len(event.buttons) > 6 else 0.0
    if throttle == 0.0 and brake == 0.0 and len(event.axes) > 1:
        throttle = max(0.0, -event.axes[1])
        brake = max(0.0, event.axes[1])
    return DriverCommand(
        throttle=throttle,
        brake=brake,
        steer=steer,
        steer_is_direct=True,
        manual_control=True,
    )


def _frame_chunk_tensor(chunk: FrameChunk, view_mode: ViewMode) -> Tensor:
    frames: list[Tensor] = []
    for frame in chunk.frames:
        if view_mode == "physx":
            value = frame.physx_rgb_host_uint8
            if value is None:
                raise ValueError("PhysX view requires a PhysX debug frame.")
        elif view_mode == "hdmap":
            value = frame.rgb_host_uint8
        else:
            value = (
                frame.model_rgb_host_uint8
                if frame.model_rgb_host_uint8 is not None
                else frame.rgb_host_uint8
            )
        array = np.asarray(value)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected HWC frame, received shape {tuple(tensor.shape)}"
            )
        frames.append(tensor.permute(2, 0, 1))
    if not frames:
        raise ValueError("The world-model backend returned an empty frame chunk.")
    return torch.stack(frames)


def _telemetry_status(vehicle: VehicleState, blocks: int) -> str:
    speed_mph = abs(vehicle.speed_mps) * 2.236936
    return (
        f"Block {blocks}; speed {speed_mph:.1f} mph; steer {vehicle.steer_rad:.2f} rad."
    )


__all__ = [
    "BackendFactory",
    "ClipGT2VApplication",
    "ClipGT2VApplicationDefaults",
    "ClipGT2VApplicationHooks",
    "ClipGT2VConfig",
    "ClipGT2VModelLoop",
    "ClipGT2VModelState",
    "ClipGT2VUILoop",
    "DriveTelemetry",
    "ManifestLoader",
    "ManifestResolver",
    "SceneLoader",
    "SceneResolver",
    "ViewMode",
]
