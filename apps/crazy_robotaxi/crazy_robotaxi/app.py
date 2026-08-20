# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi implementation of the FlashDreams application API."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from omnidreams.config import (
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
    RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
)
from omnidreams.demo.runtime import OmnidreamsRuntime, OmnidreamsRuntimeOptions
from omnidreams.demo.spec import (
    DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
    LudusBackendName,
    OmnidreamsLudusReplayScenario,
)
from omnidreams_game_engine import (
    AnalogDriverCommandConverter,
    DriverCommand,
    DriverCommandEventData,
    KeyboardDriverCommandConverter,
    driver_command_event,
)
from omnidreams_game_engine.provider import (
    APPLICATION_FRAMES_METADATA_KEY,
    OmnidreamsGameInputProvider,
)
from omnidreams_game_engine.scenario import OmnidreamsGameScenario
from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.demo import (
    ApplicationWarmupSessionInputs,
    CanonicalInputSchema,
    CanonicalInputWindow,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.results import StepResult as V1StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import DRIVER_COMMAND, InferenceConfig, StepRequirements
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult as V2StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .game import CrazyRobotaxiGame, TaxiGameConfig

RuntimeFactory = Callable[..., Any]
ProviderFactory = Callable[..., OmnidreamsGameInputProvider]

_MODEL_PRESETS = {
    "standard": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    "perf": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF,
    "native-perf": RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_NATIVE_PERF,
}
"""Integration-owned OmniDreams runner presets exposed by the game."""

_WEBRTC_WARMUP_BLOCK_COUNT = 7
"""Leading AR blocks that cover the current compiled model specializations."""


@dataclass(frozen=True, kw_only=True, slots=True)
class CrazyRobotaxiSessionConfig:
    """Resolved application settings for one Crazy Robotaxi session."""

    pipeline_config: Any
    preset_id: str
    device: str
    scene_path: Path | None
    scene_dir: Path | None
    scene_uuid: str | None
    scene_variant: str
    camera_name: str
    prompt: str | None
    pixel_height: int
    pixel_width: int
    fps: int
    total_blocks: int
    game_time_s: float
    game_seed: int


@dataclass(frozen=True, slots=True)
class CrazyRobotaxiStepResult(V2StepResult):
    """V2 video result extended with frame-synchronized game output."""

    application_frames: tuple[Mapping[str, object], ...] = ()
    """Game presentation state aligned one-to-one with generated frames."""

    model_metadata: Mapping[str, object] = field(default_factory=dict)
    """Lower-level model metadata retained across the current host adapter."""

    output_window_us: tuple[int, int] | None = None
    """Generated output time range in session-relative microseconds."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "application_frames", tuple(self.application_frames))
        object.__setattr__(
            self,
            "model_metadata",
            MappingProxyType(dict(self.model_metadata)),
        )


class CrazyRobotaxiV2Application(IApplication):
    """Own configuration and model resources for V2 game sessions."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = OmnidreamsRuntime,
        provider_factory: ProviderFactory = OmnidreamsGameInputProvider,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._provider_factory = provider_factory
        self._session_config: CrazyRobotaxiSessionConfig | None = None
        self._runtime: Any | None = None
        self._closed = False

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse application arguments without constructing GPU resources."""
        parser = argparse.ArgumentParser(prog="flashdreams-run crazy-robotaxi")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--scene-path", type=Path)
        parser.add_argument("--scene-dir", type=Path)
        parser.add_argument("--scene-uuid")
        parser.add_argument("--scene-variant", default="default")
        parser.add_argument("--camera-name", default="camera_front_wide_120fov")
        parser.add_argument("--prompt")
        parser.add_argument("--pixel-height", type=int, default=704)
        parser.add_argument("--pixel-width", type=int, default=1280)
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--total-blocks", type=int, default=2_147_483_647)
        parser.add_argument("--game-time-s", type=float, default=60.0)
        parser.add_argument("--game-seed", type=int, default=42)
        parser.add_argument(
            "--model-preset",
            choices=tuple(_MODEL_PRESETS),
            default="perf",
        )
        parser.add_argument(
            "--compile",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        args = parser.parse_args(list(commandline_args))
        if args.pixel_height <= 0 or args.pixel_width <= 0:
            raise ValueError("Pixel dimensions must be greater than zero.")
        if args.fps <= 0:
            raise ValueError("--fps must be greater than zero.")
        if args.total_blocks <= 0:
            raise ValueError("--total-blocks must be greater than zero.")
        if not math.isfinite(args.game_time_s) or args.game_time_s <= 0:
            raise ValueError("--game-time-s must be finite and greater than zero.")

        defaults = _MODEL_PRESETS[args.model_preset]
        pipeline_config = defaults.pipeline
        if args.compile is not None:
            pipeline_config = derive_config(
                pipeline_config,
                diffusion_model={
                    "transformer": {"compile_network": args.compile},
                },
            )
        self._session_config = CrazyRobotaxiSessionConfig(
            pipeline_config=pipeline_config,
            preset_id=defaults.pipeline.name,
            device=args.device,
            scene_path=args.scene_path,
            scene_dir=args.scene_dir,
            scene_uuid=args.scene_uuid,
            scene_variant=args.scene_variant,
            camera_name=args.camera_name,
            prompt=args.prompt,
            pixel_height=args.pixel_height,
            pixel_width=args.pixel_width,
            fps=args.fps,
            total_blocks=args.total_blocks,
            game_time_s=args.game_time_s,
            game_seed=args.game_seed,
        )

    def create_session(self, session_desc: SessionDesc) -> CrazyRobotaxiV2Session:
        """Create an isolated game session for the requested V2 description."""
        config = self._session_config
        if config is None:
            raise RuntimeError(
                "CrazyRobotaxiV2Application.init() must run before create_session()."
            )
        if self._closed:
            raise RuntimeError(
                "Cannot create a session from a closed Crazy Robotaxi application."
            )
        if session_desc.output_layout is not VideoTensorLayout.bvtchw:
            raise ValueError(
                "Crazy Robotaxi requires bvtchw output, got "
                f"{session_desc.output_layout.value}."
            )
        runtime = self._runtime
        if runtime is None:
            runtime = self._runtime_factory(
                config=InferenceConfig(
                    model_id="omnidreams",
                    preset_id=config.preset_id,
                    device=config.device,
                    seed=config.game_seed,
                ),
                options=OmnidreamsRuntimeOptions(
                    pipeline_config=config.pipeline_config,
                    release_oneshot_encoders_after_cache_init=False,
                ),
            )
            self._runtime = runtime
        resolved_config = replace(
            config,
            pixel_height=session_desc.video_height,
            pixel_width=session_desc.video_width,
            fps=session_desc.frames_per_second_for_step,
        )
        return CrazyRobotaxiV2Session(
            config=resolved_config,
            session_desc=replace(
                session_desc,
                metadata={**session_desc.metadata, "game": "crazy-robotaxi"},
            ),
            runtime=runtime,
            provider_factory=self._provider_factory,
        )

    def create_model_warmup_inputs(self) -> tuple[tuple[UserInputEvents, ...], ...]:
        """Return neutral V2 event windows for leading model specializations."""
        config = self._session_config
        if config is None:
            raise RuntimeError(
                "CrazyRobotaxiV2Application.init() must run before warmup planning."
            )
        warmup_blocks = min(config.total_blocks, _WEBRTC_WARMUP_BLOCK_COUNT)
        return (
            tuple(
                UserInputEvents(
                    [
                        driver_command_event(
                            timestamp_us=block_index * 1_000_000,
                            command=DriverCommand(),
                        )
                    ]
                )
                for block_index in range(warmup_blocks)
            ),
        )

    def configured_session_desc(self) -> SessionDesc:
        """Return the session description configured by application arguments."""
        config = self._session_config
        if config is None:
            raise RuntimeError(
                "CrazyRobotaxiV2Application.init() must run before session setup."
            )
        return SessionDesc(
            output_layout=VideoTensorLayout.bvtchw,
            frames_per_second_for_ui=60,
            frames_per_second_for_step=config.fps,
            video_width=config.pixel_width,
            video_height=config.pixel_height,
            metadata={"game": "crazy-robotaxi"},
        )

    def close(self) -> None:
        """Release the application-lifetime OmniDreams runtime."""
        if self._closed:
            return
        self._closed = True
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.close()


class CrazyRobotaxiV2Session(ISession):
    """Run one game through V2 input, session-description, and output types."""

    def __init__(
        self,
        *,
        config: CrazyRobotaxiSessionConfig,
        session_desc: SessionDesc,
        runtime: Any,
        provider_factory: ProviderFactory = OmnidreamsGameInputProvider,
    ) -> None:
        self.config = config
        self._session_desc = session_desc
        self._runtime = runtime
        self._provider_factory = provider_factory
        self._model_session: Any | None = None
        self._provider: OmnidreamsGameInputProvider | None = None
        self._keyboard_converter = KeyboardDriverCommandConverter()
        self._analog_converter = AnalogDriverCommandConverter()
        self._driver_command = DriverCommand()
        self._finished = False
        self._closed = False

    def init(self) -> None:
        """Construct the scene provider and first cache on the shared runtime."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed Crazy Robotaxi session.")
        if self._model_session is not None:
            return
        model_scenario = _model_scenario(self.config)
        game_scenario = OmnidreamsGameScenario(
            model=model_scenario,
            scene_path=self.config.scene_path,
        )
        provider = self._provider_factory(
            scenario=game_scenario,
            device=self.config.device,
            application=CrazyRobotaxiGame(
                TaxiGameConfig(
                    seed=self.config.game_seed,
                    game_time_s=self.config.game_time_s,
                )
            ),
        )
        try:
            model_session = self._runtime.start_session(
                provider.prepare_initial_input()
            )
        except BaseException:
            provider.close()
            raise
        self._provider = provider
        self._model_session = model_session

    @property
    def session_desc(self) -> SessionDesc:
        """Describe V2 game video for a client window."""
        return self._session_desc

    def next_step_requirements(self) -> StepRequirements | None:
        """Delegate model chunk sizing until the game or rollout completes."""
        if self._finished or self._closed:
            return None
        session = self._require_model_session()
        value = session.next_step_requirements()
        if value is not None and not isinstance(value, StepRequirements):
            raise TypeError(
                "OmniDreams session next_step_requirements() must return "
                "StepRequirements or None."
            )
        return value

    def step(
        self,
        step_index: int,
        events: UserInputEvents,
    ) -> CrazyRobotaxiStepResult:
        """Advance simulation from V2 events and return synchronized V2 output."""
        if self._finished:
            raise RuntimeError("Crazy Robotaxi session is already complete.")
        request = self.next_step_requirements()
        if request is None:
            raise RuntimeError("Crazy Robotaxi session has no remaining model step.")
        if request.step_index != step_index:
            raise ValueError(
                "V2 runner/model step index mismatch: "
                f"runner={step_index}, model={request.step_index}."
            )
        command = self._consume_driver_command(events)
        if command.reset:
            raise ValueError(
                "Use ResetUserInputEventData for lifecycle reset, not the "
                "driver-command payload."
            )
        prepared = self._require_provider().prepare_step(
            request=request,
            command=command,
        )
        model_result = self._require_model_session().step(prepared.inference_input)
        if not isinstance(model_result, V1StepResult):
            raise TypeError("OmniDreams session step() must return StepResult.")
        collisions = sorted(set(model_result.metadata) & set(prepared.result_metadata))
        if collisions:
            raise ValueError(
                "Game metadata must not overwrite model metadata: "
                + ", ".join(collisions)
            )
        if not isinstance(model_result.output, Tensor):
            raise TypeError("OmniDreams V2 output must be a torch.Tensor.")
        if model_result.layout is None:
            raise ValueError("OmniDreams V2 output requires a video layout.")
        frames = prepared.result_metadata.get(APPLICATION_FRAMES_METADATA_KEY, ())
        if not isinstance(frames, Sequence):
            raise TypeError("application_frames must be a sequence.")
        result = CrazyRobotaxiStepResult(
            step_index=model_result.step_index,
            output=model_result.output,
            frame_count=model_result.frame_count,
            output_layout=VideoTensorLayout(model_result.layout),
            metrics=dict(model_result.metrics),
            application_frames=tuple(
                cast(Mapping[str, object], frame) for frame in frames
            ),
            model_metadata=model_result.metadata,
            output_window_us=_output_window_us(model_result.output_window),
        )
        self._finished = _game_finished(result.application_frames)
        return result

    def reset(self) -> None:
        """Reset model, simulation, game, renderer timeline, and alignment state."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed Crazy Robotaxi session.")
        provider = self._require_provider()
        model_session = self._require_model_session()
        provider.reset()
        model_session.reset(provider.prepare_initial_input())
        self._keyboard_converter.reset()
        self._analog_converter.reset()
        self._driver_command = DriverCommand()
        self._finished = False

    def close(self) -> None:
        """Release session, renderer, and model resources idempotently."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._model_session is not None:
                self._model_session.close()
        finally:
            if self._provider is not None:
                self._provider.close()
        self._model_session = None
        self._provider = None

    def _require_model_session(self) -> Any:
        if self._model_session is None:
            raise RuntimeError("CrazyRobotaxiV2Session.init() must run before use.")
        return self._model_session

    def _require_provider(self) -> OmnidreamsGameInputProvider:
        if self._provider is None:
            raise RuntimeError("CrazyRobotaxiV2Session.init() must run before use.")
        return self._provider

    def _consume_driver_command(self, inputs: UserInputEvents) -> DriverCommand:
        keyboard = self._keyboard_converter.convert(inputs).command
        analog = self._analog_converter.convert(inputs)
        direct: DriverCommand | None = None
        for event in inputs.get_events():
            event_data = event.get_event_data()
            if isinstance(event_data, DriverCommandEventData):
                direct = event_data.command
        if direct is not None:
            self._driver_command = direct
        elif analog is not None:
            self._driver_command = analog.command
        else:
            self._driver_command = keyboard
        return self._driver_command


class CrazyRobotaxiV1ApplicationAdapter(IFlashDreamsApplication):
    """Adapt the current application host to the V2 game lifecycle."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory = OmnidreamsRuntime,
        provider_factory: ProviderFactory = OmnidreamsGameInputProvider,
    ) -> None:
        self.v2_application = CrazyRobotaxiV2Application(
            runtime_factory=runtime_factory,
            provider_factory=provider_factory,
        )

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Declare the V1 driving snapshot supplied by existing hosts."""
        return CanonicalInputSchema(
            modalities=(DRIVER_COMMAND,),
            description="Crazy Robotaxi V1-to-V2 driving adapter.",
        )

    @property
    def supports_session_reset(self) -> bool:
        """Return whether the delegated V2 session supports reset."""
        return True

    def init(self, commandline_args: Sequence[str]) -> None:
        """Delegate application argument parsing to the V2 application."""
        self.v2_application.init(commandline_args)

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Wrap one V2 session for the current playable application host."""
        return CrazyRobotaxiV1SessionAdapter(
            self.v2_application.create_session(
                self.v2_application.configured_session_desc()
            )
        )

    def create_model_warmup_sessions(
        self,
        spec: Any,
        scenario: Any,
    ) -> Sequence[ApplicationWarmupSessionInputs]:
        """Translate V2 neutral events into the V1 host warmup carrier."""
        del scenario
        if spec.output.mode != "webrtc":
            return ()
        return tuple(
            ApplicationWarmupSessionInputs(
                step_inputs=tuple(
                    _v1_window_from_v2_events(events, block_index=index)
                    for index, events in enumerate(session_inputs)
                )
            )
            for session_inputs in self.v2_application.create_model_warmup_inputs()
        )

    def close(self) -> None:
        """Release V2 application-lifetime resources."""
        self.v2_application.close()


class CrazyRobotaxiV1SessionAdapter(IFlashDreamsApplicationSession):
    """Adapt current host input and output to one V2 game session."""

    def __init__(self, v2_session: CrazyRobotaxiV2Session) -> None:
        self.v2_session = v2_session
        self.config = v2_session.config

    def init(self) -> None:
        """Initialize the delegated V2 session."""
        self.v2_session.init()

    def session_info(self) -> SessionInfo:
        """Project V2 session description into the current sink contract."""
        desc = self.v2_session.session_desc
        request = self.v2_session.next_step_requirements()
        return SessionInfo(
            output_layout=desc.output_layout.value,
            steady_output_frame_count=(
                None if request is None else request.input_frame_count
            ),
            frames_per_second=desc.frames_per_second_for_step,
            video_width=desc.video_width,
            video_height=desc.video_height,
            metadata=desc.metadata,
        )

    def next_step_requirements(self) -> StepRequirements | None:
        """Expose the lower-level model requirement to the current host."""
        return self.v2_session.next_step_requirements()

    def step(self, inputs: CanonicalInputWindow) -> V1StepResult:
        """Translate V1 canonical input and V2 output at the host boundary."""
        request = self.next_step_requirements()
        if request is None:
            raise RuntimeError("Crazy Robotaxi session has no remaining model step.")
        result = self.v2_session.step(
            request.step_index,
            _v2_events_from_v1_window(inputs),
        )
        metadata = {
            **result.model_metadata,
            APPLICATION_FRAMES_METADATA_KEY: result.application_frames,
        }
        return V1StepResult.from_video_chunk(
            step_index=result.step_index,
            video_chunk=result.output,
            layout=result.output_layout.value,
            output_window=_v1_output_window(result.output_window_us),
            metadata=metadata,
            metrics=result.metrics,
        )

    def reset(self) -> None:
        """Reset the delegated V2 session."""
        self.v2_session.reset()

    def close(self) -> None:
        """Close the delegated V2 session."""
        self.v2_session.close()


def _model_scenario(
    config: CrazyRobotaxiSessionConfig,
) -> OmnidreamsLudusReplayScenario:
    return OmnidreamsLudusReplayScenario(
        keyboard_events=(),
        scene_path=config.scene_path,
        scene_dir=config.scene_dir,
        scene_uuid=config.scene_uuid or DEFAULT_OMNIDREAMS_WEBRTC_SCENE_UUID,
        scene_variant=config.scene_variant,
        camera_name=config.camera_name,
        prompt=config.prompt,
        total_blocks=config.total_blocks,
        pixel_height=config.pixel_height,
        pixel_width=config.pixel_width,
        fps=config.fps,
        move_speed_per_s=6.0,
        rotate_speed_rad_per_s=math.radians(35.0),
        ludus_backend=cast(LudusBackendName, "cuda"),
    )


def _driver_command(value: object) -> DriverCommand:
    if not isinstance(value, Mapping):
        raise TypeError("driver_command must be a named field mapping.")
    return DriverCommand(
        throttle=float(str(value.get("throttle", 0.0))),
        brake=float(str(value.get("brake", 0.0))),
        steer=float(str(value.get("steer", 0.0))),
        handbrake=bool(value.get("stop", False)),
        reverse=bool(value.get("reverse", False)),
    )


def _v2_events_from_v1_window(inputs: CanonicalInputWindow) -> UserInputEvents:
    command = _driver_command(inputs.values.get(DRIVER_COMMAND.name, {}))
    return UserInputEvents(
        [
            driver_command_event(
                timestamp_us=int(round(inputs.window.end_s * 1_000_000)),
                command=command,
            )
        ]
    )


def _v1_window_from_v2_events(
    inputs: UserInputEvents,
    *,
    block_index: int,
) -> CanonicalInputWindow:
    command = DriverCommand()
    for event in inputs.get_events():
        event_data = event.get_event_data()
        if isinstance(event_data, DriverCommandEventData):
            command = event_data.command
    return CanonicalInputWindow(
        values={
            DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                {
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "steer": command.steer,
                    "stop": command.handbrake,
                    "reverse": command.reverse,
                }
            )
        },
        window=TimeWindow(
            start_s=float(block_index),
            end_s=float(block_index + 1),
        ),
    )


def _output_window_us(window: TimeWindow | None) -> tuple[int, int] | None:
    if window is None:
        return None
    return (
        int(round(window.start_s * 1_000_000)),
        int(round(window.end_s * 1_000_000)),
    )


def _v1_output_window(value: tuple[int, int] | None) -> TimeWindow | None:
    if value is None:
        return None
    return TimeWindow(start_s=value[0] / 1_000_000, end_s=value[1] / 1_000_000)


def _game_finished(frames: Sequence[Mapping[str, object]]) -> bool:
    if not frames:
        return False
    last = frames[-1]
    application = last.get("application")
    return isinstance(application, Mapping) and application.get("session_state") != (
        "playing"
    )


def create_app() -> IFlashDreamsApplication:
    """Create the temporary V1 host adapter for the V2 game application."""
    return CrazyRobotaxiV1ApplicationAdapter()


def create_v2_app() -> IApplication:
    """Create the V2 Crazy Robotaxi application."""
    return CrazyRobotaxiV2Application()


__all__ = [
    "CrazyRobotaxiStepResult",
    "CrazyRobotaxiSessionConfig",
    "CrazyRobotaxiV1ApplicationAdapter",
    "CrazyRobotaxiV1SessionAdapter",
    "CrazyRobotaxiV2Application",
    "CrazyRobotaxiV2Session",
    "create_app",
    "create_v2_app",
]
