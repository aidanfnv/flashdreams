# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crazy Robotaxi implementation of the FlashDreams application API."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
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
from omnidreams_game_engine import DriverCommand
from omnidreams_game_engine.provider import (
    APPLICATION_FRAMES_METADATA_KEY,
    OmnidreamsGameInputProvider,
)
from omnidreams_game_engine.scenario import OmnidreamsGameScenario

from flashdreams.demo import (
    ApplicationWarmupSessionInputs,
    CanonicalInputSchema,
    CanonicalInputWindow,
    IFlashDreamsApplication,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import DRIVER_COMMAND, InferenceConfig, StepRequirements

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


class CrazyRobotaxiApplication(IFlashDreamsApplication):
    """Create transport-neutral Crazy Robotaxi application sessions."""

    session_type: type[CrazyRobotaxiApplicationSession]

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

    @property
    def input_schema(self) -> CanonicalInputSchema:
        """Consume the driver command supported by stock application I/O."""
        return CanonicalInputSchema(
            modalities=(DRIVER_COMMAND,),
            description="Crazy Robotaxi driving controls.",
        )

    @property
    def supports_session_reset(self) -> bool:
        """Return whether game sessions can rebuild all per-generation state."""
        return True

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

    def create_session(self) -> IFlashDreamsApplicationSession:
        """Create an isolated game session on the retained model runtime."""
        config = self._session_config
        if config is None:
            raise RuntimeError(
                "CrazyRobotaxiApplication.init() must run before create_session()."
            )
        if self._closed:
            raise RuntimeError(
                "Cannot create a session from a closed Crazy Robotaxi application."
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
        return self.session_type(
            config=config,
            runtime=runtime,
            provider_factory=self._provider_factory,
        )

    def create_model_warmup_sessions(
        self,
        spec: Any,
        scenario: Any,
    ) -> Sequence[ApplicationWarmupSessionInputs]:
        """Warm leading autoregressive specializations for WebRTC serving."""
        del scenario
        if spec.output.mode != "webrtc":
            return ()
        config = self._session_config
        if config is None:
            raise RuntimeError(
                "CrazyRobotaxiApplication.init() must run before warmup planning."
            )
        warmup_blocks = min(config.total_blocks, _WEBRTC_WARMUP_BLOCK_COUNT)
        return (
            ApplicationWarmupSessionInputs(
                step_inputs=tuple(
                    _neutral_driver_window(block_index)
                    for block_index in range(warmup_blocks)
                )
            ),
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


class CrazyRobotaxiApplicationSession(IFlashDreamsApplicationSession):
    """Own one OmniDreams cache and one standalone game simulation."""

    def __init__(
        self,
        *,
        config: CrazyRobotaxiSessionConfig,
        runtime: Any,
        provider_factory: ProviderFactory = OmnidreamsGameInputProvider,
    ) -> None:
        self.config = config
        self._runtime = runtime
        self._provider_factory = provider_factory
        self._model_session: Any | None = None
        self._provider: OmnidreamsGameInputProvider | None = None
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

    def session_info(self) -> SessionInfo:
        """Describe generated OmniDreams video for host-owned output sinks."""
        request = self.next_step_requirements()
        return SessionInfo(
            output_layout="bvtchw",
            steady_output_frame_count=(
                None if request is None else request.input_frame_count
            ),
            frames_per_second=self.config.fps,
            video_width=self.config.pixel_width,
            video_height=self.config.pixel_height,
            metadata={"game": "crazy-robotaxi"},
        )

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

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Advance simulation, condition OmniDreams, and attach game state."""
        if self._finished:
            raise RuntimeError("Crazy Robotaxi session is already complete.")
        request = self.next_step_requirements()
        if request is None:
            raise RuntimeError("Crazy Robotaxi session has no remaining model step.")
        command = _driver_command(inputs.values.get(DRIVER_COMMAND.name, {}))
        prepared = self._require_provider().prepare_step(
            request=request,
            command=command,
        )
        result = self._require_model_session().step(prepared.inference_input)
        if not isinstance(result, StepResult):
            raise TypeError("OmniDreams session step() must return StepResult.")
        collisions = sorted(set(result.metadata) & set(prepared.result_metadata))
        if collisions:
            raise ValueError(
                "Game metadata must not overwrite model metadata: "
                + ", ".join(collisions)
            )
        result = replace(
            result,
            metadata={**result.metadata, **prepared.result_metadata},
        )
        self._finished = _game_finished(result.metadata)
        return result

    def reset(self) -> None:
        """Reset model, simulation, game, renderer timeline, and alignment state."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed Crazy Robotaxi session.")
        provider = self._require_provider()
        model_session = self._require_model_session()
        provider.reset()
        model_session.reset(provider.prepare_initial_input())
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
            raise RuntimeError(
                "CrazyRobotaxiApplicationSession.init() must run before use."
            )
        return self._model_session

    def _require_provider(self) -> OmnidreamsGameInputProvider:
        if self._provider is None:
            raise RuntimeError(
                "CrazyRobotaxiApplicationSession.init() must run before use."
            )
        return self._provider


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


def _neutral_driver_window(block_index: int) -> CanonicalInputWindow:
    return CanonicalInputWindow(
        values={
            DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                {
                    "throttle": 0.0,
                    "brake": 0.0,
                    "steer": 0.0,
                    "stop": False,
                    "reverse": False,
                }
            )
        },
        window=TimeWindow(
            start_s=float(block_index),
            end_s=float(block_index + 1),
        ),
    )


def _game_finished(metadata: Mapping[str, object]) -> bool:
    frames = metadata.get(APPLICATION_FRAMES_METADATA_KEY)
    if not isinstance(frames, Sequence) or not frames:
        return False
    last = frames[-1]
    if not isinstance(last, Mapping):
        return False
    application = last.get("application")
    return isinstance(application, Mapping) and application.get("session_state") != (
        "playing"
    )


CrazyRobotaxiApplication.session_type = CrazyRobotaxiApplicationSession


def create_app() -> IFlashDreamsApplication:
    """Create the installed Crazy Robotaxi application."""
    return CrazyRobotaxiApplication()


__all__ = [
    "CrazyRobotaxiApplication",
    "CrazyRobotaxiApplicationSession",
    "CrazyRobotaxiSessionConfig",
    "create_app",
]
