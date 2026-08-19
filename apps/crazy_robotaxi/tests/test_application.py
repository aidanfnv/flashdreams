# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomli as tomllib
from crazy_robotaxi.app import (
    CrazyRobotaxiApplication,
    CrazyRobotaxiApplicationSession,
    CrazyRobotaxiSessionConfig,
    create_app,
)
from omnidreams_game_engine import DriverCommand
from omnidreams_game_engine.provider import (
    APPLICATION_FRAMES_METADATA_KEY,
    PreparedGameStep,
)

from flashdreams.demo import (
    CanonicalInputWindow,
    IFlashDreamsApplication,
    SessionInfo,
)
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import InferenceInput, StepRequirements

pytestmark = pytest.mark.ci_cpu


class _ModelSession:
    def __init__(self, *, metadata: dict[str, object] | None = None) -> None:
        self.metadata = metadata or {}
        self.inputs: list[InferenceInput] = []
        self.reset_inputs: list[InferenceInput | None] = []
        self.closed = False

    def next_step_requirements(self) -> StepRequirements:
        return StepRequirements(step_index=len(self.inputs), input_frame_count=2)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inputs.append(inputs)
        return StepResult(step_index=len(self.inputs) - 1, metadata=self.metadata)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        self.reset_inputs.append(inputs)
        self.inputs.clear()

    def close(self) -> None:
        self.closed = True


class _Runtime:
    def __init__(
        self,
        *,
        session: _ModelSession | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.session = session
        self.sessions: list[_ModelSession] = []
        self.initial_inputs: InferenceInput | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> _ModelSession:
        self.initial_inputs = inputs
        session = self.session or _ModelSession()
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


class _Provider:
    def __init__(self, *, state: str = "playing", **kwargs: object) -> None:
        del kwargs
        self.state = state
        self.commands: list[DriverCommand] = []
        self.reset_count = 0
        self.closed = False

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput(global_conditioning={"scenario": "fake"})

    def prepare_step(
        self, *, request: StepRequirements, command: DriverCommand
    ) -> PreparedGameStep:
        self.commands.append(command)
        return PreparedGameStep(
            inference_input=InferenceInput(step={"step": request.step_index}),
            result_metadata={
                APPLICATION_FRAMES_METADATA_KEY: (
                    {"application": {"session_state": self.state}},
                )
            },
        )

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


def _config() -> CrazyRobotaxiSessionConfig:
    return CrazyRobotaxiSessionConfig(
        pipeline_config=object(),
        preset_id="test",
        device="cpu",
        scene_path=Path("scene.usdz"),
        scene_dir=None,
        scene_uuid=None,
        scene_variant="default",
        camera_name="front",
        prompt="drive",
        pixel_height=32,
        pixel_width=64,
        fps=30,
        total_blocks=10,
        game_time_s=60.0,
        game_seed=42,
    )


def _inputs() -> CanonicalInputWindow:
    return CanonicalInputWindow(
        values={
            "driver_command": {
                "throttle": 0.75,
                "brake": 0.25,
                "steer": -0.5,
                "stop": True,
                "reverse": False,
            }
        },
        window=TimeWindow(start_s=0.0, end_s=0.1),
    )


def test_application_is_registered_new_api_contract() -> None:
    app = create_app()

    assert isinstance(app, IFlashDreamsApplication)
    assert app.supports_session_reset
    assert app.input_schema.modalities[0].name == "driver_command"
    manifest = tomllib.loads(
        Path("apps/crazy_robotaxi/pyproject.toml").read_text(encoding="utf-8")
    )
    assert manifest["project"]["entry-points"]["flashdreams.applications"] == {
        "crazy-robotaxi": "crazy_robotaxi.app:create_app"
    }


def test_application_parses_without_constructing_model() -> None:
    runtime = _Runtime()
    runtime_calls: list[dict[str, object]] = []

    def runtime_factory(**kwargs: object) -> _Runtime:
        runtime_calls.append(dict(kwargs))
        return runtime

    app = CrazyRobotaxiApplication(runtime_factory=runtime_factory)
    app.init(["--device", "cpu", "--game-time-s", "15"])

    assert runtime_calls == []

    session = app.create_session()

    assert isinstance(session, CrazyRobotaxiApplicationSession)
    assert session.config.device == "cpu"
    assert session.config.game_time_s == 15
    assert session.config.preset_id.endswith("lightvae-lighttae-perf")
    options = runtime_calls[0]["options"]
    assert getattr(options, "release_oneshot_encoders_after_cache_init") is False


def test_session_maps_driver_input_and_attaches_game_metadata() -> None:
    model_session = _ModelSession()
    runtime = _Runtime(session=model_session)
    provider = _Provider()
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime=runtime,
        provider_factory=lambda **kwargs: _reuse(provider, **kwargs),
    )
    session.init()

    info = session.session_info()
    result = session.step(_inputs())

    assert info == SessionInfo(
        output_layout="bvtchw",
        steady_output_frame_count=2,
        frames_per_second=30,
        video_width=64,
        video_height=32,
        metadata={"game": "crazy-robotaxi"},
    )
    assert provider.commands == [
        DriverCommand(
            throttle=0.75,
            brake=0.25,
            steer=-0.5,
            handbrake=True,
        )
    ]
    assert APPLICATION_FRAMES_METADATA_KEY in result.metadata
    assert model_session.inputs[0].step == {"step": 0}

    session.close()
    assert provider.closed and model_session.closed
    assert not runtime.closed


def test_session_finishes_when_game_leaves_playing_state() -> None:
    runtime = _Runtime(session=_ModelSession())
    provider = _Provider(state="awaiting_name")
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime=runtime,
        provider_factory=lambda **kwargs: _reuse(provider, **kwargs),
    )
    session.init()

    session.step(_inputs())

    assert session.next_step_requirements() is None


def test_session_rejects_game_model_metadata_collision() -> None:
    runtime = _Runtime(
        session=_ModelSession(metadata={APPLICATION_FRAMES_METADATA_KEY: "model"})
    )
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime=runtime,
        provider_factory=lambda **kwargs: _reuse(_Provider(), **kwargs),
    )
    session.init()

    with pytest.raises(ValueError, match="must not overwrite"):
        session.step(_inputs())


def test_session_reset_rebuilds_model_and_game_state() -> None:
    model_session = _ModelSession()
    runtime = _Runtime(session=model_session)
    provider = _Provider(state="awaiting_name")
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime=runtime,
        provider_factory=lambda **kwargs: _reuse(provider, **kwargs),
    )
    session.init()
    session.step(_inputs())
    assert session.next_step_requirements() is None

    session.reset()
    session.reset()

    assert provider.reset_count == 2
    assert model_session.reset_inputs == [runtime.initial_inputs] * 2
    assert session.next_step_requirements() == StepRequirements(
        step_index=0,
        input_frame_count=2,
    )


def test_application_reuses_runtime_across_isolated_sessions() -> None:
    runtime = _Runtime()
    runtime_calls: list[dict[str, object]] = []
    providers: list[_Provider] = []

    def runtime_factory(**kwargs: object) -> _Runtime:
        runtime_calls.append(dict(kwargs))
        return runtime

    def provider_factory(**kwargs: object) -> Any:
        del kwargs
        provider = _Provider()
        providers.append(provider)
        return provider

    app = CrazyRobotaxiApplication(
        runtime_factory=runtime_factory,
        provider_factory=provider_factory,
    )
    app.init(["--device", "cpu", "--model-preset", "standard"])

    first = app.create_session()
    first.init()
    first.close()
    second = app.create_session()
    second.init()
    second.close()

    assert len(runtime_calls) == 1
    assert len(runtime.sessions) == 2
    assert runtime.sessions[0] is not runtime.sessions[1]
    assert all(provider.closed for provider in providers)
    assert not runtime.closed

    app.close()
    app.close()
    assert runtime.closed


def test_application_warmup_uses_neutral_driver_windows_for_webrtc() -> None:
    app = CrazyRobotaxiApplication(runtime_factory=lambda **kwargs: _Runtime(**kwargs))
    app.init(["--device", "cpu", "--total-blocks", "9"])
    webrtc = SimpleNamespace(output=SimpleNamespace(mode="webrtc"))
    local = SimpleNamespace(output=SimpleNamespace(mode="local-window"))

    sessions = app.create_model_warmup_sessions(webrtc, SimpleNamespace())

    assert len(sessions) == 1
    assert len(sessions[0].step_inputs) == 7
    assert all(
        window.values["driver_command"]
        == {
            "throttle": 0.0,
            "brake": 0.0,
            "steer": 0.0,
            "stop": False,
            "reverse": False,
        }
        for window in sessions[0].step_inputs
    )
    assert app.create_model_warmup_sessions(local, SimpleNamespace()) == ()


def test_app_packages_do_not_import_legacy_demo_or_interactive_drive() -> None:
    roots = (
        Path("apps/crazy_robotaxi/crazy_robotaxi"),
        Path("apps/omnidreams_game_engine/omnidreams_game_engine"),
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.rglob("*.py")
    )

    assert "flashdreams.runtime.demo" not in source
    assert "omnidreams.interactive_drive" not in source


def _reuse(value: Any, **kwargs: object) -> Any:
    del kwargs
    return value
