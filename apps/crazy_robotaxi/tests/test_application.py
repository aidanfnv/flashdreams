# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
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
        self.closed = False

    def next_step_requirements(self) -> StepRequirements:
        return StepRequirements(step_index=len(self.inputs), input_frame_count=2)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inputs.append(inputs)
        return StepResult(step_index=len(self.inputs) - 1, metadata=self.metadata)

    def close(self) -> None:
        self.closed = True


class _Runtime:
    def __init__(self, *, session: _ModelSession, **kwargs: object) -> None:
        del kwargs
        self.session = session
        self.initial_inputs: InferenceInput | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> _ModelSession:
        self.initial_inputs = inputs
        return self.session

    def close(self) -> None:
        self.closed = True


class _Provider:
    def __init__(self, *, state: str = "playing", **kwargs: object) -> None:
        del kwargs
        self.state = state
        self.commands: list[DriverCommand] = []
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
    assert app.input_schema.modalities[0].name == "driver_command"
    manifest = tomllib.loads(
        Path("apps/crazy_robotaxi/pyproject.toml").read_text(encoding="utf-8")
    )
    assert manifest["project"]["entry-points"]["flashdreams.applications"] == {
        "crazy-robotaxi": "crazy_robotaxi.app:create_app"
    }


def test_application_parses_without_constructing_model() -> None:
    app = CrazyRobotaxiApplication()
    app.init(["--device", "cpu", "--game-time-s", "15"])

    session = app.create_session()

    assert isinstance(session, CrazyRobotaxiApplicationSession)
    assert session.config.device == "cpu"
    assert session.config.game_time_s == 15


def test_session_maps_driver_input_and_attaches_game_metadata() -> None:
    model_session = _ModelSession()
    runtime = _Runtime(session=model_session)
    provider = _Provider()
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime_factory=lambda **kwargs: _reuse(runtime, **kwargs),
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
    assert provider.closed and model_session.closed and runtime.closed


def test_session_finishes_when_game_leaves_playing_state() -> None:
    runtime = _Runtime(session=_ModelSession())
    provider = _Provider(state="awaiting_name")
    session = CrazyRobotaxiApplicationSession(
        config=_config(),
        runtime_factory=lambda **kwargs: _reuse(runtime, **kwargs),
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
        runtime_factory=lambda **kwargs: _reuse(runtime, **kwargs),
        provider_factory=lambda **kwargs: _reuse(_Provider(), **kwargs),
    )
    session.init()

    with pytest.raises(ValueError, match="must not overwrite"):
        session.step(_inputs())


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
