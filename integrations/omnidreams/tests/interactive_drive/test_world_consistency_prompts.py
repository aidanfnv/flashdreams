# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Crazy Robotaxi physical-consistency prompts."""

from dataclasses import replace

import pytest
from omnidreams.interactive_drive._pipeline_fakes import make_trajectory
from omnidreams.interactive_drive.crazy_robotaxi.world_consistency import (
    WorldConsistencyPromptController,
)
from omnidreams.interactive_drive.types import TrajectoryChunk

pytestmark = pytest.mark.ci_cpu


def _trajectory(*speeds_mps: float, collision: bool = False) -> TrajectoryChunk:
    trajectory = make_trajectory(len(speeds_mps))
    states = tuple(
        replace(state, speed_mps=speed)
        for state, speed in zip(trajectory.vehicle_states, speeds_mps, strict=True)
    )
    return replace(
        trajectory,
        vehicle_states=states,
        boundary_state_after_chunk=states[-1],
        actor_collision_detected=collision,
        actor_collision_frame_index=0 if collision else None,
    )


def test_reverse_prompt_uses_hysteresis_and_restores_base_prompt() -> None:
    controller = WorldConsistencyPromptController("A daytime driving scene.")

    assert controller.update(_trajectory(0.0, -0.24)).active_modifiers == ()
    reverse = controller.update(_trajectory(-0.26, -1.0))
    assert reverse.active_modifiers == ("reverse",)
    assert reverse.prompt.startswith("A daytime driving scene.")
    assert "reversing backward" in reverse.prompt
    assert controller.update(_trajectory(-0.04, -0.06)).active_modifiers == ("reverse",)
    restored = controller.update(_trajectory(-0.04, 0.0))
    assert restored.active_modifiers == ()
    assert restored.prompt == "A daytime driving scene."


def test_collision_prompt_covers_impact_and_six_future_chunks() -> None:
    controller = WorldConsistencyPromptController("A daytime driving scene.")

    impact = controller.update(_trajectory(2.0, collision=True))
    assert impact.active_modifiers == ("collision",)
    for _ in range(6):
        assert controller.update(_trajectory(2.0)).active_modifiers == ("collision",)
    assert controller.update(_trajectory(2.0)).active_modifiers == ()


def test_collision_refreshes_hold_and_follows_reverse_modifier() -> None:
    controller = WorldConsistencyPromptController("A daytime driving scene.")

    update = controller.update(_trajectory(-1.0, collision=True))

    assert update.active_modifiers == ("reverse", "collision")
    assert update.prompt.index("reversing backward") < update.prompt.index(
        "Vehicles remain solid"
    )
