# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for Crazy Robotaxi physical-consistency prompts."""

from dataclasses import replace

import numpy as np
import pytest
from omnidreams.interactive_drive._pipeline_fakes import make_trajectory
from omnidreams.interactive_drive.crazy_robotaxi.world_consistency import (
    WorldConsistencyPromptController,
)
from omnidreams.interactive_drive.types import DynamicActorTrajectory, TrajectoryChunk

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


def _collision_trajectory(
    actor_xy_m: tuple[float, float],
    *,
    ego_yaw_rad: float = 0.0,
    object_type: str = "Car",
) -> TrajectoryChunk:
    trajectory = make_trajectory(1)
    state = replace(trajectory.vehicle_states[0], yaw_rad=ego_yaw_rad)
    actor = DynamicActorTrajectory(
        entity_id="struck-car",
        object_type=object_type,
        timestamps_us=trajectory.timestamps_us,
        translations_world=np.asarray([[*actor_xy_m, 0.0]], dtype=np.float32),
        orientations_xyzw=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        dimensions_lwh=np.asarray([4.5, 1.9, 1.5], dtype=np.float32),
        is_simulated=True,
    )
    return replace(
        trajectory,
        vehicle_states=(state,),
        boundary_state_after_chunk=state,
        dynamic_actors=(actor,),
        actor_collision_detected=True,
        actor_collision_frame_index=0,
        actor_collision_entity_ids=(actor.entity_id,),
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
        "A complete, solid vehicle"
    )


@pytest.mark.parametrize(
    ("actor_xy_m", "expected_location"),
    [
        ((3.0, 0.2), "immediately ahead of"),
        ((-3.0, 0.2), "immediately behind"),
        ((0.2, 3.0), "immediately beside the left side of"),
        ((0.2, -3.0), "immediately beside the right side of"),
    ],
)
def test_collision_prompt_describes_struck_car_relative_to_ego(
    actor_xy_m: tuple[float, float], expected_location: str
) -> None:
    controller = WorldConsistencyPromptController("A daytime driving scene.")

    update = controller.update(_collision_trajectory(actor_xy_m))

    assert "A complete, solid car remains" in update.prompt
    assert expected_location in update.prompt
    assert "filling the same nearby area of the view" in update.prompt


def test_collision_prompt_keeps_impact_description_during_hold() -> None:
    controller = WorldConsistencyPromptController("A daytime driving scene.")
    impact = controller.update(_collision_trajectory((3.0, 0.0)))

    held = controller.update(_trajectory(1.0))

    assert held.prompt == impact.prompt
