# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SANA-WM adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flashdreams.runtime import InferenceInput, UserInputSchema
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    PreparedScenario,
)
from sana_wm.runtime import (
    FIELD_CAMERA_INTRINSICS_VEC4,
    FIELD_CAMERA_TRAJECTORY_C2W,
    FIELD_GLOBAL_CONDITIONING_FRAME,
    FIELD_NEGATIVE_PROMPT,
    FIELD_PROMPT,
    SanaWMModelAdapter,
    inference_input_from_prepared_inputs,
)


class SanaWMDemoAdapter(SanaWMModelAdapter):
    """Model-owned SANA-WM adapter consumed by shared demo launchers."""

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "SANA-WM prepare_scenario supports input_mode='replay', "
                f"got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, (Mp4OutputSpec, NullOutputSpec)):
            raise ValueError("SANA-WM replay demo requires MP4 or null output.")
        return PreparedScenario(
            initial_inputs=_initial_inputs_from_scenario(spec.scenario),
            source_schema=UserInputSchema(description="fixed SANA-WM replay input"),
            metadata={
                "model_id": self.model_id,
                "preset_id": self.preset_id(spec.config),
            },
        )


def _initial_inputs_from_scenario(scenario: Any) -> InferenceInput:
    if isinstance(scenario, InferenceInput):
        return scenario
    if not isinstance(scenario, Mapping):
        raise TypeError("SANA-WM replay scenario must be an InferenceInput or mapping.")
    return inference_input_from_prepared_inputs(
        prompt=str(scenario[FIELD_PROMPT]),
        image=scenario[FIELD_GLOBAL_CONDITIONING_FRAME],
        poses_c2w=scenario[FIELD_CAMERA_TRAJECTORY_C2W],
        intrinsics_vec4=scenario.get(FIELD_CAMERA_INTRINSICS_VEC4),
        negative_prompt=str(scenario.get(FIELD_NEGATIVE_PROMPT, "")),
    )


__all__ = ["SanaWMDemoAdapter"]
