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

"""Causal state alignment for autoregressive world-model output."""

from __future__ import annotations

from collections.abc import Sequence

from .types import EngineFrame


class CausalStateAligner:
    """Pair generated frames with the conditioning state that produced them."""

    def __init__(self) -> None:
        self._previous: EngineFrame | None = None

    def reset(self, seed: EngineFrame | None = None) -> None:
        """Reset alignment with optional rollout-boundary state."""
        self._previous = seed

    def align(self, frames: Sequence[EngineFrame]) -> tuple[EngineFrame, ...]:
        """Delay synchronized state by one frame after the rollout boundary."""
        aligned: list[EngineFrame] = []
        for frame in frames:
            aligned.append(frame if self._previous is None else self._previous)
            self._previous = frame
        return tuple(aligned)


__all__ = ["CausalStateAligner"]
