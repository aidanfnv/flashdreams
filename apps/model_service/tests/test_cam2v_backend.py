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

"""CPU checks of camera control timing, uploaded assets, and job isolation."""

import numpy as np
import pytest

from cam2v.dummy import create_app
from model_service.flashdreams_backend import ApplicationBackend

pytestmark = pytest.mark.ci_cpu


def test_camera_blocks_release_reset_and_reopen():
    backend = ApplicationBackend({"camera": create_app}, device="cpu")
    model = backend.catalog()[0]
    assert model["interaction"] == "camera"
    assert model["input_image"] == "required_rgb24"
    parameters = {
        "prompt": "A saved room",
        "width": 32,
        "height": 16,
        "total_blocks": 3,
    }
    image = bytes([100, 120, 140]) * 32 * 16
    session = backend.open("camera", parameters, image)
    try:
        with pytest.raises(ValueError, match="requires an initial"):
            backend.open("camera", parameters, b"")
        with pytest.raises(ValueError, match="busy"):
            backend.open("camera", parameters, image)
        with pytest.raises(ValueError):
            session.step(4, {"forward": 1})
        first, pixels = session.step(4, {"forward": True})
        assert first["applied_inputs"]["forward"]
        initial_pose = session._session.model_loop.state.pose_integrator.current_pose()
        assert initial_pose[2, 3] > 0
        for _ in range(2):
            pending, _ = session.step(4, {})
            assert pending["block"] == 0 and pending["applied_inputs"]["forward"]
        released, _ = session.step(4, {})
        assert released["block"] == 1 and not any(released["applied_inputs"].values())
        np.testing.assert_allclose(
            session._session.model_loop.state.pose_integrator.current_pose(),
            initial_pose,
        )
        session.reset()
        again, repeat_pixels = session.step(4, {"forward": True})
        assert again == first and repeat_pixels == pixels
    finally:
        session.close()
    # A second job starts at identity and sees its own prompt and image.
    second = backend.open(
        "camera", {**parameters, "prompt": "Another room"}, bytes([50]) * len(image)
    )
    try:
        result, other_pixels = second.step(4, {})
        assert result["first_frame"] == 0 and other_pixels != pixels
        assert (
            second._session.model_loop.state.config.conditioning.prompt
            == "Another room"
        )
        np.testing.assert_allclose(
            second._session.model_loop.state.pose_integrator.current_pose(), np.eye(4)
        )
    finally:
        second.close()
        backend.close()
