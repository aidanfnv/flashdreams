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

"""Direct Ludus scene loading and HD-map conditioning for game runtimes."""

from __future__ import annotations

import io
import math
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from ludus_renderer import LudusCudaTimestampedContext, load_scene
from ludus_renderer.dynamic_scene import MutableObjectSceneBuffer
from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR
from omnidreams.scenes import hf_hub_download_scene
from PIL import Image

from .scenario import OmnidreamsGameScenario
from .types import DynamicActorTrajectory, SceneDefinition, VehicleState


class LudusGameScene:
    """Own one loaded Ludus scene and render arbitrary simulated rig poses."""

    def __init__(
        self, scenario: OmnidreamsGameScenario, *, device: torch.device
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("LudusGameScene requires a CUDA device.")
        self.scenario = scenario
        self.device = device
        self.scene_path = _resolve_scene_path(scenario)
        model = scenario.model
        self._scene = load_scene(
            self.scene_path,
            device=device,
            target_resolution=(model.pixel_width, model.pixel_height),
            include_ego_trajectory=False,
            include_ego_obstacle=False,
        )
        self._context = LudusCudaTimestampedContext(device=device)
        self._context.set_depth_scaling(True)
        self._context.set_msaa_samples(4)
        self._context.upload_cameras(list(self._scene.cameras))
        self._base_scene = self._scene.timestamped_scene
        self._scene_id = self._context.upload_scene(self._base_scene)
        self._object_scene = MutableObjectSceneBuffer(
            self._context,
            self._scene_id,
            self._base_scene,
            device=self.device,
        )
        self._camera_name = _resolve_camera_name(
            model.camera_name, self._scene.camera_name_to_id
        )

    def definition(self) -> SceneDefinition:
        """Return CPU scene facts used by simulation and the game application."""
        route = self._scene.ego_track.translations.detach().cpu().numpy()
        if route.shape[1] == 2:
            route = np.pad(route, ((0, 0), (0, 1)))
        first_pose = self._scene.ego_track
        translation = first_pose.translations[0].detach().cpu().numpy()
        quaternion = first_pose.quaternions[0].detach().cpu().numpy()
        yaw = _yaw_from_quaternion_xyzw(quaternion)
        model = self.scenario.model
        return SceneDefinition(
            scene_id=model.scene_uuid or self.scene_path.stem,
            scene_path=self.scene_path,
            camera_name=self._camera_name,
            prompt=_read_prompt(self.scene_path, override=model.prompt),
            first_frame_rgb=_read_first_frame(
                self.scene_path,
                camera_name=self._camera_name,
                width=model.pixel_width,
                height=model.pixel_height,
            ),
            route_world=route,
            initial_vehicle=VehicleState(
                x_m=float(translation[0]),
                y_m=float(translation[1]),
                z_m=float(translation[2]),
                yaw_rad=yaw,
            ),
            initial_timestamp_us=int(first_pose.timestamps[0].item()),
        )

    def render(
        self,
        *,
        vehicles: Sequence[VehicleState],
        timestamps_us: np.ndarray,
        dynamic_actors: Sequence[DynamicActorTrajectory] = (),
    ) -> torch.Tensor:
        """Render model-range HD-map frames as ``[1, 1, T, C, H, W]``."""
        if len(vehicles) != len(timestamps_us):
            raise ValueError("vehicles and timestamps_us must have equal length.")
        self._update_application_actors(dynamic_actors)
        self._scene_id = self._object_scene.scene_id
        rig_to_world = torch.as_tensor(
            np.stack([state.rig_to_world() for state in vehicles]),
            device=self.device,
            dtype=torch.float32,
        )
        sensor_to_rig = self._scene.sensor_to_rig[self._camera_name]
        world_to_camera = torch.linalg.inv(rig_to_world @ sensor_to_rig)
        frame_count = len(vehicles)
        scene_ids = torch.full(
            (frame_count,), self._scene_id, dtype=torch.int32, device=self.device
        )
        camera_ids = torch.full(
            (frame_count,),
            self._scene.camera_name_to_id[self._camera_name],
            dtype=torch.int32,
            device=self.device,
        )
        camera_types = torch.full(
            (frame_count,), CAMERA_TYPE_REGULAR, dtype=torch.int32, device=self.device
        )
        timestamps = torch.as_tensor(
            timestamps_us, dtype=torch.int64, device=self.device
        )
        images = self._context.render(
            scene_ids,
            camera_ids,
            timestamps,
            camera_types,
            world_to_camera,
            resolution=(
                self.scenario.model.pixel_height,
                self.scenario.model.pixel_width,
            ),
        )[..., :3]
        if getattr(self._context, "needs_vflip", True):
            images = images.flip(1)
        video = images.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(0)
        return video.to(dtype=torch.bfloat16) / 127.5 - 1.0

    def close(self) -> None:
        """Release renderer-owned CUDA resources."""
        close = getattr(self._context, "close", None)
        if callable(close):
            close()

    def _update_application_actors(
        self, actors: Sequence[DynamicActorTrajectory]
    ) -> None:
        self._object_scene.update(actors)


def _resolve_scene_path(scenario: OmnidreamsGameScenario) -> Path:
    if scenario.scene_path is not None:
        path = scenario.scene_path.expanduser()
    elif scenario.model.scene_path is not None:
        path = scenario.model.scene_path.expanduser()
    elif scenario.model.scene_dir is not None:
        path = _resolve_local_scene_path(scenario.model)
    else:
        path = hf_hub_download_scene(
            scenario.model.scene_uuid or "0d404ff7-2b66-498c-b047-1ed8cded60d4",
            scenario.model.scene_variant,
        )
    if not path.is_file():
        raise FileNotFoundError(f"OmniDreams game scene is missing: {path}")
    return path


def _resolve_local_scene_path(model: object) -> Path:
    from omnidreams.scenes import normalise_scene_uuid, scene_variant_suffix

    scene_dir = Path(str(getattr(model, "scene_dir"))).expanduser()
    if scene_dir.is_file():
        return scene_dir
    if not scene_dir.is_dir():
        raise FileNotFoundError(
            f"OmniDreams game scene directory is missing: {scene_dir}"
        )
    scene_uuid = getattr(model, "scene_uuid", None)
    variant = str(getattr(model, "scene_variant", "default"))
    if scene_uuid is not None:
        bare_uuid = normalise_scene_uuid(str(scene_uuid))
        suffix = scene_variant_suffix(variant)
        stems = [f"clipgt-{bare_uuid}{suffix}", f"{bare_uuid}{suffix}"]
        if suffix:
            stems.extend((f"clipgt-{bare_uuid}", bare_uuid))
        for stem in dict.fromkeys(stems):
            candidate = scene_dir / f"{stem}.usdz"
            if candidate.is_file():
                return candidate
    archives = sorted(scene_dir.glob("*.usdz"))
    if scene_uuid is None and len(archives) == 1:
        return archives[0]
    raise FileNotFoundError(
        f"No matching OmniDreams game scene archive found in {scene_dir}."
    )


def _resolve_camera_name(requested: str, available: dict[str, int]) -> str:
    candidates = (
        requested,
        requested.replace("_", ":"),
        requested.replace(":", "_"),
    )
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"Camera {requested!r} is unavailable; choose one of {sorted(available)}."
    )


def _read_prompt(path: Path, *, override: str | None) -> str:
    if override:
        return override
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if "/" not in name and name.startswith("prompt") and name.endswith(".txt")
        )
        return "" if not names else archive.read(names[0]).decode("utf-8").strip()


def _read_first_frame(
    path: Path, *, camera_name: str, width: int, height: int
) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        spellings = {
            camera_name,
            camera_name.replace(":", "_"),
            camera_name.replace("_", ":"),
        }
        names = [
            name
            for name in archive.namelist()
            if any(name.startswith(f"frames/{spelling}/") for spelling in spellings)
            and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not names:
            names = [
                name
                for name in archive.namelist()
                if "/" not in name and name.startswith("first_image")
            ]
        if not names:
            raise FileNotFoundError("Scene archive contains no usable first frame.")
        selected = sorted(names, key=_frame_sort_key)[0]
        with Image.open(io.BytesIO(archive.read(selected))) as image:
            rgb = image.convert("RGB").resize(
                (width, height), resample=Image.Resampling.BILINEAR
            )
            return np.asarray(rgb, dtype=np.uint8)


def _frame_sort_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (2**63 - 1, name)


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


__all__ = ["LudusGameScene"]
