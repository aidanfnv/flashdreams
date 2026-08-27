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

"""Optional file-backed settings for the interactive driving engine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from omnidreams_game_engine.cli_args import arg_was_explicit
from omnidreams_game_engine.config import BevConfig, RasterConfig
from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    overlay_dataclass,
    require_version,
)

_SettingsT = TypeVar("_SettingsT")


@dataclass(frozen=True)
class MapLaunchSettings:
    """Map selection and cache behavior."""

    path: Path | None = None
    """Selected game-map path; ``None`` requires a later CLI or UI selection."""

    directory: Path | None = None
    """Directory searched for additional game maps."""

    camera: str = "camera_front_wide_120fov"
    """Camera identifier selected from the map's available views."""

    variant: str = "default"
    """Visual variant selected from the map."""

    prompt: str | None = None
    """Base prompt override; ``None`` uses the selected map variant's prompt."""

    force_recompile: bool = False
    """Whether to rebuild the selected map's compiled cache once."""

    preload_maps: bool = False
    """Whether to parse every map in :attr:`directory` during startup."""


@dataclass(frozen=True)
class WorldModelLaunchSettings:
    """World-model selection and optional runtime features."""

    backend: Literal["raster", "omnidreams"] = "raster"
    """Rendering backend used for the main camera."""

    manifest: Path | None = None
    """World-model manifest path; required by the OmniDreams backend."""

    synthetic_model: bool | None = None
    """Synthetic-model override; ``None`` defers to the manifest."""

    official_hdmap_dir: Path | None = None
    """Optional directory of diagnostic HD-map conditioning frames."""

    offload_text_encoder: bool = False
    """Whether to release one-shot text and image encoders after warmup."""

    postprocess_preset: str = ""
    """Registered video postprocessing preset; empty disables postprocessing."""

    hf_org: str | None = None
    """Hugging Face organization override; ``None`` defers to the environment."""


@dataclass(frozen=True)
class PresentationSettings:
    """Window, stream, and HUD presentation settings."""

    hud_enabled: bool = True
    """Whether to present the native HUD around the camera view."""

    stream_mjpeg: str | None = None
    """Optional MJPEG bind address used instead of a local window."""

    stream_jpeg_quality: int = 85
    """JPEG quality used by the MJPEG presenter."""

    stream_scale: float = 1.0
    """MJPEG output scale applied after full-resolution rendering."""

    control_assets_dir: Path | None = None
    """Optional wheel and pedal overlay asset directory."""

    def __post_init__(self) -> None:
        if not 1 <= self.stream_jpeg_quality <= 100:
            raise ValueError("stream_jpeg_quality must be in [1, 100]")
        if not 0.1 <= self.stream_scale <= 1.0:
            raise ValueError("stream_scale must be in [0.1, 1.0]")


@dataclass(frozen=True)
class WheelSettings:
    """Optional steering-wheel input configuration."""

    enabled: bool = True
    """Whether steering-wheel input is enabled."""

    profile: str = "auto"
    """Wheel profile name or automatic-selection marker."""

    profiles_dir: Path | None = None
    """Directory containing wheel profile YAML documents."""

    device: Path | None = None
    """Explicit evdev device path; ``None`` enables auto-detection."""

    steering_axis: int | None = None
    """Optional steering-axis override."""

    throttle_axis: int | None = None
    """Optional throttle-axis override."""

    brake_axis: int | None = None
    """Optional brake-axis override."""

    pedals_inverted: bool | None = None
    """Pedal inversion override; ``None`` uses the selected profile."""


@dataclass(frozen=True)
class EngineRuntimeSettings:
    """Operational and diagnostic engine controls."""

    cuda_visible_devices: str = "auto"
    """CUDA visibility override; ``auto`` preserves the process environment."""

    profile_world_model: bool = False
    """Whether to collect CUDA-event timings from the world-model pipeline."""

    stop_after_chunks: int | None = None
    """Optional final chunk index for bounded diagnostic runs."""

    def __post_init__(self) -> None:
        if self.stop_after_chunks is not None and self.stop_after_chunks < 0:
            raise ValueError("stop_after_chunks must be non-negative")


@dataclass(frozen=True)
class RenderingSettings:
    """Primary-camera and BEV rendering configuration."""

    raster: RasterConfig = field(default_factory=RasterConfig)
    """Primary-camera rasterization settings."""

    bev: BevConfig = field(default_factory=BevConfig)
    """Top-down map rasterization settings."""


@dataclass(frozen=True)
class EngineSettings:
    """Complete durable engine configuration."""

    map: MapLaunchSettings = field(default_factory=MapLaunchSettings)
    """Map selection and cache behavior."""

    world_model: WorldModelLaunchSettings = field(
        default_factory=WorldModelLaunchSettings
    )
    """World-model selection and runtime behavior."""

    rendering: RenderingSettings = field(default_factory=RenderingSettings)
    """Primary-camera and BEV rendering settings."""

    presentation: PresentationSettings = field(default_factory=PresentationSettings)
    """Window, stream, and HUD presentation settings."""

    wheel: WheelSettings = field(default_factory=WheelSettings)
    """Steering-wheel input settings."""

    runtime: EngineRuntimeSettings = field(default_factory=EngineRuntimeSettings)
    """Operational and diagnostic settings."""


def engine_settings_from_args(args: argparse.Namespace) -> EngineSettings:
    """Merge internal/environment defaults, optional YAML, and explicit CLI.

    Args:
        args: Parsed engine arguments with explicit-option tracking metadata.

    Returns:
        Resolved settings, also cached on and published through ``args``.

    Raises:
        StrictConfigError: The selected YAML or merged settings are invalid.
    """
    cached = getattr(args, "_engine_settings", None)
    if cached is not None:
        return cast(EngineSettings, cached)
    settings = _settings_from_namespace(args)
    path = getattr(args, "engine_config", None)
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        doc = load_yaml_mapping(config_path)
        require_version(doc, "engine")
        values = dict(doc)
        values.pop("schema_version")
        settings = overlay_dataclass(
            settings, values, "engine", base_dir=config_path.parent
        )
        args.engine_config = config_path
    settings = _apply_explicit_cli(settings, args)
    _validate_settings(settings)
    _hydrate_namespace(args, settings)
    args._engine_settings = settings
    return settings


def _settings_from_namespace(args: argparse.Namespace) -> EngineSettings:
    raster = RasterConfig(
        compute_device=getattr(args, "compute_device", "cuda"),
        sync_gpu_timing=bool(getattr(args, "sync_gpu_timing", False)),
    )
    bev = BevConfig()
    return EngineSettings(
        map=MapLaunchSettings(
            path=getattr(args, "scene", None),
            directory=getattr(args, "scene_dir", None),
            camera=getattr(args, "camera", "camera_front_wide_120fov"),
            variant=getattr(args, "variant", "default"),
            prompt=getattr(args, "prompt", None),
            force_recompile=bool(getattr(args, "force_map_recompile", False)),
            preload_maps=bool(getattr(args, "preload_maps", False)),
        ),
        world_model=WorldModelLaunchSettings(
            backend=getattr(args, "backend", "raster"),
            manifest=getattr(args, "manifest", None),
            synthetic_model=getattr(args, "synthetic_model", None),
            official_hdmap_dir=getattr(args, "official_hdmap_dir", None),
            offload_text_encoder=bool(getattr(args, "offload_text_encoder", False)),
            postprocess_preset=getattr(args, "postprocess_preset", ""),
            hf_org=getattr(args, "hf_org", None),
        ),
        rendering=RenderingSettings(raster=raster, bev=bev),
        presentation=PresentationSettings(
            hud_enabled=not bool(getattr(args, "no_hud", False)),
            stream_mjpeg=getattr(args, "stream_mjpeg", None),
            stream_jpeg_quality=int(getattr(args, "stream_jpeg_quality", 85)),
            stream_scale=float(getattr(args, "stream_scale", 1.0)),
            control_assets_dir=getattr(args, "control_assets_dir", None),
        ),
        wheel=WheelSettings(
            enabled=not bool(getattr(args, "no_wheel", False)),
            profile=getattr(args, "wheel_profile", "auto"),
            profiles_dir=getattr(args, "wheel_profiles_dir", None),
            device=getattr(args, "wheel_device", None),
            steering_axis=getattr(args, "wheel_steering_axis", None),
            throttle_axis=getattr(args, "wheel_throttle_axis", None),
            brake_axis=getattr(args, "wheel_brake_axis", None),
            pedals_inverted=getattr(args, "wheel_pedals_inverted", None),
        ),
        runtime=EngineRuntimeSettings(
            cuda_visible_devices=getattr(args, "cuda_visible_devices", "auto"),
            profile_world_model=bool(getattr(args, "profile_world_model", False)),
            stop_after_chunks=getattr(args, "stop_after_chunks", None),
        ),
    )


def _apply_explicit_cli(
    settings: EngineSettings, args: argparse.Namespace
) -> EngineSettings:
    map_settings = settings.map
    world_model = settings.world_model
    raster = settings.rendering.raster
    bev = settings.rendering.bev
    presentation = settings.presentation
    wheel = settings.wheel
    runtime = settings.runtime

    map_fields = {
        "scene": "path",
        "scene_dir": "directory",
        "camera": "camera",
        "variant": "variant",
        "prompt": "prompt",
        "force_map_recompile": "force_recompile",
        "preload_maps": "preload_maps",
    }
    world_fields = {
        "backend": "backend",
        "manifest": "manifest",
        "synthetic_model": "synthetic_model",
        "official_hdmap_dir": "official_hdmap_dir",
        "offload_text_encoder": "offload_text_encoder",
        "postprocess_preset": "postprocess_preset",
        "hf_org": "hf_org",
    }
    presentation_fields = {
        "stream_mjpeg": "stream_mjpeg",
        "stream_jpeg_quality": "stream_jpeg_quality",
        "stream_scale": "stream_scale",
        "control_assets_dir": "control_assets_dir",
    }
    wheel_fields = {
        "wheel_profile": "profile",
        "wheel_profiles_dir": "profiles_dir",
        "wheel_device": "device",
        "wheel_steering_axis": "steering_axis",
        "wheel_throttle_axis": "throttle_axis",
        "wheel_brake_axis": "brake_axis",
        "wheel_pedals_inverted": "pedals_inverted",
    }
    runtime_fields = {
        "cuda_visible_devices": "cuda_visible_devices",
        "profile_world_model": "profile_world_model",
        "stop_after_chunks": "stop_after_chunks",
    }
    map_settings = _replace_explicit(map_settings, args, map_fields)
    world_model = _replace_explicit(world_model, args, world_fields)
    presentation = _replace_explicit(presentation, args, presentation_fields)
    wheel = _replace_explicit(wheel, args, wheel_fields)
    runtime = _replace_explicit(runtime, args, runtime_fields)
    if arg_was_explicit(args, "no_hud"):
        presentation = replace(presentation, hud_enabled=not bool(args.no_hud))
    if arg_was_explicit(args, "no_wheel"):
        wheel = replace(wheel, enabled=not bool(args.no_wheel))
    if arg_was_explicit(args, "compute_device"):
        raster = replace(raster, compute_device=args.compute_device)
    if arg_was_explicit(args, "sync_gpu_timing"):
        raster = replace(raster, sync_gpu_timing=bool(args.sync_gpu_timing))
    if arg_was_explicit(args, "bev"):
        bev = replace(bev, enabled=bool(args.bev))
    if arg_was_explicit(args, "bev_resolution"):
        width, height = _parse_resolution(args.bev_resolution)
        bev = replace(bev, width=width, height=height)
    for arg_name, field_name in (
        ("bev_height_m", "height_m"),
        ("bev_fov_deg", "fov_deg"),
        ("bev_tilt_deg", "tilt_deg"),
    ):
        if arg_was_explicit(args, arg_name):
            bev = replace(bev, **{field_name: float(getattr(args, arg_name))})
    return replace(
        settings,
        map=map_settings,
        world_model=world_model,
        rendering=replace(settings.rendering, raster=raster, bev=bev),
        presentation=presentation,
        wheel=wheel,
        runtime=runtime,
    )


def _replace_explicit(
    base: _SettingsT,
    args: argparse.Namespace,
    mapping: dict[str, str],
) -> _SettingsT:
    updates = {
        field_name: getattr(args, arg_name)
        for arg_name, field_name in mapping.items()
        if arg_was_explicit(args, arg_name)
    }
    return cast(_SettingsT, replace(cast(Any, base), **updates))


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise StrictConfigError(f"expected WIDTHxHEIGHT, got {value!r}")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise StrictConfigError(f"resolution must contain integers: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise StrictConfigError("resolution dimensions must be positive")
    return width, height


def _hydrate_namespace(args: argparse.Namespace, settings: EngineSettings) -> None:
    args.scene = settings.map.path
    if hasattr(args, "scene_dir") or settings.map.directory is not None:
        args.scene_dir = settings.map.directory
    args.camera = settings.map.camera
    args.variant = settings.map.variant
    args.prompt = settings.map.prompt
    args.force_map_recompile = settings.map.force_recompile
    args.preload_maps = settings.map.preload_maps
    args.backend = settings.world_model.backend
    args.manifest = settings.world_model.manifest
    args.synthetic_model = settings.world_model.synthetic_model
    args.official_hdmap_dir = settings.world_model.official_hdmap_dir
    args.offload_text_encoder = settings.world_model.offload_text_encoder
    args.postprocess_preset = settings.world_model.postprocess_preset
    args.hf_org = settings.world_model.hf_org
    args.compute_device = settings.rendering.raster.compute_device
    args.sync_gpu_timing = settings.rendering.raster.sync_gpu_timing
    args.bev = settings.rendering.bev.enabled
    args.bev_resolution = (
        f"{settings.rendering.bev.width}x{settings.rendering.bev.height}"
    )
    args.bev_height_m = settings.rendering.bev.height_m
    args.bev_fov_deg = settings.rendering.bev.fov_deg
    args.bev_tilt_deg = settings.rendering.bev.tilt_deg
    args.no_hud = not settings.presentation.hud_enabled
    args.stream_mjpeg = settings.presentation.stream_mjpeg
    args.stream_jpeg_quality = settings.presentation.stream_jpeg_quality
    args.stream_scale = settings.presentation.stream_scale
    args.control_assets_dir = settings.presentation.control_assets_dir
    args.no_wheel = not settings.wheel.enabled
    args.wheel_profile = settings.wheel.profile
    args.wheel_profiles_dir = settings.wheel.profiles_dir
    args.wheel_device = settings.wheel.device
    args.wheel_steering_axis = settings.wheel.steering_axis
    args.wheel_throttle_axis = settings.wheel.throttle_axis
    args.wheel_brake_axis = settings.wheel.brake_axis
    args.wheel_pedals_inverted = settings.wheel.pedals_inverted
    args.cuda_visible_devices = settings.runtime.cuda_visible_devices
    args.profile_world_model = settings.runtime.profile_world_model
    args.stop_after_chunks = settings.runtime.stop_after_chunks


def _validate_settings(settings: EngineSettings) -> None:
    raster = settings.rendering.raster
    bev = settings.rendering.bev
    if (
        raster.width <= 0
        or raster.height <= 0
        or raster.triangle_raytrace_edge_samples <= 0
        or raster.perf_log_interval_frames <= 0
    ):
        raise StrictConfigError(
            "engine.rendering.raster integer fields must be positive"
        )
    for name in (
        "near_plane_m",
        "far_plane_m",
        "fog_start_m",
        "fog_end_m",
        "fog_power",
        "triangle_raytrace_distance_m",
        "lane_segment_interval_m",
        "polyline_segment_interval_m",
        "line_width_px",
        "pole_width_px",
        "dual_line_offset_m",
        "depth_clear_m",
    ):
        if getattr(raster, name) < 0.0:
            raise StrictConfigError(
                f"engine.rendering.raster.{name} must be non-negative"
            )
    if raster.near_plane_m < 0.0 or raster.near_plane_m >= raster.far_plane_m:
        raise StrictConfigError(
            "engine.rendering.raster.near_plane_m must be non-negative and less than far_plane_m"
        )
    if raster.fog_start_m < 0.0 or raster.fog_start_m >= raster.fog_end_m:
        raise StrictConfigError(
            "engine.rendering.raster.fog_start_m must be non-negative and less than fog_end_m"
        )
    if bev.width <= 0 or bev.height <= 0 or bev.height_m <= 0.0:
        raise StrictConfigError(
            "engine.rendering.bev dimensions and height_m must be positive"
        )
    if not 0.0 < bev.fov_deg < 180.0:
        raise StrictConfigError(
            "engine.rendering.bev.fov_deg must be between 0 and 180"
        )
