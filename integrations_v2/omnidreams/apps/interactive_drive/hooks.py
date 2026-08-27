# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OmniDreams hooks for scene-driving applications."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from clipgt2v import ClipGT2VApplicationHooks
from clipgt2v.interactive_drive.backends.base import RenderBackend
from clipgt2v.interactive_drive.backends.raster import RasterRenderBackend
from clipgt2v.interactive_drive.backends.world_model import WorldModelRenderBackend
from clipgt2v.interactive_drive.config import AppConfig
from omnidreams.apps.interactive_drive.model import (
    FlashdreamsWorldModelSession,
    OmnidreamsWorldModelRuntime,
)
from omnidreams.impl.pipeline import OmnidreamsPipelineConfig
from omnidreams.impl.scenes import hf_hub_download_scene

DEFAULT_INTERACTIVE_DRIVE_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"


def _create_omnidreams_backend(
    config: AppConfig,
    *,
    pipeline_config: OmnidreamsPipelineConfig,
) -> RenderBackend:
    if config.backend == "raster":
        return RasterRenderBackend(
            config.chunk,
            config.raster,
            bev=config.bev,
            vehicle=config.vehicle,
        )
    runtime = OmnidreamsWorldModelRuntime(
        pipeline_config=pipeline_config,
        resolution_wh=config.raster.resolution_wh,
        fps=config.chunk.fps,
        num_frames_per_block=config.chunk.chunk_frames,
        device=config.world_model_device,
        seed_for_every_rollout=config.world_model_seed,
        synthetic_model=config.world_model_synthetic,
        debug_condition_frame_dir=config.world_model_debug_condition_frame_dir,
    )
    return WorldModelRenderBackend(
        model_config=runtime,
        chunk=config.chunk,
        raster=config.raster,
        bev=config.bev,
        vehicle=config.vehicle,
        offload_text_encoder=config.world_model_offload_text_encoder,
        postprocess=config.postprocess,
        session_factory=FlashdreamsWorldModelSession,
    )


def create_omnidreams_application_hooks(
    pipeline_config: OmnidreamsPipelineConfig,
) -> ClipGT2VApplicationHooks:
    """Bind scene-driving hooks to one model-owned pipeline config."""
    return ClipGT2VApplicationHooks(
        backend_factory=partial(
            _create_omnidreams_backend,
            pipeline_config=pipeline_config,
        ),
    )


def resolve_default_interactive_drive_scene() -> Path:
    """Download and return the default scene-driving scene."""
    return hf_hub_download_scene(DEFAULT_INTERACTIVE_DRIVE_SCENE_UUID)
