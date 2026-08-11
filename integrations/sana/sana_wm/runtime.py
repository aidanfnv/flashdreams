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

"""SANA-WM runtime API adapter and replay session implementation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

from flashdreams.core.distributed import init as init_distributed
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import runner_artifact_path, write_runner_stats
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    Mp4VideoOutputTarget,
    OutputArtifact,
)
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.runtime.inputs import TimeWindow
from sana_wm.camera import default_intrinsics_vec4
from sana_wm.conditioning import (
    SanaWMI2VConditioningRequest,
    SanaWMStreamingI2VConditioningRequest,
    streaming_chunk_boundaries,
)
from sana_wm.constants import (
    DEFAULT_FPS,
    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)
from sana_wm.decoder import SanaWMDecodedVideo

SANA_WM_MODEL_ID = "sana-wm"
DEFAULT_SANA_WM_PRESET = "sana-wm-bidirectional"

FIELD_PROMPT = "prompt"
FIELD_NEGATIVE_PROMPT = "negative_prompt"
FIELD_GLOBAL_CONDITIONING_FRAME = "global_conditioning_frame"
FIELD_CAMERA_TRAJECTORY_C2W = "camera_trajectory_c2w"
FIELD_CAMERA_INTRINSICS_VEC4 = "camera_intrinsics_vec4"

SanaWMVariant = Literal["bidirectional", "streaming"]
PipelineFactory = Callable[[Any, str], Any]

_VIDEO_LAYOUT = "tchw"
_INSTALL_HINT = "Install the SANA-WM plugin: pip install flashdreams-sana."


@dataclass(frozen=True, kw_only=True, slots=True)
class SanaWMSessionInputs:
    """Session-global SANA-WM conditioning state."""

    prompt: str
    image: Any
    poses_c2w: np.ndarray
    intrinsics_vec4: np.ndarray
    negative_prompt: str = ""

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses_c2w, dtype=np.float32)
        intrinsics = np.asarray(self.intrinsics_vec4, dtype=np.float32)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                "SANA-WM camera_trajectory_c2w must have shape [F, 4, 4]; "
                f"got {poses.shape}."
            )
        if intrinsics.ndim != 2 or intrinsics.shape[1:] != (4,):
            raise ValueError(
                "SANA-WM camera_intrinsics_vec4 must have shape [F, 4]; "
                f"got {intrinsics.shape}."
            )
        if intrinsics.shape[0] != poses.shape[0]:
            raise ValueError(
                "SANA-WM camera_intrinsics_vec4 length must match "
                f"camera_trajectory_c2w length; got {intrinsics.shape[0]} and "
                f"{poses.shape[0]}."
            )
        object.__setattr__(self, "prompt", " ".join(self.prompt.split()))
        object.__setattr__(self, "negative_prompt", str(self.negative_prompt))
        object.__setattr__(self, "poses_c2w", poses)
        object.__setattr__(self, "intrinsics_vec4", intrinsics)


@dataclass(frozen=True, kw_only=True, slots=True)
class SanaWMReplayRuntimeOptions:
    """Construction and rollout knobs for the SANA-WM replay runtime."""

    pipeline_config: Any
    variant: SanaWMVariant = "bidirectional"
    pipeline: Any | None = None
    pipeline_factory: PipelineFactory | None = None
    output_layout: VideoTensorLayout = _VIDEO_LAYOUT
    num_frames: int = 161
    fps: int = DEFAULT_FPS
    steps: int = 60
    cfg_scale: float = 5.0
    flow_shift: float | None = None
    sampling_algo: str = "flow_euler_ltx"
    seed: int = 42
    save_stage1: bool = False
    refiner_seed: int = 42
    sink_size: int = 1
    num_frame_per_block: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    refiner_block_size: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    refiner_kv_max_frames: int | None = None
    intrinsics_hfov_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.output_layout != _VIDEO_LAYOUT:
            raise ValueError(
                "SANA-WM runtime currently emits layout "
                f"{_VIDEO_LAYOUT!r}; got {self.output_layout!r}."
            )
        if self.variant not in {"bidirectional", "streaming"}:
            raise ValueError(f"Unsupported SANA-WM variant: {self.variant!r}.")
        if self.num_frames <= 0:
            raise ValueError("SanaWMReplayRuntimeOptions.num_frames must be > 0.")
        if self.fps <= 0:
            raise ValueError("SanaWMReplayRuntimeOptions.fps must be > 0.")
        if self.steps <= 0:
            raise ValueError("SanaWMReplayRuntimeOptions.steps must be > 0.")
        if self.num_frame_per_block <= 0:
            raise ValueError(
                "SanaWMReplayRuntimeOptions.num_frame_per_block must be > 0."
            )
        if self.refiner_block_size <= 0:
            raise ValueError(
                "SanaWMReplayRuntimeOptions.refiner_block_size must be > 0."
            )


class SanaWMModelAdapter:
    """Model adapter exposing SANA-WM through ``flashdreams.runtime``."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[..., InferenceRuntime] | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory or SanaWMReplayRuntime
        self._pipeline_factory = pipeline_factory

    @property
    def model_id(self) -> str:
        return SANA_WM_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            description="SANA-WM I2V rollout-wide model inputs.",
            global_conditioning_fields=(
                InputField(
                    name=FIELD_PROMPT,
                    input_modality="text",
                    frequency_consumed="once",
                    description="Positive text prompt for the rollout.",
                ),
                InputField(
                    name=FIELD_NEGATIVE_PROMPT,
                    required=False,
                    input_modality="text",
                    frequency_consumed="once",
                    description="Optional classifier-free guidance negative prompt.",
                ),
                InputField(
                    name=FIELD_GLOBAL_CONDITIONING_FRAME,
                    input_modality="image",
                    frequency_consumed="once",
                    description="First RGB frame, already resolved for model input.",
                ),
                InputField(
                    name=FIELD_CAMERA_TRAJECTORY_C2W,
                    input_modality="c2w_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[F,4,4]", "coordinates": "opencv_c2w"},
                    description="Full rollout camera-to-world trajectory.",
                ),
                InputField(
                    name=FIELD_CAMERA_INTRINSICS_VEC4,
                    required=False,
                    input_modality="intrinsics_vec4_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[F,4]"},
                    description="Optional full rollout fx/fy/cx/cy intrinsics.",
                ),
            ),
        )

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return None

    def default_input_mapping(self) -> IdentityInputMapping:
        """Return the pass-through mapping for fixed SANA-WM model inputs."""
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"SANA-WM adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self.runtime_options(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self._runtime_factory(
            config=config,
            options=self.runtime_options(config),
        )

    def preset_id(self, config: InferenceConfig | None) -> str:
        return (
            DEFAULT_SANA_WM_PRESET
            if config is None or config.preset_id is None
            else config.preset_id
        )

    def pipeline_config(self, config: InferenceConfig) -> Any:
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        preset_id = self.preset_id(config)
        from sana_wm.config import PIPELINE_CONFIGS  # noqa: PLC0415

        try:
            return PIPELINE_CONFIGS[preset_id]
        except KeyError as exc:
            supported = ", ".join(sorted(PIPELINE_CONFIGS))
            raise ValueError(
                f"Unsupported SANA-WM preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def runtime_options(self, config: InferenceConfig) -> SanaWMReplayRuntimeOptions:
        raw_options = config.runtime_options
        output_layout = raw_options.get("output_layout", _VIDEO_LAYOUT)
        if output_layout != _VIDEO_LAYOUT:
            raise ValueError(
                "SANA-WM runtime currently supports output_layout='tchw' only; "
                f"got {output_layout!r}."
            )
        variant = _variant_from_value(
            raw_options.get("variant"),
            fallback=self.preset_id(config),
            pipeline_config=self.pipeline_config(config),
        )
        return SanaWMReplayRuntimeOptions(
            pipeline_config=self.pipeline_config(config),
            variant=variant,
            pipeline=raw_options.get("pipeline"),
            pipeline_factory=self._pipeline_factory,
            output_layout=cast(VideoTensorLayout, output_layout),
            num_frames=int(raw_options.get("num_frames", 161)),
            fps=int(raw_options.get("fps", DEFAULT_FPS)),
            steps=int(raw_options.get("steps", raw_options.get("step", 60))),
            cfg_scale=float(raw_options.get("cfg_scale", 5.0)),
            flow_shift=_optional_float(raw_options.get("flow_shift")),
            sampling_algo=_resolve_sampling_algo(
                raw_options.get("sampling_algo", "flow_euler_ltx")
            ),
            seed=int(
                config.seed if config.seed is not None else raw_options.get("seed", 42)
            ),
            save_stage1=bool(raw_options.get("save_stage1", False)),
            refiner_seed=int(raw_options.get("refiner_seed", 42)),
            sink_size=int(raw_options.get("sink_size", 1)),
            num_frame_per_block=int(
                raw_options.get(
                    "num_frame_per_block",
                    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
                )
            ),
            refiner_block_size=int(
                raw_options.get(
                    "refiner_block_size",
                    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
                )
            ),
            refiner_kv_max_frames=(
                None
                if raw_options.get("refiner_kv_max_frames") is None
                else int(raw_options["refiner_kv_max_frames"])
            ),
            intrinsics_hfov_deg=float(raw_options.get("intrinsics_hfov_deg", 90.0)),
        )


class SanaWMReplayRuntime:
    """Heavyweight SANA-WM runtime consumed by the standard loop."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        options: SanaWMReplayRuntimeOptions,
    ) -> None:
        self.config = config
        self.options = options
        if _is_torchrun_env() and not dist.is_initialized():
            init_distributed()

        if dist.is_initialized():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
            device = f"cuda:{self.local_rank}"
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            device = _resolve_runtime_device(config.device, self.local_rank)

        self.is_rank_zero = self.global_rank == 0
        self.device = torch.device(device)
        if options.pipeline is not None:
            self.pipeline = options.pipeline
            self._owns_pipeline = False
        else:
            factory = options.pipeline_factory or _default_pipeline_factory
            self.pipeline = factory(options.pipeline_config, device)
            self._owns_pipeline = True

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        session_inputs = session_inputs_from_inference_input(inputs, self.options)
        return SanaWMReplaySession(
            pipeline=self.pipeline,
            session_inputs=session_inputs,
            options=self.options,
            device=self.device,
            is_rank_zero=self.is_rank_zero,
        )

    def close(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if self._owns_pipeline and pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()
            del self.pipeline
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class SanaWMReplaySession:
    """One SANA-WM rollout driven by fixed global conditioning inputs."""

    def __init__(
        self,
        *,
        pipeline: Any,
        session_inputs: SanaWMSessionInputs,
        options: SanaWMReplayRuntimeOptions,
        device: torch.device,
        is_rank_zero: bool,
    ) -> None:
        if options.sampling_algo != "flow_euler_ltx":
            raise ValueError(
                "SANA-WM requires sampling_algo='flow_euler_ltx'; "
                f"got {options.sampling_algo!r}."
            )
        self.pipeline = pipeline
        self.inputs = session_inputs
        self.options = options
        self.device = device
        self.is_rank_zero = is_rank_zero
        self._closed = False
        self._step_index = 0
        self._output_frame_start = 0
        self._cache: Any | None = None
        self._chunk_boundaries = _chunk_boundaries(options)
        self._initialize_cache()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if dist.is_initialized():
            dist.barrier()

    def next_step_request(self) -> StepRequest | None:
        if self._closed:
            return None
        if self._step_index >= self._total_steps:
            return None
        return StepRequest(
            step_index=self._step_index,
            inference_input_schema=InferenceInputSchema(
                description="SANA-WM replay step has no additional model inputs."
            ),
            metadata={
                "variant": self.options.variant,
                "input_frame_count": 1,
            },
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        if self._closed:
            raise RuntimeError("SANA-WM replay session is closed.")
        if self._cache is None:
            raise RuntimeError("SANA-WM replay session is not initialized.")
        if self._step_index >= self._total_steps:
            raise RuntimeError("SANA-WM replay session has no remaining steps.")

        step_index = self._step_index
        if self.is_rank_zero:
            logger.info(
                "SANA-WM runtime step {} ({})",
                step_index,
                self.options.variant,
            )

        start_t = time.perf_counter()
        with torch.inference_mode():
            decoded = self.pipeline.generate(
                step_index,
                self._cache,
                input=self._conditioning_request(),
            )
            stats = self.pipeline.finalize(step_index, self._cache)
        metrics = _numeric_metrics(stats)
        metrics.setdefault("model_step_s", time.perf_counter() - start_t)
        result = decoded_video_to_step_result(
            decoded,
            step_index=step_index,
            metrics=metrics,
            frame_start=self._output_frame_start,
            fps=self.options.fps,
        )
        self._step_index += 1
        self._output_frame_start += result.frame_count
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        if inputs is not None:
            self.inputs = session_inputs_from_inference_input(inputs, self.options)
        self._step_index = 0
        self._output_frame_start = 0
        self._initialize_cache()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache = None

    @property
    def _total_steps(self) -> int:
        if self.options.variant == "bidirectional":
            return 1
        return len(self._chunk_boundaries) - 1

    def _initialize_cache(self) -> None:
        decoder_context: dict[str, Any] = {
            "prompt": self.inputs.prompt,
            "fps": self.options.fps,
            "save_stage1": self.options.save_stage1,
            "refiner_seed": self.options.refiner_seed,
            "sink_size": self.options.sink_size,
        }
        if self.options.variant == "streaming":
            decoder_context.update(
                {
                    "block_size": self.options.refiner_block_size,
                    "refiner_kv_max_frames": self.options.refiner_kv_max_frames,
                }
            )
        self._cache = self.pipeline.initialize_cache(
            decoder_context=decoder_context,
        )

    def _conditioning_request(
        self,
    ) -> SanaWMI2VConditioningRequest | SanaWMStreamingI2VConditioningRequest:
        kwargs: dict[str, Any] = {
            "image": self.inputs.image,
            "prompt": self.inputs.prompt,
            "poses_c2w": self.inputs.poses_c2w,
            "intrinsics_vec4": self.inputs.intrinsics_vec4,
            "num_frames": self.options.num_frames,
            "fps": self.options.fps,
            "steps": self.options.steps,
            "cfg_scale": self.options.cfg_scale,
            "flow_shift": self.options.flow_shift,
            "seed": self.options.seed,
            "negative_prompt": self.inputs.negative_prompt,
        }
        if self.options.variant == "streaming":
            return SanaWMStreamingI2VConditioningRequest(
                **kwargs,
                num_frame_per_block=self.options.num_frame_per_block,
            )
        return SanaWMI2VConditioningRequest(**kwargs)


@dataclass(slots=True)
class SanaWMRunnerOutputTarget:
    """Runner-compatible MP4/stats output target for SANA-WM replay results."""

    output_stream: VideoOutputStream
    output_dir: Path
    runner_name: str
    fps: int | float
    install_hint: str = _INSTALL_HINT
    enabled: bool = True
    _opened: bool = False
    _mp4_target: Mp4VideoOutputTarget | None = None
    _stage1_target: Mp4VideoOutputTarget | None = None

    def open(self) -> None:
        video_path = runner_artifact_path(self.output_dir, self.runner_name, "mp4")
        self._mp4_target = Mp4VideoOutputTarget(
            output_path=video_path,
            fps=self.fps,
            output_layout=self.output_stream.output_layout,
            install_hint=self.install_hint,
            enabled=self.enabled,
        )
        self._mp4_target.open()
        self._opened = True

    def write(self, result: StepResult) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed SANA-WM output target.")
        if result.layout != self.output_stream.output_layout:
            raise ValueError(
                "SanaWMRunnerOutputTarget received layout "
                f"{result.layout!r}; expected {self.output_stream.output_layout!r}."
            )
        if self._mp4_target is None:
            raise RuntimeError("SANA-WM MP4 target is not open.")
        processed = self.output_stream.process(
            result.video_chunk,
            autoregressive_index=result.step_index,
            metrics=result.metrics,
            metadata=_main_metadata(result.metadata),
            output_window=result.output_window,
        )
        self._mp4_target.write(processed)
        if not self.enabled:
            return
        stage1 = result.metadata.get("stage1_video_chunk")
        if stage1 is None:
            return
        if not isinstance(stage1, torch.Tensor):
            raise TypeError("SANA-WM stage1_video_chunk metadata must be a tensor.")
        if self._stage1_target is None:
            stage1_path = runner_artifact_path(
                self.output_dir,
                f"{self.runner_name}_stage1",
                "mp4",
            )
            self._stage1_target = Mp4VideoOutputTarget(
                output_path=stage1_path,
                fps=self.fps,
                output_layout=_VIDEO_LAYOUT,
                install_hint=self.install_hint,
                enabled=self.enabled,
            )
            self._stage1_target.open()
        self._stage1_target.write(
            StepResult.from_video_chunk(
                step_index=result.step_index,
                video_chunk=stage1,
                layout=_VIDEO_LAYOUT,
                output_window=result.output_window,
                metrics=result.metrics,
                metadata={"stage1": True},
            )
        )

    def close(self) -> tuple[OutputArtifact, ...]:
        self._opened = False
        artifacts: list[OutputArtifact] = []
        target = self._mp4_target
        self._mp4_target = None
        if target is not None:
            tail = self.output_stream.finish()
            if tail is not None:
                target.write(tail)
            artifacts.extend(target.close())
        stage1_target = self._stage1_target
        self._stage1_target = None
        if stage1_target is not None:
            artifacts.extend(stage1_target.close())
        if artifacts:
            logger.info(
                "[{}] wrote video -> {}",
                self.runner_name,
                Path(artifacts[0].uri).resolve(),
            )
        stats_history = (
            artifacts[0].metadata.get("stats_history", ()) if artifacts else ()
        )
        if stats_history:
            stats_path = write_runner_stats(
                self.output_dir,
                self.runner_name,
                list(stats_history),
            )
            logger.info(
                "[{}] wrote per-AR-step stats -> {}",
                self.runner_name,
                stats_path.resolve(),
            )
            artifacts.append(
                OutputArtifact(kind="application/json", uri=str(stats_path.resolve()))
            )
        return tuple(artifacts)


def inference_config_from_runner_config(
    runner_config: Any,
    *,
    device: str,
    pipeline: Any | None = None,
    num_frames: int | None = None,
) -> InferenceConfig:
    """Build the SANA-WM runtime config from a runner config."""
    variant = _variant_from_value(
        None,
        fallback=str(runner_config.pipeline.name),
        pipeline_config=runner_config.pipeline,
    )
    runtime_options: dict[str, Any] = {
        "pipeline_config": runner_config.pipeline,
        "variant": variant,
        "output_layout": runner_config.postprocess_output_layout or _VIDEO_LAYOUT,
        "num_frames": int(
            num_frames if num_frames is not None else runner_config.num_frames
        ),
        "fps": int(runner_config.fps),
        "steps": int(runner_config.step),
        "cfg_scale": float(runner_config.cfg_scale),
        "flow_shift": runner_config.flow_shift,
        "sampling_algo": _resolve_sampling_algo(runner_config.sampling_algo),
        "seed": int(runner_config.seed),
        "save_stage1": bool(runner_config.save_stage1),
        "refiner_seed": int(runner_config.refiner_seed),
        "sink_size": int(runner_config.sink_size),
        "intrinsics_hfov_deg": float(runner_config.intrinsics_hfov_deg),
    }
    if pipeline is not None:
        runtime_options["pipeline"] = pipeline
    for name in (
        "num_frame_per_block",
        "num_cached_blocks",
        "denoising_step_list",
        "refiner_block_size",
        "refiner_kv_max_frames",
        "no_sink_token",
    ):
        if hasattr(runner_config, name):
            runtime_options[name] = getattr(runner_config, name)
    return InferenceConfig(
        model_id=SANA_WM_MODEL_ID,
        preset_id=str(runner_config.pipeline.name),
        device=device,
        seed=int(runner_config.seed),
        runtime_options=runtime_options,
    )


def inference_input_from_prepared_inputs(
    *,
    prompt: str,
    image: Any,
    poses_c2w: np.ndarray,
    intrinsics_vec4: np.ndarray | None,
    negative_prompt: str = "",
) -> InferenceInput:
    """Encode prepared SANA-WM rollout inputs into ``InferenceInput``."""
    payload: dict[str, Any] = {
        FIELD_PROMPT: prompt,
        FIELD_GLOBAL_CONDITIONING_FRAME: image,
        FIELD_CAMERA_TRAJECTORY_C2W: np.asarray(poses_c2w, dtype=np.float32),
    }
    if negative_prompt:
        payload[FIELD_NEGATIVE_PROMPT] = negative_prompt
    if intrinsics_vec4 is not None:
        payload[FIELD_CAMERA_INTRINSICS_VEC4] = np.asarray(
            intrinsics_vec4,
            dtype=np.float32,
        )
    return InferenceInput(global_conditioning=payload)


def session_inputs_from_inference_input(
    inputs: InferenceInput,
    options: SanaWMReplayRuntimeOptions,
) -> SanaWMSessionInputs:
    """Decode and validate session-global SANA-WM inputs."""
    schema = SanaWMModelAdapter().inference_input_schema
    missing = schema.missing_global_conditioning(inputs)
    if missing:
        raise ValueError(f"SANA-WM session inputs missing required fields: {missing}.")
    gc = inputs.global_conditioning
    poses = np.asarray(gc[FIELD_CAMERA_TRAJECTORY_C2W], dtype=np.float32)
    if poses.shape[0] != options.num_frames:
        raise ValueError(
            "SANA-WM camera_trajectory_c2w length must match runtime num_frames; "
            f"got {poses.shape[0]} and {options.num_frames}."
        )
    intrinsics = gc.get(FIELD_CAMERA_INTRINSICS_VEC4)
    if intrinsics is None:
        image = gc[FIELD_GLOBAL_CONDITIONING_FRAME]
        size = getattr(image, "size", None)
        if not (
            isinstance(size, tuple)
            and len(size) == 2
            and all(isinstance(item, int) for item in size)
        ):
            raise ValueError(
                "SANA-WM session inputs without camera_intrinsics_vec4 require "
                "a PIL-like global_conditioning_frame with integer .size."
            )
        intrinsics = default_intrinsics_vec4(
            cast(tuple[int, int], size),
            options.num_frames,
            hfov_deg=options.intrinsics_hfov_deg,
        )
    return SanaWMSessionInputs(
        prompt=str(gc[FIELD_PROMPT]),
        negative_prompt=str(gc.get(FIELD_NEGATIVE_PROMPT, "")),
        image=gc[FIELD_GLOBAL_CONDITIONING_FRAME],
        poses_c2w=poses,
        intrinsics_vec4=np.asarray(intrinsics, dtype=np.float32),
    )


def decoded_video_to_step_result(
    decoded: object,
    *,
    step_index: int,
    metrics: Mapping[str, float | int] | None = None,
    frame_start: int = 0,
    fps: int | float = DEFAULT_FPS,
) -> StepResult:
    """Convert a SANA-WM decoded-video payload into a layout-aware result."""
    if not isinstance(decoded, SanaWMDecodedVideo):
        raise TypeError(
            "SANA-WM pipeline decoder returned "
            f"{type(decoded).__name__}, expected SanaWMDecodedVideo."
        )
    video = video_hwc_uint8_to_tchw_normalized(decoded.video_hwc)
    metadata: dict[str, Any] = {
        "source_format": "uint8_hwc",
    }
    if decoded.stage1_video_hwc is not None:
        metadata["stage1_video_chunk"] = video_hwc_uint8_to_tchw_normalized(
            decoded.stage1_video_hwc
        )
        metadata["stage1_layout"] = _VIDEO_LAYOUT
    output_window = TimeWindow(
        start_s=frame_start / float(fps),
        end_s=(frame_start + video.shape[0]) / float(fps),
    )
    return StepResult.from_video_chunk(
        step_index=step_index,
        video_chunk=video,
        layout=_VIDEO_LAYOUT,
        output_window=output_window,
        metrics=metrics,
        metadata=metadata,
    )


def video_hwc_uint8_to_tchw_normalized(video_hwc: np.ndarray) -> torch.Tensor:
    """Convert ``uint8`` ``[T,H,W,C]`` frames to ``float32`` ``tchw`` in ``[-1,1]``."""
    video = np.asarray(video_hwc)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"SANA-WM decoded video must have shape [T, H, W, 3]; got {video.shape}."
        )
    if video.dtype != np.uint8:
        raise TypeError(f"SANA-WM decoded video must be uint8; got {video.dtype}.")
    tensor = torch.from_numpy(np.ascontiguousarray(video)).to(dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor / 127.5 - 1.0


def _default_pipeline_factory(pipeline_config: Any, device: str) -> Any:
    return pipeline_config.setup().to(device=device).eval()


def _variant_from_value(
    value: Any,
    *,
    fallback: str,
    pipeline_config: Any,
) -> SanaWMVariant:
    raw = value
    if raw is None:
        raw = getattr(pipeline_config, "name", None) or fallback
    text = str(raw)
    if text in {"bidirectional", "sana-wm-bidirectional"}:
        return "bidirectional"
    if text in {"streaming", "sana-wm-streaming"}:
        return "streaming"
    if "streaming" in text:
        return "streaming"
    if "bidirectional" in text:
        return "bidirectional"
    raise ValueError(f"Cannot infer SANA-WM variant from {text!r}.")


def _chunk_boundaries(options: SanaWMReplayRuntimeOptions) -> tuple[int, ...]:
    if options.variant == "bidirectional":
        return (0, 1)
    latent_frames = ((options.num_frames - 1) // SANA_WM_VAE_TEMPORAL_COMPRESSION) + 1
    return streaming_chunk_boundaries(latent_frames, options.num_frame_per_block)


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _resolve_runtime_device(device: str | None, local_rank: int) -> str:
    raw = device or "cuda"
    if raw == "auto":
        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
        return "cpu"
    if raw == "cuda" and torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return raw


def _resolve_sampling_algo(value: Any) -> str:
    text = str(value)
    if text == "auto":
        return "flow_euler_ltx"
    return text


def _main_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(name): value
        for name, value in metadata.items()
        if not str(name).startswith("stage1_")
    }


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


__all__ = [
    "DEFAULT_SANA_WM_PRESET",
    "FIELD_CAMERA_INTRINSICS_VEC4",
    "FIELD_CAMERA_TRAJECTORY_C2W",
    "FIELD_GLOBAL_CONDITIONING_FRAME",
    "FIELD_NEGATIVE_PROMPT",
    "FIELD_PROMPT",
    "PipelineFactory",
    "SANA_WM_MODEL_ID",
    "SanaWMModelAdapter",
    "SanaWMReplayRuntime",
    "SanaWMReplayRuntimeOptions",
    "SanaWMReplaySession",
    "SanaWMRunnerOutputTarget",
    "SanaWMSessionInputs",
    "SanaWMVariant",
    "decoded_video_to_step_result",
    "inference_config_from_runner_config",
    "inference_input_from_prepared_inputs",
    "session_inputs_from_inference_input",
    "video_hwc_uint8_to_tchw_normalized",
]
