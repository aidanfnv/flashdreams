# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import omnidreams.apps.interactive_drive.model as adapter_module
import pytest
import torch
from clipgt2v.interactive_drive.config import WorldModelProfileConfig
from omnidreams.apps.interactive_drive.model import (
    FlashdreamsWorldModelSession,
    OmnidreamsWorldModelRuntime,
    _build_pipeline_config,
    _select_config_name,
)
from omnidreams.config import (
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)
from omnidreams.impl.synthetic_fixture import (
    SyntheticWorldModelAssets,
)

from flashdreams.infra.postprocess import VideoPostprocessChainConfig
from flashdreams.infra.video_output import LazyRGBFrame

pytestmark = pytest.mark.ci_cpu


class _FakePipeline:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.initialize_calls: list[dict[str, object]] = []
        self.initialize_from_embeddings_calls: list[dict[str, object]] = []
        self.precompute_calls: list[dict[str, object]] = []
        self.generate_calls: list[dict[str, object]] = []
        self.finalize_calls: list[tuple[int, object]] = []
        self.release_calls = 0

    def get_num_frames(self, autoregressive_index: int) -> int:
        return 5 if autoregressive_index == 0 else 8

    def initialize_cache(self, **kwargs: object) -> str:
        self.initialize_calls.append(kwargs)
        return "cache"

    def initialize_cache_from_embeddings(self, **kwargs: object) -> str:
        self.initialize_from_embeddings_calls.append(kwargs)
        return "cache"

    def precompute_embeddings(self, **kwargs: object) -> dict[str, torch.Tensor | None]:
        self.precompute_calls.append(kwargs)
        return {
            "text_embeddings": torch.ones((1, 1, 2, 3), dtype=torch.float32),
            "image_embeddings": torch.ones((1, 1, 1, 2, 2, 2), dtype=torch.float32),
            "negative_text_embeddings": None,
        }

    def release_oneshot_encoders(self) -> None:
        self.release_calls += 1

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_calls.append(kwargs)
        frame_count = self.get_num_frames(int(kwargs["autoregressive_index"]))
        return torch.zeros((1, 1, frame_count, 3, 2, 3), dtype=torch.float32)

    def finalize(self, autoregressive_index: int, cache: object) -> None:
        self.finalize_calls.append((autoregressive_index, cache))


class _FakeSyntheticPipeline(_FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(synthetic_text_max_length=7)
        self.decoder = SimpleNamespace(spatial_compression_ratio=8)
        network = SimpleNamespace(
            use_crossattn_projection=True,
            crossattn_proj_in_channels=11,
            crossattn_emb_channels=13,
            in_channels=5,
        )
        transformer_config = SimpleNamespace(
            network=network,
            batch_shape=(1,),
            num_views=1,
            dtype=torch.float32,
            requires_negative_text_embeddings=False,
        )
        transformer = SimpleNamespace(config=transformer_config)
        self.diffusion_model = SimpleNamespace(transformer=transformer)
        self.V_size = 1


def _runtime(
    *,
    perf: bool = False,
) -> OmnidreamsWorldModelRuntime:
    return OmnidreamsWorldModelRuntime(
        pipeline_config=(
            OMNIDREAMS_PERF_PIPELINE_CONFIG if perf else OMNIDREAMS_PIPELINE_CONFIG
        ),
        resolution_wh=(1280, 704),
        fps=30,
        num_frames_per_block=8,
        device="cpu",
    )


def _contains_hf_url(value: object) -> bool:
    if isinstance(value, str):
        return "huggingface.co" in value
    if isinstance(value, (list, tuple, set)):
        return any(_contains_hf_url(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_hf_url(key) or _contains_hf_url(item)
            for key, item in value.items()
        )
    if hasattr(value, "__dict__"):
        return any(_contains_hf_url(item) for item in vars(value).values())
    return False


def test_select_config_name_uses_only_public_clipgt2v_slugs() -> None:
    assert _select_config_name(_runtime()) == "omnidreams"
    assert _select_config_name(_runtime(perf=True)) == "omnidreams-perf"


def test_build_pipeline_config_uses_selected_perf_config() -> None:
    config = _build_pipeline_config(
        _runtime(perf=True),
        profile=WorldModelProfileConfig(),
    )
    transformer_config = config.diffusion_model.transformer

    assert transformer_config.skip_finalize_kv_cache is True
    assert transformer_config.native_dit_acceleration == "required"
    assert transformer_config.native_dit_backend == "fp8_kvcache_cudnn"
    assert transformer_config.native_dit_attention_backend == "cudnn"


def test_build_pipeline_config_synthetic_swaps_only_weight_sources(
    monkeypatch,
    tmp_path,
) -> None:
    assets = SyntheticWorldModelAssets(
        encoder_checkpoint_path=tmp_path / "synthetic_lightvae_encoder.safetensors",
        decoder_checkpoint_path=tmp_path / "synthetic_lighttae_decoder.safetensors",
    )
    assets.encoder_checkpoint_path.touch()
    assets.decoder_checkpoint_path.touch()

    def fake_assets(*_args: object, **_kwargs: object) -> SyntheticWorldModelAssets:
        return assets

    monkeypatch.setattr(
        adapter_module,
        "build_synthetic_world_model_assets",
        fake_assets,
    )

    runtime = replace(
        _runtime(perf=True),
        synthetic_model=True,
    )
    real = _build_pipeline_config(
        replace(runtime, synthetic_model=False),
        profile=WorldModelProfileConfig(),
    )
    synthetic = _build_pipeline_config(runtime, profile=WorldModelProfileConfig())

    real_transformer = real.diffusion_model.transformer
    synthetic_transformer = synthetic.diffusion_model.transformer
    assert synthetic_transformer.checkpoint_path is None
    assert synthetic.text_encoder is None
    assert synthetic.image_encoder is None
    assert synthetic.encoder.checkpoint_path == str(assets.encoder_checkpoint_path)
    assert synthetic.decoder.checkpoint_path == str(assets.decoder_checkpoint_path)
    assert synthetic.decoder.state_dict_transform is None
    assert synthetic.synthetic_text_max_length == real.text_encoder.max_length

    for field in (
        "compile_network",
        "use_cuda_graph",
        "skip_finalize_kv_cache",
        "native_dit_acceleration",
        "native_dit_backend",
        "native_dit_attention_backend",
    ):
        assert getattr(synthetic_transformer, field) == getattr(real_transformer, field)
    assert synthetic.encoder.use_compile == real.encoder.use_compile
    assert synthetic.encoder.use_cuda_graph == real.encoder.use_cuda_graph
    assert synthetic.decoder.use_compile == real.decoder.use_compile
    assert synthetic.decoder.use_cuda_graph == real.decoder.use_cuda_graph
    assert not _contains_hf_url(synthetic)


def test_session_uses_flashdreams_pipeline_for_rollout() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _runtime(),
        pipeline_factory=lambda runtime, profile: fake_pipeline,
    )
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]
    next_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(8)]

    first = session.start(initial_rgb, first_condition_frames, "demo prompt")
    assert len(first) == 5
    assert len(fake_pipeline.initialize_calls) == 1
    assert fake_pipeline.initialize_calls[0]["text"] == [["demo prompt"]]
    assert tuple(fake_pipeline.initialize_calls[0]["image"].shape) == (1, 1, 1, 3, 2, 3)
    assert tuple(fake_pipeline.generate_calls[0]["hdmap"].shape) == (1, 1, 5, 3, 2, 3)
    assert fake_pipeline.generate_calls[0]["autoregressive_index"] == 0
    assert fake_pipeline.generate_calls[0]["cache"] == "cache"

    second = session.continue_generation(next_condition_frames)
    assert len(second) == 8
    assert fake_pipeline.finalize_calls == [(0, "cache")]
    assert fake_pipeline.generate_calls[1]["autoregressive_index"] == 1

    session.close()
    assert fake_pipeline.finalize_calls == [(0, "cache"), (1, "cache")]


def test_session_postprocesses_local_frames_and_supports_live_toggle(
    monkeypatch,
) -> None:
    streams: list[object] = []

    class _FakePostprocessStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.calls: list[int] = []
            self.finished = False
            self.last_process_stats = None
            streams.append(self)

        def process(
            self, output: torch.Tensor, *, autoregressive_index: int
        ) -> torch.Tensor:
            self.calls.append(autoregressive_index)
            return output

        def finish(self) -> None:
            self.finished = True

    monkeypatch.setattr(
        adapter_module, "VideoPostprocessStream", _FakePostprocessStream
    )
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _runtime(),
        pipeline_factory=lambda runtime, profile: fake_pipeline,
        postprocess=VideoPostprocessChainConfig(preset="fake-preset"),
    )
    session.warmup_model()
    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_conditions = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]
    next_conditions = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(8)]

    session.start(initial_rgb, first_conditions, "demo prompt")
    first_stream = streams[0]
    assert first_stream.calls == [0]
    assert first_stream.kwargs["output_layout"] == "bvtchw"
    assert first_stream.kwargs["fps"] == session.runtime.fps
    assert "collect_output" not in first_stream.kwargs
    assert "move_to_cpu" not in first_stream.kwargs

    session.set_postprocess_enabled(False)
    assert first_stream.finished is True
    session.continue_generation(next_conditions)
    assert len(streams) == 1

    session.set_postprocess_enabled(True)
    session.continue_generation(next_conditions)
    assert len(streams) == 2
    assert streams[1].calls == [2]


def test_session_synthetic_model_initializes_cache_from_synthetic_embeddings() -> None:
    fake_pipeline = _FakeSyntheticPipeline()
    runtime = replace(_runtime(), synthetic_model=True, resolution_wh=(64, 32))
    session = FlashdreamsWorldModelSession(
        runtime,
        offload_text_encoder=True,
        pipeline_factory=lambda runtime, profile: fake_pipeline,
    )
    assert session.can_prewarm is True
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "demo prompt")

    assert fake_pipeline.initialize_calls == []
    assert fake_pipeline.precompute_calls == []
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 1
    call = fake_pipeline.initialize_from_embeddings_calls[0]
    assert tuple(call["text_embeddings"].shape) == (1, 1, 7, 11)
    assert tuple(call["image_embeddings"].shape) == (1, 1, 1, 5, 4, 8)
    assert call["negative_text_embeddings"] is None
    assert call["view_names"] == ["camera_front_wide_120fov"]


def test_session_synchronizes_generated_frame_events_before_return(monkeypatch) -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _runtime(),
        pipeline_factory=lambda runtime, profile: fake_pipeline,
    )
    session.warmup_model()
    sync_calls: list[list[object]] = []

    def fake_sync(frames: list[object]) -> None:
        sync_calls.append(frames)

    monkeypatch.setattr(adapter_module, "_synchronize_cuda_frame_event", fake_sync)

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]
    next_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(8)]

    first = session.start(initial_rgb, first_condition_frames, "demo prompt")
    second = session.continue_generation(next_condition_frames)

    assert sync_calls == [first, second]


def test_lazy_rgb_frame_exposes_tensor_before_host_materialization() -> None:
    frames = torch.arange(2 * 2 * 3 * 3, dtype=torch.uint8).reshape(2, 2, 3, 3)
    lazy = LazyRGBFrame(frames, frame_index=1)

    tensor = lazy.to_cuda_tensor()

    assert torch.equal(tensor, frames[1])
    assert lazy.to_cuda_event() is None
    assert np.array_equal(lazy.to_numpy(), frames[1].numpy())


def test_session_offload_reuses_precomputed_embeddings_after_reset() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _runtime(),
        offload_text_encoder=True,
        pipeline_factory=lambda runtime, profile: fake_pipeline,
    )
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "demo prompt")
    session.reset()
    session.start(initial_rgb, first_condition_frames, "demo prompt")

    assert fake_pipeline.initialize_calls == []
    assert len(fake_pipeline.precompute_calls) == 1
    assert fake_pipeline.release_calls == 1
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 2
    assert fake_pipeline.initialize_from_embeddings_calls[0]["view_names"] == [
        "camera_front_wide_120fov"
    ]
    assert (
        fake_pipeline.initialize_from_embeddings_calls[1]["text_embeddings"]
        is (fake_pipeline.initialize_from_embeddings_calls[0]["text_embeddings"])
    )

    session.close()


def test_session_offload_reruns_embeddings_after_scene_conditioning_reset() -> None:
    fake_pipeline = _FakePipeline()
    session = FlashdreamsWorldModelSession(
        _runtime(),
        offload_text_encoder=True,
        pipeline_factory=lambda runtime, profile: fake_pipeline,
    )
    session.warmup_model()

    initial_rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    first_condition_frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(5)]

    session.start(initial_rgb, first_condition_frames, "clear prompt")
    session.reset(clear_precomputed_embeddings=True)
    session.start(initial_rgb, first_condition_frames, "snow prompt")

    assert len(fake_pipeline.precompute_calls) == 2
    assert fake_pipeline.precompute_calls[0]["text"] == [["clear prompt"]]
    assert fake_pipeline.precompute_calls[1]["text"] == [["snow prompt"]]
    assert len(fake_pipeline.initialize_from_embeddings_calls) == 2
    assert (
        fake_pipeline.initialize_from_embeddings_calls[1]["text_embeddings"]
        is not fake_pipeline.initialize_from_embeddings_calls[0]["text_embeddings"]
    )

    session.close()
