# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-safe tests for optional engine and game YAML settings."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from crazy_robotaxi import runtime_cli
from crazy_robotaxi.cli import build_parser as build_game_parser
from crazy_robotaxi.game_settings import game_settings_from_args
from omnidreams_game_engine.engine_settings import engine_settings_from_args
from omnidreams_game_engine.yaml_config import StrictConfigError

pytestmark = pytest.mark.ci_cpu

_CONFIG_ROOT = Path(__file__).parents[1] / "crazy_robotaxi" / "configs"


def _write_yaml(path: Path, values: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_packaged_example_configs_are_valid() -> None:
    """Keep both checked-in examples synchronized with their v1 schemas."""
    args = runtime_cli.build_parser().parse_args(
        [
            "--engine-config",
            str(_CONFIG_ROOT / "example_engine_config.yaml"),
            "--game-config",
            str(_CONFIG_ROOT / "example_game_config.yaml"),
        ]
    )

    engine = engine_settings_from_args(args)
    game = game_settings_from_args(args)

    assert engine.rendering.raster.line_width_px == pytest.approx(12.0)
    assert engine.rendering.bev.width == 1024
    assert game.effects.visual_flare_enabled is False
    assert game.taxi_game.vehicle.curb_forward_momentum_retention == pytest.approx(0.85)


def test_omitted_configs_use_typed_internal_defaults() -> None:
    args = runtime_cli.build_parser().parse_args([])

    engine = engine_settings_from_args(args)
    game = game_settings_from_args(args)

    assert engine.rendering.raster.width == 1280
    assert engine.rendering.bev.enabled
    assert game.mode == "taxi"
    assert game.taxi_game.vehicle.max_steer_rad == pytest.approx(0.69)


def test_partial_engine_yaml_and_explicit_cli_precedence(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "engine.yaml",
        {
            "schema_version": 1,
            "rendering": {"bev": {"enabled": False, "width": 320, "height_m": 90.0}},
        },
    )
    args = runtime_cli.build_parser().parse_args(
        [
            "--engine-config",
            str(path),
            "--bev",
            "--bev-resolution",
            "640x480",
        ]
    )

    settings = engine_settings_from_args(args)

    assert settings.rendering.bev.enabled
    assert (settings.rendering.bev.width, settings.rendering.bev.height) == (640, 480)
    assert settings.rendering.bev.height_m == pytest.approx(90.0)


def test_engine_yaml_covers_launch_presentation_wheel_and_runtime(
    tmp_path: Path,
) -> None:
    path = _write_yaml(
        tmp_path / "engine.yaml",
        {
            "schema_version": 1,
            "map": {"directory": "maps", "preload_maps": True},
            "world_model": {"offload_text_encoder": True},
            "presentation": {
                "hud_enabled": False,
                "stream_jpeg_quality": 70,
                "stream_scale": 0.5,
            },
            "wheel": {"enabled": False, "profiles_dir": "wheel-profiles"},
            "runtime": {"cuda_visible_devices": "1", "stop_after_chunks": 4},
        },
    )
    args = build_game_parser().parse_args(
        ["--engine-config", str(path), "--hud", "--wheel", "--stream-scale", "0.75"]
    )

    settings = engine_settings_from_args(args)

    assert settings.map.directory == (tmp_path / "maps").resolve()
    assert settings.map.preload_maps
    assert settings.world_model.offload_text_encoder
    assert settings.presentation.hud_enabled
    assert settings.presentation.stream_jpeg_quality == 70
    assert settings.presentation.stream_scale == pytest.approx(0.75)
    assert settings.wheel.enabled
    assert settings.wheel.profiles_dir == (tmp_path / "wheel-profiles").resolve()
    assert settings.runtime.cuda_visible_devices == "1"
    assert settings.runtime.stop_after_chunks == 4


def test_partial_game_yaml_and_explicit_cli_precedence(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "game.yaml",
        {
            "schema_version": 1,
            "effects": {"visual_flare_enabled": True},
            "vehicle": {"max_accel_mps2": 7.5},
            "live_edit": {
                "coins": {"enabled": True, "spacing_m": 31.0},
                "style": {
                    "lora_checkpoint": "checkpoints/style.pt",
                    "skins": [{"name": "sepia", "prompt": "A sepia world."}],
                },
                "weather": {"guidance_scale": 3.0},
            },
        },
    )
    args = runtime_cli.build_parser().parse_args(
        [
            "--game-config",
            str(path),
            "--no-visual-flare",
            "--no-live-edit-coins",
            "--live-edit-style",
        ]
    )

    settings = game_settings_from_args(args)

    assert not settings.effects.visual_flare_enabled
    assert settings.taxi_game.vehicle.max_accel_mps2 == pytest.approx(7.5)
    assert not settings.live_edit.coins.enabled
    assert settings.live_edit.coins.spacing_m == pytest.approx(31.0)
    assert settings.live_edit.style.skins[0].name == "sepia"
    assert settings.live_edit.style.enabled
    assert (
        settings.live_edit.style.lora_checkpoint
        == (tmp_path / "checkpoints/style.pt").resolve()
    )
    assert settings.live_edit.weather.guidance_scale == pytest.approx(3.0)


def test_yaml_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "engine.yaml",
        {
            "schema_version": 1,
            "map": {"path": "maps/city.robotaxi.yaml"},
            "world_model": {"manifest": "models/world.yaml"},
        },
    )
    args = runtime_cli.build_parser().parse_args(["--engine-config", str(path)])

    settings = engine_settings_from_args(args)

    assert settings.map.path == (tmp_path / "maps/city.robotaxi.yaml").resolve()
    assert settings.world_model.manifest == (tmp_path / "models/world.yaml").resolve()


def test_game_yaml_can_keep_inactive_mode_settings(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "game.yaml",
        {
            "schema_version": 1,
            "mode": "taxi",
            "race": {"course": "downtown", "times_path": "scores/races.csv"},
        },
    )
    args = runtime_cli.build_parser().parse_args(["--game-config", str(path)])

    settings = game_settings_from_args(args)

    assert settings.mode == "taxi"
    assert settings.race.course == "downtown"
    assert settings.race.times_path == (tmp_path / "scores/races.csv").resolve()


def test_yaml_overrides_environment_and_cli_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVE_EDIT_CORRECTOR_MODE", "unfused")
    path = _write_yaml(
        tmp_path / "game.yaml",
        {"schema_version": 1, "live_edit": {"style": {"corrector_mode": "fused"}}},
    )
    yaml_args = runtime_cli.build_parser().parse_args(["--game-config", str(path)])
    cli_args = runtime_cli.build_parser().parse_args(
        [
            "--game-config",
            str(path),
            "--live-edit-corrector-mode",
            "off",
        ]
    )

    assert game_settings_from_args(yaml_args).live_edit.style.corrector_mode == "fused"
    assert game_settings_from_args(cli_args).live_edit.style.corrector_mode == "off"


def test_unknown_and_secret_engine_keys_are_rejected(tmp_path: Path) -> None:
    typo = _write_yaml(
        tmp_path / "typo.yaml",
        {"schema_version": 1, "rendering": {"bev": {"widht": 320}}},
    )
    secret = _write_yaml(
        tmp_path / "secret.yaml",
        {"schema_version": 1, "presentation": {"stream_token": "secret"}},
    )

    with pytest.raises(StrictConfigError, match="widht"):
        engine_settings_from_args(
            runtime_cli.build_parser().parse_args(["--engine-config", str(typo)])
        )
    with pytest.raises(StrictConfigError, match="stream_token"):
        engine_settings_from_args(
            runtime_cli.build_parser().parse_args(["--engine-config", str(secret)])
        )


def test_explicitly_missing_config_is_an_error(tmp_path: Path) -> None:
    args = runtime_cli.build_parser().parse_args(
        ["--engine-config", str(tmp_path / "missing.yaml")]
    )

    with pytest.raises(StrictConfigError, match="does not exist"):
        engine_settings_from_args(args)
