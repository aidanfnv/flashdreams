# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Strict optional Crazy Robotaxi configuration loading."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Literal, cast

from omnidreams_game_engine.cli_args import arg_was_explicit
from omnidreams_game_engine.yaml_config import (
    StrictConfigError,
    load_yaml_mapping,
    overlay_dataclass,
    require_mapping,
    require_version,
)

from crazy_robotaxi.driving import TaxiVehicleConfig
from crazy_robotaxi.game import TaxiGameConfig
from crazy_robotaxi.live_edit.config import (
    LiveEditConfig,
    apply_live_edit_cli,
)

_RUNTIME_GAME_FIELDS = {
    "enabled",
    "seed",
    "high_scores_path",
    "alignment_diagnostics_enabled",
}
_RULE_FIELDS = (
    {item.name for item in fields(TaxiGameConfig)} - _RUNTIME_GAME_FIELDS - {"vehicle"}
)
_VEHICLE_FIELDS = {item.name for item in fields(TaxiVehicleConfig)}
_VEHICLE_BOOL_FIELDS = {
    "speed_limit_enabled",
    "actor_collision_enabled",
    "static_collision_enabled",
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "mode",
    "effects",
    "rules",
    "vehicle",
    "taxi",
    "race",
    "live_edit",
    "diagnostics",
}


@dataclass(frozen=True)
class GameEffectsSettings:
    """Game-directed visual effects."""

    visual_flare_enabled: bool = False
    """Whether collisions may trigger the full-screen visual flare."""


@dataclass(frozen=True)
class TaxiSessionSettings:
    """Taxi-mode session and persistence settings."""

    seed: int | None = None
    """Optional deterministic fare-layout seed."""

    high_scores_path: Path | None = None
    """Taxi leaderboard path; ``None`` uses the cache-directory default."""


@dataclass(frozen=True)
class RaceSessionSettings:
    """Race-mode selection and persistence settings."""

    course: str | None = None
    """Race-course identifier; ``None`` selects the map's first course."""

    times_path: Path | None = None
    """Race leaderboard path; ``None`` uses the cache-directory default."""


@dataclass(frozen=True)
class GameDiagnosticsSettings:
    """Optional Crazy Robotaxi diagnostic outputs."""

    alignment_directory: Path | None = None
    """Optional frame-alignment diagnostic output directory."""


@dataclass(frozen=True)
class CrazyRobotaxiSettings:
    """Complete durable game configuration."""

    mode: Literal["taxi", "race"] = "taxi"
    """Gameplay mode selected for the session."""

    effects: GameEffectsSettings = field(default_factory=GameEffectsSettings)
    """Game-directed visual effects."""

    taxi_game: TaxiGameConfig = field(default_factory=TaxiGameConfig)
    """Taxi rules and player-vehicle configuration."""

    taxi: TaxiSessionSettings = field(default_factory=TaxiSessionSettings)
    """Taxi session and persistence settings."""

    race: RaceSessionSettings = field(default_factory=RaceSessionSettings)
    """Race selection and persistence settings."""

    live_edit: LiveEditConfig = field(default_factory=LiveEditConfig)
    """Map-context prompting and live-edit ability settings."""

    diagnostics: GameDiagnosticsSettings = field(
        default_factory=GameDiagnosticsSettings
    )
    """Crazy Robotaxi diagnostic outputs."""


def load_game_settings(path: Path) -> TaxiGameConfig:
    """Load the rules and vehicle portion of a partial game YAML.

    Args:
        path: Game configuration path.

    Returns:
        Taxi rules and player-vehicle configuration.
    """
    return load_crazy_robotaxi_settings(path).taxi_game


def load_crazy_robotaxi_settings(
    path: Path,
    *,
    base: CrazyRobotaxiSettings | None = None,
) -> CrazyRobotaxiSettings:
    """Strictly overlay a partial game YAML onto ``base``.

    Args:
        path: Game configuration path.
        base: Lower-precedence settings; ``None`` uses typed defaults.

    Returns:
        Resolved game settings.

    Raises:
        StrictConfigError: The YAML or merged settings are invalid.
    """
    config_path = path.expanduser().resolve()
    doc = load_yaml_mapping(config_path)
    require_version(doc, "game")
    unknown = sorted(doc.keys() - _TOP_LEVEL_FIELDS)
    if unknown:
        raise StrictConfigError(f"game has unknown keys: {', '.join(unknown)}")
    settings = base or CrazyRobotaxiSettings()
    base_dir = config_path.parent
    if "mode" in doc:
        settings = overlay_dataclass(
            settings, {"mode": doc["mode"]}, "game", base_dir=base_dir
        )
    for yaml_name, field_name in (
        ("effects", "effects"),
        ("taxi", "taxi"),
        ("race", "race"),
        ("live_edit", "live_edit"),
        ("diagnostics", "diagnostics"),
    ):
        if yaml_name not in doc:
            continue
        nested = overlay_dataclass(
            getattr(settings, field_name),
            require_mapping(doc[yaml_name], f"game.{yaml_name}"),
            f"game.{yaml_name}",
            base_dir=base_dir,
        )
        settings = replace(settings, **{field_name: nested})
    taxi_game = settings.taxi_game
    if "rules" in doc:
        raw_rules = require_mapping(doc["rules"], "game.rules")
        _reject_unknown(raw_rules, _RULE_FIELDS, "game.rules")
        taxi_game = overlay_dataclass(
            taxi_game, raw_rules, "game.rules", base_dir=base_dir
        )
    if "vehicle" in doc:
        raw_vehicle = require_mapping(doc["vehicle"], "game.vehicle")
        _reject_unknown(raw_vehicle, _VEHICLE_FIELDS, "game.vehicle")
        vehicle = overlay_dataclass(
            taxi_game.vehicle, raw_vehicle, "game.vehicle", base_dir=base_dir
        )
        taxi_game = replace(taxi_game, vehicle=vehicle)
    taxi_game = replace(
        taxi_game,
        enabled=True,
        seed=settings.taxi.seed,
        alignment_diagnostics_enabled=(
            settings.diagnostics.alignment_directory is not None
        ),
        **(
            {"high_scores_path": settings.taxi.high_scores_path}
            if settings.taxi.high_scores_path is not None
            else {}
        ),
    )
    settings = replace(settings, taxi_game=taxi_game)
    _validate_game_settings(settings)
    return settings


def game_settings_from_args(args: argparse.Namespace) -> CrazyRobotaxiSettings:
    """Merge code/environment defaults, optional YAML, and explicit CLI.

    Args:
        args: Parsed Crazy Robotaxi arguments with explicit-option metadata.

    Returns:
        Resolved settings, also cached on and published through ``args``.

    Raises:
        StrictConfigError: The selected YAML or merged settings are invalid.
    """
    cached = getattr(args, "_crazy_robotaxi_settings", None)
    if cached is not None:
        return cast(CrazyRobotaxiSettings, cached)
    taxi_game = TaxiGameConfig(
        enabled=True,
        seed=getattr(args, "taxi_seed", None),
        **(
            {"high_scores_path": args.taxi_highscores.expanduser()}
            if getattr(args, "taxi_highscores", None) is not None
            else {}
        ),
    )
    base = CrazyRobotaxiSettings(
        mode=getattr(args, "game_mode", "taxi"),
        effects=GameEffectsSettings(
            visual_flare_enabled=bool(getattr(args, "visual_flare", False))
        ),
        taxi_game=taxi_game,
        taxi=TaxiSessionSettings(
            seed=getattr(args, "taxi_seed", None),
            high_scores_path=getattr(args, "taxi_highscores", None),
        ),
        race=RaceSessionSettings(
            course=getattr(args, "race_course", None),
            times_path=getattr(args, "race_times", None),
        ),
        live_edit=_default_live_edit_from_args(args),
        diagnostics=GameDiagnosticsSettings(
            alignment_directory=getattr(args, "taxi_alignment_diagnostics", None)
        ),
    )
    path = getattr(args, "game_config", None)
    settings = (
        load_crazy_robotaxi_settings(Path(path), base=base)
        if path is not None
        else base
    )
    if path is not None:
        args.game_config = Path(path).expanduser().resolve()
    settings = _apply_explicit_cli(settings, args)
    _validate_game_settings(settings)
    _hydrate_namespace(args, settings)
    args._crazy_robotaxi_settings = settings
    args._game_settings = settings.taxi_game
    args._live_edit_settings = settings.live_edit
    return settings


def _apply_explicit_cli(
    settings: CrazyRobotaxiSettings, args: argparse.Namespace
) -> CrazyRobotaxiSettings:
    if arg_was_explicit(args, "game_mode"):
        settings = replace(settings, mode=args.game_mode)
    if arg_was_explicit(args, "visual_flare"):
        settings = replace(
            settings,
            effects=replace(
                settings.effects, visual_flare_enabled=bool(args.visual_flare)
            ),
        )
    taxi = settings.taxi
    taxi_game = settings.taxi_game
    if arg_was_explicit(args, "taxi_seed"):
        taxi = replace(taxi, seed=args.taxi_seed)
        taxi_game = replace(taxi_game, seed=args.taxi_seed)
    if arg_was_explicit(args, "taxi_highscores"):
        taxi = replace(taxi, high_scores_path=args.taxi_highscores)
        if args.taxi_highscores is not None:
            taxi_game = replace(
                taxi_game, high_scores_path=args.taxi_highscores.expanduser()
            )
    race = settings.race
    if arg_was_explicit(args, "race_course"):
        race = replace(race, course=args.race_course)
    if arg_was_explicit(args, "race_times"):
        race = replace(race, times_path=args.race_times)
    diagnostics = settings.diagnostics
    if arg_was_explicit(args, "taxi_alignment_diagnostics"):
        diagnostics = replace(
            diagnostics, alignment_directory=args.taxi_alignment_diagnostics
        )
    live_edit = apply_live_edit_cli(settings.live_edit, args, explicit_only=True)
    taxi_game = replace(
        taxi_game,
        enabled=True,
        seed=taxi.seed,
        alignment_diagnostics_enabled=(diagnostics.alignment_directory is not None),
        **(
            {"high_scores_path": taxi.high_scores_path}
            if taxi.high_scores_path is not None
            else {}
        ),
    )
    return replace(
        settings,
        taxi=taxi,
        taxi_game=taxi_game,
        race=race,
        diagnostics=diagnostics,
        live_edit=live_edit,
    )


def _default_live_edit_from_args(args: argparse.Namespace) -> LiveEditConfig:
    """Return live-edit defaults after applying environment-backed values."""
    base = LiveEditConfig()
    return replace(
        base,
        style=replace(base.style, corrector_mode=str(args.live_edit_corrector_mode)),
        perf_log_every_frames=int(args.live_edit_perf_log),
    )


def _hydrate_namespace(
    args: argparse.Namespace, settings: CrazyRobotaxiSettings
) -> None:
    args.game_mode = settings.mode
    args.visual_flare = settings.effects.visual_flare_enabled
    args.taxi_seed = settings.taxi.seed
    args.taxi_highscores = settings.taxi.high_scores_path
    args.race_course = settings.race.course
    args.race_times = settings.race.times_path
    args.taxi_alignment_diagnostics = settings.diagnostics.alignment_directory


def _reject_unknown(values: dict[str, object], allowed: set[str], context: str) -> None:
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise StrictConfigError(f"{context} has unknown keys: {', '.join(unknown)}")


def _validate_game_settings(settings: CrazyRobotaxiSettings) -> None:
    config = settings.taxi_game
    for name in _RULE_FIELDS:
        if getattr(config, name) < 0:
            raise StrictConfigError(f"game.rules.{name} must be non-negative")
    for name in _VEHICLE_FIELDS - _VEHICLE_BOOL_FIELDS:
        if getattr(config.vehicle, name) < 0:
            raise StrictConfigError(f"game.vehicle.{name} must be non-negative")
    if config.fare_min_route_distance_m > config.fare_max_route_distance_m:
        raise StrictConfigError(
            "game.rules.fare_min_route_distance_m must not exceed fare_max_route_distance_m"
        )
    if config.min_time_s > config.max_time_s:
        raise StrictConfigError("game.rules.min_time_s must not exceed max_time_s")
    for name in (
        "wheel_base_m",
        "max_speed_mps",
        "aabb_length_m",
        "aabb_width_m",
        "aabb_height_m",
        "input_dt_cap_s",
        "speed_taper_knee_fraction",
    ):
        if getattr(config.vehicle, name) <= 0.0:
            raise StrictConfigError(f"game.vehicle.{name} must be positive")
    for name in (
        "speed_taper_knee_fraction",
        "speed_taper_low_floor",
        "speed_taper_high_floor",
        "curb_collision_restitution",
        "curb_forward_momentum_retention",
    ):
        value = getattr(config.vehicle, name)
        if not 0.0 <= value <= 1.0:
            raise StrictConfigError(f"game.vehicle.{name} must be in [0, 1]")
