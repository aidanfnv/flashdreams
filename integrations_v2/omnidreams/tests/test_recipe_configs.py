# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe configuration and application binding checks for OmniDreams."""

from collections.abc import Callable
from pathlib import Path

import pytest
import tomli as tomllib
from clipgt2v import ClipGT2VApplication, ClipGT2VConfig
from interactive_drive import InteractiveDriveConfig
from omnidreams.apps.clipgt2v.adapter import (
    OMNIDREAMS_APPLICATION_DEFAULTS,
    OMNIDREAMS_PERF_APPLICATION_DEFAULTS,
    create_app,
    create_perf_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS,
    OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_app as create_interactive_drive_app,
)
from omnidreams.apps.interactive_drive.adapter import (
    create_perf_app as create_interactive_drive_perf_app,
)
from omnidreams.config import (
    OMNIDREAMS_CONFIGS,
    OMNIDREAMS_PERF_PIPELINE_CONFIG,
    OMNIDREAMS_PIPELINE_CONFIG,
)

from flashdreams.api_v2.application import IApplication

pytestmark = pytest.mark.ci_cpu


def test_pipeline_configs_are_keyed_by_name() -> None:
    """Expose exactly the two model-owned OmniDreams pipeline configs."""
    assert OMNIDREAMS_CONFIGS == {
        "omnidreams": OMNIDREAMS_PIPELINE_CONFIG,
        "omnidreams-perf": OMNIDREAMS_PERF_PIPELINE_CONFIG,
    }


def test_application_defaults_are_owned_by_each_adapter() -> None:
    """Keep demo-specific configuration beside each application factory."""
    assert OMNIDREAMS_APPLICATION_DEFAULTS.slug == "clipgt2v"
    assert OMNIDREAMS_PERF_APPLICATION_DEFAULTS.slug == "clipgt2v-perf"
    assert OMNIDREAMS_APPLICATION_DEFAULTS.width == 1280
    assert OMNIDREAMS_PERF_APPLICATION_DEFAULTS.width == 1168
    assert OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.slug == "interactive-drive"
    assert OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS.slug == "interactive-drive-perf"
    assert OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS.width == 1280
    assert OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS.width == 1168
    for defaults, pipeline_config in (
        (OMNIDREAMS_APPLICATION_DEFAULTS, OMNIDREAMS_PIPELINE_CONFIG),
        (OMNIDREAMS_PERF_APPLICATION_DEFAULTS, OMNIDREAMS_PERF_PIPELINE_CONFIG),
        (OMNIDREAMS_INTERACTIVE_DRIVE_DEFAULTS, OMNIDREAMS_PIPELINE_CONFIG),
        (
            OMNIDREAMS_INTERACTIVE_DRIVE_PERF_DEFAULTS,
            OMNIDREAMS_PERF_PIPELINE_CONFIG,
        ),
    ):
        assert defaults.pipeline_config is pipeline_config


@pytest.mark.parametrize(
    ("factory", "resolution_wh"),
    [
        (create_app, (1280, 704)),
        (create_perf_app, (1168, 640)),
        (create_interactive_drive_app, (1280, 704)),
        (create_interactive_drive_perf_app, (1168, 640)),
    ],
)
def test_each_application_owns_its_parsed_config(
    factory: Callable[[], IApplication],
    resolution_wh: tuple[int, int],
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.usdz"
    scene.touch()
    app = factory()

    app.init(["--scene", str(scene)])

    assert isinstance(app, ClipGT2VApplication)
    assert app._config is not None
    assert app._config.app.raster.resolution_wh == resolution_wh
    expected_type = (
        InteractiveDriveConfig
        if factory
        in {
            create_interactive_drive_app,
            create_interactive_drive_perf_app,
        }
        else ClipGT2VConfig
    )
    assert type(app._config) is expected_type


def test_pyproject_registers_model_owned_app_adapters() -> None:
    """Expose applications through nested adapters and no runner entry points."""
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(path.read_text())
    entry_points = project["project"]["entry-points"]

    assert "flashdreams.runner_configs" not in entry_points
    assert entry_points["flashdreams.applications_v2"] == {
        "interactive-drive-omnidreams": (
            "omnidreams.apps.interactive_drive.adapter:create_app"
        ),
        "interactive-drive-omnidreams-perf": (
            "omnidreams.apps.interactive_drive.adapter:create_perf_app"
        ),
        "clipgt2v-omnidreams": "omnidreams.apps.clipgt2v.adapter:create_app",
        "clipgt2v-omnidreams-perf": (
            "omnidreams.apps.clipgt2v.adapter:create_perf_app"
        ),
    }
